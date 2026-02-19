import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import torch


def sample_connected_regular_graph(n_nodes: int, degree: int, seed: int = 42) -> nx.Graph:
    trial_seed = seed
    while True:
        graph = nx.random_regular_graph(degree, n_nodes, seed=trial_seed)
        if nx.is_connected(graph):
            return graph
        trial_seed += 1


def cond_prob_states_unweighted(
    chosen_node: int,
    nodes_current_states: np.ndarray,
    adj: np.ndarray,
    affinity: np.ndarray,
) -> np.ndarray:
    n_states = affinity.shape[0]
    neighbors = np.where(adj[chosen_node] == 1)[0]
    neighbor_vals = nodes_current_states[neighbors]

    def state_product(state_num: int) -> float:
        val = 1.0
        for neighbor_state in neighbor_vals:
            val *= affinity[state_num, neighbor_state]
        return val

    probs = np.array([state_product(s) for s in range(n_states)], dtype=float)
    normalizer = probs.sum()
    if normalizer == 0:
        return np.ones(n_states, dtype=float) / n_states
    return probs / normalizer


def generate_trajectory(
    adj_true: np.ndarray,
    affinity: np.ndarray,
    t_steps: int,
    init_state: np.ndarray | None = None,
) -> np.ndarray:
    n_nodes = adj_true.shape[0]
    n_states = affinity.shape[0]
    trajectory = np.zeros((t_steps, n_nodes), dtype=np.int64)
    if init_state is not None:
        trajectory[0] = init_state

    for t in range(1, t_steps):
        x_curr = trajectory[t - 1].copy()
        target_node = np.random.randint(0, n_nodes)
        cond_prob_vector = cond_prob_states_unweighted(
            target_node, x_curr, adj_true, affinity
        )
        new_state = np.random.choice(np.arange(n_states), p=cond_prob_vector)
        x_curr[target_node] = new_state
        trajectory[t] = x_curr

    return trajectory


def expected_complete_log_likelihood(
    Phi: torch.Tensor,
    beta: torch.Tensor,
    x_prev: torch.Tensor,
    x_next: torch.Tensor,
    gamma: torch.Tensor,
) -> torch.Tensor:
    # phi_lookup[t, j, k] = Phi[k, x_j(t)]
    phi_lookup = Phi[:, x_prev].permute(1, 2, 0)
    logits = torch.einsum("ij,tjk->tik", beta, phi_lookup)
    pos_term = logits.gather(2, x_next.unsqueeze(-1)).squeeze(-1)
    log_partition = torch.logsumexp(logits, dim=2)
    return torch.sum(gamma * (pos_term - log_partition))


def e_step_gamma(
    x_prev: torch.Tensor,
    x_next: torch.Tensor,
    beta: torch.Tensor,
    Phi: torch.Tensor,
) -> torch.Tensor:
    device = x_prev.device
    dtype = Phi.dtype
    t_transitions, n_nodes = x_prev.shape
    gamma = torch.zeros((t_transitions, n_nodes), dtype=dtype, device=device)

    diff = x_prev.ne(x_next)
    diff_count = diff.sum(dim=1)

    one_change_rows = torch.where(diff_count == 1)[0]
    if one_change_rows.numel() > 0:
        changed_nodes = diff[one_change_rows].to(torch.int64).argmax(dim=1)
        gamma[one_change_rows, changed_nodes] = 1.0

    zero_change_rows = torch.where(diff_count == 0)[0]
    if zero_change_rows.numel() > 0:
        prev_zero = x_prev[zero_change_rows]
        phi_lookup = Phi[:, prev_zero].permute(1, 2, 0)
        logits = torch.einsum("ij,mjk->mik", beta, phi_lookup)
        probs = torch.softmax(logits, dim=2)
        stay_prob = probs.gather(2, prev_zero.unsqueeze(-1)).squeeze(-1)
        norm = stay_prob.sum(dim=1, keepdim=True)
        normalized = stay_prob / torch.clamp(norm, min=1e-12)
        uniform = torch.full_like(stay_prob, 1.0 / n_nodes)
        gamma[zero_change_rows] = torch.where(norm > 0, normalized, uniform)

    bad_rows = torch.where(diff_count > 1)[0]
    if bad_rows.numel() > 0:
        raise ValueError(
            "Found transitions with more than one changed node. "
            "The model assumes single-site updates."
        )

    return gamma


def optimize_phi_given_fixed_beta(
    Phi_init: torch.Tensor,
    beta_fixed: torch.Tensor,
    x_prev: torch.Tensor,
    x_next: torch.Tensor,
    gamma: torch.Tensor,
    lr: float = 5e-2,
    max_steps: int = 150,
    tol: float = 1e-5,
    symmetrize: bool = True,
) -> tuple[torch.Tensor, float]:
    Phi = Phi_init.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([Phi], lr=lr)
    prev_q = None
    q_value = None

    for _ in range(max_steps):
        optimizer.zero_grad()
        q = expected_complete_log_likelihood(Phi, beta_fixed, x_prev, x_next, gamma)
        loss = -q
        loss.backward()
        optimizer.step()

        if symmetrize:
            with torch.no_grad():
                Phi.copy_(0.5 * (Phi + Phi.t()))

        q_value = float(q.item())
        if prev_q is not None and abs(q_value - prev_q) <= tol * (1.0 + abs(prev_q)):
            break
        prev_q = q_value

    return Phi.detach(), float(q_value if q_value is not None else 0.0)


def optimize_beta_global(
    beta_init: torch.Tensor,
    Phi_fixed: torch.Tensor,
    x_prev: torch.Tensor,
    x_next: torch.Tensor,
    gamma: torch.Tensor,
    lambda_reg: float = 0.15,
    lr: float = 1e-2,
    max_steps: int = 300,
    tol: float = 1e-5,
) -> tuple[torch.Tensor, float]:
    device = beta_init.device
    dtype = beta_init.dtype
    n_nodes = beta_init.shape[0]
    mask = torch.ones((n_nodes, n_nodes), device=device, dtype=dtype) - torch.eye(
        n_nodes, device=device, dtype=dtype
    )

    beta = beta_init.detach().clone().requires_grad_(True)
    prev_obj = None
    obj_value = None

    for _ in range(max_steps):
        if beta.grad is not None:
            beta.grad.zero_()

        beta_used = 0.5 * (beta + beta.t())
        beta_used = beta_used * mask

        q = expected_complete_log_likelihood(Phi_fixed, beta_used, x_prev, x_next, gamma)
        penalty = lambda_reg * beta_used.sum()
        loss = -q + penalty
        loss.backward()

        with torch.no_grad():
            beta -= lr * beta.grad
            beta.clamp_(0.0, 1.0)
            beta.mul_(mask)
            beta.copy_(0.5 * (beta + beta.t()))

        obj_value = float(loss.item())
        if prev_obj is not None and abs(obj_value - prev_obj) <= tol * (1.0 + abs(prev_obj)):
            break
        prev_obj = obj_value

    beta_final = beta.detach()
    beta_final = 0.5 * (beta_final + beta_final.t())
    beta_final.mul_(mask)
    return beta_final, float(obj_value if obj_value is not None else 0.0)


def alternating_em_and_graph_recovery(
    trajectory: np.ndarray,
    n_states: int,
    outer_iterations: int = 10,
    em_iterations_per_outer: int = 3,
    phi_lr: float = 5e-2,
    phi_steps: int = 120,
    beta_lr: float = 1e-2,
    beta_steps: int = 250,
    lambda_reg: float = 0.15,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[dict]]:
    x_all = torch.as_tensor(trajectory, dtype=torch.long, device=device)
    x_prev = x_all[:-1]
    x_next = x_all[1:]
    n_nodes = x_all.shape[1]

    a_start = np.random.uniform(low=0.5, high=1.0, size=(n_states, n_states))
    a_start = 0.5 * (a_start + a_start.T)
    Phi = torch.tensor(np.log(a_start), dtype=torch.float32, device=device)

    beta = torch.full((n_nodes, n_nodes), 0.5, dtype=torch.float32, device=device)
    beta.fill_diagonal_(0.0)
    beta = 0.5 * (beta + beta.t())

    history = []

    for outer in range(outer_iterations):
        print(f"\nOuter iteration {outer + 1}/{outer_iterations}")

        for em_iter in range(em_iterations_per_outer):
            with torch.no_grad():
                gamma = e_step_gamma(x_prev, x_next, beta, Phi)

            Phi, q_phi = optimize_phi_given_fixed_beta(
                Phi,
                beta,
                x_prev,
                x_next,
                gamma,
                lr=phi_lr,
                max_steps=phi_steps,
            )
            print(
                f"  EM (fixed beta) {em_iter + 1}/{em_iterations_per_outer} | "
                f"Q(Phi)={q_phi:.4f}"
            )

        with torch.no_grad():
            gamma = e_step_gamma(x_prev, x_next, beta, Phi)

        beta, beta_obj = optimize_beta_global(
            beta,
            Phi,
            x_prev,
            x_next,
            gamma,
            lambda_reg=lambda_reg,
            lr=beta_lr,
            max_steps=beta_steps,
        )

        with torch.no_grad():
            q_total = expected_complete_log_likelihood(Phi, beta, x_prev, x_next, gamma).item()
            offdiag_mean = (beta.sum() / (n_nodes * (n_nodes - 1))).item()

        history.append(
            {
                "outer_iter": outer + 1,
                "Q_value": float(q_total),
                "graph_objective": float(beta_obj),
                "mean_offdiag_beta": float(offdiag_mean),
            }
        )
        print(
            f"  Graph recovery (global beta) | Obj={beta_obj:.4f} | "
            f"Q={q_total:.4f} | mean beta={offdiag_mean:.4f}"
        )

    return Phi, beta, gamma, history


def beta_to_graph(beta: torch.Tensor, threshold: float = 0.5) -> tuple[nx.Graph, np.ndarray]:
    beta_np = beta.detach().cpu().numpy()
    adjacency = (beta_np > threshold).astype(int)
    np.fill_diagonal(adjacency, 0)
    graph = nx.from_numpy_array(adjacency)
    return graph, beta_np


def main() -> None:
    np.random.seed(42)
    torch.manual_seed(42)

    n_nodes = 10
    n_states = 2
    t_steps = 3000

    g_true = sample_connected_regular_graph(n_nodes=n_nodes, degree=3, seed=42)
    adj_true = nx.to_numpy_array(g_true)

    lambda_true = 3.0
    affinity_true = np.array(
        [
            [1.0 / (1.0 + lambda_true), lambda_true / (1.0 + lambda_true)],
            [lambda_true / (1.0 + lambda_true), 1.0 / (1.0 + lambda_true)],
        ],
        dtype=float,
    )

    trajectory = generate_trajectory(
        adj_true=adj_true,
        affinity=affinity_true,
        t_steps=t_steps,
        init_state=np.zeros(n_nodes, dtype=np.int64),
    )

    Phi_em, beta_em, _, _ = alternating_em_and_graph_recovery(
        trajectory=trajectory,
        n_states=n_states,
        outer_iterations=10,
        em_iterations_per_outer=3,
        phi_lr=5e-2,
        phi_steps=120,
        beta_lr=1e-2,
        beta_steps=250,
        lambda_reg=0.15,
        device="cpu",
    )

    g_learned, beta_np = beta_to_graph(beta_em, threshold=0.5)
    learned_adj = nx.to_numpy_array(g_learned)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    pos = nx.spring_layout(g_true, seed=42)

    nx.draw(
        g_true,
        pos,
        ax=ax[0],
        with_labels=True,
        node_color="lightblue",
        edge_color="black",
        width=2.0,
        node_size=500,
    )
    ax[0].set_title("True Graph (3-Regular)")

    learned_edges = list(g_learned.edges())
    edge_widths = [1.0 + 4.0 * beta_np[i, j] for i, j in learned_edges]
    nx.draw(
        g_learned,
        pos,
        ax=ax[1],
        with_labels=True,
        node_color="lightgreen",
        edge_color="red",
        width=edge_widths if learned_edges else 1.0,
        node_size=500,
    )
    ax[1].set_title("Learned Graph from Weighted Beta")

    fig.tight_layout()
    fig.savefig("alternating_em_weighted_graph.png", dpi=150)

    print("\nFinal Phi:")
    print(Phi_em.detach().cpu().numpy())
    print("\nFinal beta (rounded):")
    print(np.round(beta_np, 3))
    print(f"\nLearned edges above threshold: {int(np.sum(learned_adj) // 2)}")
    print("Saved plot to alternating_em_weighted_graph.png")


if __name__ == "__main__":
    main()
