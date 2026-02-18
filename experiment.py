#generating a trajectory for a Markov Random Field with affinity matrix A
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

np.random.seed(42)

n_nodes = 12
n_states = 2
t_steps = 500

trajectory = np.zeros((t_steps, n_nodes), dtype=int)
#this will be filled in below

#Generating a 3-regular graph
while True:
    G_true = nx.random_regular_graph(3, n_nodes)
    if nx.is_connected(G_true):
        break
adj_true = nx.to_numpy_array(G_true)

#defining affinity matrix A
lambda_true = 3
A = np.zeros((n_states, n_states))
A[0, 1] = A[1, 0] = lambda_true/(1+lambda_true)
A[0, 0] = 1/(1+lambda_true)

#Gives conidtional probability of each state (color) for a choosen node
def cond_prob_states(chosen_node,nodes_current_states, adj, affinity):
  neighbors = np.where(adj[chosen_node] == 1)[0]
  neighbor_vals = nodes_current_states[neighbors]

  #product for a particular color/state
  def state_product(state_num):
    affinity_prod = 1
    for neighbor_state in neighbor_vals:
      affinity_prod *= affinity[state_num, neighbor_state]
    return affinity_prod

  # Calculate probabilities for all possible colors
  probs = np.array([state_product(s) for s in range(n_states)])
  if np.sum(probs) == 0: return np.ones(n_states) / n_states
  return probs / np.sum(probs)

#Generate trajectory
trajectory[0] = np.zeros(n_nodes) #intial state assignment

for t in np.arange(1,t_steps):
  x_curr = trajectory[t-1].copy()

  target_node = np.random.randint(0, n_nodes) #uniformly sample from nodes
  cond_prob_vector = cond_prob_states(target_node,x_curr,adj_true,A) #conditonal probability of each state
  new_state = np.random.choice(np.arange(n_states), p=cond_prob_vector) #pick new state
  x_curr[target_node] = new_state
  trajectory[t] = x_curr

import torch
import networkx as nx

#Goal is to optimize Phi matrix

def create_edge_index(G_em):
    # Map nodes to 0...N-1 to ensure they work as tensor indices
    mapping = {node: i for i, node in enumerate(G_em.nodes())}
    G_mapped = nx.relabel_nodes(G_em, mapping)

    edges = list(G_mapped.edges())
    # Convert to LongTensor [2, num_edges]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    # Add reverse edges for undirected behavior: sum_{j in N(i)}
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    return edge_index

def Q_function(Phi, edge_index, gamma_matrix, trajectory):
    n_nodes = trajectory.shape[1]
    t_steps = trajectory.shape[0]
    K = Phi.size(0)
    total_Q = 0.0

    row, col = edge_index

    for t in range(t_steps - 1):
        x_curr = trajectory[t]      # shape: [n_nodes]
        x_next = trajectory[t+1]    # shape: [n_nodes]

        # 1. Map current neighbor states to their contributions in Phi
        # Note: Phi[k, x_j] means we want the k-th row, x_j-th column
        # We take all rows (all possible k) for the specific observed neighbor states
        phi_lookup = Phi[:, x_curr].t() # shape: [n_nodes, K]

        # 2. Aggregate neighbor features: sum_{j in N(i)} Phi_{*, x_j}
        aggregated_phi = torch.zeros(n_nodes, K, device=Phi.device)
        aggregated_phi.index_add_(0, row, phi_lookup[col])

        # 3. Positive Term: sum_{j in N(i)} Phi_{x_i(t+1), x_j(t)}
        # We extract the entry corresponding to the actual next state x_i(t+1)
        pos_term = aggregated_phi[torch.arange(n_nodes), x_next]

        # 4. Negative Term: Log-Sum-Exp over all possible states k
        neg_term = torch.logsumexp(aggregated_phi, dim=1)

        # 5. Weight by gamma and accumulate
        q_t = torch.sum(gamma_matrix[t] * (pos_term - neg_term))
        total_Q = total_Q + q_t # Avoid in-place += for backprop stability

    return total_Q

def gradient_ascent_phi(trajectory, G_em, gamma_matrix, n_states):
    # Pre-convert data to tensors
    trajectory = torch.as_tensor(trajectory, dtype=torch.long)
    gamma_matrix = torch.as_tensor(gamma_matrix, dtype=torch.float)

    edge_index = create_edge_index(G_em)

    # Initialize Phi
    Phi = torch.randn(n_states, n_states, requires_grad=True)
    optimizer = torch.optim.Adam([Phi], lr=1e-2)

    epoch = 0
    Q_Value = -1000

    #when to stop gradient
    while np.abs(Q_Value) > 80 and epoch < 500:
        optimizer.zero_grad()

        q_val = Q_function(Phi, edge_index, gamma_matrix, trajectory)

        # Gradient Ascent: Minimize the negative of the objective
        loss = -q_val
        loss.backward()
        optimizer.step()

        epoch += 1

        if epoch % 20 == 0:
            print(f"Epoch {epoch:03d} | Q-Value: {q_val.item():.4f}")

        Q_Value = q_val.item()

    return Phi

#optimized_phi = gradient_ascent_phi(trajectory, G_em, gamma_matrix, n_states)

# Guess a random graph to start
G_em = nx.gnp_random_graph(n_nodes, 0.3)
A_em = np.ones((n_states, n_states)) * 0.5
adj_em = nx.to_numpy_array(G_em)
n_em_iterations = 1

for em_iteration in range(n_em_iterations):
  print(f"Iteration: {em_iteration + 1}")

  #E-step
  #Estimations of latent probabilities for every change
  #in the trajectory
  gamma_matrix = np.zeros((t_steps - 1, n_nodes))
  for t in range(1, t_steps):
    prev = trajectory[t-1]
    curr = trajectory[t]
    if np.array_equal(prev, curr):
      # No change, calculate using current guesses
      for i in range(n_nodes):
        probs = cond_prob_states(i, prev, adj_em, A_em)
        current_color = prev[i]
        gamma_matrix[t-1, i] = probs[current_color]
      gamma_matrix[t-1] /= np.sum(gamma_matrix[t-1])
    else:
      # Change
      changed_node = np.where(prev != curr)[0]
      gamma_matrix[t-1, changed_node] = 1.0

  #M-step
  new_G = np.zeros((n_nodes, n_nodes))
  new_Am = np.ones((n_states, n_states)) * 0.5

  #update the graph G via nodewise LASSO
  #for i in range(n_nodes):

    #set up data to run LASSO on


    # model = x
    # model.fit(y)


    # thresholding tells us our estimated neighbors
    #new_adj_row = (np.abs(model.coef_) > varepsilon).astype(int)
    #new_G[i, :] = new_adj_row

  #symmetrize the graph using OR
  #G_em = np.maximum(new_G, new_G.T)

  #update the phi
  #Phi_new = gradient_ascent_phi(trajectory, G_em, gamma_matrix, n_states)