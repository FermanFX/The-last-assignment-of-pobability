"""
Assignment 7: Markov Chains & Sampling Methods
Math4AI: Probability & Statistics

Fill in the algorithm implementations below
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy import stats


# =============================================================================
# Task 7.1: PageRank (Power Iteration)
# =============================================================================

def create_mini_internet_graph():
    """
    Create a mini internet graph for PageRank demonstration.
    Returns adjacency matrix.
    """
    G = nx.DiGraph()
    
    nodes = list(range(10))
    G.add_nodes_from(nodes)
    
    edges = [
        (0, 1), (0, 2), (0, 3),
        (1, 0), (1, 2), (1, 4),
        (2, 0), (2, 5),
        (3, 1), (3, 4), (3, 6),
        (4, 3), (4, 5),
        (5, 2), (5, 4), (5, 7),
        (6, 3), (6, 7),
        (7, 5), (7, 6), (7, 8),
        (8, 7), (8, 9),
        (9, 8)
    ]
    G.add_edges_from(edges)
    
    return nx.to_numpy_array(G, nodelist=nodes)


def pagerank_power_iteration(adj_matrix, d=0.85, tol=1e-6, max_iter=1000):
    """
    Implement PageRank using power iteration method.
    
    The PageRank formula:
        new = d * old * P + (1-d) * N
        
    Where:
        - d is the damping factor (0.85)
        - P is the transition matrix (column-stochastic)
        - N is the teleportation vector (uniform 1/n)
    
    Steps:
        1. Convert adjacency matrix to transition matrix (column-stochastic)
        2. Initialize uniform probability vector
        3. Iterate until convergence (||new - old|| < tol)
        4. Return the stationary distribution
    
    Args:
        adj_matrix: Adjacency matrix of the graph
        d: Damping factor
        tol: Convergence tolerance
        max_iter: Maximum number of iterations
    
    Returns:
        numpy array: PageRank scores for each node
    """
    '''DONE'''# TODO: Implement PageRank power iteration
    # HINT: 
    # 1. Make column-stochastic: divide each column by its sum
    # 2. Handle dangling nodes (columns with sum 0)
    # 3. Apply the PageRank formula iteratively
    # 4. Check for convergence using L1 or L2 norm
    
    n = adj_matrix.shape[0]
    
    # Code start here
    column_sums = adj_matrix.sum(axis=0)

    '''Bu tam düzgün sayılmır, çünki dangling node (çıxışı olmayan node) bütün səhifələrə bərabər ehtimalla keçmir.'''
    # column_sums[column_sums == 0] = 1
    # P = adj_matrix / column_sums
    '''Bu səbəbdən dangling node-lar üçün bütün səhifələrə bərabər ehtimalla keçmək üçün aşağıdakı kimi düzəliş edəyin'''
    column_sums = adj_matrix.sum(axis=0)
    P = adj_matrix.astype(float)

    dangling = (column_sums == 0)

    P[:, ~dangling] /= column_sums[~dangling]
    P[:, dangling] = 1.0 / n

    # Initialize PageRank vector
    pagerank = np.ones(n) / n
    teleport = np.ones(n) / n
    for iteration in range(max_iter):
        new_pagerank = d * P @ pagerank + (1 - d) * teleport
        if np.linalg.norm(new_pagerank - pagerank, ord=1) < tol:
            print(f"Converged after {iteration+1} iterations.")
            break
        pagerank = new_pagerank.copy()


    return pagerank


def visualize_pagerank(adj_matrix, pagerank_scores, save_path='pagerank_graph.png'):
    """
    Visualize the graph with PageRank scores as node sizes and colors.
    """
    G = nx.DiGraph(adj_matrix)
    
    plt.figure(figsize=(12, 8))
    
    pos = nx.spring_layout(G, seed=42, k=2)
    
    node_sizes = pagerank_scores * 3000 + 300
    
    nodes = plt.scatter(
        [pos[i][0] for i in range(len(pagerank_scores))],
        [pos[i][1] for i in range(len(pagerank_scores))],
        s=node_sizes,
        c=pagerank_scores,
        cmap=plt.cm.YlOrRd,
        alpha=0.8,
        edgecolors='black',
        linewidths=2
    )
    
    nx.draw_networkx_edges(
        G, pos,
        arrowstyle='->',
        arrowsize=15,
        alpha=0.5,
        edge_color='gray',
        width=1.5
    )
    
    nx.draw_networkx_labels(
        G, pos,
        font_size=14,
        font_weight='bold',
        font_color='black'
    )
    
    plt.colorbar(nodes, label='PageRank Score', shrink=0.8)
    plt.title('PageRank: Mini-Internet Graph\n(Node size and color = importance)', fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"PageRank graph saved to {save_path}")


# =============================================================================
# Task 7.2: Metropolis-Hastings (MCMC)
# =============================================================================

def double_well_density(x):
    """
    Unnormalized double well potential density:
    f(x) = exp(-(x-4)^2/2) + exp(-(x+4)^2/2)
    """
    return np.exp(-(x - 4)**2 / 2) + np.exp(-(x + 4)**2 / 2)


def metropolis_hastings(target_density, x0, n_samples, proposal_std=2.0):
    """
    This is the Metropolis-Hastings MCMC algorithm.
    
    The acceptance probability (for symmetric proposal):
        A(x -> x') = min(1, target_density(x') / target_density(x))
    
    Algorithm:
        1. Start at x0
        2. For each iteration:
           - Propose x' ~ N(x, proposal_std)
           - Calculate acceptance probability
           - Accept or reject based on random draw
        3. Return all accepted samples (including current state)
    
    Args:
        target_density: Unnormalized density function P(x) ∝ f(x)
        x0: Initial state
        n_samples: Number of samples to generate
        proposal_std: Standard deviation of Gaussian proposal
    
    Returns:
        numpy array: Samples from the target distribution
    """
    '''DONE'''# TODO: Implement Metropolis-Hastings
    # HINT:
    # 1. Initialize samples array and current state x = x0
    # 2. For each sample:
    #    - Propose new x' from normal distribution with mean=x and std=proposal_std
    #    - Calculate acceptance ratio: alpha = min(1, P(x')/P(x))
    #    - Accept with probability alpha (use np.random.random() < alpha)
    #    - If accepted, set x = x', else keep current x
    #    - Store current x in samples
    # 3. Return samples
    
    samples = np.zeros(n_samples)
    x = x0
    
    # Code start here
    for i in range(n_samples):
        x_proposal = np.random.normal(x, proposal_std)
        
        acceptance_ratio = target_density(x_proposal) / target_density(x)
        alpha = min(1, acceptance_ratio)
        
        if np.random.random() < alpha:
            x = x_proposal
        else:
            x = x 
        
        samples[i] = x
    return samples

def visualize_metropolis_hastings(samples, save_path='mh_trace_hist.png'):
    """
    Create trace plot and histogram for Metropolis-Hastings samples.
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Trace plot
    axes[0].plot(samples, alpha=0.7, linewidth=0.5, color='steelblue')
    axes[0].axhline(y=4, color='red', linestyle='--', alpha=0.5, label='Mode 1 (+4)')
    axes[0].axhline(y=-4, color='green', linestyle='--', alpha=0.5, label='Mode 2 (-4)')
    axes[0].set_xlabel('Iteration', fontsize=12)
    axes[0].set_ylabel('x', fontsize=12)
    axes[0].set_title('Metropolis-Hastings: Trace Plot', fontsize=14, fontweight='bold')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)
    
    # Histogram
    x_range = np.linspace(-10, 10, 1000)
    true_density = double_well_density(x_range)
    true_density_normalized = true_density / (np.sum(true_density) * (x_range[1] - x_range[0]))
    
    axes[1].hist(samples, bins=50, density=True, alpha=0.6, color='steelblue', 
                 edgecolor='white', label='MCMC Samples')
    axes[1].plot(x_range, true_density_normalized, 'r-', linewidth=2, label='True Density')
    axes[1].set_xlabel('x', fontsize=12)
    axes[1].set_ylabel('Density', fontsize=12)
    axes[1].set_title('Metropolis-Hastings: Histogram vs True Density', fontsize=14, fontweight='bold')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(-10, 10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Metropolis-Hastings plot saved to {save_path}")


# =============================================================================
# Task 7.3: Gibbs Sampling
# =============================================================================

def gibbs_sampler(rho, n_samples):
    """
    Implement Gibbs Sampling for Bivariate Gaussian.
    
    For a bivariate Gaussian with correlation rho:
        P(x | y) = N(y * rho, 1 - rho^2)
        P(y | x) = N(x * rho, 1 - rho^2)
    
    Algorithm:
        1. Initialize (x0, y0) = (0, 0)
        2. For t = 1 to n_samples:
           - Sample x_t ~ N(y_{t-1} * rho, 1 - rho^2)
           - Sample y_t ~ N(x_t * rho, 1 - rho^2)
        3. Return all samples
    
    Args:
        rho: Correlation coefficient (0.8 for this task)
        n_samples: Number of samples to generate
    
    Returns:
        tuple: (x_samples, y_samples)
    """
    '''DONE'''# TODO: Implement Gibbs Sampling
    # HINT:
    # 1. Initialize x = 0, y = 0
    # 2. For each iteration:
    #    - Sample x from N(y * rho, 1 - rho^2)
    #    - Sample y from N(x * rho, 1 - rho^2)
    #    - Store x and y
    # 3. Return x_samples and y_samples
    
    x_samples = np.zeros(n_samples)
    y_samples = np.zeros(n_samples)
    x, y = 0.0, 0.0
    
    # Code starts here
    for i in range(n_samples):
        x = np.random.normal(loc=y * rho, scale=np.sqrt(1 - rho**2))
        y = np.random.normal(loc=x * rho, scale=np.sqrt(1 - rho**2))
        
        x_samples[i] = x
        y_samples[i] = y

    return x_samples, y_samples


def visualize_gibbs(x_samples, y_samples, rho, save_path='gibbs_scatter.png'):
    """
    Create scatter plot for Gibbs sampling results.
    """
    plt.figure(figsize=(10, 8))
    
    plt.scatter(x_samples, y_samples, alpha=0.3, s=10, c='steelblue', label='Gibbs Samples')
    
    mean = [0, 0]
    cov = [[1, rho], [rho, 1]]
    x_line = np.linspace(-4, 4, 100)
    y_line = np.linspace(-4, 4, 100)
    X, Y = np.meshgrid(x_line, y_line)
    pos = np.dstack((X, Y))
    Z = stats.multivariate_normal(mean, cov).pdf(pos)
    
    plt.contour(X, Y, Z, levels=5, colors='red', alpha=0.6, linewidths=1.5)
    
    plt.xlabel('X', fontsize=12)
    plt.ylabel('Y', fontsize=12)
    plt.title(f'Gibbs Sampling: Bivariate Gaussian (ρ = {rho})', fontsize=14, fontweight='bold')
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Gibbs sampling plot saved to {save_path}")


# =============================================================================
# Main: Run All Tasks
# =============================================================================

def main():
    print("=" * 60)
    print("Assignment 7: Markov Chains & Sampling Methods")
    print("=" * 60)
    
    # -------------------------------------------------------------------------
    # Task 7.1: PageRank
    # -------------------------------------------------------------------------
    print("\n--- Task 7.1: PageRank ---")
    
    adj_matrix = create_mini_internet_graph()
    
    print("Adjacency Matrix:")
    print(adj_matrix)
    
    pagerank_scores = pagerank_power_iteration(adj_matrix, d=0.85, tol=1e-6)
    
    print("\nPageRank Scores:")
    for i, score in enumerate(pagerank_scores):
        print(f"  Node {i}: {score:.4f}")
    
    most_important = np.argmax(pagerank_scores)
    print(f"\nMost important node: {most_important} (score: {pagerank_scores[most_important]:.4f})")
    
    visualize_pagerank(adj_matrix, pagerank_scores)
    
    # -------------------------------------------------------------------------
    # Task 7.2: Metropolis-Hastings
    # -------------------------------------------------------------------------
    print("\n--- Task 7.2: Metropolis-Hastings ---")
    
    x0 = 0.0
    n_samples = 10000
    proposal_std = 2.0
    
    samples = metropolis_hastings(double_well_density, x0, n_samples, proposal_std)
    
    print(f"Generated {len(samples)} samples")
    print(f"Sample mean: {np.mean(samples):.4f}")
    print(f"Sample std: {np.std(samples):.4f}")
    
    visualize_metropolis_hastings(samples)
    
    # -------------------------------------------------------------------------
    # Task 7.3: Gibbs Sampling
    # -------------------------------------------------------------------------
    print("\n--- Task 7.3: Gibbs Sampling ---")
    
    rho = 0.8
    n_samples = 5000
    
    x_samples, y_samples = gibbs_sampler(rho, n_samples)
    
    print(f"Generated {n_samples} samples")
    print(f"X mean: {np.mean(x_samples):.4f}, std: {np.std(x_samples):.4f}")
    print(f"Y mean: {np.mean(y_samples):.4f}, std: {np.std(y_samples):.4f}")
    print(f"Correlation: {np.corrcoef(x_samples, y_samples)[0,1]:.4f}")
    
    visualize_gibbs(x_samples, y_samples, rho)
    
    print("\n" + "=" * 60)
    print("All tasks completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
