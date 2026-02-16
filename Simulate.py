import random
import math
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


# ---------- Fast adjacency utilities ----------
def to_adjlist(G):
    """Convert a NetworkX graph with nodes 0..n-1 into list-of-lists adjacency."""
    n = G.number_of_nodes()
    adj = [[] for _ in range(n)]
    for u, v in G.edges():
        adj[u].append(v)
        adj[v].append(u)
    return adj

def rw_endpoint(adj, start, steps, rng):
    x = start
    for _ in range(steps):
        nbrs = adj[x]
        if not nbrs:
            break
        x = nbrs[rng.randrange(len(nbrs))]
    return x

def sp_distance_bfs(adj, s, t):
    if s == t:
        return 0
    n = len(adj)
    q = [s]
    dist = [-1] * n
    dist[s] = 0
    for x in q:
        dx = dist[x] + 1
        for y in adj[x]:
            if dist[y] == -1:
                dist[y] = dx
                if y == t:
                    return dx
                q.append(y)
    return n  # fallback if disconnected


# ---------- Calibrate L to hit target k on initial graph ----------
def estimate_Edist(adj, steps, samples, rng):
    n = len(adj)
    total = 0.0
    for _ in range(samples):
        s = rng.randrange(n)
        t = rw_endpoint(adj, s, steps, rng)
        total += sp_distance_bfs(adj, s, t)
    return total / samples

def choose_L_for_k(adj, target_k, max_steps=30, samples=80, tol=0.35, rng=None):
    rng = rng or random.Random(0)
    bestL, bestErr, bestVal = 1, float("inf"), None
    for L in range(1, max_steps + 1):
        val = estimate_Edist(adj, L, samples, rng)
        err = abs(val - target_k)
        if err < bestErr:
            bestL, bestErr, bestVal = L, err, val
        if err <= tol:
            break
    return bestL, bestVal


# ---------- Model dynamics on adjacency lists ----------
def infect(adj, types, m, L, rng):
    # types: bytearray with 1=a, 0=b
    n = len(adj)
    a_nodes = [i for i in range(n) if types[i] == 1]
    to_inf = set()
    for x in a_nodes:
        for _ in range(m):
            y = rw_endpoint(adj, x, L, rng)
            to_inf.add(y)
    for y in to_inf:
        types[y] = 1  # irreversible

def reproduce_and_lift(adj, types, n_a, n_b, rng):
    n = len(adj)
    # offspring ranges
    offspring = [None] * n
    new_types = []
    next_id = 0
    for v in range(n):
        mult = n_a if types[v] == 1 else n_b
        kids = list(range(next_id, next_id + mult))
        offspring[v] = kids
        new_types.extend([types[v]] * mult)
        next_id += mult

    new_adj = [[] for _ in range(next_id)]

    # lift each parent edge locally
    for u in range(n):
        for v in adj[u]:
            if v < u:
                continue
            ku, kv = offspring[u], offspring[v]
            # each child of u connects to one random child of v
            for cu in ku:
                cv = kv[rng.randrange(len(kv))]
                new_adj[cu].append(cv)
                new_adj[cv].append(cu)
            # each child of v connects to one random child of u
            for cv in kv:
                cu = ku[rng.randrange(len(ku))]
                new_adj[cv].append(cu)
                new_adj[cu].append(cv)

    # sibling chain to reduce fragmentation
    idx = 0
    for v in range(n):
        kids = offspring[v]
        for i in range(len(kids) - 1):
            a = kids[i]; b = kids[i + 1]
            new_adj[a].append(b); new_adj[b].append(a)

    return new_adj, bytearray(new_types)

def downsample(adj, types, cap, rng):
    n = len(adj)
    if n <= cap:
        return adj, types
    keep = rng.sample(range(n), cap)
    keep_set = set(keep)
    mapping = {old:i for i, old in enumerate(keep)}
    new_adj = [[] for _ in range(cap)]
    for old_u in keep:
        u = mapping[old_u]
        for old_v in adj[old_u]:
            if old_v in keep_set:
                v = mapping[old_v]
                new_adj[u].append(v)
    new_types = bytearray(types[old] for old in keep)
    return new_adj, new_types


# ---------- One run ----------
def one_run(k, m, generations=5, cap=800, seed=0, p_a0=0.15, mean_deg=6):
    rng = random.Random(seed)
    # initial WS graph (small-world-ish)
    n0 = 140
    k_ws = mean_deg + (mean_deg % 2)  # even
    G0 = nx.watts_strogatz_graph(n0, k_ws, 0.08, seed=seed)
    if not nx.is_connected(G0):
        # stitch quickly
        comps = list(nx.connected_components(G0))
        for i in range(len(comps)-1):
            u = rng.choice(list(comps[i])); v = rng.choice(list(comps[i+1]))
            G0.add_edge(u, v)

    adj = to_adjlist(G0)
    L, est = choose_L_for_k(adj, k, rng=rng)  # calibrate once

    # types: 1=a, 0=b
    types = bytearray(1 if rng.random() < p_a0 else 0 for _ in range(len(adj)))

    for _ in range(generations):
        infect(adj, types, m=m, L=L, rng=rng)
        adj, types = reproduce_and_lift(adj, types, n_a=2, n_b=8, rng=rng)
        adj, types = downsample(adj, types, cap=cap, rng=rng)

    p = sum(types) / len(types)
    return p, L, est


# ---------- Sweep + plot ----------
def sweep_and_plot():
    n, N = 2, 8
    k_values = np.linspace(1.0, 6.0, 9)   # adjust as you like
    m_values = list(range(0, 11))         # 0..10
    reps = 5                               # increase for smoother boundary

    mean_p = np.zeros((len(m_values), len(k_values)))

    for i, m in enumerate(m_values):
        for j, k in enumerate(k_values):
            ps = []
            for r in range(reps):
                p, L, est = one_run(k, m, generations=5, cap=800, seed=10000 + 97*r + 13*i + j)
                ps.append(p)
            mean_p[i, j] = float(np.mean(ps))

    plt.figure()
    plt.imshow(mean_p, origin="lower", aspect="auto")
    plt.xticks(range(len(k_values)), [f"{x:.1f}" for x in k_values], rotation=45)
    plt.yticks(range(len(m_values)), m_values)
    plt.xlabel("k (target expected RW distance)")
    plt.ylabel("m (infections per a-node per generation)")
    plt.title("Mean final p(a) after 5 generations (n=2, N=8)")
    plt.colorbar(label="mean final p(a)")
    plt.tight_layout()
    plt.savefig("heatmap.png", dpi=150)
    print("Saved heatmap.png")

    # Report rough critical m per k for p(a) >= 0.5 and >= 0.9
    for j, k in enumerate(k_values):
        m50 = next((m_values[i] for i in range(len(m_values)) if mean_p[i, j] >= 0.5), None)
        m90 = next((m_values[i] for i in range(len(m_values)) if mean_p[i, j] >= 0.9), None)
        print(f"k={k:.1f}: m@0.5={m50}, m@0.9={m90}")

if __name__ == "__main__":
    sweep_and_plot()
