import argparse
import random
from collections import deque

import matplotlib.pyplot as plt
import networkx as nx


# ----------------------------
# Random-walk helpers
# ----------------------------
def random_walk_endpoint(G: nx.Graph, start, steps: int, rng: random.Random):
    """Endpoint of a simple random walk of length `steps` starting at `start`."""
    x = start
    for _ in range(steps):
        nbrs = list(G.neighbors(x))
        if not nbrs:
            break
        x = rng.choice(nbrs)
    return x


def estimate_expected_distance(G: nx.Graph, steps: int, samples: int, rng: random.Random) -> float:
    """Estimate E[d(X, Y)] where Y is endpoint of steps-walk from X, X uniform."""
    nodes = list(G.nodes())
    if len(nodes) < 2:
        return 0.0

    # Precompute APSP distances for speed on small graphs; fallback to single-source BFS per sample if needed.
    # For growth this can explode; keep initial graphs modest or generations small.
    # We'll do BFS distances per sample to avoid O(|V|^2) memory.
    total = 0.0
    used = 0
    for _ in range(samples):
        x = rng.choice(nodes)
        y = random_walk_endpoint(G, x, steps, rng)
        if x == y:
            total += 0.0
            used += 1
            continue

        # BFS from x to get distance to y
        dist = nx.shortest_path_length(G, source=x, target=y)
        total += float(dist)
        used += 1

    return total / max(1, used)


def choose_walk_length_for_k(
    G: nx.Graph,
    target_k: float,
    max_steps: int,
    samples: int,
    rng: random.Random,
    tol: float = 0.2,
):
    """
    Choose smallest L such that estimated E[d(X, Y_L)] is close to target_k.
    Uses a simple increasing scan.
    """
    best_L = 1
    best_err = float("inf")
    best_val = None

    for L in range(1, max_steps + 1):
        val = estimate_expected_distance(G, L, samples=samples, rng=rng)
        err = abs(val - target_k)
        if err < best_err:
            best_err = err
            best_L = L
            best_val = val
        # Stop early if close enough
        if err <= tol:
            break

    return best_L, best_val, best_err


# ----------------------------
# Model steps
# ----------------------------
def infect_types(
    G: nx.Graph,
    types: dict,
    m: int,
    walk_steps: int,
    rng: random.Random,
):
    """
    For each node currently type 'a', infect m targets sampled as random-walk endpoints.
    Infection is irreversible: b -> a, a stays a.
    """
    a_nodes = [v for v, t in types.items() if t == "a"]
    to_infect = set()

    for x in a_nodes:
        for _ in range(m):
            y = random_walk_endpoint(G, x, walk_steps, rng)
            to_infect.add(y)

    for y in to_infect:
        types[y] = "a"


def reproduce_and_rewire_local(
    G: nx.Graph,
    types: dict,
    n: int,
    N: int,
    rng: random.Random,
):
    """
    Reproduction:
      - each node v spawns n or N children, inheriting type
    Locality-preserving rewiring:
      - for each parent edge (u,v), connect each child of u to 1 random child of v
        and each child of v to 1 random child of u.
    """
    offspring = {}
    new_types = {}
    next_id = 0

    for v in G.nodes():
        mult = n if types[v] == "a" else N
        kids = list(range(next_id, next_id + mult))
        next_id += mult
        offspring[v] = kids
        for c in kids:
            new_types[c] = types[v]

    G_next = nx.Graph()
    G_next.add_nodes_from(range(next_id))

    for (u, v) in G.edges():
        kids_u = offspring[u]
        kids_v = offspring[v]

        # each child of u connects to one random child of v
        for cu in kids_u:
            cv = rng.choice(kids_v)
            if cu != cv:
                G_next.add_edge(cu, cv)

        # each child of v connects to one random child of u
        for cv in kids_v:
            cu = rng.choice(kids_u)
            if cu != cv:
                G_next.add_edge(cu, cv)

    # Optional: keep siblings lightly connected (helps avoid fragmentation)
    # Comment out if you don't want it.
    for v, kids in offspring.items():
        if len(kids) >= 2:
            # connect as a chain
            for i in range(len(kids) - 1):
                G_next.add_edge(kids[i], kids[i + 1])

    # Ensure connectedness by stitching components if needed
    if not nx.is_connected(G_next) and G_next.number_of_nodes() > 0:
        comps = list(nx.connected_components(G_next))
        for i in range(len(comps) - 1):
            a = rng.choice(list(comps[i]))
            b = rng.choice(list(comps[i + 1]))
            G_next.add_edge(a, b)

    return G_next, new_types


# ----------------------------
# Initialization
# ----------------------------
def make_initial_graph(num_nodes: int, mean_degree: int, p_rewire: float, seed: int):
    rng = random.Random(seed)
    k_ws = max(2, min(mean_degree, num_nodes - 1))
    if k_ws % 2 == 1:
        k_ws += 1
    k_ws = min(k_ws, num_nodes - 1)

    G = nx.watts_strogatz_graph(num_nodes, k_ws, p_rewire, seed=seed)

    # Ensure connected by stitching components
    if not nx.is_connected(G):
        comps = list(nx.connected_components(G))
        for i in range(len(comps) - 1):
            u = rng.choice(list(comps[i]))
            v = rng.choice(list(comps[i + 1]))
            G.add_edge(u, v)
    return G


def assign_initial_types(G: nx.Graph, p_a0: float, seed: int):
    rng = random.Random(seed)
    return {v: ("a" if rng.random() < p_a0 else "b") for v in G.nodes()}


# ----------------------------
# Simulation
# ----------------------------
def simulate(k, n, N, m, generations, num_nodes0=80, mean_degree0=6, p_rewire=0.05, p_a0=0.5,
             seed=0, L_max=30, L_samples=400, L_tol=0.2):
    rng = random.Random(seed)

    G = make_initial_graph(num_nodes0, mean_degree0, p_rewire, seed)
    types = assign_initial_types(G, p_a0, seed)

    proportions = []
    L_used = []
    L_estEdist = []

    for t in range(generations + 1):
        A = sum(1 for v in types.values() if v == "a")
        B = len(types) - A
        proportions.append(A / (A + B) if (A + B) else 0.0)

        # pick L so that E[d(x, endpoint)] ≈ k on current graph
        L, est_val, _err = choose_walk_length_for_k(
            G, target_k=k, max_steps=L_max, samples=L_samples, rng=rng, tol=L_tol
        )
        L_used.append(L)
        L_estEdist.append(est_val)

        if t == generations:
            break

        # Infection step (uses current G and types)
        infect_types(G, types, m=m, walk_steps=L, rng=rng)

        # Reproduction + locality rewiring
        G, types = reproduce_and_rewire_local(G, types, n=n, N=N, rng=rng)

    return proportions, L_used, L_estEdist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=float, required=True, help="Target E[shortest-path distance] between start and L-step RW endpoint.")
    ap.add_argument("--n", type=int, required=True, help="Offspring count for type a.")
    ap.add_argument("--N", type=int, required=True, help="Offspring count for type b.")
    ap.add_argument("--m", type=int, required=True, help="Infections per type-a node per generation.")
    ap.add_argument("--generations", type=int, default=8)
    ap.add_argument("--num_nodes0", type=int, default=80)
    ap.add_argument("--mean_degree0", type=int, default=6)
    ap.add_argument("--p_rewire", type=float, default=0.05, help="WS rewiring probability for initial graph.")
    ap.add_argument("--p_a0", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()

    proportions, L_used, L_est = simulate(
        k=args.k, n=args.n, N=args.N, m=args.m,
        generations=args.generations,
        num_nodes0=args.num_nodes0,
        mean_degree0=args.mean_degree0,
        p_rewire=args.p_rewire,
        p_a0=args.p_a0,
        seed=args.seed
    )

    plt.figure()
    plt.plot(range(len(proportions)), proportions, marker="o")
    plt.ylim(0, 1)
    plt.xlabel("Generation t")
    plt.ylabel("Proportion type a  (A / (A + B))")
    plt.title(f"Type-a proportion over time (k={args.k}, n={args.n}, N={args.N}, m={args.m})")
    plt.grid(True)
    plt.show()

    # Diagnostics
    print("t\tprop_a\tL_used\tE[d] est")
    for t, (p, L, ed) in enumerate(zip(proportions, L_used, L_est)):
        print(f"{t}\t{p:.4f}\t{L}\t{ed:.3f}")


if __name__ == "__main__":
    main()
