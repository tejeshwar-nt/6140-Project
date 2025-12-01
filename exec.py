#!/usr/bin/env python3
import sys
import math
import time
import random
import itertools
import os
from typing import List, Tuple

# ==========================
#   Data loading & distances
# ==========================

def load_tsp_file(filename: str) -> Tuple[List[int], List[Tuple[float, float]]]:
    """
    Load a TSP instance from a file.
    Assumes lines with three tokens: node_id x y
    and skips non-data lines.

    Returns:
        node_ids: list of original vertex IDs as in file
        coords:   list of (x, y) coordinates in the same order
    """
    node_ids = []
    coords = []
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            # Try to parse first token as an int node id
            try:
                node_id = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
            except ValueError:
                # Not a data line (e.g., header), skip
                continue
            node_ids.append(node_id)
            coords.append((x, y))
    return node_ids, coords


def build_distance_matrix(coords: List[Tuple[float, float]]) -> List[List[int]]:
    """
    Build an NxN matrix of rounded Euclidean distances.
    """
    n = len(coords)
    dist = [[0] * n for _ in range(n)]
    for i in range(n):
        x1, y1 = coords[i]
        for j in range(i + 1, n):
            x2, y2 = coords[j]
            d = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            w = int(round(d))
            dist[i][j] = w
            dist[j][i] = w
    return dist


def tour_cost(tour: List[int], dist: List[List[int]]) -> int:
    """
    Compute cost of a tour visiting each city exactly once.
    We assume the tour is a cycle, so we close it back to the start city.
    """
    total = 0
    n = len(tour)
    for i in range(n):
        u = tour[i]
        v = tour[(i + 1) % n]
        total += dist[u][v]
    return total


# ==========================
#   Algorithm 1: Brute Force
# ==========================

def brute_force_tsp(dist: List[List[int]], cutoff: float) -> Tuple[int, List[int]]:
    """
    Brute force TSP (exact) with time cutoff.

    Fix city 0 as the start to avoid equivalent permutations.
    Explore permutations of the remaining cities. Stop when
    cutoff seconds have elapsed and return the best tour found.

    Returns:
        (best_cost, best_tour_indices)
    """
    n = len(dist)
    if n == 0:
        return 0, []

    start_time = time.time()
    cities = list(range(n))
    best_cost = float("inf")
    best_tour = None

    # We'll fix city 0 at the start of the tour
    others = cities[1:]

    for perm in itertools.permutations(others):
        # Check cutoff
        if time.time() - start_time > cutoff:
            break

        tour = [0] + list(perm)
        cost = tour_cost(tour, dist)
        if cost < best_cost:
            best_cost = cost
            best_tour = tour

    # If cutoff was so small no permutation was fully evaluated,
    # fall back on a trivial tour [0,1,2,...,n-1]
    if best_tour is None:
        best_tour = list(range(n))
        best_cost = tour_cost(best_tour, dist)

    return best_cost, best_tour


# ====================================
#   Algorithm 2: MST-based 2-Approx TSP
# ====================================

def prim_mst(dist: List[List[int]]) -> List[List[int]]:
    """
    Prim's algorithm to compute the MST of a complete graph
    given by distance matrix 'dist'.

    Returns adjacency list 'mst_adj', where mst_adj[u] is a list
    of neighbors of u in the MST.
    """
    n = len(dist)
    in_mst = [False] * n
    key = [float("inf")] * n
    parent = [-1] * n

    key[0] = 0
    for _ in range(n):
        # Pick the vertex with minimum key not yet in MST
        u = -1
        min_val = float("inf")
        for v in range(n):
            if not in_mst[v] and key[v] < min_val:
                min_val = key[v]
                u = v

        if u == -1:
            break
        in_mst[u] = True

        # Update keys
        for w in range(n):
            if not in_mst[w] and dist[u][w] < key[w]:
                key[w] = dist[u][w]
                parent[w] = u

    # Build adjacency list
    mst_adj = [[] for _ in range(n)]
    for v in range(1, n):
        p = parent[v]
        if p != -1:
            mst_adj[p].append(v)
            mst_adj[v].append(p)

    return mst_adj


def preorder_traversal(adj: List[List[int]], start: int = 0) -> List[int]:
    """
    Preorder DFS traversal of a tree (MST).
    """
    n = len(adj)
    visited = [False] * n
    order = []

    def dfs(u: int):
        visited[u] = True
        order.append(u)
        for v in adj[u]:
            if not visited[v]:
                dfs(v)

    dfs(start)
    return order


def mst_approx_tsp(dist: List[List[int]]) -> Tuple[int, List[int]]:
    """
    2-Approximation of TSP via MST preorder walk.

    Steps:
    1. Build MST (e.g., Prim's).
    2. Do a preorder traversal of the MST.
    3. Use that traversal order as the tour.

    Returns:
        (cost, tour_indices)
    """
    if not dist:
        return 0, []

    mst_adj = prim_mst(dist)
    tour = preorder_traversal(mst_adj, 0)
    cost = tour_cost(tour, dist)
    return cost, tour


# =====================================
#   Algorithm 3: Local Search - 2-opt
# =====================================

def two_opt_swap(tour: List[int], i: int, k: int) -> List[int]:
    """
    Perform a 2-opt swap by reversing the tour segment between i and k (inclusive).
    """
    new_tour = tour[:i] + tour[i:k+1][::-1] + tour[k+1:]
    return new_tour


def two_opt_local_search(dist: List[List[int]], cutoff: float, seed: int) -> Tuple[int, List[int]]:
    """
    2-opt local search for TSP within a time cutoff.

    - Start with a random tour determined by 'seed'.
    - Iteratively apply 2-opt moves that improve the tour.
    - Stop when no improving move is found or cutoff expires.

    Returns:
        (best_cost, best_tour)
    """
    random.seed(seed)
    n = len(dist)
    if n == 0:
        return 0, []

    # Initial random tour
    tour = list(range(n))
    random.shuffle(tour)

    best_tour = tour
    best_cost = tour_cost(best_tour, dist)

    start_time = time.time()

    improved = True
    while improved and (time.time() - start_time) < cutoff:
        improved = False
        # Try all pairs i < k for 2-opt
        for i in range(1, n - 1):       # don't move the first city index 0
            for k in range(i + 1, n):
                if time.time() - start_time >= cutoff:
                    break

                # Compute the cost difference of reversing between i and k
                a, b = best_tour[i - 1], best_tour[i]
                c, d = best_tour[k], best_tour[(k + 1) % n]

                # Current edges: (a,b) + (c,d)
                # New edges:     (a,c) + (b,d)
                delta = dist[a][c] + dist[b][d] - dist[a][b] - dist[c][d]
                if delta < 0:
                    # Improvement found
                    best_tour = two_opt_swap(best_tour, i, k)
                    best_cost += delta
                    improved = True
                    break  # restart scanning from beginning of neighbors
            if improved:
                break

    return best_cost, best_tour


# ==========================
#   Output writer
# ==========================

def write_solution_file(
    instance_name: str,
    method: str,
    cutoff: float = None,
    seed: int = None,
    best_cost: int = 0,
    best_tour_indices: List[int] = None,
    node_ids: List[int] = None,
):
    """
    Write solution to a .sol file.

    File naming conventions (you can tweak to match the grader):
    - BF:     <instance> BF <cutoff>.sol
    - Approx: <instance> Approx.sol
    - LS:     <instance> LS <cutoff> <seed>.sol

    File content:
    - line 1: cost
    - line 2: list of vertex IDs (original IDs from input), comma-separated
    """
    if best_tour_indices is None:
        best_tour_indices = []
    if node_ids is None:
        node_ids = []

    if method == "BF":
        file_name = f"{instance_name} BF {int(cuttoff)}.sol"
    elif method == "Approx":
        file_name = f"{instance_name} Approx.sol"
    elif method == "LS":
        if cutoff is None or seed is None:
            raise ValueError("LS requires both cutoff and seed for naming.")
        file_name = f"{instance_name} LS {int(cutoff)} {seed}.sol"
    else:
        raise ValueError(f"Unknown method {method}")

    # Map internal indices back to original vertex IDs
    # If node_ids is [id0, id1, ..., id(n-1)], and tour is [0,2,1,...],
    # then solution line is [node_ids[0], node_ids[2], node_ids[1], ...]
    vertex_id_tour = [str(node_ids[i]) for i in best_tour_indices]

    with open(file_name, "w") as f:
        f.write(str(best_cost) + "\n")
        f.write(",".join(vertex_id_tour) + "\n")

    print(f"Wrote solution to {file_name}")


# ==========================
#   Main / CLI
# ==========================

def main():
    """
    Simple CLI wrapper.

    Usage (you can adapt to your professor's exact spec):

        Brute force:
            python exec.py <instance_file> BF <cutoff_seconds>

        Approximation (MST 2-approx):
            python exec.py <instance_file> Approx

        Local search (2-opt):
            python exec.py <instance_file> LS <cutoff_seconds> <seed>
    """
    if len(sys.argv) < 3:
        print("Usage:")
        print("  BF:     python exec.py <instance_file> BF <cutoff_seconds>")
        print("  Approx: python exec.py <instance_file> Approx")
        print("  LS:     python exec.py <instance_file> LS <cutoff_seconds> <seed>")
        sys.exit(1)

    instance_file = sys.argv[1]
    method = sys.argv[2]

    # Load instance
    node_ids, coords = load_tsp_file(instance_file)
    if not node_ids:
        print(f"Error: could not parse any nodes from {instance_file}")
        sys.exit(1)

    dist = build_distance_matrix(coords)
    instance_name = os.path.splitext(os.path.basename(instance_file))[0]

    if method == "BF":
        if len(sys.argv) < 4:
            print("Error: BF requires cutoff_seconds.")
            sys.exit(1)
        cutoff = float(sys.argv[3])
        best_cost, best_tour = brute_force_tsp(dist, cutoff)
        write_solution_file(
            instance_name,
            method,
            cutoff=cutoff,
            seed=None,
            best_cost=best_cost,
            best_tour_indices=best_tour,
            node_ids=node_ids,
        )

    elif method == "Approx":
        best_cost, best_tour = mst_approx_tsp(dist)
        write_solution_file(
            instance_name,
            method,
            cutoff=None,
            seed=None,
            best_cost=best_cost,
            best_tour_indices=best_tour,
            node_ids=node_ids,
        )

    elif method == "LS":
        if len(sys.argv) < 5:
            print("Error: LS requires cutoff_seconds and seed.")
            sys.exit(1)
        cutoff = float(sys.argv[3])
        seed = int(sys.argv[4])
        best_cost, best_tour = two_opt_local_search(dist, cutoff, seed)
        write_solution_file(
            instance_name,
            method,
            cutoff=cutoff,
            seed=seed,
            best_cost=best_cost,
            best_tour_indices=best_tour,
            node_ids=node_ids,
        )

    else:
        print(f"Unknown method '{method}'. Use BF, Approx, or LS.")
        sys.exit(1)


if __name__ == "__main__":
    main()
