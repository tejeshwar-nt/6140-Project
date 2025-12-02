#!/usr/bin/env python3
import sys
import math
import os
import time
import csv
import itertools
import random
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
    """
    n = len(dist)
    if n == 0:
        return 0, []

    start_time = time.time()
    cities = list(range(n))
    best_cost = float("inf")
    best_tour = None

    others = cities[1:]

    for perm in itertools.permutations(others):
        if time.time() - start_time > cutoff:
            break
        tour = [0] + list(perm)
        cost = tour_cost(tour, dist)
        if cost < best_cost:
            best_cost = cost
            best_tour = tour

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
        u = -1
        min_val = float("inf")
        for v in range(n):
            if not in_mst[v] and key[v] < min_val:
                min_val = key[v]
                u = v

        if u == -1:
            break
        in_mst[u] = True

        for w in range(n):
            if not in_mst[w] and dist[u][w] < key[w]:
                key[w] = dist[u][w]
                parent[w] = u

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
    return tour[:i] + tour[i:k+1][::-1] + tour[k+1:]


def two_opt_local_search(dist: List[List[int]], cutoff: float, seed: int) -> Tuple[int, List[int]]:
    """
    2-opt local search for TSP within a time cutoff.

    - Start with a random tour determined by 'seed'.
    - Iteratively apply 2-opt moves that improve the tour.
    - Stop when no improving move is found or cutoff expires.
    """
    random.seed(seed)
    n = len(dist)
    if n == 0:
        return 0, []

    tour = list(range(n))
    random.shuffle(tour)

    best_tour = tour[:]
    best_cost = tour_cost(best_tour, dist)

    start_time = time.time()
    improved = True

    while improved and (time.time() - start_time) < cutoff:
        improved = False
        for i in range(1, n - 1):
            for k in range(i + 1, n):
                if time.time() - start_time >= cutoff:
                    break
                a, b = best_tour[i - 1], best_tour[i]
                c, d = best_tour[k], best_tour[(k + 1) % n]
                delta = dist[a][c] + dist[b][d] - dist[a][b] - dist[c][d]
                if delta < 0:
                    best_tour = two_opt_swap(best_tour, i, k)
                    best_cost += delta
                    improved = True
                    break
            if improved:
                break

    return best_cost, best_tour


# ==========================
#   Output: .sol files
# ==========================

def write_solution(instance_name: str,
                   method: str,
                   best_cost: int,
                   best_tour: List[int],
                   node_ids: List[int],
                   cutoff: float = None,
                   seed: int = None):
    """
    Write solution to a .sol file.

    Naming:
      BF:     <instance> BF <cutoff>.sol
      Approx: <instance> Approx.sol
      LS:     <instance> LS <cutoff> <seed>.sol
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))

    if method == "BF":
        if cutoff is None:
            raise ValueError("BF requires cutoff for naming.")
        file_name = f"{instance_name} BF {int(cutoff)}.sol"
    elif method == "Approx":
        file_name = f"{instance_name} Approx.sol"
    elif method == "LS":
        if cutoff is None or seed is None:
            raise ValueError("LS requires cutoff and seed for naming.")
        file_name = f"{instance_name} LS {int(cutoff)} {seed}.sol"
    else:
        raise ValueError(f"Unknown method {method}")

    output_path = os.path.join(script_dir, file_name)

    # Map internal indices back to original vertex IDs
    vertex_id_tour = [str(node_ids[i]) for i in best_tour]

    with open(output_path, "w") as f:
        f.write(str(best_cost) + "\n")
        f.write(",".join(vertex_id_tour) + "\n")

    print(f"  → Wrote .sol: {output_path}")


# ==========================
#   results.csv handling
# ==========================

def append_results_row(instance_name: str,
                       algorithm: str,
                       cutoff_str: str,
                       runtime_sec: float,
                       cost: int,
                       full_tour_flag: bool,
                       rel_error: float or str):
    """
    Append a row to results.csv in the script directory.

    Columns:
      instance, algorithm, cutoff, runtime_sec, cost, full_tour, rel_error
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_path = os.path.join(script_dir, "results.csv")

    file_exists = os.path.exists(results_path)

    with open(results_path, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(["instance", "algorithm", "cutoff",
                             "runtime_sec", "cost", "full_tour", "rel_error"])

        rel_error_str = rel_error if isinstance(rel_error, str) else f"{rel_error:.6f}"

        writer.writerow([
            instance_name,
            algorithm,
            cutoff_str,
            f"{runtime_sec:.6f}",
            cost,
            int(full_tour_flag),
            rel_error_str
        ])

    print(f"  → Appended to results.csv: {instance_name}, {algorithm}")


# ==========================
#   Batch mode for all .tsp
# ==========================

def batch_main():
    """
    Batch mode:
      - Runs on ALL .tsp files in a given folder.
      - For each instance:
          * Runs LS multiple times (different seeds), averages results (LS baseline).
          * Runs Approx once.
          * Runs BF once.
          * Writes .sol files for each run.
          * Writes 3 rows (BF, Approx, LS) into results.csv with RelError
            computed relative to best LS solution on that instance.
    """
    # <<< CONFIG: change these if you want >>>
    tsp_folder = r"E:\GATECH\2025-2026 Fall\Project\DATA"
    BF_CUTOFF = 10.0           # seconds
    LS_CUTOFF = 10.0           # seconds
    LS_SEEDS = list(range(10)) # 10 seeds for averaging

    print("Python executable:", sys.executable)
    print("Current working directory:", os.getcwd())
    print("Batch TSP folder:", tsp_folder)
    print("BF cutoff:", BF_CUTOFF, "LS cutoff:", LS_CUTOFF, "LS seeds:", LS_SEEDS)
    print("")

    if not os.path.isdir(tsp_folder):
        print("ERROR: TSP folder does not exist:", tsp_folder)
        input("Press Enter to exit...")
        return

    tsp_files = [f for f in os.listdir(tsp_folder) if f.lower().endswith(".tsp")]

    if not tsp_files:
        print("ERROR: No .tsp files found in folder.")
        input("Press Enter to exit...")
        return

    print(f"Found {len(tsp_files)} TSP instances:")
    for f in tsp_files:
        print(" •", f)
    print("")

    for tsp_file in tsp_files:
        full_path = os.path.join(tsp_folder, tsp_file)
        instance_name = os.path.splitext(tsp_file)[0]

        print(f"=== Processing {tsp_file} ===")

        # Load instance
        node_ids, coords = load_tsp_file(full_path)
        print(f"  Loaded {len(node_ids)} nodes.")
        if len(node_ids) == 0:
            print("  WARNING: No nodes parsed, skipping this file.")
            continue

        dist = build_distance_matrix(coords)
        print("  Distance matrix built.")

        n = len(node_ids)

        # ---- 1) Local Search baseline (multiple seeds) ----
        ls_cost_sum = 0.0
        ls_time_sum = 0.0
        best_ls_cost = float("inf")
        full_tour_ls = True

        for seed in LS_SEEDS:
            start = time.time()
            cost_ls, tour_ls = two_opt_local_search(dist, LS_CUTOFF, seed)
            end = time.time()
            runtime_ls = end - start

            ls_cost_sum += cost_ls
            ls_time_sum += runtime_ls
            if cost_ls < best_ls_cost:
                best_ls_cost = cost_ls

            # full tour?
            if len(tour_ls) != n:
                full_tour_ls = False

            # Write individual LS .sol for this seed
            write_solution(instance_name, "LS", cost_ls, tour_ls, node_ids,
                           cutoff=LS_CUTOFF, seed=seed)

        ls_avg_cost = ls_cost_sum / len(LS_SEEDS)
        ls_avg_time = ls_time_sum / len(LS_SEEDS)

        print(f"  LS avg cost: {ls_avg_cost:.2f}, best LS cost: {best_ls_cost}, avg time: {ls_avg_time:.6f}s")

        # ---- 2) Approx (MST 2-approx) ----
        start = time.time()
        cost_ap, tour_ap = mst_approx_tsp(dist)
        end = time.time()
        runtime_ap = end - start
        full_tour_ap = (len(tour_ap) == n)
        print(f"  Approx cost: {cost_ap}, time: {runtime_ap:.6f}s")

        # ---- 3) Brute Force ----
        start = time.time()
        cost_bf, tour_bf = brute_force_tsp(dist, BF_CUTOFF)
        end = time.time()
        runtime_bf = end - start
        full_tour_bf = (len(tour_bf) == n)
        print(f"  BF cost: {cost_bf}, time: {runtime_bf:.6f}s")

        # ---- write BF & Approx .sol files ----
        write_solution(instance_name, "Approx", cost_ap, tour_ap, node_ids)
        write_solution(instance_name, "BF", cost_bf, tour_bf, node_ids,
                       cutoff=BF_CUTOFF)

        # ---- RelError using best LS cost as baseline ----
        if best_ls_cost <= 0 or best_ls_cost == float("inf"):
            # degenerate case, can't compute meaningful RelError
            rel_ap = "NA"
            rel_bf = "NA"
            rel_ls = "NA"
        else:
            rel_ap = (cost_ap - best_ls_cost) / best_ls_cost
            rel_bf = (cost_bf - best_ls_cost) / best_ls_cost
            rel_ls = (ls_avg_cost - best_ls_cost) / best_ls_cost

        # ---- Append results rows (one per algorithm) ----
        append_results_row(instance_name, "LS",
                           cutoff_str=str(int(LS_CUTOFF)),
                           runtime_sec=ls_avg_time,
                           cost=int(ls_avg_cost),
                           full_tour_flag=full_tour_ls,
                           rel_error=rel_ls)

        append_results_row(instance_name, "Approx",
                           cutoff_str="",
                           runtime_sec=runtime_ap,
                           cost=cost_ap,
                           full_tour_flag=full_tour_ap,
                           rel_error=rel_ap)

        append_results_row(instance_name, "BF",
                           cutoff_str=str(int(BF_CUTOFF)),
                           runtime_sec=runtime_bf,
                           cost=cost_bf,
                           full_tour_flag=full_tour_bf,
                           rel_error=rel_bf)

        print("")

    print("=== Batch processing complete ===")
    print("results.csv is in the same folder as exec.py.")
    input("Done. Press Enter to exit...")


# ==========================
#   CLI single-run mode
# ==========================

def cli_main():
    """
    Single-run mode, closer to the project spec:

      python exec.py <instance_file> BF <cutoff_seconds>
      python exec.py <instance_file> Approx
      python exec.py <instance_file> LS <cutoff_seconds> <seed>

    This will:
      - Run the chosen algorithm on that instance.
      - Write the appropriate .sol file.
      - Append a single row to results.csv with rel_error = NA
        (since we don't know LS baseline from just one run).
    """
    if len(sys.argv) < 3:
        print("Usage:")
        print("  BF:     python exec.py <instance_file> BF <cutoff_seconds>")
        print("  Approx: python exec.py <instance_file> Approx")
        print("  LS:     python exec.py <instance_file> LS <cutoff_seconds> <seed>")
        input("Press Enter to exit...")
        sys.exit(1)

    instance_file = sys.argv[1]
    method = sys.argv[2]

    print("Python executable:", sys.executable)
    print("Current working directory:", os.getcwd())
    print("Instance file:", instance_file)
    print("Method:", method)

    if not os.path.isfile(instance_file):
        print("ERROR: Instance file does not exist.")
        input("Press Enter to exit...")
        sys.exit(1)

    node_ids, coords = load_tsp_file(instance_file)
    if not node_ids:
        print("ERROR: Could not parse any nodes from file.")
        input("Press Enter to exit...")
        sys.exit(1)

    dist = build_distance_matrix(coords)
    n = len(node_ids)
    instance_name = os.path.splitext(os.path.basename(instance_file))[0]

    if method == "BF":
        if len(sys.argv) < 4:
            print("ERROR: BF requires cutoff_seconds.")
            input("Press Enter to exit...")
            sys.exit(1)
        cutoff = float(sys.argv[3])
        start = time.time()
        cost_bf, tour_bf = brute_force_tsp(dist, cutoff)
        end = time.time()
        runtime_bf = end - start
        full_tour_bf = (len(tour_bf) == n)
        write_solution(instance_name, "BF", cost_bf, tour_bf, node_ids,
                       cutoff=cutoff)
        append_results_row(instance_name, "BF", str(int(cutoff)),
                           runtime_bf, cost_bf, full_tour_bf, rel_error="NA")

    elif method == "Approx":
        start = time.time()
        cost_ap, tour_ap = mst_approx_tsp(dist)
        end = time.time()
        runtime_ap = end - start
        full_tour_ap = (len(tour_ap) == n)
        write_solution(instance_name, "Approx", cost_ap, tour_ap, node_ids)
        append_results_row(instance_name, "Approx", "",
                           runtime_ap, cost_ap, full_tour_ap, rel_error="NA")

    elif method == "LS":
        if len(sys.argv) < 5:
            print("ERROR: LS requires cutoff_seconds and seed.")
            input("Press Enter to exit...")
            sys.exit(1)
        cutoff = float(sys.argv[3])
        seed = int(sys.argv[4])
        start = time.time()
        cost_ls, tour_ls = two_opt_local_search(dist, cutoff, seed)
        end = time.time()
        runtime_ls = end - start
        full_tour_ls = (len(tour_ls) == n)
        write_solution(instance_name, "LS", cost_ls, tour_ls, node_ids,
                       cutoff=cutoff, seed=seed)
        append_results_row(instance_name, "LS", str(int(cutoff)),
                           runtime_ls, cost_ls, full_tour_ls, rel_error="NA")

    else:
        print(f"ERROR: Unknown method '{method}'. Use BF, Approx, or LS.")
        input("Press Enter to exit...")
        sys.exit(1)


# ==========================
#   Main entry
# ==========================

def main():
    # If called with arguments -> CLI single-run mode.
    # If double-clicked (no args) -> batch mode on all .tsp files.
    if len(sys.argv) >= 3:
        cli_main()
    else:
        batch_main()


if __name__ == "__main__":
    main()
