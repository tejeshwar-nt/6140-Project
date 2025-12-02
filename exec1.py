#!/usr/bin/env python3
import sys
import math
import os
from typing import List, Tuple

# ==========================
#   Data loading & distances
# ==========================

def load_tsp_file(filename: str) -> Tuple[List[int], List[Tuple[float, float]]]:
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
                continue
            node_ids.append(node_id)
            coords.append((x, y))
    return node_ids, coords


def build_distance_matrix(coords: List[Tuple[float, float]]) -> List[List[int]]:
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
    total = 0
    n = len(tour)
    for i in range(n):
        u = tour[i]
        v = tour[(i + 1) % n]
        total += dist[u][v]
    return total


# ====================================
#   MST-based 2-Approx TSP
# ====================================

def prim_mst(dist: List[List[int]]):
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


def preorder_traversal(adj, start=0):
    n = len(adj)
    visited = [False] * n
    order = []

    def dfs(u):
        visited[u] = True
        order.append(u)
        for v in adj[u]:
            if not visited[v]:
                dfs(v)

    dfs(start)
    return order


def mst_approx_tsp(dist):
    if not dist:
        return 0, []

    mst_adj = prim_mst(dist)
    tour = preorder_traversal(mst_adj, 0)
    cost = tour_cost(tour, dist)
    return cost, tour


# ==========================
#   Write .sol file
# ==========================

def write_solution(instance_name, best_cost, best_tour, node_ids):
    file_name = f"{instance_name} Approx.sol"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, file_name)

    with open(output_path, "w") as f:
        f.write(str(best_cost) + "\n")
        f.write(",".join(str(node_ids[i]) for i in best_tour) + "\n")

    print(f"  → Wrote: {output_path}")


# ==========================
#   Batch run on all TSPs
# ==========================

def main():
    # Folder containing tsp files
    tsp_folder = r"E:\GATECH\2025-2026 Fall\Project\DATA"

    print("Python executable:", sys.executable)
    print("Current working directory:", os.getcwd())
    print("Scanning folder:", tsp_folder)
    print("")

    tsp_files = [f for f in os.listdir(tsp_folder) if f.lower().endswith(".tsp")]

    if not tsp_files:
        print("ERROR: No .tsp files found. Check the folder path.")
        input("Press Enter to exit...")
        return

    print(f"Found {len(tsp_files)} TSP instances:")
    for f in tsp_files:
        print(" •", f)
    print("")

    # Process each .tsp file
    for tsp_file in tsp_files:
        full_path = os.path.join(tsp_folder, tsp_file)
        instance_name = os.path.splitext(tsp_file)[0]

        print(f"Processing {tsp_file}...")

        # 1. Load
        node_ids, coords = load_tsp_file(full_path)
        print(f"  Loaded {len(node_ids)} nodes.")

        # 2. Distances
        dist = build_distance_matrix(coords)
        print("  Distance matrix built.")

        # 3. Approx
        cost, tour = mst_approx_tsp(dist)
        print(f"  Approximate cost: {cost}")

        # 4. Output
        write_solution(instance_name, cost, tour, node_ids)

        print("")

    print("=== All instances processed ===")
    input("Done. Press Enter to exit...")


if __name__ == "__main__":
    main()
