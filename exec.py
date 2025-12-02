#!/usr/bin/env python3

import argparse
import itertools
import math
import os
import random
import time
from typing import List, Tuple


Coord = Tuple[float, float]
Tour = List[int]

# Data loading
def read_tsp_coordinates(path: str) -> List[Coord]:
    coordinate_pairs: List[Coord] = []
    reading_coordinates = False

    with open(path, "r") as file_reader:
        for current_line in file_reader:
            stripped_line = current_line.strip()
            if stripped_line.upper().startswith("NODE_COORD_SECTION"):
                reading_coordinates = True
                continue
            if stripped_line.upper().startswith("EOF"):
                break
            if not reading_coordinates:
                continue
            if not stripped_line:
                continue
            # just get x and y
            split_parts = stripped_line.split()
            x_coord = float(split_parts[1])
            y_coord = float(split_parts[2])
            coordinate_pairs.append((x_coord, y_coord))
    return coordinate_pairs


# create distance matrix
def build_distance_matrix(coords: List[Coord]) -> List[List[int]]:
    num_points = len(coords)
    # initialize with zeros
    distance_matrix = []
    for i in range(num_points):
        row = [0] * num_points
        distance_matrix.append(row)
    for i in range(num_points):
        point_x, point_y = coords[i]
        for j in range(i + 1, num_points):
            other_x, other_y = coords[j]
            x_diff = point_x - other_x
            y_diff = point_y - other_y
            # euclidean distance formula
            raw_distance = math.sqrt(x_diff * x_diff + y_diff * y_diff)
            rounded_dist = int(round(raw_distance))
            distance_matrix[i][j] = rounded_dist
            distance_matrix[j][i] = rounded_dist
    return distance_matrix

# calculate tour length
def tour_length(tour: Tour, dist: List[List[int]]) -> int:
    tour_size = len(tour)
    total_distance = 0
    for index in range(tour_size - 1):
        current_point = tour[index]
        next_point = tour[index + 1]
        total_distance += dist[current_point][next_point]
    last_point = tour[-1]
    first_point = tour[0]
    total_distance += dist[last_point][first_point]
    return total_distance

# create random tour list
def random_tour(n: int) -> Tour:
    point_list = list(range(n))
    random.shuffle(point_list)
    return point_list

# 2-opt swap
def two_opt_swap(tour: Tour, i: int, j: int) -> Tour:
    before_segment = tour[:i]
    reversed_segment = list(reversed(tour[i:j+1]))
    after_segment = tour[j+1:]
    return before_segment + reversed_segment + after_segment

# brute force tsp
def brute_force_tsp(dist: List[List[int]], cutoff: float) -> Tuple[int, List[int]]:
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
        cost = tour_length(tour, dist)
        if cost < best_cost:
            best_cost = cost
            best_tour = tour
    if best_tour is None:
        best_tour = list(range(n))
        best_cost = tour_length(best_tour, dist)
    return best_cost, best_tour

# prim's minimum spanning tree
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

# preorder traversal
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

# mst approximation
def mst_approx_tsp(dist):
    if not dist:
        return 0, []

    mst_adj = prim_mst(dist)
    tour = preorder_traversal(mst_adj, 0)
    cost = tour_length(tour, dist)
    return cost, tour


# local search - hill climbing with 2-opt
def hill_climb_2opt(dist: List[List[int]], time_limit: float, first_improvement: bool = True) -> Tuple[Tour, int]:
    num_points = len(dist)
    algorithm_start = time.time()

    best_tour_found = None
    best_distance_found = math.inf

    # keep trying new random starting points until time runs out
    while time.time() - algorithm_start < time_limit:
        # get random starting tour
        working_tour = random_tour(num_points)
        working_distance = tour_length(working_tour, dist)
        keep_improving = True
        # climb from this starting point
        while keep_improving and (time.time() - algorithm_start < time_limit):
            keep_improving = False
            best_move_tour = working_tour
            best_move_distance = working_distance
            # try all possible 2-opt swaps
            for i in range(1, num_points - 1):
                for j in range(i + 1, num_points):
                    if time.time() - algorithm_start >= time_limit:
                        break
                    new_tour = two_opt_swap(working_tour, i, j)
                    new_distance = tour_length(new_tour, dist)
                    if new_distance < best_move_distance:
                        best_move_tour = new_tour
                        best_move_distance = new_distance
                        keep_improving = True
                        if first_improvement:
                            # take the first improvement (greedy)
                            break
                if first_improvement and keep_improving:
                    break
            # accept improvement
            if keep_improving:
                working_tour = best_move_tour
                working_distance = best_move_distance
        # update global best
        if working_distance < best_distance_found:
            best_distance_found = working_distance
            best_tour_found = working_tour
    return best_tour_found, best_distance_found



# write solution file
def write_solution_file(instance_path: str, method: str, cutoff: int, seed: int, tour: Tour, length: int) -> str:
    filename_with_ext = os.path.basename(instance_path)
    instance_name, file_extension = os.path.splitext(filename_with_ext)
    instance_name_lower = instance_name.lower()
    method_upper = method.upper()
    cutoff_value = str(int(cutoff))
    seed_value = str(int(seed)) if seed is not None else "0"

    # construct output filename
    output_filename = f"{instance_name_lower}_{method_upper}_{cutoff_value}_{seed_value}.sol"
    # convert from 0-based to 1-based indexing
    tour_one_based = []
    for vertex in tour:
        tour_one_based.append(vertex + 1)
    # add the first vertex at the end
    if tour_one_based:
        tour_one_based.append(tour_one_based[0])
    # create comma-separated string of the tour
    tour_string = ", ".join(str(vertex_id) for vertex_id in tour_one_based)
    # write the solution file
    with open(output_filename, "w") as output_file:
        output_file.write(f"{float(length)}\n")
        output_file.write(tour_string + "\n")
    return output_filename

# argument parsing
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-inst", required=True, help="path to .tsp")
    parser.add_argument("-alg", required=True, help="algorithm name")
    parser.add_argument("-time", required=False, type=int, help="time limit in seconds")
    parser.add_argument("-seed", required=False, type=int, help="random seed")

    args = parser.parse_args()

    alg_upper = args.alg.upper()
    if alg_upper not in ["LS", "APPROX", "BF"]:
        raise ValueError("This executable only supports LS, APPROX, or BF")

    if alg_upper == "LS" and args.time is None:
        raise ValueError("Time limit is required for LS")
    
    if alg_upper == "BF" and args.time is None:
        raise ValueError("Time limit is required for BF")

    if alg_upper in ["LS", "APPROX"] and args.seed is None:
        raise ValueError("Seed is required for LS and APPROX")

    if args.seed is not None:
        random.seed(args.seed)
    
    coords = read_tsp_coordinates(args.inst)
    dist = build_distance_matrix(coords)
    
    if alg_upper == "LS":
        best_tour, best_len = hill_climb_2opt(dist, time_limit=float(args.time))
        cutoff_value = args.time
        seed_value = args.seed
    elif alg_upper == "BF":
        best_len, best_tour = brute_force_tsp(dist, cutoff=float(args.time))
        cutoff_value = args.time
        seed_value = args.seed if args.seed is not None else 0
    else:  # APPROX
        best_len, best_tour = mst_approx_tsp(dist)
        cutoff_value = args.time if args.time is not None else 0
        seed_value = args.seed
    
    sol_path = write_solution_file(instance_path=args.inst, method=args.alg, cutoff=cutoff_value, seed=seed_value, tour=best_tour, length=best_len)

    print(f"Best tour length: {best_len}")
    print(f"Solution written to: {sol_path}")


if __name__ == "__main__":
    main()