import time
from ortools.linear_solver import pywraplp
from ortools.sat.python import cp_model
from collections import Counter
from pulp import LpProblem, LpVariable, lpSum, LpMinimize, LpStatusOptimal, value, PULP_CBC_CMD
from pysat.solvers import Minisat22
from itertools import combinations
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
import functools
import multiprocessing
import signal

class TimeoutException(Exception):
    pass

def _target_func(queue, func, args, kwargs):
    try:
        result = func(*args, **kwargs)
        queue.put((True, result))
    except Exception as e:
        queue.put((False, e))

def run_with_timeout(func, args=(), kwargs={}, timeout=10):
    queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=_target_func, args=(queue, func, args, kwargs))
    process.start()
    process.join(timeout)

    if process.is_alive():
        process.terminate()
        process.join()
        return None, f"Timed out after {timeout}s"
    else:
        if not queue.empty():
            success, result = queue.get()
            return (result, None) if success else (None, f"Exception: {result}")
        else:
            return None, "No result returned"
        
def solve_bin_packing_ilp(item_groups, bin_capacity):
    # Flatten items: (value, group_id)
    items = []
    for group_id, group in enumerate(item_groups):
        for item in group:
            items.append((item, group_id))

    num_items = len(items)
    max_bins = num_items

    solver = pywraplp.Solver.CreateSolver('SCIP')
    if not solver:
        raise Exception("Solver not available")

    x = {}
    y = {}
    for i in range(num_items):
        for j in range(max_bins):
            x[i, j] = solver.IntVar(0, 1, f'x_{i}_{j}')
    for j in range(max_bins):
        y[j] = solver.IntVar(0, 1, f'y_{j}')

    # Constraints
    for i in range(num_items):
        solver.Add(solver.Sum(x[i, j] for j in range(max_bins)) == 1)
    for j in range(max_bins):
        solver.Add(solver.Sum(x[i, j] * items[i][0] for i in range(num_items)) <= bin_capacity * y[j])
    for j in range(max_bins):
        groups_in_bin = set(gid for _, gid in items)
        for group_id in groups_in_bin:
            solver.Add(solver.Sum(x[i, j] for i in range(num_items) if items[i][1] == group_id) <= 1)

    solver.Minimize(solver.Sum(y[j] for j in range(max_bins)))

    status = solver.Solve()
    if status != pywraplp.Solver.OPTIMAL:
        return None

    bins = [[] for _ in range(max_bins)]
    for i in range(num_items):
        for j in range(max_bins):
            if x[i, j].solution_value() > 0.5:
                bins[j].append(items[i])
                break

    used_bins = [b for b in bins if b]
    return used_bins

def best_fit_with_group_constraint(item_groups, bin_capacity):
    # Flatten item_groups into list of (size, group_id)
    items = []
    for group_id, group in enumerate(item_groups):
        for item in group:
            items.append((item, group_id))

    # Sort items by size descending (Best Fit Decreasing)
    items.sort(key=lambda x: -x[0])

    bins = []

    for item in items:
        placed = False
        for b in bins:
            b_sum = sum(i[0] for i in b)
            b_groups = {i[1] for i in b}
            if b_sum + item[0] <= bin_capacity and item[1] not in b_groups:
                b.append(item)
                placed = True
                break
        if not placed:
            bins.append([item])

    return bins


  
def prepare_items(item_groups):
    items = []
    for gid, group in enumerate(item_groups):
        for v in group:
            items.append((v, gid))
    return items

def cp_bin_packing(item_groups, bin_capacity):
    items = prepare_items(item_groups)
    n = len(items)
    max_bins = n

    model = cp_model.CpModel()
    bins = [model.NewIntVar(0, max_bins - 1, f'bin_{i}') for i in range(n)]

    # At most one item from each group per bin
    groups = set(g for _, g in items)
    for group_id in groups:
        indices = [i for i, (_, g) in enumerate(items) if g == group_id]
        for i1 in range(len(indices)):
            for i2 in range(i1 + 1, len(indices)):
                model.Add(bins[indices[i1]] != bins[indices[i2]])

    # Capacity constraints per bin
    for b in range(max_bins):
        in_bin = []
        for i in range(n):
            in_bin_var = model.NewBoolVar(f'in_bin_{i}_{b}')
            model.Add(bins[i] == b).OnlyEnforceIf(in_bin_var)
            model.Add(bins[i] != b).OnlyEnforceIf(in_bin_var.Not())
            in_bin.append(in_bin_var)
        model.Add(
            sum(items[i][0] * in_bin[i] for i in range(n)) <= bin_capacity
        )

    # Minimize number of bins used
    bin_used = [model.NewBoolVar(f'bin_used_{b}') for b in range(max_bins)]
    for b in range(max_bins):
        # Link bin_used[b] to whether any item is assigned to bin b
        model.AddMaxEquality(bin_used[b], [model.NewBoolVar(f'item_{i}_in_bin_{b}') for i in range(n)])
        # But to connect those vars, we reuse in_bin variables:
        # We'll relax and just link bin_used[b] to OR of in_bin variables:
        model.AddMaxEquality(bin_used[b], in_bin)

    model.Minimize(sum(bin_used))

    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        result_bins = [[] for _ in range(max_bins)]
        for i in range(n):
            b = solver.Value(bins[i])
            result_bins[b].append(items[i])
        return [b for b in result_bins if b]
    else:
        return None
    




# Test cases with bin capacity, max-min group values ≤ 1
test_cases = [
    { #0
        "groups": [
            [2,2,3,3],
            [1,1,2,2],
            [6,6,7,7],
            [8,8,9,9],
            [10,10,11,11],
            [3,3,3,4],
        ],
        "bin_capacity": 15
    },
    {#1
        "groups": [
            [1]*5 + [2]*4,
            [3]*5 + [4]*4,
            [5]*5 + [6]*4,
            [7]*5 + [8]*4,
        ],
        "bin_capacity": 20
    },
    {#2
        "groups": [
            [1,1,2],
            [2,3,3],
            [4,4,5],
            [6,7,7],
        ],
        "bin_capacity": 10
    },
    {#3
        "groups": [
            [9]*6 + [10]*6,
            [1]*5,
            [2]*5 + [3]*5,
            [5,5,5,5,5,6],
        ],
        "bin_capacity": 18
    },
    {#4
        "groups": [
            [1],
            [2],
            [3],
            [4],
            [5],
            [6],
            [7],
            [8],
            [9],
            [10],
        ],
        "bin_capacity": 10
    },
    {#5
        "groups": [
            [1,1,2,2],
            [5,5,6,6],
            [8,8,9,9],
            [2,2,2,2],
            [11,12,12,12],
        ],
        "bin_capacity": 15
    },
    {#6
        "groups": [
            [1]*5 + [2]*4,
            [7]*7 + [8]*6,
            [9]*8 + [10]*5,
            [4]*6 + [5]*6,
            [7]*7 + [8]*8,
            [8]*3 + [9]*2,
            [3]*5 + [4]*4,
            [5]*5 + [6]*4,
            [7]*5 + [8]*4,
        ],
        "bin_capacity": 50
    },
    {#7
        "groups": [
            [1]*15 + [2]*14,
            [8]*17 + [7]*16,
            [9]*18 + [10]*15,
            [4]*16 + [5]*16,
            [7]*17 + [8]*18,
            [8]*13 + [9]*12,
            [3]*15 + [4]*14,
            [5]*15 + [6]*14,
            [7]*15 + [8]*14,
        ],
        "bin_capacity": 200
    },
     {#8
        "groups": [
            [1]*7 + [2]*6,
            [5]*6 + [6]*5,
            [9]*7 + [10]*6,
            [3]*5 + [4]*5,
            [6]*5 + [7]*6,
            [8]*4 + [9]*3,
            [2]*6 + [3]*5,
            [4]*6 + [5]*5,
            [6]*5 + [7]*4,
            [1]*5 + [3]*5,
            [4]*4 + [6]*6,
        ],
        "bin_capacity": 48
    },
    {#9
        "groups": [
            [3]*6 + [4]*5,
            [7]*6 + [8]*6,
            [9]*5 + [10]*6,
            [4]*5 + [5]*5,
            [6]*3 + [7]*5,
            [7]*3 + [8]*2,
            [2]*5 + [3]*6,
            [5]*4 + [6]*6,
            [7]*6 + [9]*4,
            [8]*4 + [10]*5,
            [2]*6 + [3]*6,
            [4]*5 + [5]*5,
            [6]*12 + [7]*5,
            [7]*3 + [8]*4,
            [2]*5 + [3]*6,
            [5]*4 + [6]*4,
            [2]*15 + [3]*6,
            [5]*4 + [6]*16,
            [7]*6 + [9]*4,
            [8]*4 + [10]*5,
            [2]*6 + [3]*6,
            [4]*5 + [5]*5,
            [6]*6 + [7]*5,
        ],
        "bin_capacity": 200
    },
    {#10
        "groups": [
            [2]*5 + [3]*6,
            [8]*6 + [9]*5,
            [10]*5 + [11]*4,
            [5]*5 + [6]*5,
            [7]*6 + [8]*6,
            [6]*3 + [7]*3,
            [3]*6 + [4]*5,
            [5]*6 + [6]*4,
            [7]*6 + [8]*5,
            [9]*4 + [10]*4,
            [4]*6 + [5]*6,
            [5]*5 + [6]*5,
            [7]*6 + [8]*6,
            [6]*3 + [7]*3,
            [3]*6 + [4]*5,
        ],
        "bin_capacity": 100
    },
    {#11
        "groups": [
            [2]*5 + [3]*6,
            [8]*6 + [9]*5,
            [10]*5 + [11]*4,
            [5]*5 + [6]*5,
            [7]*6 + [8]*6,
            [6]*3 + [7]*3,
            [3]*6 + [4]*5,
            [5]*6 + [6]*4,
            [7]*6 + [8]*5,
            [9]*4 + [10]*4,
            [4]*6 + [5]*6,
            [5]*5 + [6]*5,
            [7]*6 + [8]*6,
            [6]*3 + [7]*3,
            [3]*6 + [4]*5,
        ],
        "bin_capacity": 300
    },
    {#12
        "groups": [
            [3]*6 + [4]*5,
            [7]*6 + [8]*6,
            [9]*5 + [10]*6,
            [4]*5 + [5]*5,
            [6]*6 + [7]*5,
            [7]*3 + [8]*4,
            [2]*5 + [3]*6,
            [5]*4 + [6]*6,
            [7]*6 + [9]*4,
            [8]*4 + [10]*5,
            [2]*6 + [3]*6,
            [4]*5 + [5]*5,
            [6]*6 + [7]*5,
            [7]*3 + [8]*4,
            [2]*5 + [3]*6,
            [5]*4 + [6]*6,
            [2]*5 + [3]*6,
            [5]*4 + [6]*6,
            [7]*6 + [9]*4,
            [8]*4 + [10]*5,
            [2]*6 + [3]*6,
            [4]*5 + [5]*5,
            [6]*6 + [7]*5,
        ],
        "bin_capacity": 300
    },
]


def print_bins(label, bins, show_items=True):
    print(f"\n{label} solution: {len(bins)} bins")
    for i, b in enumerate(bins):
        vals = [v for v, _ in b]
        if show_items:
            print(f"  {label} Bin {i + 1}: items = {vals}, sum = {sum(vals)}")

if __name__ == "__main__":
    for idx, case in enumerate(test_cases, 1):
        groups = case["groups"]
        capacity = case["bin_capacity"]


        print(f"\n=== Test Case {idx} (Bin capacity = {capacity}) ===")

        timeout_sec = 60  # Change as needed

        start = time.time()
        bf_bins = best_fit_with_group_constraint(groups, capacity)
        bf_time = time.time() - start

        start = time.time()
        ilp_bins, ilp_err = run_with_timeout(solve_bin_packing_ilp, args=(groups, capacity), timeout=timeout_sec)
        ilp_time = time.time() - start

        start = time.time()
        cp_bins, cp_err = run_with_timeout(cp_bin_packing, args=(groups, capacity), timeout=timeout_sec)

        cp_time = time.time() - start

        #times out
        # start = time.time()
        # bnb_bins = branch_and_bound(groups, capacity)
        # bnb_time = time.time() - start

        #wrong answer
        # start = time.time()
        # colg_bins = column_generation_binpacking(groups, capacity)
        # colg_time = time.time() - start

        #times out
        # start = time.time()
        # sat_bins = sat_binpacking(groups, capacity)
        # sat_time = time.time() - start
        

        

        print(f"Best Fit: bins used = {len(bf_bins)}, time = {bf_time:.4f}s")
        print(f"ILP     : bins used = {len(ilp_bins) if ilp_bins else 'No solution'}, time = {ilp_time:.4f}s")
        print(f"CP      : bins used = {len(cp_bins) if cp_bins else 'No solution'}, time = {cp_time:.4f}s")

        # print_bins("Best Fit", bf_bins)
        # print_bins("ILP", ilp_bins)
        # print_bins("CP", cp_bins)


    
