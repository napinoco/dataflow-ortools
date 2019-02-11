from ortools.linear_solver import pywraplp
import numpy as np
import pandas as pd

A = [[10, 10, 0], [0, 10, 10], [5, 10, 5], [2, 0, 0], [0, 2, 0], [0, 0, 2]]
# stock = [100, 100, 1000, 1000, 1000]
np.random.seed(1)
R = np.random.randint(0, 50, (100, 3))

sol_x = {}
sol_y = {}
for store, req in enumerate(R):
    solver = pywraplp.Solver('prob_{0}'.format(store), pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)
    x = {i_pack: solver.IntVar(0, solver.infinity(), 'pack_{0}'.format(i_pack)) for i_pack, ai in enumerate(A)}
    y = {j_sku: solver.Sum([A[i_pack][j_sku] * x[i_pack] for i_pack in range(len(A))]) for j_sku in
         range(len(A[0]))}
    abs_diff = {}
    for j_sku in y:
        abs_diff[j_sku] = solver.NumVar(0, solver.infinity(), 'diff_{0}'.format(j_sku))
        diff = req[j_sku] - y[j_sku]
        solver.Add(-abs_diff[j_sku] <= diff)
        solver.Add(diff <= abs_diff[j_sku])
    solver.Minimize(solver.Sum(abs_diff.values()) + 0.1 * solver.Sum(x.values()))
    status = solver.Solve()
    if status == solver.OPTIMAL:
        print('optimal_{0}'.format(store))
    else:
        print('error')
    sol_x[store] = {i: x[i].solution_value() for i in x}
    sol_y[store] = {j: y[j].solution_value() for j in y}

if (pd.Series([43, 64, 59, 826, 286, 693]) == pd.DataFrame(sol_x).sum(axis=1)).all():
    print('success')
else:
    print('error')