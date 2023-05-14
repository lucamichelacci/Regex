import gurobipy as gp
from gurobipy import GRB
import numpy as np

def find_optimal_weights(integers):
    n = len(integers)
    m = gp.Model('miqp')

    # Set up decision variables
    w = m.addVars(n, lb=-1, ub=1, vtype=GRB.CONTINUOUS, name="w")
    z = m.addVars(n, vtype=GRB.BINARY, name="z")

    # Set up objective function
    obj = gp.QuadExpr()
    for i in range(n):
        obj += integers[i] * w[i]
    m.setObjective(obj, GRB.MAXIMIZE)

    # Set up cardinality constraint
    card = m.addConstr(z.sum() <= 4)

    # Set up constraints to enforce that z_i * w_i = 0 if z_i = 0
    for i in range(n):
        m.addConstr(w[i] <= z[i])
        m.addConstr(w[i] >= -z[i])

    # Solve the optimization problem
    m.setParam('OutputFlag', 0)  # Turn off Gurobi output
    m.optimize()

    # Return the optimal weights
    weights = [w[i].X for i in range(n)]
    return np.array(weights)
