import numpy as np
from scipy.optimize import linprog
from tabulate import tabulate

def run_linear_optimization(objective, constraints, bounds, maximize=True):
    c = np.array(objective, dtype=float)
    if maximize:
        c = -c

    A = []
    b = []

    for coeffs, sign, rhs in constraints.values():
        if sign == "<=":
            A.append(coeffs)
            b.append(rhs)
        elif sign == ">=":
            # flip sign to turn into <=
            A.append([-c for c in coeffs])
            b.append(-rhs)
        else:
            raise ValueError("Only <= or >= constraints supported")

    res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method="highs")

    solution = {
        "success": res.success,
        "objective_value": -res.fun if maximize else res.fun,
        "variables": res.x
    }
    return solution


def print_linear_results(result, var_names):
    print("\n=== Linear Optimization Results ===")
    if not result["success"]:
        print("Optimization failed")
        return

    rows = [(name, f"{val:.4f}") for name, val in zip(var_names, result["variables"])]
    print(tabulate(rows, headers=["Variable", "Value"], tablefmt="grid"))
    print(f"\nOptimal Objective Value = {result['objective_value']:.4f}")