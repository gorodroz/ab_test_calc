import numpy as np
from scipy.stats import beta, norm
from tabulate import tabulate


def decision_analysis(data, kpi_type="conversion", utility=None, alpha=0.05, simulations=5000):
    if utility is None:
        utility = {"gain": 1.0, "loss": -1.0}

    group_names = list(data.keys())
    sims = {}

    for g, vals in data.items():
        if kpi_type == "conversion":
            a = vals["conversions"] + 1
            b = vals["users"] - vals["conversions"] + 1
            sims[g] = beta.rvs(a, b, size=simulations)
        else:
            mean = vals["total"] / vals["users"]
            sd = vals.get("sd", mean)
            sims[g] = norm.rvs(loc=mean, scale=sd / np.sqrt(vals["users"]), size=simulations)

    eu = {g: 0.0 for g in group_names}
    sim_matrix = np.column_stack([sims[g] for g in group_names])

    for row in sim_matrix:
        best_idx = np.argmax(row)
        best_group = group_names[best_idx]
        for i, g in enumerate(group_names):
            if g == best_group:
                eu[g] += utility["gain"]
            else:
                eu[g] += utility["loss"]

    eu = {g: val / simulations for g, val in eu.items()}

    #Best decision
    leader = max(eu, key=eu.get)

    return {
        "expected_utilities": eu,
        "decision": leader,
        "utility": utility,
        "alpha": alpha
    }


def print_decision_results(results):
    print("\n=== Bayesian Decision-Theoretic Analysis ===")
    rows = [(g, f"{eu:.4f}") for g, eu in results["expected_utilities"].items()]
    print(tabulate(rows, headers=["Group", "Expected Utility"], tablefmt="grid"))
    print(f"\n>>> Recommended Decision: Choose **{results['decision']}**")
    print(f"Utility function: gain={results['utility']['gain']}, loss={results['utility']['loss']}")
