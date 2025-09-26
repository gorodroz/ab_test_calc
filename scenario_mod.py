from tabulate import tabulate
from visual_mod import plot_scenarios

def run_scenario_analysis(base_params, scenarios, model_func):
    results = {}
    for scen_name, multipliers in scenarios.items():
        params = {k: base_params[k] * multipliers.get(k, 1.0) for k in base_params}
        outcome = model_func(params)
        results[scen_name] = {"params": params, "outcome": outcome}

    return results


def print_scenario_results(results):
    print("\n=== Scenario Analysis Results ===")
    rows = [
        (
            scen,
            f"{vals['outcome']:.4f}",
            str({k: round(v, 2) for k, v in vals['params'].items()})
        )
        for scen, vals in results.items()
    ]
    print(tabulate(rows, headers=["Scenario", "Outcome", "Parameters"], tablefmt="grid"))
    plot_scenarios(results)
