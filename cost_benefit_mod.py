import numpy as np
from tabulate import tabulate

def run_cost_benefit_simulation(cost_params, benefit_params, simulations=10000, discount_rate=0.0, horizon=1):
    costs = np.random.normal(loc=cost_params["mean"], scale=cost_params.get("sd", 0), size=simulations)
    benefits = np.random.normal(loc=benefit_params["mean"], scale=benefit_params.get("sd", 0), size=simulations)

    discounted_benefits = benefits * (1 - (1 + discount_rate) ** -horizon) / discount_rate if discount_rate > 0 else benefits * horizon
    npv = discounted_benefits - costs

    roi = (discounted_benefits - costs) / costs

    results = {
        "npv_mean": np.mean(npv),
        "npv_median": np.median(npv),
        "npv_p5": np.percentile(npv, 5),
        "npv_p95": np.percentile(npv, 95),
        "prob_positive": np.mean(npv > 0),
        "roi_mean": np.mean(roi),
        "roi_p5": np.percentile(roi, 5),
        "roi_p95": np.percentile(roi, 95)
    }

    return results


def print_cost_benefit_results(results):
    print("\n=== Cost-Benefit Simulation Results ===")
    rows = [
        ("NPV (mean)", f"{results['npv_mean']:.2f}"),
        ("NPV (median)", f"{results['npv_median']:.2f}"),
        ("NPV (5th pct)", f"{results['npv_p5']:.2f}"),
        ("NPV (95th pct)", f"{results['npv_p95']:.2f}"),
        ("Probability NPV > 0", f"{results['prob_positive']:.2%}"),
        ("ROI (mean)", f"{results['roi_mean']:.2f}"),
        ("ROI (5th pct)", f"{results['roi_p5']:.2f}"),
        ("ROI (95th pct)", f"{results['roi_p95']:.2f}")
    ]
    print(tabulate(rows, headers=["Metric", "Value"], tablefmt="grid"))