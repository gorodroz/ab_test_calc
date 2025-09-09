import numpy as np
from tabulate import tabulate

def run_montecarlo(data, kpi_type="conversion", n_sim=10000, alpha=0.05):
    results={}
    sims={}

    if kpi_type == "conversion":
        for group, vals in data.items():
            p_hat = vals["conversions"]/vals["users"]
            sims[group]=np.random.binomial(n=vals["users"], p=p_hat, size=n_sim)/vals["users"]

    else:
        for group, vals in data.items():
            mean=vals["total"]/vals["users"]
            sd=vals.get("sd", mean)
            sims[group]= np.random.normal(loc=mean, scale=sd, size=(n_sim,))

    groups = list(sims.keys())
    comparisons=[]
    for i in range(len(groups)):
        for j in range(i+1, len(groups)):
            g1, g2 = groups[i], groups[j]
            diff = sims[g1] - sims[g2]
            prob = (diff > 0).mean()
            ci_low, ci_high = np.percentile(diff, [alpha/2*100, (1-alpha/2)*100])
            comparisons.append({
                "group1": g1,
                "group2": g2,
                "prob_g1_better": prob,
                "ci_low": ci_low,
                "ci_high": ci_high
            })
    results["comparisons"]=comparisons
    results["raw_sims"]=sims
    return results

def print_montecarlo_results(results):
    print("\n=== Monte Carlo Simulation Results ===")
    rows= [
        (
            r["group1"], r["group2"],
            f"{r["prob_g1_better"]:.3f}",
            f"[{r["ci_low"]:.4f}, {r["ci_high"]:.4f}"
        )
        for r in results["comparisons"]
    ]
    print(tabulate(rows, headers=["Group1", "Group2", "P(g1>g2)", "CI for diff"], tablefmt="grid"))