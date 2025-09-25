import numpy as np
from scipy.stats import beta, norm
from tabulate import tabulate

def bayesian_test(groups, kpi_type="conversion", simulations=100000, alpha=0.05, utility=None):
    results = {}
    samples = {}

    if kpi_type == "conversion":
        for g, vals in groups.items():
            a = vals["conversions"] + 1
            b = vals["users"] - vals["conversions"] + 1
            samples[g] = beta.rvs(a, b, size=simulations)
    else:
        for g, vals in groups.items():
            mean = vals["total"] / vals["users"]
            sd = vals.get("sd", mean)
            samples[g] = norm.rvs(loc=mean, scale=sd / np.sqrt(vals["users"]), size=simulations)

    # compute pairwise probabilities and intervals
    posteriors = {}
    for g, s in samples.items():
        ci_low = np.percentile(s, 100 * (alpha / 2))
        ci_high = np.percentile(s, 100 * (1 - alpha / 2))
        posteriors[g] = {
            "mean": np.mean(s),
            "median": np.median(s),
            f"{int((1-alpha)*100)}%_CI": (ci_low, ci_high)
        }

    # compute probability that each group is the best
    best_counts = {g: 0 for g in groups}
    all_samples = np.vstack([samples[g] for g in groups])
    for i in range(simulations):
        values = {g: samples[g][i] for g in groups}
        best = max(values, key=values.get)
        best_counts[best] += 1
    probs_best = {g: best_counts[g] / simulations for g in groups}

    # Expected Utility if provided
    expected_utilities = None
    if utility:
        expected_utilities = {}
        for g in groups:
            util = 0
            for h in groups:
                if g == h:
                    continue
                diff = samples[g] - samples[h]
                p_win = np.mean(diff > 0)
                util += p_win * utility["gain"] + (1 - p_win) * utility["loss"]
            expected_utilities[g] = util

    results["posteriors"] = posteriors
    results["prob_best"] = probs_best
    results["expected_utility"] = expected_utilities
    return results


def print_bayesian_results(results):
    print("\n=== Bayesian Test Results ===")
    rows = []
    for g, stats in results["posteriors"].items():
        ci = stats["95%_CI"]
        rows.append([
            g,
            f"{stats['mean']:.4f}",
            f"{stats['median']:.4f}",
            f"[{ci[0]:.4f}, {ci[1]:.4f}]",
            f"{results['prob_best'][g]:.2%}"
        ])
    print(tabulate(rows, headers=["Group", "Mean", "Median", "95% CI", "Prob Best"], tablefmt="grid"))

    if results["expected_utility"]:
        print("\nExpected Utility Analysis:")
        util_rows = [(g, f"{u:.2f}") for g, u in results["expected_utility"].items()]
        print(tabulate(util_rows, headers=["Group", "Expected Utility"], tablefmt="grid"))