import numpy as np
from scipy.stats import chi2_contingency, f_oneway, ttest_ind
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests
from tabulate import tabulate
from itertools import combinations
import math
from visual_mod import plot_abn_results
from visual_mod import plot_cumulative_results
from scipy.stats import norm


def run_n_test(data, kpi_type="conversion", alpha=0.05):
    """
    Run A/B/n test for conversions or mean-based KPIs (ARPU, LTV, Churn).
    """

    results = {}
    is_conv = kpi_type.lower().startswith("conv")

    if is_conv:
        table = []
        for group, vals in data.items():
            convr = int(vals["conversions"])
            users = int(vals["users"])
            if users <= 0 or convr < 0 or convr > users:
                raise ValueError(f"Bad inputs for group {group}: users={users}, convr={convr}")
            nonconv = users - convr
            table.append([convr, nonconv])
        table = np.array(table, dtype=float)

        chi2, p, dof, exp = chi2_contingency(table)
        results["global_test"] = {
            "test": "Chi²",
            "chi2": chi2,
            "dof": dof,
            "p_value": p,
            "significant": p < alpha
        }

        # Conversion rates
        convr_rates = {g: vals["conversions"] / vals["users"] for g, vals in data.items()}
        results["conversion_rates"] = convr_rates

        # Pairwise with Holm correction
        pairwise = pairwise_chi2(data, alpha=alpha, method="holm")
        results["pairwise"] = pairwise

        # Choose leader
        results["leader"] = choose_leader(convr_rates, pairwise)

    else:
        groups = []
        labels = []
        group_means = {}
        for group, vals in data.items():
            mean = vals["total"] / vals["users"]
            sd = vals.get("sd", mean)
            sample = np.random.normal(mean, sd, size=vals["users"])
            groups.append(sample)
            labels.extend([group] * vals["users"])
            group_means[group] = mean

        # Run ANOVA
        f_stat, p = f_oneway(*groups)
        results["global_test"] = {
            "test": "ANOVA (f-test)",
            "F": f_stat,
            "p_value": p,
            "significant": p < alpha
        }

        # Post-hoc Tukey test
        flat = np.concatenate(groups)
        tukey = pairwise_tukeyhsd(endog=flat, groups=np.array(labels), alpha=alpha)
        results["posthoc"] = str(tukey)

        # Pairwise t-tests with correction
        pairwise = pairwise_ttests(groups, list(data.keys()), alpha=alpha, method="holm")
        results["pairwise"] = pairwise

        # Leader
        results["leader"] = choose_leader(group_means, pairwise)

    return results


def choose_leader(means_dict, pairwise_results):
    top = max(means_dict, key=means_dict.get)
    others = [g for g in means_dict if g != top]

    wins = set()
    for r in pairwise_results:
        g1, g2 = r["group1"], r["group2"]
        if top not in (g1, g2):
            continue

        if not r["significant"]:
            continue

        other = g2 if g1 == top else g1
        if means_dict[top] > means_dict[other]:
            wins.add(other)
        else:
            return f'No clear leader (best by mean: {top})'

    if len(wins) == len(others) and len(others) > 0:
        return top
    else:
        return f'No clear leader (best by mean: {top})'


def print_n_results(results, kpi_type="conversion"):
    is_conv = kpi_type.lower().startswith("conv")

    if is_conv:
        print("\nPairwise chi2 tests with Holm-Bonferroni correction:")
        rows = [
            (
                r["group1"], r["group2"],
                f"{r['chi2']:.4f}",
                f"{r['p_value_raw']:.4f}",
                f"{r['p_value_corrected']:.4f}",
                "Yes" if r["significant"] else "No"
            )
            for r in results["pairwise"]
        ]
        print(tabulate(
            rows,
            headers=["Group1", "Group2", "chi2", "p_raw", "p_corrected", "Significant?"],
            tablefmt="grid"
        ))
    else:
        print("\nPost-hoc Analysis (Tukey HSD)")
        print(results["posthoc"])

        if "pairwise" in results:
            print("\nPairwise Welch t-tests with Holm-Bonferroni correction:")
            rows = [
                (
                    r["group1"], r["group2"],
                    f"{r['t_stat']:.4f}",
                    f"{r['p_value_raw']:.4f}",
                    f"{r['p_value_corrected']:.4f}",
                    "Yes" if r["significant"] else "No"
                )
                for r in results["pairwise"]
            ]
            print(tabulate(
                rows,
                headers=["Group1", "Group2", "t", "p_raw", "p_corrected", "Significant?"],
                tablefmt="grid"
            ))

    if "leader" in results:
        print(f"\n>>> Leader: {results['leader']}")

    plot_abn_results(results, kpi_type=kpi_type)


def pairwise_chi2(groups, alpha=0.05, method="holm"):
    """Run pairwise chi2 tests with multiple comparisons correction"""
    results = []
    names = list(groups.keys())
    p_values = []

    for g1, g2 in combinations(names, 2):
        convr1, users1 = groups[g1]["conversions"], groups[g1]["users"]
        convr2, users2 = groups[g2]["conversions"], groups[g2]["users"]

        table = [
            [convr1, users1 - convr1],
            [convr2, users2 - convr2]
        ]

        chi2, p, _, _ = chi2_contingency(table)
        results.append([g1, g2, chi2, p])
        p_values.append(p)

    reject, p_corr, _, _ = multipletests(p_values, alpha=alpha, method=method)

    final = []
    for (g1, g2, chi2, p), p_corr, rej in zip(results, p_corr, reject):
        final.append({
            "group1": g1,
            "group2": g2,
            "chi2": chi2,
            "p_value_raw": p,
            "p_value_corrected": p_corr,
            "significant": rej
        })
    return final

def pairwise_ttests(groups, names, alpha=0.05, method="holm"):
    results = []
    p_values = []

    for (i, g1), (j, g2) in combinations(enumerate(names), 2):
        t, p = ttest_ind(groups[i], groups[j], equal_var=False)
        results.append([g1, g2, t, p])
        p_values.append(p)

    reject, p_corr, _, _ = multipletests(p_values, alpha=alpha, method=method)

    final = []
    for (g1, g2, t, p), p_corr, rej in zip(results, p_corr, reject):
        final.append({
            "group1": g1,
            "group2": g2,
            "t_stat": t,
            "p_value_raw": p,
            "p_value_corrected": p_corr,
            "significant": rej
        })
    return final

def run_sequential_test(data_over_time, kpi_type="conversion", alpha=0.05, method="pocock"):
    if not data_over_time:
        return []

    K = len(data_over_time)

    if method == "bonferroni":
        thresholds = [alpha / K] * K
    elif method == "pocock":
        per_alpha = 1 - (1 - alpha) ** (1 / K)
        thresholds = [per_alpha] * K
    elif method == "obrien":
        z_overall = norm.ppf(1 - alpha / 2)
        thresholds = []
        for i in range(1, K + 1):
            z_i = z_overall / math.sqrt(i / K)
            p_i = 2 * (1 - norm.cdf(abs(z_i)))
            thresholds.append(p_i)
    else:
        raise ValueError("Unknown sequential method. Use 'pocock', 'bonferroni' or 'obrien'.")

    results = []
    for i, (day, data) in enumerate(data_over_time.items(), start=1):
        res = run_n_test(data, kpi_type=kpi_type, alpha=alpha)
        p = res.get("global_test", {}).get("p_value")
        if p is None:
            p = res.get("p_value", None)

        threshold = thresholds[i - 1]
        significant = (p is not None) and (p < threshold)

        results.append({
            "day": day,
            "p_value": float(p) if p is not None else None,
            "threshold": float(threshold),
            "significant": bool(significant)
        })

        if significant:
            # stop on first significant look
            break

    return results

def print_sequential_results(results, plot=False, data_over_time=None, kpi_type=None):
    print("\n=== Sequential Test Results ===")
    if not results:
        print("No results to show.")
        return

    rows = []
    for r in results:
        p_str = f"{r['p_value']:.4f}" if r['p_value'] is not None else "N/A"
        thr_str = f"{r['threshold']:.4f}"
        decision = "STOP" if r["significant"] else "continue"
        rows.append((r["day"], p_str, thr_str, decision))

    print(tabulate(rows, headers=["Day", "P-value", "Threshold", "Decision"], tablefmt="grid"))

    if plot and (data_over_time is not None) and (kpi_type is not None):
        try:
            p_values = {r["day"]: r["p_value"] for r in results}
            plot_cumulative_results(data_over_time, kpi_type=kpi_type, p_values=p_values)
        except Exception as e:
            print(f"[plotting error] Could not plot cumulative results: {e}")