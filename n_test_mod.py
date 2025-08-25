import numpy as np
from scipy.stats import chi2_contingency, f_oneway, ttest_ind
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests
from tabulate import tabulate
from itertools import combinations


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