import numpy
from scipy.stats import chi2_contingency, f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from tabulate import tabulate

def run_n_test(data, kpi_type="conversion", alpha=0.05):
    """
    Run A/B/n test for conversions or mean-based KPIs (ARPU, LTV, Churn).

    Parameters

    data : dict
        Example (conversions):
        {
            "A": {"users": 200, "conversions": 40},
            "B": {"users": 180, "conversions": 25},
            "C": {"users": 190, "conversions": 50}
        }

        Example (means):
        {
            "A": {"users": 200, "total": 5000, "sd": 20},
            "B": {"users": 180, "total": 4200, "sd": 25},
            "C": {"users": 190, "total": 5300, "sd": 22}
        }

    kpi_type : str
        "conversion" or "mean" (ARPU, LTV, Churn)

    alpha : float
        Significance level

    Returns
    dict with results
    """

    results={}
    if kpi_type=="conversion":
        table=[]
        for group, vals in data.items():
            convr=vals["conversions"]
            nonconvr=vals["users"]-convr
            table.append([convr, nonconvr])
        table=numpy.array(table)

        chi2, p, dof, exp = chi2_contingency(table)
        results["global_test"]={
            "test":"Chi²",
            "chi2":chi2,
            "dof":dof,
            "p_value":p,
            "significant": p<alpha
        }

        #Conversion rates
        convr_rates={g: vals["conversions"]/vals["users"] for g, vals in data.items()}
        results["conversion_rates"]=convr_rates


    else:
        groups=[]
        labels=[]
        for group, vals in data.items():
            mean=vals["total"]/vals["users"]
            sd=vals.get("sd", mean)
            sample=numpy.random.normal(mean, sd, size=vals["users"])
            groups.append(sample)
            labels.extend([group]*vals["users"])
#Run ANOVA
        f_stat, p=f_oneway(*groups)
        results["global_test"]={
            "test":"ANOVA (f-test)",
            "F": f_stat,
            "p_value": p,
            "significant": p<alpha
        }

#Post-hoc Tukey test
        flat=numpy.concatenate(groups)
        tukey=pairwise_tukeyhsd(endog=flat, groups=numpy.array(labels), alpha=alpha)
        results["posthoc"]=str(tukey)

    return results
##########################################################
def print_n_results(results, kpi_type="conversions"):
    print("\n===A/B/n Test Results===")
    if "global_test" in results:
        print(tabulate(
            [[
                results["global_test"].get("test"),
                results["global_test"].get("p_value"),
                "Yes" if results["global_test"]["significant"] else "No"
            ]],
            headers=["Test", "P-value", "Significant?"],
            tablefmt="grid"
        ))

    if kpi_type=="conversion":
        rows=[(g, f"{cr:.2%}") for g, cr in results["conversion_rates"].items()]
        print("\nConversion Rates by Group")
        print(tabulate(rows, headers=["Group", "Rate"], tablefmt="grid"))
    else:
        print("\nPost-hoc Analysis (Tukey HSD)")
        print(results["posthoc"])