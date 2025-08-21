import math
import numpy
from scipy.stats import norm
import matplotlib
import matplotlib.pyplot as plt
from tabulate import tabulate
import json
import csv
import logging

logging.basicConfig(
    filename="ab_test_calc_log.txt",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8"
)

matplotlib.use('Agg')

def sample_size(p, mde, alpha=0.05, power=0.8):
    if not (0<p<1):
        raise ValueError("Baseline conversion rate (p) must be between 0 and 1 (exclusive).")
    if mde <= 0 or mde >= 1:
        raise  ValueError("Minimum detectable effect (mde) must be between 0 and 1 (exclusive).")
    if not (0<alpha<1):
        raise ValueError("Significance level (alpha) must be between 0 and 1.")
    if not (0<power<1):
        raise   ValueError("Statistical power must be between 0 and 1.")

    z_alpha=norm.ppf(1-alpha/2)
    z_beta=norm.ppf(power)

    p_var=2*p*(1-p)
    n_per_grp=((z_alpha+z_beta)**2*p_var)/(mde**2)

    return math.ceil(n_per_grp)


def ab_test(visitors_a, conversions_a, visitors_b, conversions_b, alpha=0.05, revenue_per_conversion=1.0):
    if visitors_a <= 0 or visitors_b <= 0:
        raise ValueError("Visitors must be greater than zero")
    if conversions_a <0 or conversions_b < 0:
        raise ValueError("Conversions cannot be negative")
    if conversions_a > visitors_a or conversions_b > visitors_b:
        raise ValueError("Conversions cannot be greater than visitors.")

    cr_a=conversions_a/visitors_a
    cr_b=conversions_b/visitors_b

    se_a=math.sqrt(cr_a * (1 - cr_a) / visitors_a)
    se_b=math.sqrt(cr_b * (1 - cr_b) / visitors_b)

    se_diff=math.sqrt(se_a**2 + se_b**2)
    z_score=(cr_b -cr_a)/se_diff

    p_value=2*(1-norm.cdf(abs(z_score)))

    significant=p_value < alpha

    z_crit=norm.ppf(1-alpha/2)
    margin_error=z_crit*se_diff
    ci_lower=(cr_b-cr_a)-margin_error
    ci_upper=(cr_b-cr_a)+margin_error

    relative_uplift=(cr_b-cr_a)/cr_a if cr_a > 0 else float("inf")

    #Simulation for Expected loss

    sims=100_000
    a_samples=numpy.random.beta(conversions_a+1, visitors_a-conversions_a+1, sims)
    b_samples=numpy.random.beta(conversions_b+1, visitors_b-conversions_b+1, sims)
    expected_loss=numpy.mean((a_samples-b_samples)[b_samples>a_samples])

    expected_value=(cr_b-cr_a)*visitors_b*revenue_per_conversion

    return {
        "cr_a": cr_a,
        "cr_b": cr_b,
        "diff": cr_b-cr_a,
        "p_value": p_value,
        "significant": significant,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "relative_uplift":relative_uplift,
        "expected_loss":expected_loss,
        "expected_value":expected_value
    }


def bayesian_ab_test(visitors_a, conversions_a, visitors_b, conversions_b, samples=100_000):
    a_alpha, a_beta = 1+conversions_a, 1+(visitors_a-conversions_a)
    b_alpha, b_beta = 1+conversions_b, 1+(visitors_b-conversions_b)

    a_dist=numpy.random.beta(a_alpha, a_beta, samples)
    b_dist=numpy.random.beta(b_alpha, b_beta, samples)

    prob_b_better=(b_dist>a_dist).mean()

    return prob_b_better

def bayesian_expected_metrics(visitors_a, conversions_a, visitors_b, conversions_b,samples=100_000, value_per_conversion=100):
    a_alpha, a_beta = 1 + conversions_a, 1 + (visitors_a - conversions_a)
    b_alpha, b_beta = 1 + conversions_b, 1 + (visitors_b - conversions_b)

    a_dist = numpy.random.beta(a_alpha, a_beta, samples)
    b_dist = numpy.random.beta(b_alpha, b_beta, samples)

    loss_b = numpy.mean((a_dist - b_dist) * (a_dist > b_dist)) * value_per_conversion
    loss_a = numpy.mean((b_dist - a_dist) * (b_dist > a_dist)) * value_per_conversion

    expected_value_b = numpy.mean(b_dist - a_dist) * value_per_conversion
    expected_value_a = numpy.mean(a_dist - b_dist) * value_per_conversion

    return {
        "expected_loss_b": float(loss_b),
        "expected_loss_a": float(loss_a),
        "expected_value_b": float(expected_value_b),
        "expected_value_a": float(expected_value_a),
        "a_dist": a_dist,
        "b_dist": b_dist,
    }

def bayesian_decision_analysis(visitors_a, conversions_a, visitors_b, conversions_b,samples=100_000, value_per_conversion=100):
    a_alpha, a_beta = 1 + conversions_a, 1 + (visitors_a - conversions_a)
    b_alpha, b_beta = 1 + conversions_b, 1 + (visitors_b - conversions_b)

    a_dist = numpy.random.beta(a_alpha, a_beta, samples)
    b_dist = numpy.random.beta(b_alpha, b_beta, samples)

    diff = b_dist - a_dist
    prob_b_better = (diff > 0).mean()

    expected_value = numpy.mean(diff) * value_per_conversion * visitors_b
    regret = numpy.mean((-diff) * (diff < 0)) * value_per_conversion * visitors_b
    expected_utility = prob_b_better * expected_value - (1 - prob_b_better) * regret

    return {
        "prob_b_better": float(prob_b_better),
        "expected_value": float(expected_value),
        "expected_regret": float(regret),
        "expected_utility": float(expected_utility),
    }

def sequential_bayesian_monitoring(visitors_a, conversions_a, visitors_b, conversions_b, samples=50_000, threshold_high=0.99, threshold_low=0.01):
    """
    Perform sequential Bayesian monitoring for an A/B test.

    Returns:
        decision (str), prob_b_better (float)
    """
    #Posterior for both groups
    a_alpha, a_beta = 1 + conversions_a, 1 + (visitors_a - conversions_a)
    b_alpha, b_beta = 1 + conversions_b, 1 + (visitors_b - conversions_b)

    a_dist = numpy.random.beta(a_alpha, a_beta, samples)
    b_dist = numpy.random.beta(b_alpha, b_beta, samples)

    prob_b_better = (b_dist > a_dist).mean()
    posterior_mean_a = a_alpha / (a_alpha + a_beta)
    posterior_mean_b = b_alpha / (b_alpha + b_beta)
    #Decision making
    if prob_b_better > threshold_high:
        decision = "Stop: Variant B is clearly better"
    elif prob_b_better < threshold_low:
        decision = "Stop: Variant A is clearly better"
    else:
        decision = "Continue: Not enough evidence yet"

    return {
        "decision": decision,
        "prob_b_better": float(prob_b_better),
        "posterior_mean_a": float(posterior_mean_a),
        "posterior_mean_b": float(posterior_mean_b),
    }

def plot_expected_metrics(a_dist, b_dist, loss_b, expected_value):
    plt.figure(figsize=(6,4))

    plt.hist(a_dist, bins=100, alpha=0.5, label="Group A", density=True, color="blue")
    plt.hist(b_dist, bins=100, alpha=0.5, label="Group B", density=True, color="green")

    plt.title("Posterior Distributions of Conversion Rates")
    plt.xlabel("Conversion RAte")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig("bayesian_posteriors.png")
    print("Graph saved as bayesian_posteriors.png")

    #Expected loss
    plt.figure(figsize=(5, 3))
    plt.bar(["Expected Loss", "Expected Value"], [loss_b, expected_value], color=["red", "green"])
    plt.title("Bayesian Risk&Value")
    plt.ylabel("Value ($)")
    plt.tight_layout()
    plt.savefig("bayesian_metrics.png")
    print("Graph saved as bayesian_metrics.png")

def plot_confidence_interval(diff, ci_lower, ci_upper):
    plt.figure(figsize=(5, 2))
    plt.errorbar(x=diff, y=0, xerr=[[diff-ci_lower], [ci_upper-diff]], fmt="o", color="blue", ecolor="black",
                 elinewidth=2, capsize=5)
    plt.axvline(0, color="red", linestyle="--", label="No difference")
    plt.title("95% Confidence Interval for Conversion Difference")
    plt.yticks([])
    plt.legend()
    plt.grid(True, axis="x", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("ab_test_ci.png")
    print("Graph saved as ab_test_ci.png")

def main():
    while True:
        print("\n=== A/B test calculator ===")
        print("1. Sample size calculator")
        print("2. Classic A/B test")
        print("3. Bayesian A/B test")
        print("4. Advanced Metrics (Relative Uplift, Expected Loss/Value)")
        print("5. Bayesian Expected Metrics (Loss & Value)")
        print("6. Bayesian Decision Analysis (Utility & Regret)")
        print("7. Sequential Bayesian Monitoring")
        print("0. Exit")
        choice = input("Choose an option: ")

        if choice == "1":
            baseline=float(input("Enter baseline conversion rate (e.g. 0.1 for 10%): "))
            mde=float(input("Enter minimum detectable effect (e.g. 0.02 for +2%): "))
            alpha=float(input("Enter significance level (default 0.05): ") or 0.05)
            power=float(input("Enter statistical power (default 0.8): ") or 0.8)
            needed_n=sample_size(baseline, mde, alpha, power)
            print(f"You need at least {needed_n} visitors per group.")

            logging.info(
                f"Sample size calculation | baseline={baseline}, mde={mde}, alpha={alpha}, power={power} -> needed_n={needed_n}"
            )


        elif choice == "2":
            visitors_a=int(input("Enter visitors in Group A: "))
            conversions_a=int(input("Enter conversions in Group A: "))
            visitors_b=int(input("Enter visitors in Group B: "))
            conversions_b=int(input("Enter conversions in Group B: "))
            result=ab_test(visitors_a, conversions_a, visitors_b, conversions_b)
            table=[
                ["Metric", "Value"],
                ["Conversion Rate A", f"{result['cr_a']:.2%}"],
                ["Conversion Rate B", f"{result['cr_b']:.2%}"],
                ["Difference", f"{result['diff']:.2%}"],
                ["95% CI Lower", f"{result['ci_lower']:.2%}"],
                ["95% CI Upper", f"{result['ci_upper']:.2%}"],
                ["p-value", f"{result['p_value']:.4f}"],
                ["Significant?", "Yes" if result['significant'] else "No"],
                ["Relative Uplift", f"{result['relative_uplift']:.2%}"],
                ["Expected Loss", f"{result['expected_loss']:.4f}"],
                ["Expected Value", f"${result['expected_value']:.2f}"]
            ]

            print("\nClassic A/B Test Results")
            print(tabulate(table, headers="firstrow", tablefmt="grid"))

            plot_confidence_interval(result['diff'], result['ci_lower'], result['ci_upper'])

            save = input("Save results? (csv/json/skip): ").strip().lower()
            if save in ["csv", "json"]:
                from csvjsonm import save_results
                save_results(result, "ab_test_results", save)

            logging.info(
                f"Classic A/B Test | A: {visitors_a} visitors, {conversions_a} conversions "
                f"| B: {visitors_b} visitors, {conversions_b} conversions "
                f"| Results: CR_A={result['cr_a']:.4f}, CR_B={result['cr_b']:.4f}, "
                f"diff={result['diff']:.4f}, p={result['p_value']:.4f}, sig={result['significant']}"
            )

        elif choice == "3":
            visitors_a=int(input("Enter visitors in Group A: "))
            conversions_a=int(input("Enter conversions in Group A: "))
            visitors_b=int(input("Enter visitors in Group B: "))
            conversions_b=int(input("Enter conversions in Group B: "))

            prob_b_better=bayesian_ab_test(visitors_a, conversions_a, visitors_b, conversions_b)
            print(f"\nBayesian Result: Probability B > A = {prob_b_better:.2%}")

            logging.info(
                f"Bayesian A/B Test | A: {visitors_a}/{conversions_a}, B: {visitors_b}/{conversions_b} "
                f"| Probability B>A = {prob_b_better:.4f}"
            )

        elif choice == "4":
            visitors_a=int(input("Enter visitors in Group A: "))
            conversions_a=int(input("Enter conversions in Group A: "))
            visitors_b=int(input("Enter visitors in Group B: "))
            conversions_b=int(input("Enter conversions in Group B: "))
            value_per_conversion=float(input("Enter value per conversion ($): "))
            cost_a=float(input("Enter total cost for Group A ($): "))
            cost_b=float(input("Enter total cost for Group B ($): "))

            result=ab_test(visitors_a, conversions_a, visitors_b, conversions_b)

            revenue_a=conversions_a*value_per_conversion
            revenue_b=conversions_b*value_per_conversion

            roi_a=(revenue_a - cost_a)/cost_a if cost_a>0 else 0
            roi_b=(revenue_b - cost_b) / cost_b if cost_b > 0 else 0

            payback_a=cost_a/revenue_a if revenue_a > 0 else float("inf")
            payback_b=cost_b/revenue_b if revenue_b > 0 else float("inf")

            rel_uplift=(result['cr_b']-result['cr_a'])/result['cr_a']

            table = [
                ["Metric", "Group A", "Group B"],
                ["Conversion Rate", f"{result['cr_a']:.2%}", f"{result['cr_b']:.2%}"],
                ["Revenue ($)", f"{revenue_a:.2f}", f"{revenue_b:.2f}"],
                ["ROI", f"{roi_a:.2%}", f"{roi_b:.2%}"],
                ["Payback Period", f"{payback_a:.2f}", f"{payback_b:.2f}"],
                ["Relative Uplift", f"{rel_uplift:.2%}", ""]
            ]

            print("\nAdvanced Metrics (with ROI & Payback)")
            print(tabulate(table, headers="firstrow", tablefmt="fancy_grid"))

            save = input("Save results? (csv/json/skip): ").strip().lower()
            if save in ["csv", "json"]:
                from csvjsonm import save_results
                save_results(result, "advanced_metrics", save)

            logging.info(
                f"Advanced Metrics | A: ROI={roi_a:.4f}, Payback={payback_a:.4f} | "
                f"B: ROI={roi_b:.4f}, Payback={payback_b:.4f}"
            )

        elif choice == "5":
            visitors_a=int(input("Enter visitors in Group A: "))
            conversions_a=int(input("Enter conversions in Group A: "))
            visitors_b=int(input("Enter visitors in Group B: "))
            conversions_b=int(input("Enter conversions in Group B: "))
            value_per_conversion=float(input("Enter value per conversion ($): "))
            cost_a=float(input("Enter total cost for Group A ($): "))
            cost_b=float(input("Enter total cost for Group B ($): "))

            metrics=bayesian_expected_metrics(visitors_a, conversions_a, visitors_b, conversions_b)

            roi_a=(metrics['expected_value_a']-cost_a)/cost_a if cost_a>0 else 0
            roi_b=(metrics['expected_value_b']-cost_b)/cost_b if cost_b>0 else 0

            payback_a=cost_a/metrics['expected_value_a'] if metrics['expected_value_a']>0 else float("inf")
            payback_b=cost_b/metrics['expected_value_b'] if metrics['expected_value_b']>0 else float("inf")

            table = [
                ["Metric", "Group A", "Group B"],
                ["Expected Loss", f"{metrics['expected_loss_a']:.2f}", f"{metrics['expected_loss_b']:.2f}"],
                ["Expected Value ($)", f"{metrics['expected_value_a']:.2f}", f"{metrics['expected_value_b']:.2f}"],
                ["ROI", f"{roi_a:.2%}", f"{roi_b:.2%}"],
                ["Payback Period", f"{payback_a:.2f}", f"{payback_b:.2f}"],
            ]

            print("\nBayesian Expected Metrics (with ROI & Payback)")
            print(tabulate(table, headers="firstrow", tablefmt="fancy_grid"))

            save = input("Save results? (csv/json/skip): ").strip().lower()
            if save in ["csv", "json"]:
                from csvjsonm import save_results
                save_results(result, "bayesian_expected_metrics", save)

            logging.info(
                f"Bayesian Expected Metrics | A: ROI={roi_a:.4f}, Payback={payback_a:.4f} | "
                f"B: ROI={roi_b:.4f}, Payback={payback_b:.4f}"
            )

        elif choice == "6":
            visitors_a=int(input("Enter visitors in Group A: "))
            conversions_a=int(input("Enter conversions in Group A: "))
            visitors_b=int(input("Enter visitors in Group B: "))
            conversions_b=int(input("Enter conversions in Group B: "))
            value_per_conversion=float(input("Enter value per conversion ($): "))
            cost_b=float(input("Enter total cost for Group B ($): "))

            decision=bayesian_decision_analysis(visitors_a, conversions_a, visitors_b, conversions_b, value_per_conversion=value_per_conversion)

            roi_b=(decision['expected_value']-cost_b)/cost_b if cost_b>0 else 0
            payback_b=cost_b/decision['expected_value'] if decision['expected_value']>0 else float("inf")

            table = [
                ["Metric", "Value"],
                ["Probability B > A", f"{decision['prob_b_better']:.2%}"],
                ["Expected Value ($)", f"{decision['expected_value']:.2f}"],
                ["Expected Regret ($)", f"{decision['expected_regret']:.2f}"],
                ["Expected Utility ($)", f"{decision['expected_utility']:.2f}"],
                ["ROI (B)", f"{roi_b:.2%}"],
                ["Payback (B)", f"{payback_b:.2f}"],
            ]

            print("\nBayesian Decision Analysis (with ROI & Payback)")
            print(tabulate(table, headers="firstrow", tablefmt="grid"))

            save = input("Save results? (csv/json/skip): ").strip().lower()
            if save in ["csv", "json"]:
                from csvjsonm import save_results
                save_results(result, "bayesian_decision_analysis", save)

            logging.info(f"Bayesian Decision Analysis | ROI(B)={roi_b:.4f}, Payback(B)={payback_b:.4f}")

        elif choice == "7":
            visitors_a=int(input("Enter visitors in Group A: "))
            conversions_a=int(input("Enter conversions in Group A: "))
            visitors_b=int(input("Enter visitors in Group B: "))
            conversions_b=int(input("Enter conversions in Group B: "))
            value_per_conversion=float(input("Enter value per conversion ($): "))
            cost_a=float(input("Enter total cost for Group A ($): "))
            cost_b=float(input("Enter total cost for Group B ($): "))

            monitoring=sequential_bayesian_monitoring(visitors_a, conversions_a, visitors_b, conversions_b)

            revenue_a=conversions_a*value_per_conversion
            revenue_b=conversions_b*value_per_conversion

            roi_a=(revenue_a-cost_a)/cost_a if cost_a>0 else 0
            roi_b=(revenue_b-cost_b)/cost_b if cost_b>0 else 0

            payback_a=cost_a/revenue_a if revenue_a>0 else float("inf")
            payback_b=cost_b/revenue_b if revenue_b>0 else float("inf")

            print("\nSequential Bayesian Monitoring (with ROI & Payback)")
            print(f"Posterior Mean A: {monitoring['posterior_mean_a']:.4f}")
            print(f"Posterior Mean B: {monitoring['posterior_mean_b']:.4f}")
            print(f"Probability B > A: {monitoring['prob_b_better']:.2%}")
            print(f"ROI A: {roi_a:.2%}, ROI B: {roi_b:.2%}")
            print(f"Payback A: {payback_a:.2f}, Payback B: {payback_b:.2f}")

            logging.info(
                f"Sequential Monitoring | ROI(A)={roi_a:.4f}, Payback(A)={payback_a:.4f} | "
                f"ROI(B)={roi_b:.4f}, Payback(B)={payback_b:.4f}"
            )

        elif choice == "0":
            print("Exit")
            break
        else:
            print("Invalid choice, try again.")

if __name__ == "__main__":
    main()