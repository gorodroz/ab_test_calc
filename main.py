import math
import numpy
from scipy.stats import norm
import matplotlib
import matplotlib.pyplot as plt
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
            baseline = float(input("Enter baseline conversion rate (e.g. 0.1 for 10%): "))
            mde = float(input("Enter minimum detectable effect (e.g. 0.02 for +2%): "))
            alpha = float(input("Enter significance level (default 0.05): ") or 0.05)
            power = float(input("Enter statistical power (default 0.8): ") or 0.8)
            needed_n = sample_size(baseline, mde, alpha, power)
            print(f"You need at least {needed_n} visitors per group.")

        elif choice == "2":
            visitors_a = int(input("Enter visitors in Group A: "))
            conversions_a = int(input("Enter conversions in Group A: "))
            visitors_b = int(input("Enter visitors in Group B: "))
            conversions_b = int(input("Enter conversions in Group B: "))

            result = ab_test(visitors_a, conversions_a, visitors_b, conversions_b)
            print("\nClassic A/B Test Results")
            print(f"Conversion Rate A: {result['cr_a']:.2%}")
            print(f"Conversion Rate B: {result['cr_b']:.2%}")
            print(f"Difference: {result['diff']:.2%}")
            print(f"95% CI: [{result['ci_lower']:.2%}, {result['ci_upper']:.2%}]")
            print(f"p-value: {result['p_value']:.4f}")
            print("Significant?", "Yes" if result['significant'] else "No")
            plot_confidence_interval(result['diff'], result['ci_lower'], result['ci_upper'])

        elif choice == "3":
            visitors_a = int(input("Enter visitors in Group A: "))
            conversions_a = int(input("Enter conversions in Group A: "))
            visitors_b = int(input("Enter visitors in Group B: "))
            conversions_b = int(input("Enter conversions in Group B: "))

            prob_b_better = bayesian_ab_test(visitors_a, conversions_a, visitors_b, conversions_b)
            print(f"\nBayesian Result: Probability B > A = {prob_b_better:.2%}")

        elif choice == "4":
            visitors_a = int(input("Enter visitors in Group A: "))
            conversions_a = int(input("Enter conversions in Group A: "))
            visitors_b = int(input("Enter visitors in Group B: "))
            conversions_b = int(input("Enter conversions in Group B: "))

            result = ab_test(visitors_a, conversions_a, visitors_b, conversions_b)
            rel_uplift = (result['cr_b'] - result['cr_a']) / result['cr_a']
            print("\nAdvanced Metrics")
            print(f"Relative Uplift: {rel_uplift:.2%}")

        elif choice == "5":
            visitors_a = int(input("Enter visitors in Group A: "))
            conversions_a = int(input("Enter conversions in Group A: "))
            visitors_b = int(input("Enter visitors in Group B: "))
            conversions_b = int(input("Enter conversions in Group B: "))

            metrics = bayesian_expected_metrics(visitors_a, conversions_a, visitors_b, conversions_b)
            print("\nBayesian Expected Metrics")
            print(f"Expected Loss choosing B: {metrics['expected_loss_b']:.4f}")
            print(f"Expected Loss choosing A: {metrics['expected_loss_a']:.4f}")
            print(f"Expected Value choosing B: {metrics['expected_value_b']:.4f}")
            print(f"Expected Value choosing A: {metrics['expected_value_a']:.4f}")

        elif choice == "6":
            visitors_a = int(input("Enter visitors in Group A: "))
            conversions_a = int(input("Enter conversions in Group A: "))
            visitors_b = int(input("Enter visitors in Group B: "))
            conversions_b = int(input("Enter conversions in Group B: "))
            value_per_conversion = float(input("Enter value per conversion ($): "))

            decision = bayesian_decision_analysis(
                visitors_a, conversions_a,
                visitors_b, conversions_b,
                value_per_conversion=value_per_conversion
            )
            print("\nBayesian Decision Analysis")
            print(f"Expected Utility: ${decision['expected_utility']:.2f}")
            print(f"Expected Regret: ${decision['expected_regret']:.2f}")

        elif choice == "7":
            visitors_a = int(input("Enter visitors in Group A: "))
            conversions_a = int(input("Enter conversions in Group A: "))
            visitors_b = int(input("Enter visitors in Group B: "))
            conversions_b = int(input("Enter conversions in Group B: "))

            monitoring = sequential_bayesian_monitoring(visitors_a, conversions_a, visitors_b, conversions_b)
            print("\nSequential Bayesian Monitoring")
            print(f"Posterior Mean A: {monitoring['posterior_mean_a']:.4f}")
            print(f"Posterior Mean B: {monitoring['posterior_mean_b']:.4f}")
            print(f"Probability B > A: {monitoring['prob_b_better']:.2%}")

        elif choice == "0":
            print("Exit")
            break
        else:
            print("Invalid choice, try again.")

if __name__ == "__main__":
    main()