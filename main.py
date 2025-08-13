import math

from matplotlib.lines import lineStyles
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


def ab_test(visitors_a, conversions_a, visitors_b, conversions_b, alpha=0.05):
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

    return {
        "cr_a": cr_a,
        "cr_b": cr_b,
        "diff": cr_b-cr_a,
        "p_value": p_value,
        "significant": significant,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper
    }

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

if __name__ == "__main__":
    print("Sample Size Calculator")
    baseline = float(input("Enter baseline conversion rate (e.g. 0.1 for 10%): "))
    mde = float(input("Enter minimum detectable effect (e.g. 0.02 for +2%): "))
    alpha = float(input("Enter significance level (default 0.05): ") or 0.05)
    power = float(input("Enter statistical power (default 0.8): ") or 0.8)

    needed_n = sample_size(baseline, mde, alpha, power)
    print(f"\nYou need at least {needed_n} visitors per group for the test.\n")

    print("A/B test calculator")
    visitors_a = int(input("Enter the visitors in Group A: "))
    conversions_a = int(input("Enter the conversions in Group A: "))
    visitors_b = int(input("Enter the visitors in Group B: "))
    conversions_b = int(input("Enter the conversions in Group B: "))

    try:
        result = ab_test(visitors_a, conversions_a, visitors_b, conversions_b)
    except ValueError as e:
        print(f"Error: {e}")
        exit(1)

    result = ab_test(visitors_a, conversions_a, visitors_b, conversions_b)

    print("\n Results")
    print(f"Conversion Rate A: {result['cr_a']:.2%}")
    print(f"Conversion Rate B: {result['cr_b']:.2%}")
    print(f"Difference: {result['diff']:.2%}")
    print(f"95% CI for difference: [{result['ci_lower']:.2%}, {result['ci_upper']:.2%}]")
    print(f"p-value: {result['p_value']:.4f}")
    print("Statistically Significant?", "Yes" if result['significant'] else "No")

    plot_confidence_interval(result['diff'], result['ci_lower'], result['ci_upper'])