import math
import numpy
from scipy.stats import norm
import matplotlib
import matplotlib.pyplot as plt
from tabulate import tabulate
import logging
from n_test_mod import run_n_test, print_n_results
from n_test_mod import run_sequential_test, print_sequential_results
from montecarlo_mod import run_montecarlo, print_montecarlo_results
from bayesian_mod import bayesian_test, print_bayesian_results
from decision_mod import decision_analysis, print_decision_results
from markov_mod import build_transition_matrix, simulate_markov, steady_state, expected_ltv, print_markov_results
from linear_opt_mod import run_linear_optimization, print_linear_results
from scenario_mod import run_scenario_analysis, print_scenario_results
from cost_benefit_mod import run_cost_benefit_simulation, print_cost_benefit_results

logging.basicConfig(
    filename="ab_test_calc_log.txt",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8"
)

matplotlib.use('Agg')

#Core Functions

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
    #binominal A/B test for CR/Churn
    if visitors_a <= 0 or visitors_b <= 0:
        raise ValueError("Visitors must be greater than zero")
    if conversions_a <0 or conversions_b < 0:
        raise ValueError("Conversions cannot be negative")
    if conversions_a > visitors_a or conversions_b > visitors_b:
        raise ValueError("Conversions cannot be greater than visitors.")

    cr_a=conversions_a/visitors_a
    cr_b=conversions_b/visitors_b

    se_a=math.sqrt(cr_a*(1-cr_a)/visitors_a)
    se_b=math.sqrt(cr_b*(1-cr_b)/visitors_b)

    se_diff=math.sqrt(se_a**2 + se_b**2)
    z_score=(cr_b -cr_a)/se_diff if se_diff>0 else 0.0

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
    expected_loss=float(numpy.mean((a_samples-b_samples)[b_samples>a_samples]))

    expected_value=(cr_b-cr_a)*visitors_b*revenue_per_conversion

    return {
        "cr_a": float(cr_a),
        "cr_b": float(cr_b),
        "diff": float(cr_b-cr_a),
        "p_value": float(p_value),
        "significant": bool(significant),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "relative_uplift":float(relative_uplift),
        "expected_loss":float(expected_loss),
        "expected_value":float(expected_value)
    }

def ab_test_means(mean_a, sd_a, n_a, mean_b, sd_b, n_b, alpha=0.05):
    #for ARPU/LTV (diference of means, normal approximation)
    if n_a<=1 or n_b<=1:
        raise ValueError("Sample size per group must be > 1 for means-based KPI.")
    if sd_a <0 or sd_b<0:
        raise ValueError("Standart deviation cannot be negative")
    se_diff=math.sqrt((sd_a**2)/n_a+(sd_b**2)/n_b)
    z_score=(mean_b-mean_a)/se_diff if se_diff >0 else 0.0
    p_value=2*(1-norm.cdf(abs(z_score)))

    z_crit=norm.ppf(1-alpha/2)
    margin_error=z_crit*se_diff
    ci_lower=(mean_b-mean_a)-margin_error
    ci_upper=(mean_b-mean_a)+margin_error
    significant=p_value<alpha

    relative_uplift=(mean_b-mean_a)/mean_a if mean_a != 0 else float("inf")

    #Expected value - incremental mean*size of B
    expected_value=(mean_b-mean_a)*n_b

    return{
        "mean_a":float(mean_a),
        "mean_b":float(mean_b),
        "sd_a": float(sd_a),
        "sd_b": float(sd_b),
        "diff": float(mean_b-mean_a),
        "p_value": float(p_value),
        "significant":bool(significant),
        "ci_lower":float(ci_lower),
        "ci_upper": float(ci_upper),
        "relative_uplift": float(relative_uplift),
        "expected_value": float(expected_value)
    }

def bayesian_ab_test(visitors_a, conversions_a, visitors_b, conversions_b, samples=100_000):
    #only for proportion KPIs
    a_alpha, a_beta = 1+conversions_a, 1+(visitors_a-conversions_a)
    b_alpha, b_beta = 1+conversions_b, 1+(visitors_b-conversions_b)

    a_dist=numpy.random.beta(a_alpha, a_beta, samples)
    b_dist=numpy.random.beta(b_alpha, b_beta, samples)

    prob_b_better=(b_dist>a_dist).mean()

    return float(prob_b_better)

def bayesian_expected_metrics(visitors_a, conversions_a, visitors_b, conversions_b,samples=100_000, value_per_conversion=100):
    #Only fpr proportion KPIs
    a_alpha, a_beta = 1 + conversions_a, 1 + (visitors_a - conversions_a)
    b_alpha, b_beta = 1 + conversions_b, 1 + (visitors_b - conversions_b)

    a_dist = numpy.random.beta(a_alpha, a_beta, samples)
    b_dist = numpy.random.beta(b_alpha, b_beta, samples)

    loss_b = float(numpy.mean((a_dist - b_dist) * (a_dist > b_dist)) * value_per_conversion)
    loss_a = float(numpy.mean((b_dist - a_dist) * (b_dist > a_dist)) * value_per_conversion)

    expected_value_b = float(numpy.mean(b_dist - a_dist) * value_per_conversion)
    expected_value_a = float(numpy.mean(a_dist - b_dist) * value_per_conversion)

    return {
        "expected_loss_b": loss_b,
        "expected_loss_a": loss_a,
        "expected_value_b": expected_value_b,
        "expected_value_a": expected_value_a
    }

def bayesian_decision_analysis(visitors_a, conversions_a, visitors_b, conversions_b,samples=100_000, value_per_conversion=100):
    #only for proportion KPIs
    a_alpha, a_beta = 1 + conversions_a, 1 + (visitors_a - conversions_a)
    b_alpha, b_beta = 1 + conversions_b, 1 + (visitors_b - conversions_b)

    a_dist = numpy.random.beta(a_alpha, a_beta, samples)
    b_dist = numpy.random.beta(b_alpha, b_beta, samples)

    diff = b_dist - a_dist
    prob_b_better = float((diff > 0).mean())

    expected_value = float(numpy.mean(diff) * value_per_conversion * visitors_b)
    regret = float(numpy.mean((-diff) * (diff < 0)) * value_per_conversion * visitors_b)
    expected_utility = float(prob_b_better * expected_value - (1 - prob_b_better) * regret)

    return {
        "prob_b_better": prob_b_better,
        "expected_value": expected_value,
        "expected_regret": regret,
        "expected_utility": expected_utility
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

    prob_b_better = float((b_dist > a_dist).mean())
    posterior_mean_a = float(a_alpha / (a_alpha + a_beta))
    posterior_mean_b = float(b_alpha / (b_alpha + b_beta))
    #Decision making
    if prob_b_better > threshold_high:
        decision = "Stop: Variant B is clearly better"
    elif prob_b_better < threshold_low:
        decision = "Stop: Variant A is clearly better"
    else:
        decision = "Continue: Not enough evidence yet"

    return {
        "decision": decision,
        "prob_b_better": prob_b_better,
        "posterior_mean_a": posterior_mean_a,
        "posterior_mean_b": posterior_mean_b
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

def plot_confidence_interval(diff, ci_lower, ci_upper, title="95% Confidence Interval"):
    plt.figure(figsize=(6, 2))
    plt.errorbar(
        x=diff,
        y=0,
        xerr=[[diff - ci_lower], [ci_upper - diff]],
        fmt='o',
        color='blue',
        ecolor='red',
        capsize=5
    )
    plt.axvline(0, color='black', linestyle='--')
    plt.title(title)
    plt.xlabel("Difference")
    plt.yticks([])
    plt.show()

#KPI input

def choose_kpi():
    print("\nChoose KPI type:")
    print("1. Conversion Rate(CR)")
    print("2. ARPU")
    print("3. LTV")
    print("4. Churn")

    k=input("Enter KPI (1/2/3/4): ").strip()
    if k == "1":
        return "cr"
    elif k == "2":
        return "arpu"
    elif k == "3":
        return "ltv"
    elif k == "4":
        return "churn"
    else:
        print("Invalid KPI, defaulting to Conversion Rate.")
        return "cr"

def get_inputs_for_kpi(kpi_type):
    """
    Returns a dict with standardized fields for later use.
    For proportions (cr/churn): visitors_a, conversions_a, visitors_b, conversions_b
    For means (arpu/ltv): users_a, total_a, sd_a(optional), users_b, total_b, sd_b(optional)
    """
    if kpi_type in ("cr", "churn"):
        visitors_a = int(input("Enter users/visitors in Group A: "))
        success_a = int(input("Enter conversions (CR) or lost users (Churn) in Group A: "))
        visitors_b = int(input("Enter users/visitors in Group B: "))
        success_b = int(input("Enter conversions (CR) or lost users (Churn) in Group B: "))
        return{
            "visitors_a": visitors_a,
            "converisons_a": success_a,
            "visitors_b": visitors_b,
            "converisons_b": success_b
        }
    else:
        users_a=int(input("Enter users in Group A: "))
        total_a=float(input("Enter TOTAL value for KPI in Group A (e.g. total ravenue or total LTV): "))
        users_b=int(input("Enter users in Group B: "))
        total_b=float(input("Enter TOTAL value for KPI in Group B: "))
        #Optinal SDs
        sd_a_in = input("Enter sample SD per user for Group A (optional, press Enter to skip): ").strip()
        sd_b_in = input("Enter sample SD per user for Group B (optional, press Enter to skip): ").strip()
        sd_a=float(sd_a_in) if sd_a_in else None
        sd_b = float(sd_b_in) if sd_b_in else None
        return {
            "users_a": users_a,
            "total_a": total_a,
            "sd_a": sd_a,
            "users_b": users_b,
            "total_b": total_b,
            "sd_b": sd_b
        }

def infer_means_payload(kpi_inputs, kpi_type):
    #Build mean/sd/n payload for ARPU/LTV based tests
    n_a=kpi_inputs["users_a"]
    n_b=kpi_inputs["users_b"]
    mean_a=kpi_inputs["total_a"]/n_a if n_a >0 else 0.0
    mean_b=kpi_inputs["total_b"]/n_b if n_b >0 else 0.0

    sd_a=kpi_inputs["sd_a"]
    sd_b=kpi_inputs["sd_b"]

    warning=None
    if sd_a is None:
        sd_a=mean_a
        warning="Sd for Group A not provided - assumed SD = mean"
    if sd_b is None:
        sd_b=mean_b
        warning = ("SDs not provided — assumed SD≈mean for both groups."
                   if warning is None else f"{warning} SD≈mean for Group B")

    return mean_a, sd_a, n_a, mean_b, sd_b, n_b, warning

def run_n_test_cli():
    print("\n=== Multi-variant A/B/n test ===")
    print("Choose KPI type:")
    print("1. Conversion Rate (CR)")
    print("2. ARPU")
    print("3. LTV")
    print("4. Churn")
    kpi_choice=input("Enter KPI (1/2/3/4): ").strip()

    if kpi_choice == "1":
        kpi_type = "conversion"
    else:
        kpi_type = "mean"

    groups = {}
    while True:
        name = input("\nEnter group name (or press Enter to finish): ").strip()
        if not name:
            break

        users = int(input(f"Enter users in Group {name}: "))

        if kpi_type == "conversion":
            convr = int(input(f"Enter conversions in Group {name}: "))
            groups[name] = {"users": users, "conversions": convr}
        else:
            total = float(input(f"Enter TOTAL value for KPI in Group {name}: "))
            sd = input(f"Enter sample SD per user for Group {name} (optional, press Enter to skip): ")
            sd = float(sd) if sd.strip() != "" else None
            groups[name] = {"users": users, "total": total, "sd": sd}

    alpha = input("Enter significance level (default 0.05): ").strip()
    alpha = float(alpha) if alpha else 0.05

    results = run_n_test(groups, kpi_type=kpi_type, alpha=alpha)

    print_n_results(results, kpi_type=kpi_type)

def run_sequential_cli():
    print("\n=== Sequential Testing ===")
    print("Choose KPI type:")
    print("1. Conversion Rate (CR)")
    print("2. ARPU")
    print("3. LTV")
    print("4. Churn")
    kpi_choice = input("Enter KPI (1/2/3/4):").strip()

    if kpi_choice == "1":
        kpi_type = "conversion"
    else:
        kpi_type = "mean"

    data_over_time={}
    day = 1
    while True:
        print(f"\n+++ Data for Day {day} +++")
        day_name=f"day{day}"
        groups={}

        while True:
            name = input("\nEnter group name (or press Enter to finish day):").strip()
            if not name:
                break

            users = int(input(f"Enter users in Group {name}: "))

            if kpi_type == "conversion":
                conv = int(input(f"Enter conversions in Group {name}: "))
                groups[name] = {"users":users, "conversions":conv}
            else:
                total = float(input(f"Enter TOTAL value for KPI in Group {name}: "))
                sd = input(f"Enter sample SD per user for group {name} (optional, press Enter to skip): ")
                sd = float(sd) if sd.strip() != "" else None
                groups[name] = {"users":users, "total": total, "sd":sd}

        if not groups:
            break

        data_over_time[day_name] = groups
        day += 1

        cont = input("Add another day? (y/n): ").strip().lower()
        if cont != "y":
            break

    alpha = input("Enter significance level (default 0.05): ").strip()
    alpha = float(alpha) if alpha else 0.05

    print("\nChoose sequential method:")
    print("1. Pocock")
    print("2. OBrien-Fleming")
    method_choice = input("Enter method (1/2): ").strip()
    method = "pocock" if method_choice == "1" else "obrien"

    results = run_sequential_test(data_over_time, kpi_type=kpi_type, alpha=alpha, method=method)
    print_sequential_results(results)

def run_decision_cli():
    print("\n=== Bayesian Decision-Theoretic Analysis ===")
    print("Choose KPI type:")
    print("1. Conversion Rate (CR)")
    print("2. ARPU")
    print("3. LTV")
    print("4. Churn")
    kpi_choice = input("Enter KPI (1/2/3/4): ").strip()

    kpi_type = "conversion" if kpi_choice == "1" else "mean"

    groups = {}
    while True:
        name = input("\nEnter group name (or press Enter to finish): ").strip()
        if not name:
            break

        users = int(input(f"Enter users in Group {name}: "))

        if kpi_type == "conversion":
            conv = int(input(f"Enter conversions in Group {name}: "))
            groups[name] = {"users": users, "conversions": conv}
        else:
            total = float(input(f"Enter TOTAL value for KPI in Group {name}: "))
            sd = input(f"Enter sample SD per user for Group {name} (optional, press Enter to skip): ")
            sd = float(sd) if sd.strip() else None
            groups[name] = {"users": users, "total": total, "sd": sd}

    alpha = input("Enter credible interval level (default 0.05): ").strip()
    alpha = float(alpha) if alpha else 0.05

    gain = input("Enter utility gain (default 1): ").strip()
    loss = input("Enter utility loss (default -1): ").strip()
    utility = {
        "gain": float(gain) if gain else 1.0,
        "loss": float(loss) if loss else -1.0
    }

    sims = input("Number of Monte Carlo simulations (default 5000): ").strip()
    sims = int(sims) if sims else 5000

    results = decision_analysis(groups, kpi_type=kpi_type, alpha=alpha, utility=utility, simulations=sims)
    print_decision_results(results)

def run_markov_cli():
    print("\n=== Markov Chain Economic Model ===")

    states_input = input("Enter states (comma-separated, e.g., Active,Churned): ").strip()
    states = [s.strip() for s in states_input.split(",")]

    print("\nNow enter transition counts between states:")
    data = {}
    for s_from in states:
        for s_to in states:
            count = input(f"Count of transitions {s_from} -> {s_to} (default 0): ").strip()
            count = int(count) if count else 0
            data[(s_from, s_to)] = count

    P = build_transition_matrix(data, states)

    initial_state_name = input(f"Enter initial state from {states}: ").strip()
    initial_state = states.index(initial_state_name)
    steps = int(input("Number of steps to simulate (default 10): ") or 10)
    history = simulate_markov(P, initial_state=initial_state, steps=steps)

    steady = steady_state(P)

    print("\nEnter revenue per state (e.g., Active=100, Churned=0):")
    revenues = []
    for s in states:
        val = input(f"Revenue for state {s}: ").strip()
        revenues.append(float(val) if val else 0.0)
    revenues = numpy.array(revenues)

    discount = float(input("Enter discount factor (default 0.95): ") or 0.95)
    ltv = expected_ltv(P, revenues, initial_state=initial_state, discount=discount)

    print_markov_results(P, states, history, steady, ltv)

def run_linear_opt_cli():
    print("\n=== Linear Optimization ===")

    n = int(input("Enter number of decision variables: "))
    var_names = [input(f"Name of variable {i+1}: ") for i in range(n)]

    print("\nEnter objective function coefficients (profit per variable):")
    objective = [float(input(f"{var}: ")) for var in var_names]

    maximize = input("Maximize? (y/n, default y): ").strip().lower() != "n"

    constraints = {}
    m = int(input("\nEnter number of constraints: "))
    for i in range(m):
        coeffs = [float(input(f"Coefficient for {var} in constraint {i+1}: ")) for var in var_names]
        sign = input("Sign (<= or >=): ").strip()
        rhs = float(input("Right-hand side value: "))
        constraints[f"constr_{i+1}"] = (coeffs, sign, rhs)

    print("\nEnter variable bounds:")
    bounds = []
    for var in var_names:
        lb = float(input(f"Lower bound for {var} (default 0): ") or 0)
        ub_input = input(f"Upper bound for {var} (default None): ").strip()
        ub = float(ub_input) if ub_input else None
        bounds.append((lb, ub))

    result = run_linear_optimization(objective, constraints, bounds, maximize=maximize)
    print_linear_results(result, var_names)

def run_scenario_cli():
    print("\n=== Scenario Analysis ===")
    n = int(input("Enter number of base parameters: "))
    base_params = {}
    for i in range(n):
        key = input(f"Name of parameter {i+1}: ").strip()
        val = float(input(f"Value of {key}: "))
        base_params[key] = val

    m = int(input("\nEnter number of scenarios: "))
    scenarios = {}
    for j in range(m):
        scen_name = input(f"\nScenario {j+1} name: ").strip()
        multipliers = {}
        for k in base_params:
            mult = float(input(f"Multiplier for {k} in {scen_name} (default 1.0): ") or 1.0)
            multipliers[k] = mult
        scenarios[scen_name] = multipliers

    # Simple default model: profit = (price - cost) * users
    def default_model(params):
        if "price" in params and "cost" in params and "users" in params:
            return (params["price"] - params["cost"]) * params["users"]
        else:
            return sum(params.values())  # fallback simple model

    results = run_scenario_analysis(base_params, scenarios, default_model)
    print_scenario_results(results)

def run_cost_benefit_cli():
    print("\n=== Cost-Benefit Simulation ===")

    mean_cost = float(input("Enter mean cost: "))
    sd_cost = float(input("Enter SD of cost (0 if deterministic): ") or 0)
    mean_benefit = float(input("Enter mean benefit: "))
    sd_benefit = float(input("Enter SD of benefit (0 if deterministic): ") or 0)

    discount_rate = float(input("Enter discount rate (default 0.0): ") or 0.0)
    horizon = int(input("Enter time horizon (years, default 1): ") or 1)

    sims = int(input("Number of Monte Carlo simulations (default 10000): ") or 10000)

    cost_params = {"mean": mean_cost, "sd": sd_cost}
    benefit_params = {"mean": mean_benefit, "sd": sd_benefit}

    results = run_cost_benefit_simulation(cost_params, benefit_params, simulations=sims,
                                          discount_rate=discount_rate, horizon=horizon)

    print_cost_benefit_results(results)

############################################################################################

#Main CLI
def main():
    while True:
        print("\n=== A/B test calculator ===")
        print("1. Sample size calculator")
        print("2. Classic A/B test")
        print("3. Bayesian A/B test")
        print("4. Advanced Metrics (Relative Uplift, ROI, Payback)")
        print("5. Bayesian Expected Metrics (Loss & Value)")
        print("6. Bayesian Decision Analysis (Utility & Regret)")
        print("7. Sequential Bayesian Monitoring")
        print("8. Multi-varian A/B/n test")
        print("9. Sequential testing")
        print("10. Monte Carlo simulation")
        print("11. Bayesian credible intervals & Expected Utility")
        print("12. Bayesian Decision-Theoretic Analysis")
        print("13. Markov chain modelling")
        print("14. Linear optimisation")
        print("15. Scenario analysis")
        print("16. Cost-benefit simulation")
        print("0. Exit")
        choice = input("Choose an option: ").strip()

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
            kpi_type = choose_kpi()
            if kpi_type in ("cr", "churn"):
                data = get_inputs_for_kpi(kpi_type)
                result = ab_test(
                    data["visitors_a"], data["conversions_a"],
                    data["visitors_b"], data["conversions_b"]
                )
                result["kpi_type"] = kpi_type

                table = [
                    ["Metric", "Value"],
                    ["Rate A", f"{result['cr_a']:.2%}"],
                    ["Rate B", f"{result['cr_b']:.2%}"],
                    ["Difference", f"{result['diff']:.2%}"],
                    ["95% CI Lower", f"{result['ci_lower']:.2%}"],
                    ["95% CI Upper", f"{result['ci_upper']:.2%}"],
                    ["p-value", f"{result['p_value']:.4f}"],
                    ["Significant?", "Yes" if result['significant'] else "No"],
                    ["Relative Uplift", f"{result['relative_uplift']:.2%}"],
                    ["Expected Loss", f"{result['expected_loss']:.4f}"],
                    ["Expected Value", f"{result['expected_value']:.2f}"]
                ]

                print("\nClassic A/B Test Results")
                print(tabulate(table, headers="firstrow", tablefmt="grid"))
                plot_confidence_interval(result['diff'], result['ci_lower'], result['ci_upper'], title="95% CI for Rate Difference")

                save = input("Save results? (csv/json/skip): ").strip().lower()
                if save in ["csv", "json"]:
                    from csvjsonm import save_results
                    save_results(result, "ab_test_results", save)

                logging.info(
                    f"[{kpi_type}] Classic A/B | A: {data['visitors_a']}/{data['conversions_a']} "
                    f"| B: {data['visitors_b']}/{data['conversions_b']} "
                    f"| diff={result['diff']:.4f}, p={result['p_value']:.4f}, sig={result['significant']}"
                )

            else:
                # Means-based KPI (ARPU/LTV)
                data = get_inputs_for_kpi(kpi_type)
                mean_a, sd_a, n_a, mean_b, sd_b, n_b, warn = infer_means_payload(data, kpi_type)
                alpha = float(input("Enter significance level (default 0.05): ") or 0.05)
                result = ab_test_means(mean_a, sd_a, n_a, mean_b, sd_b, n_b, alpha=alpha)
                result["kpi_type"] = kpi_type

                table = [
                    ["Metric", "Value"],
                    ["Mean A", f"{result['mean_a']:.4f}"],
                    ["Mean B", f"{result['mean_b']:.4f}"],
                    ["Difference", f"{result['diff']:.4f}"],
                    ["95% CI Lower", f"{result['ci_lower']:.4f}"],
                    ["95% CI Upper", f"{result['ci_upper']:.4f}"],
                    ["p-value", f"{result['p_value']:.4f}"],
                    ["Significant?", "Yes" if result['significant'] else "No"],
                    ["Relative Uplift", f"{result['relative_uplift']:.2%}"],
                    ["Expected Value (Δ×nB)", f"{result['expected_value']:.4f}"]
                ]

                print("\nClassic A/B Test (Means KPI) Results")
                print(tabulate(table, headers="firstrow", tablefmt="grid"))
                if warn:
                    print(f"Note: {warn}")

                plot_confidence_interval(result['diff'], result['ci_lower'], result['ci_upper'], title="95% CI for Mean Difference")

                save = input("Save results? (csv/json/skip): ").strip().lower()
                if save in ["csv", "json"]:
                    from csvjsonm import save_results
                    save_results(result, f"ab_test_results_{kpi_type}", save)

                logging.info(
                    f"[{kpi_type}] Classic A/B (means) | "
                    f"A(n={n_a}, mean={mean_a:.4f}, sd={sd_a:.4f}) "
                    f"| B(n={n_b}, mean={mean_b:.4f}, sd={sd_b:.4f}) "
                    f"| diff={result['diff']:.4f}, p={result['p_value']:.4f}, sig={result['significant']}"
                )

#3.
        elif choice == "3":
            kpi_type = choose_kpi()
            if kpi_type not in ("cr", "churn"):
                print(
                    "Bayesian A/B currently supports only proportion KPIs (CR/Churn). Use option 2 or 4 for ARPU/LTV.")
                continue

            data = get_inputs_for_kpi(kpi_type)
            prob_b_better = bayesian_ab_test(
                data["visitors_a"], data["conversions_a"],
                data["visitors_b"], data["conversions_b"]
            )
            print(f"\nBayesian Result: Probability B > A = {prob_b_better:.2%}")

            logging.info(
                f"[{kpi_type}] Bayesian A/B | A: {data['visitors_a']}/{data['conversions_a']}, "
                f"B: {data['visitors_b']}/{data['conversions_b']} | P(B>A)={prob_b_better:.4f}"
            )

        elif choice == "4":
            kpi_type = choose_kpi()
            if kpi_type in ("cr", "churn"):
                data = get_inputs_for_kpi(kpi_type)
                value_per_conversion = float(input("Enter value per success/conversion ($): "))
                cost_a = float(input("Enter total cost for Group A ($): "))
                cost_b = float(input("Enter total cost for Group B ($): "))

                result = ab_test(
                    data["visitors_a"], data["conversions_a"],
                    data["visitors_b"], data["conversions_b"],
                    revenue_per_conversion=value_per_conversion
                )
                result["kpi_type"] = kpi_type

                revenue_a = data["conversions_a"] * value_per_conversion
                revenue_b = data["conversions_b"] * value_per_conversion

                roi_a = (revenue_a - cost_a) / cost_a if cost_a > 0 else 0.0
                roi_b = (revenue_b - cost_b) / cost_b if cost_b > 0 else 0.0

                payback_a = cost_a / revenue_a if revenue_a > 0 else float("inf")
                payback_b = cost_b / revenue_b if revenue_b > 0 else float("inf")

                rel_uplift = (result['cr_b'] - result['cr_a']) / result['cr_a'] if result['cr_a'] > 0 else float("inf")

                table = [
                    ["Metric", "Group A", "Group B"],
                    ["Rate", f"{result['cr_a']:.2%}", f"{result['cr_b']:.2%}"],
                    ["Revenue ($)", f"{revenue_a:.2f}", f"{revenue_b:.2f}"],
                    ["ROI", f"{roi_a:.2%}", f"{roi_b:.2%}"],
                    ["Payback Period", f"{payback_a:.2f}", f"{payback_b:.2f}"],
                    ["Relative Uplift", f"{rel_uplift:.2%}", ""]
                ]

                print("\nAdvanced Metrics (with ROI & Payback)")
                print(tabulate(table, headers="firstrow", tablefmt="fancy_grid"))

                logging.info(
                    f"[{kpi_type}] Advanced | ROI_A={roi_a:.4f}, Payback_A={payback_a:.4f} | "
                    f"ROI_B={roi_b:.4f}, Payback_B={payback_b:.4f}"
                )

            else:
                # ARPU / LTV advanced: ROI/Payback via means
                data = get_inputs_for_kpi(kpi_type)
                mean_a, sd_a, n_a, mean_b, sd_b, n_b, warn = infer_means_payload(data, kpi_type)
                alpha = float(input("Enter significance level (default 0.05): ") or 0.05)
                cost_a = float(input("Enter total cost for Group A ($): "))
                cost_b = float(input("Enter total cost for Group B ($): "))

                result = ab_test_means(mean_a, sd_a, n_a, mean_b, sd_b, n_b, alpha=alpha)
                result["kpi_type"] = kpi_type

                revenue_a = mean_a * n_a
                revenue_b = mean_b * n_b
                roi_a = (revenue_a - cost_a) / cost_a if cost_a > 0 else 0.0
                roi_b = (revenue_b - cost_b) / cost_b if cost_b > 0 else 0.0
                payback_a = cost_a / revenue_a if revenue_a > 0 else float("inf")
                payback_b = cost_b / revenue_b if revenue_b > 0 else float("inf")

                table = [
                    ["Metric", "Group A", "Group B"],
                    ["Mean", f"{mean_a:.4f}", f"{mean_b:.4f}"],
                    ["Revenue ($)", f"{revenue_a:.2f}", f"{revenue_b:.2f}"],
                    ["ROI", f"{roi_a:.2%}", f"{roi_b:.2%}"],
                    ["Payback Period", f"{payback_a:.2f}", f"{payback_b:.2f}"],
                    ["Δ Mean (CI)", f"{result['diff']:.4f}", f"[{result['ci_lower']:.4f}; {result['ci_upper']:.4f}]"],
                ]

                print("\nAdvanced Metrics (Means KPI: ROI & Payback)")
                print(tabulate(table, headers="firstrow", tablefmt="fancy_grid"))
                if warn:
                    print(f"Note: {warn}")

                logging.info(
                    f"[{kpi_type}] Advanced (means) | ROI_A={roi_a:.4f}, Payback_A={payback_a:.4f} | "
                    f"ROI_B={roi_b:.4f}, Payback_B={payback_b:.4f}"
                )

        elif choice == "5":
            kpi_type = choose_kpi()
            if kpi_type not in ("cr", "churn"):
                print("Bayesian Expected Metrics currently support only CR/Churn. Use option 4 for ARPU/LTV.")
                continue

            data = get_inputs_for_kpi(kpi_type)
            metrics = bayesian_expected_metrics(
                data["visitors_a"], data["conversions_a"],
                data["visitors_b"], data["conversions_b"]
            )
            metrics["kpi_type"] = kpi_type

            table = [
                ["Metric", "Group A", "Group B", "Note"],
                ["Expected Loss", f"{metrics['expected_loss_a']:.2f}", f"{metrics['expected_loss_b']:.2f}", ""],
                ["Expected Value ($)", f"{metrics['expected_value_a']:.2f}", f"{metrics['expected_value_b']:.2f}", ""]
            ]

            print("\nBayesian Expected Metrics (Table)")
            print(tabulate(table, headers="firstrow", tablefmt="fancy_grid"))

            save = input("Save results? (csv/json/skip): ").strip().lower()
            if save in ["csv", "json"]:
                from csvjsonm import save_results
                save_results(metrics, "bayesian_expected_metrics", save)

            logging.info(
                f"[{kpi_type}] Bayesian Expected | "
                f"Loss_A={metrics['expected_loss_a']:.4f}, Loss_B={metrics['expected_loss_b']:.4f}, "
                f"EV_A={metrics['expected_value_a']:.2f}, EV_B={metrics['expected_value_b']:.2f}"
            )

        elif choice == "6":
            kpi_type = choose_kpi()
            if kpi_type not in ("cr", "churn"):
                print("Bayesian Decision Analysis currently supports only CR/Churn. Use option 4 for ARPU/LTV.")
                continue

            data = get_inputs_for_kpi(kpi_type)
            value_per_conversion = float(input("Enter value per success/conversion ($): "))
            decision = bayesian_decision_analysis(
                data["visitors_a"], data["conversions_a"],
                data["visitors_b"], data["conversions_b"],
                value_per_conversion=value_per_conversion
            )
            decision["kpi_type"] = kpi_type

            table = [
                ["Metric", "Value"],
                ["Probability B > A", f"{decision['prob_b_better']:.2%}"],
                ["Expected Value ($)", f"{decision['expected_value']:.2f}"],
                ["Expected Regret ($)", f"{decision['expected_regret']:.2f}"],
                ["Expected Utility ($)", f"{decision['expected_utility']:.2f}"]
            ]

            print("\nBayesian Decision Analysis (Table)")
            print(tabulate(table, headers="firstrow", tablefmt="grid"))

            save = input("Save results? (csv/json/skip): ").strip().lower()
            if save in ["csv", "json"]:
                from csvjsonm import save_results
                save_results(decision, "bayesian_decision_analysis", save)

            logging.info(
                f"[{kpi_type}] Bayesian Decision | "
                f"P(B>A)={decision['prob_b_better']:.4f}, "
                f"EV={decision['expected_value']:.2f}, "
                f"Regret={decision['expected_regret']:.2f}, "
                f"Utility={decision['expected_utility']:.2f}"
            )

        elif choice == "7":
            kpi_type = choose_kpi()
            if kpi_type not in ("cr", "churn"):
                print("Sequential Bayesian Monitoring currently supports only CR/Churn.")
                continue

            data = get_inputs_for_kpi(kpi_type)
            monitoring = sequential_bayesian_monitoring(
                data["visitors_a"], data["conversions_a"],
                data["visitors_b"], data["conversions_b"]
            )
            monitoring["kpi_type"] = kpi_type

            print("\nSequential Bayesian Monitoring")
            print(f"Posterior Mean A: {monitoring['posterior_mean_a']:.4f}")
            print(f"Posterior Mean B: {monitoring['posterior_mean_b']:.4f}")
            print(f"Probability B > A: {monitoring['prob_b_better']:.2%}")
            print(f"Decision: {monitoring['decision']}")

            logging.info(
                f"[{kpi_type}] Sequential Monitoring | "
                f"PostMeanA={monitoring['posterior_mean_a']:.4f}, "
                f"PostMeanB={monitoring['posterior_mean_b']:.4f}, "
                f"P(B>A)={monitoring['prob_b_better']:.4f}, Decision={monitoring['decision']}"
            )

        elif choice == "8":
            run_n_test_cli()

        elif choice == "9":
            run_sequential_cli()

        elif choice == "10":
            print("\n=== Monte Carlo Simulation ===")
            print("Choose KPI type:")
            print("1. Conversion Rate (CR)")
            print("2. ARPU")
            print("3. LTV")
            print("4. Churn")
            kpi_choice = input("Enter KPI (1/2/3/4): ").strip()
            kpi_type = "conversion" if kpi_choice == "1" else "mean"

            groups = {}
            while True:
                name = input("\nEnter group name (or press Enter to finish): ").strip()
                if not name:
                    break
                users = int(input(f"Enter users in Group {name}: "))
                if kpi_type == "conversion":
                    conv = int(input(f"Enter conversions in Group {name}: "))
                    groups[name] = {"users": users, "conversions": conv}
                else:
                    total = float(input(f"Enter TOTAL value for KPI in Group {name}: "))
                    sd = input(f"Enter sample SD per user for Group {name} (optional, press Enter to skip): ")
                    sd = float(sd) if sd.strip() != "" else None
                    groups[name] = {"users": users, "total": total, "sd": sd}

            n_sim = input("Enter number of simulations (default 10000): ").strip()
            n_sim = int(n_sim) if n_sim else 10000

            results = run_montecarlo(groups, kpi_type=kpi_type, n_sim=n_sim)
            print_montecarlo_results(results)

        elif choice == "11":
            print("\n=== Bayesian Test with Credible Intervals & Expected Utility ===")
            print("Choose KPI type:")
            print("1. Conversion Rate (CR)")
            print("2. ARPU")
            kpi_choice = input("Enter KPI (1/2): ").strip()
            kpi_type = "conversion" if kpi_choice == "1" else "mean"

            groups = {}
            while True:
                name = input("\nEnter group name (or press Enter to finish): ").strip()
                if not name:
                    break
                users = int(input(f"Enter users in Group {name}: "))
                if kpi_type == "conversion":
                    conv = int(input(f"Enter conversions in Group {name}: "))
                    groups[name] = {"users": users, "conversions": conv}
                else:
                    total = float(input(f"Enter TOTAL value for KPI in Group {name}: "))
                    sd = input(f"Enter sample SD per user for Group {name} (optional, press Enter to skip): ")
                    sd = float(sd) if sd.strip() != "" else None
                    groups[name] = {"users": users, "total": total, "sd": sd}

            alpha = input("Enter significance level (default 0.05): ").strip()
            alpha = float(alpha) if alpha else 0.05

            use_utility = input("Run Expected Utility Analysis? (y/n): ").strip().lower()
            utility = None
            if use_utility == "y":
                gain = float(input("Enter gain if better option chosen: "))
                loss = float(input("Enter loss if worse option chosen: "))
                utility = {"gain": gain, "loss": loss}

            results = bayesian_test(groups, kpi_type=kpi_type, alpha=alpha, utility=utility)
            print_bayesian_results(results)

        elif choice == "12":
            run_decision_cli()

        elif choice == "13":
            run_markov_cli()

        elif choice == "14":
            run_linear_opt_cli()

        elif choice == "15":
            run_scenario_cli()

        elif choice == "16":
            run_cost_benefit_cli()

        elif choice == "0":
            print("Exit")
            break
        else:
            print("Invalid choice, try again.")

if __name__ == "__main__":
    main()