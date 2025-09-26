import matplotlib.pyplot as plt
import numpy as np

# === 1. Sequential Testing (p-values over time) ===
def plot_cumulative_results(data_over_time, kpi_type="conversion", p_values=None):
    days = list(data_over_time.keys())
    if not p_values:
        print("No p-values provided for plotting.")
        return

    plt.figure(figsize=(8, 5))
    plt.plot(list(p_values.keys()), list(p_values.values()), marker="o", label="P-value")
    plt.axhline(0.05, color="red", linestyle="--", label="α=0.05")
    plt.title(f"Sequential Test Results ({kpi_type})")
    plt.xlabel("Day")
    plt.ylabel("P-value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# === 2. A/B/n Test (bar chart of CR or means) ===
def plot_abn_results(results, kpi_type="conversion"):
    if kpi_type == "conversion":
        values = results.get("conversion_rates", {})
        title = "Conversion Rates by Group"
        ylabel = "Conversion Rate"
    else:
        values = results.get("group_means", {})
        title = "Group Means"
        ylabel = "Mean Value"

    if not values:
        return

    plt.figure(figsize=(7, 5))
    plt.bar(values.keys(), values.values(), color="skyblue")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(axis="y", alpha=0.3)
    plt.show()

# === 3. Bayesian Decision (expected utility) ===
def plot_decision_results(results):
    eu = results.get("expected_utilities", {})
    if not eu:
        return

    plt.figure(figsize=(7, 5))
    plt.bar(eu.keys(), eu.values(), color="green")
    plt.title("Expected Utility per Group")
    plt.ylabel("Expected Utility")
    plt.grid(axis="y", alpha=0.3)
    plt.show()

# === 4. Cost-Benefit Simulation (profit distribution) ===
def plot_cost_benefit(results):
    sims = results.get("simulations", [])
    if not len(sims):
        return

    plt.figure(figsize=(8, 5))
    plt.hist(sims, bins=30, color="orange", alpha=0.7)
    plt.title("Cost-Benefit Simulation")
    plt.xlabel("Profit")
    plt.ylabel("Frequency")
    plt.grid(alpha=0.3)
    plt.show()

# === 5. Scenario Analysis ===
def plot_scenarios(results):
    scenarios = results.get("scenarios", {})
    if not scenarios:
        return

    plt.figure(figsize=(7, 5))
    plt.bar(scenarios.keys(), scenarios.values(), color="purple")
    plt.title("Scenario Analysis")
    plt.ylabel("Outcome")
    plt.grid(axis="y", alpha=0.3)
    plt.show()

# === 6. Markov Chains (state evolution) ===
def plot_markov(states_over_time):
    plt.figure(figsize=(8, 5))
    for state, values in states_over_time.items():
        plt.plot(range(len(values)), values, marker="o", label=state)

    plt.title("Markov Chain State Evolution")
    plt.xlabel("Step")
    plt.ylabel("Probability")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()