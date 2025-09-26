import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

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
def plot_markov(history, states=None):
    history = np.array(history)
    steps = range(len(history))
    if states is None:
        states = [f"State{i}" for i in range(history.shape[1])]

    plt.figure(figsize=(8,5))
    for i, state in enumerate(states):
        plt.plot(steps, history[:, i], marker="o", label=state)

    plt.xlabel("Step")
    plt.ylabel("Probability")
    plt.title("Markov Chain State Evolution")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()

def plot_decision_scatter(eu, er):
    plt.figure(figsize=(7, 6))
    for g in eu.keys():
        plt.scatter(eu[g], er[g], s=120, label=g)
        plt.text(eu[g] * 1.01, er[g] * 1.01, g, fontsize=10)

    plt.xlabel("Expected Utility (EV)")
    plt.ylabel("Expected Regret (ER)")
    plt.title("Expected Value vs Expected Regret (Scatter)")
    plt.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    plt.axvline(max(eu.values()), color="gray", linestyle="--", linewidth=0.8)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()


def plot_decision_heatmap(eu, er):
    df = pd.DataFrame({
        "Group": list(eu.keys()),
        "Expected Utility": list(eu.values()),
        "Expected Regret": list(er.values())
    })
    df = df.set_index("Group")

    plt.figure(figsize=(7, 4))
    sns.heatmap(df, annot=True, fmt=".4f", cmap="RdYlGn_r", cbar=True)
    plt.title("Expected Value vs Expected Regret (Heatmap)")
    plt.show()