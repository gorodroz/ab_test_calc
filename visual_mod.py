import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize

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

# === 7. Heatmap of probabilities (for different mde/alpha) ===
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize

def plot_power_heatmap(baseline_rate=0.1,sample_size=500,mde_range=None,alpha_range=None,savepath=None):
    if mde_range is None:
        mde_range = np.linspace(0.01, 0.1, 10)
    if alpha_range is None:
        alpha_range = [0.01, 0.05, 0.1]

    analysis = NormalIndPower()
    power_matrix = np.zeros((len(alpha_range), len(mde_range)))

    for i, a in enumerate(alpha_range):
        for j, m in enumerate(mde_range):
            p1 = float(baseline_rate)
            p2 = float(baseline_rate * (1.0 + m))
            p2 = min(max(p2, 1e-8), 1 - 1e-8)

            try:
                effect_size = proportion_effectsize(p1, p2)
            except Exception:
                effect_size = 2 * (np.arcsin(np.sqrt(p2)) - np.arcsin(np.sqrt(p1)))

            power = analysis.power(effect_size=abs(effect_size),
                                   nobs1=sample_size,
                                   alpha=a,
                                   ratio=1.0)
            power_matrix[i, j] = power

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(power_matrix, aspect='auto', origin='lower', interpolation='nearest', cmap='YlGnBu')

    ax.set_xticks(np.arange(len(mde_range)))
    ax.set_xticklabels([f"{m*100:.1f}%" for m in mde_range], rotation=45, ha='right')
    ax.set_yticks(np.arange(len(alpha_range)))
    ax.set_yticklabels([f"α={a}" for a in alpha_range])

    ax.set_xlabel("Minimum Detectable Effect (MDE)")
    ax.set_ylabel("Significance level (α)")
    ax.set_title(f"Power Heatmap (baseline={baseline_rate:.1%}, n/group={sample_size})")

    for i in range(power_matrix.shape[0]):
        for j in range(power_matrix.shape[1]):
            ax.text(j, i, f"{power_matrix[i, j]:.2f}", ha="center", va="center", fontsize=9, color="black")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Power")

    plt.tight_layout()

    if savepath:
        plt.savefig(savepath, dpi=150)
        plt.close(fig)
        print(f"Saved power heatmap to {savepath}")
    else:
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