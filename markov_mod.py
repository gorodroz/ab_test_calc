import numpy as np
from tabulate import tabulate
from visual_mod import plot_markov

def build_transition_matrix(data, states):
    n = len(states)
    mat = np.zeros((n, n))
    for (s_from, s_to), count in data.items():
        i, j = states.index(s_from), states.index(s_to)
        mat[i, j] += count

    mat = mat / mat.sum(axis=1, keepdims=True)
    return mat

def simulate_markov(P, initial_state, steps=10):
    n = P.shape[0]
    state = np.zeros(n)
    state[initial_state] = 1.0
    history = [state]

    for _ in range(steps):
        state = state @ P
        history.append(state)

    return np.array(history)

def steady_state(P, tol=1e-8, max_iter=1000):
    n = P.shape[0]
    pi = np.ones(n) / n
    for _ in range(max_iter):
        new_pi = pi @ P
        if np.allclose(new_pi, pi, atol=tol):
            break
        pi = new_pi
    return pi

def expected_ltv(P, revenues, initial_state, discount=0.95, steps=50):
    history = simulate_markov(P, initial_state, steps)
    discounted = [discount**t * (h @ revenues) for t, h in enumerate(history)]
    return sum(discounted)

def print_markov_results(P, states, history, steady, ltv=None):
    print("\n=== Markov Chain Analysis ===")
    print("Transition Matrix:")
    print(tabulate(P, headers=states, showindex=states, tablefmt="grid"))
    print("\nDistribution over time:")
    for t, dist in enumerate(history):
        row = [f"{p:.4f}" for p in dist]
        print(f"Step {t}: {row}")
    print("\nSteady-state distribution:")
    print({s: f"{p:.4f}" for s, p in zip(states, steady)})
    if ltv is not None:
        print(f"\nExpected discounted LTV: {ltv:.2f}")
    #plot_markov()
