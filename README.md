# A/B Test Calculator

A simple Python tool for conducting A/B tests. Supports both the classical (frequentist) approach and Bayesian A/B testing with visualisation of results.

## ❒ Features

✧ Calculation of sample size (the number of users required for the test)

✧ Classic A/B test:

- conversions

- p-value

- confidence interval (95% CI)

 - statistical significance test

✧ Bayesian A/B test:

- probability that B > A

- Beta distribution graph for A and B

✧ Visualisations:

- 95% CI as a graph

- probability distributions for the Bayesian approach (saved as PNG)

✧ Automatic input checks (incorrect values are not accepted

## ❒ Installation

```
git clone https://github.com/username/ab-test-calculator.git
cd ab-test-calculator
python -m venv .venv
source .venv/bin/activate   # або .venv\Scripts\activate для Windows
pip install -r requirements.txt
```
## ❒ Usage

#### 1.Sample Size Calculator
`python main.py`

**Input example:**
```
Sample Size Calculator
Enter baseline conversion rate (e.g. 0.1 for 10%): 0.1
Enter minimum detectable effect (e.g. 0.02 for +2%): 0.02
Enter significance level (default 0.05): 0.05
Enter statistical power (default 0.8): 0.8
```
**output**
```
You need at least 2401 visitors per group for the test.
```
#### 2. A/B test
```
A/B test calculator
Enter the visitors in Group A: 1000
Enter the conversions in Group A: 120
Enter the visitors in Group B: 980
Enter the conversions in Group B: 150
```
**output**
```
Results
Conversion Rate A: 12.00%
Conversion Rate B: 15.31%
Difference: 3.31%
95% CI for difference: [0.34%, 6.28%]
p-value: 0.0291
Statistically Significant? Yes
Graph saved as ab_test_ci.png
```
#### 3. Bayesian A/B test
```
Bayesian A/B Test
Probability that B > A: 96.4%
Graph saved as bayesian_ab.png
```

## ❒ Roadmap
### ✧  Stage 1. Basic calculator
- [x] Sample size calculation
- [x] Classic A/B test (z-score, p-value, CI)
- [x] Bayesian A/B test (posterior, prob B > A)
- [x] Confidence interval graph

### ✧  Stage 2. User convenience
- [x] CLI menu with mode selection (Sample size / Classic AB / Bayesian AB / Advanced metrics)
- [x] Output of results in tabular form (rich or tabulate)
- [x] Saving results in .csv / .json
- [x] Logging test runs to a file

### ✧  Stage 3. New metrics
- [x] Relative Uplift
- [x] Expected Loss & Expected Value in $
- [x] ROI / Payback period
- [x] Support for various KPIs: not only conversions, but also ARPU, LTV, Churn

### ✧  Stage 4. Multi-variant experiments
- [x] A/B/n tests (more than 2 options)
- [x] Correction for multiple comparisons (Bonferroni, Holm-Bonferroni)
- [x] Choosing ‘leaders’ from among 3+ options

### ✧  Stage 5. Statistical extensions
- [x] Sequential testing
- [x] Monte Carlo simulations
- [x] Bayesian credible intervals + Expected Utility
- [x] Decision-theoretic approach

### ✧  Stage 6. Economic and mathematical modelling
- [x] Markov chains
- [x] Linear optimisation
- [x] Scenario analysis
- [x] Cost-benefit simulation

### ✧  Stage 7. Visualisation and reports
- [x] Best graphs (cumulative results over time)
- [x] Scatter Expected Value vs Expected Regret
- [x] Heatmap of probabilities (for different mde/alpha)
- [x] Automatic PDF report with graphs + conclusions

## ❒ License

MIT License