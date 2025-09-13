# Time-Series Forecasting Toolkit: ARIMA & GARCH with Walk-Forward CV

## Overview
This project implements ARIMA and GARCH models with a **scikit-learn–style API**, built from scratch using NumPy, pandas, and SciPy.  
It includes **walk-forward cross-validation** for out-of-sample testing on financial and macroeconomic data.

---

## Key Features
- ARIMA(p,d,q) with differencing and Gaussian likelihood.
- GARCH(p,q) with conditional variance recursion.
- `fit()`, `predict()`, `score()` API aligned with scikit-learn standards.
- Walk-forward expanding-window cross-validation.
- Optional comparisons to `statsmodels` (ARIMA) and `arch` (GARCH), if installed.
- All in a **single Python file** (`timeseries_toolkit.py`).

---

## Example Output

### ARIMA(1,1,1) Forecast
In-sample MSE: **1.0212**

| Step |   Mean   |  Lower  |  Upper  |
|------|----------|---------|---------|
| 300  | 7.800518 | 5.789454 | 9.811582 |
| 301  | 7.807451 | 5.796387 | 9.818515 |
| 302  | 7.808067 | 5.797002 | 9.819131 |
| 303  | 7.808121 | 5.797057 | 9.819186 |
| 304  | 7.808126 | 5.797062 | 9.819191 |

---

### GARCH(1,1) Volatility Forecast
Variance fit MSE: **1.7498**

| Step Ahead |   Forecasted Volatility (σ)  |
|------|----------|
| 1000  | 0.772446 |
| 1001  | 0.755948 |
| 1002  | 0.743538 |
| 1003  | 0.734244 |
| 1004  | 0.727310 |



Name: volatility, dtype: float64

---

### Walk-Forward CV Results

```python
{
  'fold_metrics': [
    0.4567, 1.6636, 0.9189, 0.3341, 0.9442,
    0.1933, 1.4054, 0.8815, 0.7015, 1.6461,
    0.6614, 0.6181, 1.2589, 0.3851, 0.3867,
    0.3101, 0.3905, 0.5847, 0.7988, 1.1348,
    1.3643, 0.5668, 0.1015, 0.1448, 0.7037,
    0.1461, 0.5902, 0.9970, 1.1598, 0.2257,
    0.8123, 0.4274, 0.5938, 0.1669, 0.4599,
    0.4813, 1.8446, 1.6745, 0.5643, 0.7349
  ],
  'mean': 0.7359,
  'std': 0.4630,
  'n_folds': 40,
  'params': {
    'p': 1, 'd': 1, 'q': 1,
    'enforce_stationarity': True,
    'enforce_invertibility': True,
    'trend': None,
    'optimizer': 'L-BFGS-B',
    'maxiter': 200,
    'tol': 1e-06,
    'random_state': 42
  },
  'elapsed_sec': 1.1375
}
```

---

## Skills Demonstrated
- Time-series econometrics (ARIMA, GARCH).
- Forecast evaluation with walk-forward CV.
- Likelihood-based estimation with SciPy optimizers.
- Python software engineering with sklearn-style API design.
- Numerical linear algebra and recursion for innovations & volatility.

---

## How to Run

Install dependencies:
```bash
pip install numpy pandas scipy
