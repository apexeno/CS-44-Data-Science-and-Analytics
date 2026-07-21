# Stock Analysis & Prediction

End-to-end exploratory data analysis, feature engineering, and machine-learning
prediction on five tickers using two years of daily market data.

## Tickers

| Ticker | Asset | Market |
|---|---|---|
| ADANIPOWER.NS | Adani Power Ltd | NSE (India) |
| TCS.NS | Tata Consultancy Services | NSE (India) |
| WIPRO.NS | Wipro Ltd | NSE (India) |
| GOLDBEES.NS | Nippon India ETF Gold BeES | NSE (India) |
| RELIANCE.NS | Reliance Industries | NSE (India) |

## What the project does

1. **Data collection** : pulls ~499 trading days of OHLCV data per ticker via
   `yfinance`.
2. **Cleaning** : null checks, forward-fill, dtype verification.
3. **EDA**
   - Normalized price chart (base = 100) to compare all five NSE-listed assets.
   - Daily return distributions with summary statistics.
   - Correlation heatmap — GOLDBEES' average correlation with the four equities
     is **0.037** (near zero, TCS is slightly negative at −0.004), confirming
     gold's diversification role.
   - 7-day / 30-day SMA overlay on ADANIPOWER.NS.
4. **Feature engineering** : `pct_change`, `SMA_7`, `SMA_30`, `SMA_ratio`,
   `RSI_14`, `rolling_std_20`, and three lag-return features.
5. **Regression** — predict next-day close price for TCS.NS.
6. **Classification** : predict next-day price direction (up / down) for
   ADANIPOWER.NS.

## Models used

| Task | Model | Validation |
|---|---|---|
| Regression (TCS.NS next-day close) | `LinearRegression` | `TimeSeriesSplit` (5 folds, shuffle=False) |
| Classification (ADANIPOWER.NS direction) | `RandomForestClassifier` (200 trees, max_depth=5) | `TimeSeriesSplit` (5 folds, shuffle=False) |

## Actual results (from the run)

### Regression : TCS.NS next-day close

| Metric | Value |
|---|---|
| Mean R² | 0.877739 |
| Mean RMSE | ₹54.9788 |
| Residual mean | −2.3733 |
| Residual std | 60.2810 |

### Classification — ADANIPOWER.NS direction

| Metric | Value |
|---|---|
| Mean Accuracy | 55.90% |
| Majority-class baseline | 56.29% |
| Best fold accuracy | 58.97% (Fold 4) |

**Confusion matrix (last fold):**

|  | Predicted Down | Predicted Up |
|---|---|---|
| **Actual Down** | 37 | 2 |
| **Actual Up** | 35 | 4 |

**Top feature importances:** `pct_change` (0.18), `RSI_14` (0.13),
`rolling_std_20` (0.11), `SMA_30` (0.11).

### EDA highlights

- GOLDBEES.NS had the best 2-year normalized return at ~190 (+90%), followed by
  ADANIPOWER.NS at ~158 (+58%).
- TCS.NS was the worst performer, ending at ~56 (−44%).
- ADANIPOWER.NS is the most volatile (daily std ~2.77%), almost double the next
  ticker; RELIANCE.NS is the least volatile (~1.30%).

## What didn't work well

- **Regression R² is misleading.** The 0.878 R² is mostly autocorrelation, stock
  prices are serially correlated, so the model learns "tomorrow ≈ today."  The
  RMSE of ₹55 on a ₹2,200 stock (~2.5% error) is too noisy to trade on.
- **Classification doesn't beat the baseline.** 55.9% vs 56.3% : the model
  actually underperforms always predicting "Down." It's heavily biased toward
  that class (Up recall is only 10%). Price-only features lack the signal to
  reliably call daily direction.
- **No external features.** Limiting ourselves to price-derived indicators
  (SMA, RSI, lagged returns) leaves out volume dynamics, macro data, and
  sentiment, all of which could help but are out of scope for this submission.

## How to run

```bash
pip install -r requirements.txt
jupyter notebook stock_analysis.ipynb
```

Run all cells top-to-bottom. The notebook pulls live data from Yahoo Finance, so
results will update with each run.

## Project structure

```
.
├── stock_analysis.ipynb   # Main notebook (all analysis + models)
├── requirements.txt       # Python dependencies
└── README.md              # This file
```
