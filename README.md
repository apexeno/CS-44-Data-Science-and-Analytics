# Stock Analysis & Prediction

End-to-end exploratory data analysis, statistical testing, feature engineering,
machine-learning prediction, and sentiment analysis on five tickers using two
years of daily market data.

## Tickers

| Ticker | Asset | Market |
|---|---|---|
| ADANIPOWER.NS | Adani Power Ltd | NSE (India) |
| TCS.NS | Tata Consultancy Services | NSE (India) |
| WIPRO.NS | Wipro Ltd | NSE (India) |
| GOLDBEES.NS | Nippon India ETF Gold BeES | NSE (India) |
| RELIANCE.NS | Reliance Industries | NSE (India) |

## What the project does

1. **Data collection** — pulls ~499 trading days of OHLCV data per ticker via
   `yfinance`.
2. **Cleaning** — null checks, forward-fill, dtype verification.
3. **EDA**
   - Normalized price chart (base = 100) to compare all five NSE-listed assets.
   - Daily return distributions with summary statistics.
   - Correlation heatmap — GOLDBEES' average correlation with the four equities
     is **0.037** (near zero, TCS is slightly negative at −0.004), confirming
     gold's diversification role.
   - 7-day / 30-day SMA overlay on ADANIPOWER.NS.
4. **Statistics (Week 5)** — annualized volatility, Sharpe ratios, rolling
   30-day risk metrics, and a two-sample t-test.
5. **Feature engineering** — `pct_change`, `SMA_7`, `SMA_30`, `SMA_ratio`,
   `RSI_14`, `rolling_std_20`, and three lag-return features.
6. **Regression** — predict next-day close price for TCS.NS.
7. **Classification** — predict next-day price direction (up / down) for
   ADANIPOWER.NS.
8. **Web scraping + Sentiment (Week 8)** — scrape live headlines, score with
   TextBlob, and test whether sentiment improves classification.

## Models used

| Task | Model | Validation |
|---|---|---|
| Regression (TCS.NS next-day close) | `LinearRegression` | `TimeSeriesSplit` (5 folds, shuffle=False) |
| Classification (ADANIPOWER.NS direction) | `RandomForestClassifier` (200 trees, max_depth=5) | `TimeSeriesSplit` (5 folds, shuffle=False) |

## Actual results (from the run)

### Week 5 — Statistical Analysis

#### Annualized Volatility

| Ticker | Annualized Volatility |
|---|---|
| ADANIPOWER.NS | 43.94% |
| TCS.NS | 23.63% |
| WIPRO.NS | 26.37% |
| GOLDBEES.NS | 25.11% |
| RELIANCE.NS | 20.63% |

ADANIPOWER.NS is by far the most volatile at 43.94%, more than double
RELIANCE.NS (20.63%). This is consistent with Adani Power being a mid-cap
power utility with high speculative interest.

#### Sharpe Ratios (Rf = 6.0% — Indian 10yr G-Sec)

| Ticker | Sharpe Ratio |
|---|---|
| ADANIPOWER.NS | +0.5843 |
| TCS.NS | −1.3952 |
| WIPRO.NS | −0.6825 |
| GOLDBEES.NS | +1.1996 |
| RELIANCE.NS | −0.5315 |

Only GOLDBEES.NS (Sharpe = 1.20) and ADANIPOWER.NS (Sharpe = 0.58) delivered
positive risk-adjusted excess returns over the 6% risk-free benchmark. TCS.NS
was the worst risk-adjusted performer at −1.40, reflecting its steep ~44% price
decline over the period.

#### Two-Sample T-Test (ADANIPOWER.NS vs RELIANCE.NS)

| Metric | Value |
|---|---|
| ADANIPOWER.NS mean daily return | +0.1257% |
| RELIANCE.NS mean daily return | −0.0197% |
| T-statistic | 1.0621 |
| P-value | 0.2885 |

At α = 0.05, **the difference is NOT statistically significant** (p = 0.2885).
Despite ADANIPOWER averaging a higher daily return, the wide variance in both
stocks means we cannot reject the null hypothesis that they have the same mean
return.

### Regression — TCS.NS next-day close

| Metric | Value |
|---|---|
| Mean R² | 0.888594 |
| Mean RMSE | ₹52.7593 |
| Residual mean | −4.8506 |
| Residual std | 59.7112 |

### Classification — ADANIPOWER.NS direction

| Metric | Value |
|---|---|
| Mean Accuracy | 56.41% |
| Majority-class baseline | 56.38% |
| Best fold accuracy | 61.54% (Fold 4) |

**Confusion matrix (last fold):**

|  | Predicted Down | Predicted Up |
|---|---|---|
| **Actual Down** | 39 | 1 |
| **Actual Up** | 36 | 2 |

**Top feature importances:** `pct_change` (0.18), `RSI_14` (0.13),
`rolling_std_20` (0.11), `SMA_30` (0.11).

### Week 8 — Web Scraping + Sentiment

#### Scraping outcome

| Detail | Value |
|---|---|
| Source used | Google News RSS |
| ADANIPOWER.NS headlines retrieved | 20 |
| TCS.NS headlines retrieved | 20 |
| Scraping status | **Success** |

Headlines were scraped from Google News RSS (`news.google.com/rss/search`)
using `requests` + `BeautifulSoup` with the `lxml` XML parser. Both tickers
returned 20 headlines each from recent Indian financial news sources (NDTV,
Economic Times, Moneycontrol, etc.).

#### Sentiment Scores (ADANIPOWER.NS, TextBlob polarity)

| Metric | Value |
|---|---|
| Average polarity | −0.0005 |
| Min polarity | −0.2000 |
| Max polarity | +0.3500 |
| Std polarity | 0.0999 |
| Positive headlines | 2 |
| Negative headlines | 5 |
| Neutral headlines | 13 |

The average sentiment is essentially neutral (−0.0005). Most financial
headlines are factual/neutral in tone, which is reflected in the scores.

#### Before/After Accuracy (with sentiment feature)

| Model | Mean Accuracy |
|---|---|
| Without sentiment | 56.41% |
| With sentiment | 56.92% |
| Difference | +0.51% |

The +0.51% improvement is noise, not signal. The sentiment feature is a single
constant value applied uniformly across all historical rows (since we only have
today's headlines, not historical daily sentiment). A constant column cannot
provide meaningful time-series information to a tree-based model. To make
sentiment genuinely useful, you would need historical daily sentiment scores
aligned to each trading day.

### EDA highlights

- GOLDBEES.NS had the best 2-year normalized return at ~190 (+90%), followed by
  ADANIPOWER.NS at ~158 (+58%).
- TCS.NS was the worst performer, ending at ~56 (−44%).
- ADANIPOWER.NS is the most volatile (daily std ~2.77%), almost double the next
  ticker; RELIANCE.NS is the least volatile (~1.30%).

## What didn't work well

- **Regression R² is misleading.** The 0.889 R² is mostly autocorrelation — stock
  prices are serially correlated, so the model learns "tomorrow ≈ today."  The
  RMSE of ₹52.76 on a ₹4,000+ stock (~1.3% error) is too noisy to trade on.
- **Classification barely matches the baseline.** 56.41% vs 56.38% — the model
  is virtually indistinguishable from always predicting the majority class ("Down").
  Price-only features lack the signal to reliably call daily direction.
- **Sentiment didn't help.** Scraping succeeded (20 headlines from Google News
  RSS), but the resulting constant polarity score (−0.0005) added no
  time-varying signal. Historical daily sentiment would be needed for a
  meaningful test.

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
