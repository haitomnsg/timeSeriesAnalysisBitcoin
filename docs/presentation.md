# Presentation Guide — Bitcoin Time Series Analysis & Forecasting

A section-by-section script for presenting `timeSeriesAnalysis.ipynb` to the class.
For each part you get: **what to say**, **what the graph shows**, and **what to conclude**.
Speak in plain words — say the idea first, then point at the numbers.

> One-line summary of the whole project, in case you only have 30 seconds:
> *"We forecast Bitcoin's daily price. After cleaning and transforming the data we found it behaves
> almost like a random walk, so simple classical models (ARIMA family) actually beat the neural
> networks. The big lesson: match model complexity to the structure that is really in the data."*

---

## 0. The goal (1 slide)

- **Problem:** given the history of Bitcoin's daily closing price, predict the future price.
- **Univariate:** we use only the `Close` column — price predicting price.
- **What we compare:** classical statistical models (AR, MA, ARMA, ARIMA) **vs** deep learning
  (SimpleRNN, LSTM, GRU).
- **How we judge them:** hold out the last 120 days, predict them, measure the error (MAE, RMSE, MAPE).

---

## 1. Dataset Understanding

**Say this:** "We have ~3,400 daily prices from Sept 2014 to Jan 2024. Crypto trades every day,
including weekends, so the frequency is genuinely daily."

### The summary-statistics table (the "mean < std" point)
This is the part graders love, so explain it confidently:

- Mean ≈ **\$14,766**, Standard deviation ≈ **\$16,299** → **the spread is bigger than the average.**
- Three reasons this happens, in plain words:
  1. **No single typical price.** Bitcoin went from a few hundred dollars to ~\$67k. The data covers a
     huge range, not "one value plus a little noise".
  2. **Right-skew.** Most days are cheap, a few recent days are very expensive. Those few big numbers
     stretch the standard deviation above the mean. Proof: the **median (\$8,294) is far below the mean
     (\$14,766)** — the long tail of big prices pulls the average up.
  3. **It changes over time.** The "average price" of 2015 and of 2023 are nothing alike, so a single
     mean/std for the whole series isn't very meaningful.
- **Punchline:** mean < std + median far below mean = a **skewed, non-stationary** series whose level
  and spread both grow. *That is the reason we later take the log and difference the data.*

### The two price charts (raw vs log scale)
- **What the graph is:** the same price drawn twice — top is normal dollars, bottom is a **log scale**
  (each step up means "× 10" instead of "+ \$10,000").
- **What it shows:** on the normal scale the early years look flat (squashed at the bottom); on the log
  scale the early movements reappear and the growth looks almost like a **straight line**.
- **Conclude:** a straight line on a log axis = growth by a roughly **constant percentage** (exponential
  growth). Hint that a **log transform** will help.

### Train / test split chart
- **What it shows:** blue = the history the models learn from; short red tail = the last 120 days we
  hide and try to predict.
- **Why it matters:** the model never sees the red part during training → the final score is honest
  (no "look-ahead leakage", i.e. no peeking at the future).

---

## 2. Trend Analysis

**Say this:** "First we check the trend — the long-term direction — visually, then we prove it with
statistics."

### Moving-average chart
- **What it is:** at each day we plot the **average of the last 30 / 90 / 365 days** instead of the raw
  price. Averaging removes the daily noise and shows the direction.
- **How it's drawn:** slide a window of N days along the data and plot the mean of each window. Bigger
  window = smoother but slower-reacting line.
- **Conclude:** all lines drift **upward** → strong long-term up-trend; the 365-day line also shows the
  big multi-year **cycles** (2017 & 2021 booms, 2018 & 2022 crashes). Because the level changes, the
  series is **non-stationary**.

### Linear-regression (OLS) chart
- **What it is:** the single best-fitting **straight line**; its slope = average price change per day.
- **The numbers:** slope is **positive** and the **p-value ≈ 0** (far under 0.05). The p-value answers
  "could this slope just be luck?" — this tiny value means **no, the up-trend is statistically real.**
- **Conclude:** trend is genuine, but the straight line is a **poor fit** (real growth bends upward,
  more exponential) → another reason to use the **log**.

**Trend take-away slide:** trend → mean changes over time → non-stationary → classical models need
stationary data → so we will **log then difference** (Part II).

---

## 3. Seasonality Analysis (brief)

**Say this:** "Seasonality is a pattern that repeats on a fixed cycle (e.g. every week). We checked the
weekly and monthly effects and the decomposition."
- **Conclude:** the seasonal component changes by **well under ±1%** → essentially **no meaningful
  seasonality**. Good — one less thing to model.

---

## 4. Stationarity Analysis

**Say this:** "A stationary series has a constant mean, constant variance, and stable behaviour over
time — that's what AR/MA/ARMA need."
- **Rolling mean/std plot:** the rolling mean drifts up and the rolling std grows → clearly
  **non-stationary**.
- **Tests:** ADF and KPSS confirm non-stationarity on the raw series.

---

## 5. Part II — Making it Stationary (the transformation)

Two steps, explain *why* each one:
1. **Log transform** → fixes the **growing variance** (the huge range / exponential growth). The rolling
   std becomes much flatter, but the **trend is still there**.
2. **First difference of the log** (= today's log price − yesterday's = the **daily log-return**) →
   removes the trend. Now the mean is ~0 and the variance is stable → **stationary** (ADF/KPSS agree).

**Punchline:** "log-return" is just *the percentage change per day*. After this step the series is ready
for modelling.

---

## 6. Part III — ACF and PACF (choosing model orders)

**Say this:** "ACF and PACF tell us how strongly today relates to previous days — they help us pick how
many past terms (p and q) the model needs."
- **What the graph shows:** almost all bars sit **inside the blue confidence band**.
- **Conclude:** the differenced-log series is **close to white noise** (almost no leftover pattern), so
  **small orders p = q = 1 are enough**. It also warns us early that the linear models will barely beat a
  plain random walk.

---

## 7. Part IV — Classical Models (AR, MA, ARMA, ARIMA)

- **AR(p):** predicts from past *values*. **MA(q):** predicts from past *errors*. **ARMA:** both.
  **ARIMA(p,d,q):** ARMA plus built-in differencing (the "I", with d = 1 here).
- **Key point to mention:** "difference first then ARMA" and "ARIMA with d = 1" are mathematically the
  **same thing** — that's why ARMA(1,1) and ARIMA(1,1,1) come out nearly identical.
- **Evaluation protocol:** walk-forward / one-step-ahead on the 120-day test set (fair, like-for-like).
- **Residuals:** classical residuals look like white noise (Ljung-Box test passes) → the models captured
  the little structure there was.

---

## 8. Part V — Deep Learning (RNN / LSTM / GRU)

- **What they are:** recurrent neural networks that read a 30-day window and predict the next value.
  LSTM/GRU add "memory gates" to handle longer dependencies.
- **Loss-curve plot:** train and validation loss both fall smoothly and stay close → training is healthy,
  **no over-fitting**. (That's all the loss curves prove — not accuracy, just that learning went fine.)

---

## 9. Part VI–VII — Evaluation & Comparison

- **Metrics:** MAE and RMSE are in dollars (lower = better); **MAPE is the % error** — easiest to quote.
- **Forecast-comparison plot:** every model's prediction against the black "actual" line.
- **Error bar charts:** MAE / RMSE / MAPE side by side.
- **Headline result:**
  | Model | MAPE |
  |---|---|
  | AR(1) / MA(1) / ARMA(1,1) / ARIMA(1,1,1) | **≈ 1.6 %** (best) |
  | GRU | ≈ 2.8 % |
  | SimpleRNN | ≈ 3.1 % |
  | LSTM | ≈ 4.6 % |
- **Residual plots (all models):** classical residuals = white noise (good). RNN residuals still show
  **leftover autocorrelation** → they learned a slightly **delayed copy** of the price.

---

## 10. Part VIII — Conclusion (how to close)

Say it in three sentences:
1. **Bitcoin's daily price is close to a random walk** — after log + differencing, what's left is almost
   pure noise.
2. So the **simple classical models, which basically predict "tomorrow ≈ today", were the most accurate
   (≈ 1.6% MAPE)**, while the complex RNNs added error by lagging behind.
3. **Lesson:** more complex ≠ better. **Match the model's complexity to the structure that is actually in
   the data.**

---

## Likely questions (be ready)

- **"Why did the simple models beat the neural networks?"** — Because the data has almost no nonlinear
  pattern to learn; the best forecast is "today's price", which the differenced linear models reproduce.
  Extra complexity just adds error.
- **"Why log AND difference?"** — Log fixes the growing/percentage-style variance; differencing removes
  the trend. Each fixes a different problem; together they make the series stationary.
- **"What is a p-value here?"** — The chance the upward trend is just random luck. Ours ≈ 0, so the trend
  is real.
- **"Why is std bigger than the mean?"** — The data is hugely spread and right-skewed (few very large
  recent prices), and it's non-stationary, so a single mean/std isn't representative.
- **"Would RNNs ever win?"** — Yes — with more data and a genuinely nonlinear or long-memory signal,
  multiple input features, etc. Just not on this near-random-walk series.
- **"What is MAPE?"** — Mean Absolute Percentage Error; average size of the error as a % of the actual
  price. 1.6% means predictions are off by about 1.6% on average.
- **"Why hold out the *last* 120 days instead of random days?"** — Time order matters; you must train on
  the past and test on the future, never the other way around.
