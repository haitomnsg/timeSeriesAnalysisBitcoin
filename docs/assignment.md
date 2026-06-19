Assignment: Time Series Analysis and Forecasting using Classical and Deep Learning Models

Objective
The objective of this assignment is to apply the theoretical concepts covered in the lecture notes to a real-world time series dataset and perform a comprehensive forecasting study using both classical statistical models and Recurrent Neural Network (RNN)-based models.

Dataset Selection
Select a real-world univariate time series dataset. Suitable datasets include:
Stock market prices
Cryptocurrency prices
Weather measurements
Energy consumption
Traffic flow data
Retail sales data
Air quality measurements
Sensor data
Students may use publicly available datasets from Kaggle, UCI Repository, Yahoo Finance, or other reliable sources.
Part I: Exploratory Time Series Analysis
1. Dataset Understanding
Perform the following:
Load and visualize the raw time series.
Describe the dataset and forecasting objective.
Identify the observation frequency (daily, weekly, monthly, etc.).
Split the dataset into training and testing sets.
2. Trend Analysis
Perform:
Visual trend inspection.
Moving Average smoothing.
Trend estimation using Linear Regression.
Trend significance analysis.
Provide plots and interpretation.3. Seasonality Analysis
Perform:
Seasonal visualization.
Seasonal decomposition.
Seasonal pattern identification.
Explain whether the series exhibits seasonality and determine its seasonal period.4. Stationarity Analysis
Perform:
Rolling Mean Analysis.
Rolling Standard Deviation Analysis.
Augmented Dickey-Fuller (ADF) Test.
KPSS Test.
Interpret all statistical test results.
Part II: Time Series Transformation
If required, apply:
Log Transformation
Trend Removal
First Differencing
Seasonal Differencing
After each transformation:
Re-plot the series.
Re-run ADF and KPSS tests.
Justify whether stationarity has been achieved.
Part III: ACF and PACF Analysis
Generate and analyze:Autocorrelation Function (ACF)
Plot the ACF of the stationary series.
Interpret significant lags.
Explain how ACF helps identify MA(q) behavior.
Partial Autocorrelation Function (PACF)
Plot the PACF of the stationary series.
Interpret significant lags.
Explain how PACF helps identify AR(p) behavior.
Model Order Selection
Using ACF and PACF analysis, determine suitable values for:
p (AR order)
q (MA order)
Provide justification for all selected values.


Part IV: Classical Time Series Forecasting Models
Implement the following forecasting models:

1. Autoregressive Model (AR)
Perform:
Order selection using PACF.
Model fitting.
Forecast generation.
Residual analysis.
2. Moving Average Model (MA)
Perform:
Order selection using ACF.
Model fitting.
Forecast generation.
Residual analysis.
3. Autoregressive Moving Average Model (ARMA)
Perform:
Selection of p and q.
Model fitting.
Forecast generation.
Residual analysis.
4. Autoregressive Integrated Moving Average Model (ARIMA)
Perform:
Selection of p, d, q.
Model fitting.
Forecast generation.
Residual analysis.
Explain how differencing contributes to ARIMA.
Part V: Deep Learning-Based Forecasting

Implement Recurrent Neural Network model.
Simple RNN
LSTM
GRU
For each model:
Explain the architecture.
Create input sequences.
Train the network.
Generate forecasts on the test set.
Plot training and validation loss curves.
Part VI: Model Evaluation
Evaluate all implemented models using:
MAE (Mean Absolute Error)
RMSE (Root Mean Squared Error)
MAPE (Mean Absolute Percentage Error)
Create a comparison table containing the performance metrics of:
AR
MA
ARMA
ARIMA
RNN
LSTM (if implemented)
GRU (if implemented)
Part VII: Comparative Visualization


Generate the following visualizations:Forecast Comparison Plot
Display on a single graph:
Actual values
AR predictions
MA predictions
ARMA predictions
ARIMA predictions
RNN predictions
Error Comparison Chart
Create bar charts comparing:
MAE
RMSE
MAPE
across all models.Residual Analysis
For each model:
Plot residuals.
Plot residual ACF.
Discuss whether residuals behave like white noise.
Part VIII: Discussion and Conclusions
Discuss the following:Which model achieved the best forecasting performance?Did ACF and PACF correctly guide model order selection?How did AR, MA, ARMA, and ARIMA compare against RNN-based forecasting?What are the advantages and limitations of classical statistical models?What are the advantages and limitations of deep learning models?Under what circumstances would you prefer ARIMA over RNN, and vice versa?

Deliverables
Present in class:Jupyter Notebook (.ipynb)Dataset usedAll generated plots and visualizations
The notebook must contain clear explanations, code comments, interpretations, and conclusions for every stage of the analysis.

The assignment will be graded based on your presentation and carries marks toward your final evaluation. There is no need to prepare separate slides; you may demonstrate your work directly using the same Jupyter Notebook (.ipynb) file during your presentation.