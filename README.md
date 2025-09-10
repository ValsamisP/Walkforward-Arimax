# Walkforward-ARIMAX Financial Time Series Forecasting

 ## Overview
This project implements a **walk-forward validation framework** for financial time series forecasting using the ARIMAX(AutoRegressive Integrated Moving Average with Exogenous Variables) model.

The aim is to evaluate the performance of ARIMAX under different configurations, feature sets, scalers and to compare the resulting trading strategy against Buy-And-Hold and market benchmarks(NASDAQ QQQ).

The project is designed with reusable modules for:
- **Data Downloading**
- **Feature Engineering**
- **Model training and walk forward validation**
- **Performance evaluation and visualization**

## Project Structure

- data_prep.py -> Downloading stock/market data via yfinance package
- feature_engineering.py -> Constructing different features such as volatility, lags etc
- model_arimax.py -> Model training
- evaluation_visualization.py -> Performance Evaluation
- main.ipynb -> End to End workflow

  ## Key Features

  - This project simulates **out of sample trading** with rolling windows of **1,5 and 20 days**.
  - Includes returns,realized volatility,volume z-score,market returns,MACD indicators and lag features.
  - **Automated Grid Search over ARIMA** with scaling options
  - The performance is evaluated based on different metrics (RMSE,MAE,hit-rate,Sharpe Ratio,Drawdown)
  - There are plenty visualizations such as:
     - Cumulative Returns of the Strategy vs Buy & Hold vs Market
     - Rolling Sharpe Ratio, Hit-Rate ,Drawdown for every rolling window strategy
     - Monthly heatmaps of profitability, to better understand in which months the strategy is not performing very good
     - Detailed per window strategy charts
   

## Focus Areas
- **Time Series Forecasting**
- **Machine Learning for Finance**
- **Risk-adjusted Performance Metrics**

Developed by **Panagiotis Valsamis** , M.Sc in Data Science canditate and aspiring Data Scientist.

  


