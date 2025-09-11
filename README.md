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
- results.ipynb -> End to End workflow/Final Results

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

## Notable Results
- Achieved two strategies, one with 5 day window and one with 20 window, where both of them **provide better total return than the total return of the market** (more than twice).

<img width="1489" height="990" alt="image" src="https://github.com/user-attachments/assets/c242fa6a-27c2-4f1f-9675-2f11431edf51" />

<img width="1489" height="990" alt="image" src="https://github.com/user-attachments/assets/70561928-cd26-4760-be1b-1bba86a0dfd6" />

- Created Heatmaps of all of the three strategies,in where it's visualizing the performance of each strategy in every year and every month.
<img width="1129" height="790" alt="image" src="https://github.com/user-attachments/assets/93bb8681-e2d6-461e-861b-93d046fcac66" />

- Designed different visualization of metrics to evaluate the performance of each strategy.
<img width="1489" height="989" alt="image" src="https://github.com/user-attachments/assets/aa715df5-e612-4eb6-a96b-1b8cde1007b0" />




## Author
Developed by **Panagiotis Valsamis**, M.Sc. in Data Science candidate and aspiring Data Scientist.

## Contributor
**Panagiotis Akidis**, Master in Finance & Economics providing valuable theoretical insights in finance.


  


