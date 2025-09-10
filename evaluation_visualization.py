from data_prep import load_prices
from feature_engineering import build_features, make_nextday_target, add_lags
from model_arimax import walk_forward_arimax, analyze_residuals 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# Configuration Parameters
# These parameters control the whole process
ASSET   = "NVDA"
MARKET  = "QQQ" # For NASDAQ
START   = "2020-01-01"
END     = "2025-12-31"

TRAIN_LEN = 504                 # ~ 2 years (trading days)
# ARIMAX model orders to test - (p,d,q) where:
# p=autoregressive terms, d=differencing order, q=moving average terms
ORDER_GRID = ((1,0,0),(0,0,1),(1,0,1),(2,0,1),(2,0,2))
# Different sets of exogenous variables to use (X in ARIMAX)
EXOG_SETS     = [
    ["mkt_ret_l1", "macd_hist_l1"],
    ["mkt_ret_l1","macd_hist_l1","macd_hist_l2","vol_z_l1"]
]
TARGET    = "y_next"
SCALERS = ("zscore", "robust") #Feature scaling methods to test

#Different configurations to test, these are controlling how the model is walking through time
CFG1 = {"name": "1-day", "test_len": 1, "step": 1}
CFG5  = {"name": "5-day",  "test_len": 5,  "step": 5}
CFG20 = {"name": "20-day", "test_len": 20, "step": 20}

SAVE_CSV = False   # whether to save or not the results



# Visualization Functions

def plot_detailed_strategy_window(per_point: pd.DataFrame, market_data: pd.DataFrame = None, title: str = "Daily returns inside the window", 
                                window_start: str = None, window_end: str = None, max_days: int = 30):
    """
    Creates a detailed strategy visualization showing asset returns, market returns, and strategy performance
    over a specific time window.

    How it works:
    1. Takes the strategy predictions and actual returns
    2. Calculatess strategy returns as: signal*actual_return
        - If i predict positive return (+1 signal) and actual is positive so i have profit
        - If i predict negative return (-1 signal) and actual is negative so i have profit (short position)
    3. Plots all three series for comparison

    
    Parameters:
    per_point : pd.DataFrame
        Results from walk_forward_arimax with columns ['y_true','y_pred'] and date index
        - y_true : actual next day returns
        - y_pred : predicted next day returns
    market_data : pd.DataFrame, optional
        Market benchmark data with 'ret' column for comparison
    title : str 
        Plot title
    window start/end : str
        Date Range 
    max_days : int 
        Maximum number of days to show (To prevent overcrowding plots)
    
    The idea is the strategy bars to show how much we would have made or lost if we followed the model's prediction. Blue bars = profitable days , light blues = loss days
    """
    df = per_point[["y_true","y_pred"]].dropna().copy()
    
    # Filter date range if specified
    if window_start:
        df = df[df.index >= window_start]
    if window_end:
        df = df[df.index <= window_end]
        
    # Limit to max_days for readability (as referred before to prevent overcrowding plots)
    if len(df) > max_days:
        df = df.tail(max_days) # Show more recent days
    
    ### IDEA
    # Convert predictions to trading signals: +1 = buy/long , -1 = sell/short, 0=hold
    df["signal"] = np.sign(df["y_pred"]).replace(0, 0)

    # Calculate Strategy returns signal * actual_return
    df["strategy_ret"] = df["signal"] * df["y_true"]
    
    fig, ax = plt.subplots(figsize=(15, 6))
    
    # Plot asset returns as line with circle markers
    ax.plot(df.index, df["y_true"], 'o-', color='steelblue', linewidth=2, markersize=6, label='Asset ret', alpha=0.8)
    
    # Plot market returns if available
    if market_data is not None:
        market_aligned = market_data.reindex(df.index) #align dates
        if 'ret' in market_aligned.columns:
            ax.plot(df.index, market_aligned['ret'], 's-', color='orange', linewidth=2, markersize=5, label='Market ret', alpha=0.8)
        else:
            # If no 'ret' column, use the first numeric column
            numeric_cols = market_aligned.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                ax.plot(df.index, market_aligned[numeric_cols[0]], 's-', color='orange', linewidth=2, markersize=5, label='Market ret', alpha=0.8)
    
    # Plot strategy returns as bars
    # Color dark blue for gains light blue for losses
    colors = ['darkblue' if x >= 0 else 'lightblue' for x in df["strategy_ret"]]
    ax.bar(df.index, df["strategy_ret"], alpha=0.7, color=colors, label='Strategy', width=0.8)
    
    # Formatting and styling
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel('Daily Returns', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Format x-axis dates better
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.show()


def plot_strategy_heatmap(per_point: pd.DataFrame, title: str = "Strategy Performance Heatmap"):
    """
    Creates a heatmap showing strategy performance by year and month
    This helps identify seasonal patterns and periods of strong or weak performance of the model

    The heatmap uses a color scale where:
    - Green = profitable months
    - Red = loss months
    - White/neutral = break-even

    This heatmap help us to reveal the specific months in where the strategy failed, and to understand some seasonal patterns

    Parameters:

    per_point : pd.DataFrame
        Strategy results with y_true, y_pred columns
    title :  str
        Heatmap Title
    """


    df = per_point[["y_true","y_pred"]].dropna().copy()
    df["signal"] = np.sign(df["y_pred"]).replace(0, 0)
    df["strategy_ret"] = df["signal"] * df["y_true"]
    
    # Group by month and year
    df['year'] = df.index.year
    df['month'] = df.index.month
    
    # Calculate monthly returns (sum of daily returns within each month)
    monthly_rets = df.groupby(['year', 'month'])['strategy_ret'].sum().unstack(fill_value=0)
    
    if monthly_rets.empty:
        print("Not enough data for heatmap")
        return
    
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(monthly_rets.values, cmap='RdYlGn', aspect='auto', vmin=-0.15, vmax=0.15)
    
    # ticks and labels
    ax.set_xticks(range(len(monthly_rets.columns)))
    ax.set_xticklabels([f'Month {i}' for i in monthly_rets.columns])
    ax.set_yticks(range(len(monthly_rets.index)))
    ax.set_yticklabels(monthly_rets.index)
    
    # colorbar
    plt.colorbar(im, ax=ax, label='Monthly Strategy Returns')
    
    # text annotations (adding numerical values in each cell for precise reading)
    for i in range(len(monthly_rets.index)):
        for j in range(len(monthly_rets.columns)):
            text = ax.text(j, i, f'{monthly_rets.iloc[i, j]:.3f}',
                          ha="center", va="center", color="black", fontsize=8)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_rolling_performance_metrics(per_point: pd.DataFrame, window: int = 60, title: str = "Rolling Performance Metrics"):
    """
    Creates a 2x2 subplot showing rolling performance metrics over time.
    This provides insights into strategy stability and evolving performance.

    The four metrics shown are:
    1. Rolling Sharpe Ratio - Risk-adjusted returns (higher is better)
    2. Rolling Hit Rate - Percentage of correct predictions (Trying to achieve > 50%)
    3. Cumulate returns - Total strategy performance over time
    4. Drawdown - Peak to-trough losses (shows worst losing streaks)

    Key information:
    - Sharpe Ratio > 1 is generally considered good for trading
    - Hit rate > 50% means model predictions are better than random
    - Drawdown shows maximum pain points useful to measure and manage risk
    - Cumulative returns show overall wealth creation

    Parameters:

    per_point : pd.DataFrame
        Strategy results
    window : int
        Rolling window size for metrics calculations
    title : str
        Overall plot title
       
    """


    df = per_point[["y_true","y_pred"]].dropna().copy()
    df["signal"] = np.sign(df["y_pred"]).replace(0, 0)
    df["strategy_ret"] = df["signal"] * df["y_true"]
    
    # Calculate rolling metrics

    # Sharpe ratio (mean_return)/(std_return) *sqrt(252)
    df['rolling_sharpe'] = df['strategy_ret'].rolling(window).mean() / df['strategy_ret'].rolling(window).std() * np.sqrt(252) # 252 typical number of trading days per year 

    # Percentage of signal prediction matches the actual   
    df['rolling_hit_rate'] = (df["signal"] * np.sign(df["y_true"]) > 0).rolling(window).mean()

    # Wealth accumulation (1+r1)*(1+r2)*.....*(1+rn)
    df['cumulative_ret'] = (1 + df['strategy_ret']).cumprod()

    # How much we are down from the previous peak
    df['rolling_max'] = df['cumulative_ret'].rolling(window).max()
    df['drawdown'] = (df['cumulative_ret'] - df['rolling_max']) / df['rolling_max']
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Rolling Sharpe Ratio
    ax1.plot(df.index, df['rolling_sharpe'], color='blue', alpha=0.7)
    ax1.set_title(f'Rolling Sharpe Ratio ({window}-day)', fontweight='bold')
    ax1.set_ylabel('Sharpe Ratio')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    # Rolling Hit Rate
    ax2.plot(df.index, df['rolling_hit_rate'], color='green', alpha=0.7)
    ax2.set_title(f'Rolling Hit Rate ({window}-day)', fontweight='bold')
    ax2.set_ylabel('Hit Rate')
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
    
    # Cumulative Returns
    ax3.plot(df.index, df['cumulative_ret'], color='purple', alpha=0.7)
    ax3.set_title('Cumulative Strategy Returns', fontweight='bold')
    ax3.set_ylabel('Cumulative Return')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=1, color='red', linestyle='--', alpha=0.5)
    
    # Drawdown
    ax4.fill_between(df.index, df['drawdown'], 0, color='red', alpha=0.3)
    ax4.plot(df.index, df['drawdown'], color='red', alpha=0.7)
    ax4.set_title('Strategy Drawdown', fontweight='bold')
    ax4.set_ylabel('Drawdown')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def enhanced_plot_cum_pnl(per_point: pd.DataFrame, market_ret_series: pd.Series = None, title: str = "Enhanced Strategy Performance"):
    """
    Two panel layout:
    - Top : Cummulative returns comparing Strategy vs Buy and Hold vs Market
    - Bottom : Daily returns bar chart

    Key Comparison:
    1. Strategy : Following model signals (can be long or short)
    2. Buy & Hold : Simple buying and holding the asset
    3. Market : Benchmark performance (NASDAQ)

    The performance metrics box shows:
    - Total returns over the period
    - Strategy volatility 
    - Sharpe Ratio


    """
    df = per_point[["y_true","y_pred"]].dropna().copy()
    df["signal"] = np.sign(df["y_pred"]).replace(0, 0)
    df["strategy_ret"] = df["signal"] * df["y_true"]
    
    # Calculate cumulative returns
    cum_strategy = (1 + df["strategy_ret"]).cumprod()
    cum_bh = (1 + df["y_true"]).cumprod()
    
    # If market returns provided, calculate market cumulative return
    if market_ret_series is not None:
        market_aligned = market_ret_series.reindex(df.index).fillna(0)
        cum_market = (1 + market_aligned).cumprod()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Top plot: Cumulative returns
    cum_strategy.plot(ax=ax1, label="Strategy", linewidth=2, color='blue')
    cum_bh.plot(ax=ax1, label="Buy & Hold", linewidth=2, color='orange')
    if market_ret_series is not None:
        cum_market.plot(ax=ax1, label="Market (QQQ)", linewidth=2, color='green')
    
    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.set_ylabel('Cumulative Return')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Bottom plot: Daily returns (sample to avoid overcrowding)
    sample_df = df.iloc[::max(1, len(df)//100)]  # Sample max 100 points
    sample_df["strategy_ret"].plot(ax=ax2, kind='bar', alpha=0.6, color='lightblue', width=0.8)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2.set_title('Daily Strategy Returns (sampled)', fontweight='bold')
    ax2.set_ylabel('Daily Return')
    ax2.grid(True, alpha=0.3)
    
    # Calculate and display performance metrics
    if len(cum_strategy) > 0 and len(cum_bh) > 0:
        strategy_total_ret = cum_strategy.iloc[-1] - 1
        bh_total_ret = cum_bh.iloc[-1] - 1
        strategy_vol = df["strategy_ret"].std() * np.sqrt(252)
        strategy_sharpe = df["strategy_ret"].mean() / df["strategy_ret"].std() * np.sqrt(252) if df["strategy_ret"].std() > 0 else 0
        market_line = ""
        if market_ret_series is not None:
            market_total_ret = cum_market.iloc[-1] - 1
            market_line = f"\nMarket (QQQ) Total Return: {market_total_ret:.2%}"
        
        metrics_text = f"""Strategy Total Return: {strategy_total_ret:.2%}
Buy & Hold Total Return: {bh_total_ret:.2%}
Market Return: {market_line}
Strategy Sharpe: {strategy_sharpe:.3f}"""
        
        ax1.text(0.02, 0.98, metrics_text, transform=ax1.transAxes, 
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.show()


# Helper Functions
# These functions handle the backtesting workflow and basic analysis.

def run_cfg(supervised: pd.DataFrame, cfg: dict, scaler: str = "zscore", exogs: list | None = None, name_suffix: str = ""):
    """
    Run a single forward backtesting configuration
    
    This is the main backtesting engine that:
    1. Take prepared features and targets.
    2. Runs walk-forward validation with specified parameters
    3. Returns comprehensive results.
    
    """
    exog_cols = exogs if exogs is not None else EXOG_SETS[0]

    # Walk forward Validation
    per_point, overall, per_window, diagnostics = walk_forward_arimax(
        features=supervised,
        order_grid=ORDER_GRID, # ARIMAX orders to try
        exog_cols=exog_cols, # Features to use
        train_len=TRAIN_LEN, 
        test_len=cfg["test_len"],
        step=cfg["step"],
        expanding=True,
        max_train_len=TRAIN_LEN,
        scaling_method=scaler,
        use_aicc=True,
        target_col=TARGET,
        verbose=False,
        save_diagnostics=True,
    )
    return dict(
        name=f"{cfg['name']} {name_suffix} [{scaler}]".strip(),
        per_point=per_point,
        overall=overall,
        per_window=per_window,
        diagnostics=diagnostics,
        exogs=exog_cols,
        scaler=scaler,
        cfg=cfg["name"],
    )


def print_overall_metrics(results):
    """ 
    Print a summary table of overall performance metrics.
    
    -RMSE
    -MAE 
    -HIT Rate
    -Windows
    """
    print("=== Overall metrics ===")
    for r in results:
        m = r["overall"]
        print(f"{r['name']}: RMSE={m.rmse:.6f}  MAE={m.mae:.6f}  Hit={m.hit_rate:.3f}  Windows={m.n_windows}")


def plot_hitrate_timeline(res, ax=None, marker='o'):
    """
    Plot hit rate (directional accuracy) over time for a single configuration.
    
    This shows how the model's ability to predict direction changes over time.
    Values above 0.5 (50%) indicate better than random performance.
    
    Look for:
    - Consistent performance above 50%
    - Periods of deteriorating performance  
    - Market regime changes affecting accuracy
    
    Parameters:
    -----------
    res : dict
        Single result dictionary from run_cfg()
    ax : matplotlib axis, optional
        Existing axis to plot on
    marker : str
        Matplotlib marker style
    """
    pw = res["per_window"].copy()
    s = pw.set_index("test_end")["hit_rate"].sort_index()
    if ax is None:
        fig, ax = plt.subplots(figsize=(10,3))
    s.plot(ax=ax, marker=marker, linestyle='-')
    ax.set_ylim(0,1.05)
    ax.set_title(f"Hit rate by window — {res['name']}")
    ax.set_ylabel("Hit rate")
    ax.grid(True, alpha=0.3)
    return ax


def plot_cum_pnl(per_point: pd.DataFrame, title: str):
    """
    Simple cumulative P&L plot comparing strategy vs buy-and-hold.
    
    This is a basic version of the wealth creation visualization.
    Shows two lines:
    1. Strategy: Following model signals  
    2. Buy & Hold: Simple buy and hold benchmark
    
    Parameters:
    -----------
    per_point : pd.DataFrame
        Per-point results with y_true, y_pred columns
    title : str
        Plot title
    """
    df = per_point[["y_true","y_pred"]].dropna().copy()
    df["signal"] = np.sign(df["y_pred"]).replace(0, 0)
    df["strategy_ret"] = df["signal"] * df["y_true"]
    cum = (1 + df["strategy_ret"]).cumprod()
    bh  = (1 + df["y_true"]).cumprod()
    ax = cum.plot(figsize=(10,3), label="Strategy", legend=True)
    bh.plot(ax=ax, label="Buy&Hold")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.show()
