import numpy as np
import pandas as pd 
from typing import Optional, Tuple, List, Union

def log_return(close: pd.Series)->pd.Series:
    """r_t = ln(C_t / C_{t-1})"""
    return np.log(close).diff()

def realized_vol(ret: pd.Series, window: int=30, min_periods: Optional[int] = None, annualized: bool = True, trading_days: int = 252) -> pd.Series:
    """
    Calculate rolling realized volatility of returns
    
    Parameters:
    ret: pd.Series 
    return Series
    window: int
    rolling window size
    min_periods: int, optional
        Minimum periods required for calculation
    annualized: bool
        Whether to annualize the volatility
    trading_days: int
        Number of trading days per year for annualization
     
     Returns
     pd.Series
        Rolling volatility series
    """
    if min_periods is None:
        min_periods=max(1, window // 2)
    vol = ret.rolling(window, min_periods=min_periods).std(ddof=1)
    
    if annualized:
        vol = vol * np.sqrt(trading_days)
    
    return vol


def volume_zscore(
    volume: pd.Series,
    window: int=60,
    min_periods: Optional[int] = None,
    eps: float = 1e-8
    ) -> pd.Series:
    """
    Calculate rolling z-score of volume: (V - mean) / std
    
    Parameters
    
    Volume : pd.Series
        Volume series
    window : int
        Rolling window size
    min_periods : int, optional
        Minimum periods required for calculation
    eps : float 
        Small value to avoid division by zero
    
    Returns
    pd.Series
        Volume z-score series
    """

    if min_periods is None:
        min_periods = max(1, window // 2)
    m=volume.rolling(window, min_periods=min_periods).mean()
    s=volume.rolling(window, min_periods=min_periods).std(ddof=1)

    s_safe = s.fillna(eps).replace(0,eps) #to avoid division by zero
    return (volume-m)/s_safe

def add_nasdaq_feature(
    features: pd.DataFrame,
    nasdaq_df: pd.DataFrame,
    *,
    close_col: str = "Close",
    out_col: str = "mkt_ret",
    overwrite: bool = True,
    drop_missing: bool = False
) -> pd.DataFrame:
    """
    Add a Nasdaq-based market return feature to an existing features DataFrame.

    Parameters

    features : pd.DataFrame
         Current features (indexed by dates).
    nasdaq_df : pd.DataFrame
        DataFrame with at least a 'Close' column for the Nasdaq 
    close_col : str
        Name of the close column in nasdaq_df.
    out_col : str
        Output column name for the market return feature.
    overwrite : bool
        If False and out_col already exists in `features`, raise an error.
    drop_missing : bool
        If True, drop rows where market return is missing
        If False, keep all rows (market return will be NaN for missing dates).

    Returns
    
    pd.DataFrame
        A copy of features with the updated out_col.
    """
    if close_col not in nasdaq_df.columns:
        raise ValueError(f"nasdaq_df must contain '{close_col}' column.")
    if (out_col in features.columns) and not overwrite:
        raise ValueError(f"Column '{out_col}' already exists in features; set overwrite=True to replace it.")

    # Compute market return
    mkt_ret = log_return(nasdaq_df[close_col]).rename(out_col)

    # Align to features' index (date intersection), no look-ahead
    mkt_ret_aligned = mkt_ret.reindex(features.index)

    out = features.copy()
    out[out_col] = mkt_ret_aligned

    # Only drop rows with missing market data
    if drop_missing:
        out = out.dropna(subset=[out_col])
        out = out[np.isfinite(out[out_col])]

    return out


def end_of_quarter_flag(index: pd.DatetimeIndex) -> pd.Series:
    """
    Binary flag for quarter ending months
    Returns a series named 'eoq' aligned with the given index

    Parameters 

    index : pd.DatetimeIndex
        Date index

    Returns
    
    pd.Series
        Binary series with 1 for end-of-quarter months, 0 otherwise
    
    
    """
    months=index.month
    eoq=np.isin(months, [3,6,9,12]).astype(int)
    return pd.Series(eoq, index=index, name="eoq")

def compute_macd(
        close: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
        adjust: bool=False,
) -> pd.DataFrame:
    """
    Compute MACD (Moving Average Convergence Divergence) indicator
    MACD is a momentum oscillator helping to identify trends and momentum shifts
    It is a lagging indicator meaning it confirms trends rathen than predicting them
    
    Parameters
     
    close : pd.Series
        close prices
    fast : int
        Fast EMA period
    slow : int
        Slow EMA period
    signal : int
        Signal line EMA period
    adjust : bool
        EMA adjustment parameter
         
    Returns 
    
    pd.Dataframe
        dataframe with columns: macd, macd_signal, macd_hist
    
     """
    
    if fast >= slow:
        raise ValueError("Fast period must be less than slow period")
    
    close = close.astype(float)
    ema_fast = close.ewm(span=fast, adjust=adjust).mean()
    ema_slow = close.ewm(span=slow,adjust=adjust).mean()

    macd_line = (ema_fast-ema_slow).rename("macd")
    macd_signal = macd_line.ewm(span=signal,adjust=adjust).mean().rename("macd_signal")
    macd_hist = (macd_line - macd_signal).rename("macd_hist")

    return pd.concat([macd_line, macd_signal, macd_hist],axis=1)

def add_eoq_and_macd(
    features: pd.DataFrame,
    close: pd.Series,
    *,
    add_signal: bool = False, 
    add_hist: bool = True,      # macd_hist is usually the most informative single MACD feature
    drop_missing: bool = False,
) -> pd.DataFrame:
    """
    Append 'eoq' and MACD-based features to the given features DataFrame.

    Parameters
    
    features : DataFrame
        Existing features indexed by dates.
    close : Series
        Close prices (same index as features).
    add_signal : bool
        If True, also include 'macd_signal'.
    add_hist : bool
        If True, include 'macd_hist'.
    drop_missing : bool
        If true, drop rows where new features are missing
    

    Returns
    
    DataFrame
        New DataFrame with added columns.
    """
    # Align close to features index (no look-ahead)
    close_aligned = close.reindex(features.index)

    # Compute features

    macd_df = compute_macd(close_aligned)
    cols_to_keep = ["macd"]
    if add_signal:
        cols_to_keep.append("macd_signal")
    if add_hist:
        cols_to_keep.append("macd_hist")
    macd_df = macd_df.loc[:, cols_to_keep]

    eoq = end_of_quarter_flag(features.index)

    # Joining features

    out = features.copy()
    out = out.join(eoq).join(macd_df)

    # Handling missing values 

    if drop_missing:
        out = out.dropna(subset=["eoq"] + cols_to_keep)
        for c in ["eoq"] + cols_to_keep:
            out = out[np.isfinite(out[c])]
    
    return out


def build_features(
    stock_close: pd.Series,
    stock_volume: pd.Series,
    *,
    # market options
    market_close: Optional[pd.Series] = None,   # QQQ['Close']
    market_out_col: str = "mkt_ret",
    # volatility / volume params
    vol_window: int = 20,
    volz_window: int = 60,
    # calendar/technical toggles
    include_eoq: bool = True,
    macd: str = "hist",
    #data handling options
    handle_missing : str = "conservative",
    min_valid_ratio: float = 0.7,
) -> pd.DataFrame:
    """
    Assembling a comprehensive feature set for financial modelling.

    Parameters

    stock_close : pd.Series
        Stock closing prices
    stock_volume : pd.Series
        Stock Volume
    market_close : pd.Series, optional
        Market index closing prices (NASDAQ)
    market_out_col : str
        Name for market return column
    vol_window : int
        Window for realized volatility calculation
    volz_window : int
        Window for volume z_score calculation
    include_eoq : bool
        Whether to include end-of-quarter 
    macd : str
        MACD components to include
    handle_missing : str
        How to handle missing values (Conservative(only drop columns where target is missing) , aggresive ( Drop rows with any missing features), target_only (only require target to be valid))
    min_valid_ratio : float
        Minimum ratio of valid observations required to proceed
    
    Returns

    pd.DataFrame
        Feature matrix with columns depending on options:
        - 'ret' (target)
        - 'mkt_ret' (market_Close)
        - 'vol_realized'
        - 'vol_z'
        - 'eoq'
        - MACD columns 
    
    """

    if len(stock_close) == 0 or len(stock_volume) == 0:
        raise ValueError("Input series cannot be empty")
    
    if not stock_close.index.equals(stock_volume.index):
        raise ValueError("stock_close and stock_volume must have the same index")
    
    # Base index
    idx = stock_close.index

    #Target variable
    ret = log_return(stock_close).rename("ret")

    # Core features
    vol_real = realized_vol(ret,window=vol_window).rename("vol_realized")
    vol_z = volume_zscore(stock_volume, window=volz_window).rename("vol_z")

    parts = [ret, vol_real, vol_z]

    # Market feature
    if market_close is not None:
        mkt_ret = log_return(market_close).reindex(idx).rename(market_out_col)
        parts.append(mkt_ret)

    if include_eoq:
        eoq = end_of_quarter_flag(idx)
        parts.append(eoq)


    # Macd features

    if macd and macd.lower() != "none":
        macd_df = compute_macd(stock_close)
        macd_choice = macd.lower()
        if macd_choice == "line":
            macd_df = macd_df[["macd"]]
        elif macd_choice == "signal":
            macd_df = macd_df[["macd_signal"]]
        elif macd_choice == "hist":
            macd_df = macd_df[["macd_hist"]]
        elif macd_choice == "both":
            macd_df = macd_df[["macd","macd_signal","macd_hist"]]
        else:
            raise ValueError("macd must be one of {'none','line','signal','hist','both'}")
        parts.append(macd_df)
    
    # Assemble features
    df = pd.concat(parts,axis=1)

    # Handle missing values
    if handle_missing == "aggressive":
        df = df.dropna() #Drop any row with any missing value
    elif handle_missing == "conservative":
        df = df.dropna(subset=["ret"]) # Drop rows where target (ret) is missing
        feature_cols = [col for col in df.columns if col != 'ret']
        df[feature_cols] = df[feature_cols].ffill(limit=5) #max 5 days forward fill
        df = df.dropna()
    elif handle_missing == "target_only":
        df = df.dropna(subset=['ret']) # only require target to be valid
    else:
        raise ValueError("handle missing must be one of conservative, aggresive, target only")
    
    # Ensure finite values
    df=df[np.isfinite(df).all(axis=1)]

    # Check for sufficient data
    if len(df) / len(idx) < min_valid_ratio:
        raise ValueError(
            f"Too much data lost due to missing values"
            f"Only {len(df)}/{len(idx)} ({100*len(df)/len(idx):.1f}%) observations remain"
        )
    
    df.index.name = "Date"
    return df





def add_lags(
        df: pd.DataFrame,
        cols: Union[str, List[str]],
        lags: Union[int, Tuple[int, ...]] = (1,)
) -> pd.DataFrame:
    """
    Add lagged versions of specified columns
    
    Parameters

    df : pd.DataFrame
        Input dataframe
    cols : str or list of str
        Column name(s) to lag
    lags : int or tuple of int
        Lag periods to create

    Returns

    pd.DataFrame
        DataFrame with added lagged columns
    """

    if isinstance(cols,str):
        cols = [cols]
    if isinstance(lags, int):
        lags = (lags, )
    
    # Validate columns exist
    missing_cols = [c for c in cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Columns not found in dataframe: {missing_cols}")
    
    out = df.copy()
    for c in cols:
        for L in lags:
            if L <= 0:
                raise ValueError("Lag periods must be positive")
            out[f"{c}_l{L}"] = out[c].shift(L)
    return out


def make_nextday_target(
        features: pd.DataFrame,
        target_col: str = "ret",
        out_col: str = "y_next"
) -> pd.DataFrame:
    """
    Create a supervised dataset where X_t predicts y_{t+1}
    
    Parameters
    
    features : pd.DataFrame
        Feature matrix
    target_col : str
        Name of target column to shift
    out_col : str
        Name for the next-day target column
        
    Returns
    
    pd.DataFrame
        Dataset with next-day targets
    """

    if target_col not in features.columns:
        raise ValueError(f"Target column '{target_col}' not found in features")
    
    df = features.copy()
    df[out_col]=df[target_col].shift(-1)
    df = df.dropna(subset=[out_col])

    return df


