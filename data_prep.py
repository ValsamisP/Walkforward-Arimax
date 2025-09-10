import numpy as np
import pandas as pd
import yfinance as yf
from typing import Optional, Tuple, Union
from datetime import date, datetime

DateLike = Optional[Union[str, date, datetime]]

def _parse_date(d: DateLike, *, dayfirst: bool = True) -> Optional[str]:
    """Parse many date formats into 'YYYY-MM-DD' strings for yfinance."""
    if d is None:
        return None
    if isinstance(d, (date, datetime)):
        return pd.Timestamp(d).strftime("%Y-%m-%d")
    ts = pd.to_datetime(d, dayfirst=dayfirst, errors="raise")
    return ts.strftime("%Y-%m-%d")


def load_prices(
    ticker: str = "NVDA",
    start: DateLike = None,
    end: DateLike = None,
    interval: str = "1d",
    auto_adjust: bool = True, #This parameter controls whether the yfinance automatically adjust the OHLC columns
    columns: Tuple[str, ...] = ("Open","Close","High","Low", "Volume"),
    *,
    dayfirst: bool = True, #when False follows european date formats
    group_by: str = "column",
) -> pd.DataFrame:
    """
    Download stock data from yfinance.

    start/end accept formats:
      - '2020-01-01' (ISO)
      - '01/01/2020', '01-01-2020'
      - 'Jan 1, 2020', datetime/date objects
    """
    start_str = _parse_date(start, dayfirst=dayfirst) #using the function created in the beggining of the script
    end_str   = _parse_date(end,   dayfirst=dayfirst)

    try:
        df = yf.download(
            tickers=ticker,
            start=start_str,
            end=end_str,
            interval=interval,
            auto_adjust=auto_adjust,
            progress=False, #controlling if progress bar will be displaying when i am downloading the data
            group_by=group_by, #depending on which version of the yfinance package i have
        )
    except TypeError: #there no exist the parameter group_by (so it helps as a compatibility mechanism) to handle different versions of the yfinance library
        df = yf.download(
            tickers=ticker,
            start=start_str,
            end=end_str,
            interval=interval,
            auto_adjust=auto_adjust,
            progress=False,
        )

    if df.empty:
        raise ValueError(f"No data returned for ticker {ticker} with given parameters") #just in case if no data downloaded

    # --- Normalize columns across yfinance variants ---
    if isinstance(df.columns, pd.MultiIndex): #checking if i have multiindex columns
        # 1) If a level contains the ticker, slice that level
        lvl_with_ticker = None
        for lvl in range(df.columns.nlevels):
            if str(ticker) in df.columns.get_level_values(lvl):
                lvl_with_ticker = lvl
                break

        if lvl_with_ticker is not None:
            df = df.xs(ticker, axis=1, level=lvl_with_ticker, drop_level=True)
        else:
            # 2) If top level looks like standard OHLCV names, just keep that level
            top = df.columns.get_level_values(0)
            common_cols = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}
            if set(common_cols).issubset(set(top)):
                df.columns = top
            else:
                # 3) Flatten and pick requested columns by best match
                df.columns = ["|".join(map(str, tup)).strip("|") for tup in df.columns]

                def find_col(name: str) -> str:
                    cands = [
                        c for c in df.columns
                        if c.split("|")[0] == name or c.endswith(f"|{name}")
                    ]
                    if not cands:
                        preview = ", ".join(list(df.columns)[:10])
                        raise ValueError(
                            f"Could not find column '{name}' in flattened columns: {preview} ..."
                        )
                    return cands[0]

                colmap = {c: find_col(c) for c in columns}
                df = df.loc[:, list(colmap.values())]
                df.columns = list(colmap.keys())

    # Ensure requested columns exist
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in downloaded data: {missing}")

    df = df.loc[:, list(columns)].copy()
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    return df

