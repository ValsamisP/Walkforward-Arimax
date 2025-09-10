from __future__ import annotations
import warnings
import logging
from dataclasses import dataclass
from typing import List, Tuple, Optional, Union, Dict, Any
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller

# exogenous features 
DEFAULT_EXOG = ["mkt_ret", "vol_realized", "vol_z", "eoq", "macd_hist"]

@dataclass
class WalkMetrics:
    """Metrics for walk-forward validation results"""
    rmse: float
    mae: float
    hit_rate: float
    n_windows: int
    sharpe_ratio: Optional[float] = None # trading performance
    max_drawdown: Optional[float] = None 
    avg_successful_fits: Optional[float] = None
    model_stability: Optional[float] = None 

@dataclass
class ModelDiagnostics:
    """Diagnostic information for model fitting."""
    convergence_rate: float
    avg_log_likelihood: float
    residual_autocorr_pval: Optional[float] = None
    stationarity_pval: Optional[float] = None

def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Square Error"""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) == 0 or len(y_pred) == 0:
        return np.nan
    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}")
    
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not mask.any():
        return np.nan 
    
    return float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2)))

def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error"""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if len(y_true) == 0 or len(y_pred) == 0:
        return np.nan 
    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}")
    
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not mask.any():
        return np.nan
    
    return float(np.mean(np.abs(y_true - y_pred)))

def _aicc(llf: float, nobs: int, k_params: int) -> float:
    """AICc: small-sample corrected AIC.
    
    Formula : AIC + (2k(k+1))/(n-k-1) where k=param and n=observations
    """
    if k_params <=0 or nobs <= 0:
        return np.inf
    aic = -2.0 * llf + 2.0 * k_params
    if nobs - k_params - 1 <= 0:
        return np.inf
    return aic + (2.0 * k_params * (k_params + 1)) / (nobs - k_params - 1)

def _calculate_max_drawdown(returns: pd.Series) -> float:
    """Calculate maximum drawdown from a return series."""

    if len(returns) == 0:
        return np.nan
    
    cumulative = (1+returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative-running_max)/running_max
    return float(drawdown.min())

def _robust_scaling(X_train: pd.DataFrame, X_test: pd.DataFrame,method: str = "zscore") -> Tuple[pd.DataFrame, pd.DataFrame]:

    """
    Enhanced scaling with multiple methods
    
    Parameters
    
    method : str
        "zscore"  -- standard normalization
        "robust"  -- median/IQR scaling
        "minmax"  -- min-max scaling
    """

    if method == "zscore":
        mu = X_train.mean()
        sigma = X_train.std().replace(0, 1.0)
        return (X_train - mu) / sigma, (X_test - mu) / sigma 
    
    elif method == "robust":
        median = X_train.median()
        q75, q25 = X_train.quantile(0.75), X_train.quantile(0.25)
        iqr = (q75 - q25 ).replace(0, 1.0)
        return (X_train - median) / iqr, (X_test - median) /iqr 
    
    elif method == "minmax":
        x_min, x_max = X_train.min(), X_train.max()
        x_range = (x_max - x_min).replace(0, 1.0)
        return (X_train - x_min) / x_range , (X_test - x_min) / x_range
    
    else:
        raise ValueError("method must be zscore, robust or minmax")
    

def _validate_model_inputs(features: pd.DataFrame, target_col: str, 
                          exog_cols: List[str], train_len: int, 
                          test_len: int) -> None:
    """Comprehensive input validation."""
    if features.empty:
        raise ValueError("Features DataFrame is empty")
    
    if target_col not in features.columns:
        raise ValueError(f"Target column '{target_col}' not found")
    
    missing_exog = [col for col in exog_cols if col not in features.columns]
    if missing_exog:
        raise ValueError(f"Exogenous columns missing: {missing_exog}")
    
    if train_len <= 0 or test_len <= 0:
        raise ValueError("train_len and test_len must be positive")
    
    if len(features) < train_len + test_len:
        raise ValueError(
            f"Insufficient data: need {train_len + test_len}, got {len(features)}"
        )

def _fit_arimax_robust(
        y_tr: pd.Series,
        X_tr: pd.DataFrame,
        order: Tuple[int,int,int],
        *,
        trend: str = "n",
        maxiter: int = 500,
        methods: List[str] = None,
        validate_residuals: bool = True  
) -> Optional[Dict[str, Any]]:
    """
    Robust ARIMAX fitting  with multiple optimization methods and diagnostics.
    
    Returns:
    Dict with keys: "model","converged", "llf", "diagnostics", or None if all methods fail
    """
    if methods is None:
        methods = ["lbfgs" , "bfgs", "nm", "powell"]
    
    best_result = None
    best_llf = -np.inf 

    for method in methods:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                warnings.simplefilter("ignore", UserWarning)

                model = SARIMAX(
                    y_tr,
                    exog=X_tr,
                    order=order,
                    seasonal_order=(0,0,0,0),
                    trend=trend,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                    simple_differencing=False
                )

                result = model.fit(
                    disp=False,
                    maxiter=maxiter,
                    method=method,
                    full_output=False
                )

                # Checking Convergence
                converged = True
                if hasattr(result , 'mle_retvals') and result.mle_retvals:
                    converged = result.mle_retvals.get('converged',True)

                # Validating result quality
                if np.isfinite(result.llf) and result.llf > best_llf:
                    diagnostics = {}

                    # Residual Analysis

                    if validate_residuals and converged: 
                        try :
                            residuals = result.resid
                            if len(residuals) > 10:
                                ljung_box = acorr_ljungbox(residuals, lags=min(10, len(residuals)//4))
                                diagnostics['ljung_box_pval'] = ljung_box['lb_pvalue'].iloc[-1]
                        except Exception:
                            pass
                    
                    best_result = {
                        'model': result,
                        'converged': converged,
                        'llf': result.llf,
                        'method': method,
                        'diagnostics': diagnostics
                    }
                    best_llf = result.llf

                    if converged:
                        break
        except Exception as e:
            logging.debug(f"Method {method} failed for order {order}: {str(e)}")
            continue
    return best_result



def walk_forward_arimax(
    features: pd.DataFrame,
    *,
    order_grid: Optional[List[Tuple[int, int, int]]] = None,
    exog_cols: Optional[List[str]] = None,
    train_len: int = 120,
    test_len: int = 20,
    step: int = 20,                 # step between windows (use test_len for non-overlap)
    expanding: bool = False,
    max_train_len: Optional[int] = None,  # cap for expanding window
    min_train_len: int = 60,
    scaling_method: str = "zscore",
    use_aicc: bool = True,
    target_col: str = "ret",
    verbose: bool = False,
    save_diagnostics: bool = True,
    early_stopping_patience: Optional[int] = None, #Stopping if RMSE does not improve
    ensemble_method: Optional[str] = None, 
) -> Tuple[pd.DataFrame, WalkMetrics, pd.DataFrame, Optional[ModelDiagnostics]]:
    """
    Rolling ARIMAX with optional order search and per-window scaling.

    Parameters
    ----------
    features : DataFrame
        Must contain the target column and requested exogenous columns.
    order_grid : list of (p,d,q)
        Orders to try per train window. Best chosen by AICc (or AIC).
    exog_cols : list[str]
        Which exogenous columns to use. Defaults to DEFAULT_EXOG.
    train_len : int
        Number of observations in each training window.
    test_len : int
        Number of observations to forecast each iteration.
    step : int
        How far to move the window each iteration.
    expanding : bool
        If True, training set grows by `step` each iteration (up to max_train_len).
        If False, use a fixed-length sliding window.
    max_train_len : int | None
        Cap the training length if expanding=True.
    scale_exog : bool
        Standardize exogenous variables using training mean/std, apply to test.
    use_aicc : bool
        Use AICc to select best order (recommended with short windows).
    target_col : str
        Target column name (e.g., "y_next").
    min_train_len : int
        Minimum training window size
    scaling_method : str
        Feature scaling method
    early_stopping_patience: int , Optional
        Stop if validation RMSE does not improve for N windows.
    ensemble_method : str, optional
        How to combine predictions from multiple good models.
    

    Returns
    
    per_point : DataFrame
        With confidence intervals and ensempble info
    overall  : WalkMetrics
        Including stability measures.
    per_window: DataFrame
        Window diagnostics
    diagnostics: ModelDiagnostics
        OVerall model health metrics.
    """
    
    # Order Grid
    if order_grid is None:
        order_grid= [
        (1,0,0),(0,0,1),(1,0,1),# Basic AR/MA
        (2,0,0),(0,0,2),(2,0,1),(1,0,2), # Higher order
        (1,1,0),(0,1,1),(1,1,1),(2,1,1), # With differencing
        (3,0,0),(0,0,3)
        ]
    
    if exog_cols is None:
        exog_cols = DEFAULT_EXOG
    
    # Input Validation
    _validate_model_inputs(features, target_col, exog_cols, train_len , test_len)

    # Data preparation

    need_cols = [target_col] + list(exog_cols)
    df = features[need_cols].copy()

    for col in need_cols:
        df[col] = pd.to_numeric(df[col], errors= 'coerce')

    df = df.replace([np.inf, -np.inf],np.nan)

    initial_len = len(df)
    df = df.dropna()


    if len(df) / initial_len < 0.5:
        warnings.warn(f"Lost {100*(1-len(df)/initial_len):.1f}% of data due to missing values")
    
    if len(df) < train_len + test_len:
        raise ValueError(f"After cleaning: {len(df)} < {train_len + test_len}")
    
    y_all = df[target_col]
    X_all = df[exog_cols]

    i=0
    n = len(df)
    preds =[]
    win_rows = []
    current_train_len = max(train_len, min_train_len)

    all_convergence = []
    all_llf = []
    window_rmses = []

    #Early stopping
    best_rolling_rmse = np.inf
    patience_counter = 0

    while True:
        start_tr = i
        end_tr = i + current_train_len
        end_te = end_tr + test_len 

        if end_te > n:
            break

        y_tr = y_all.iloc[start_tr:end_tr]
        X_tr = X_all.iloc[start_tr:end_tr]
        y_te = y_all.iloc[end_tr:end_te]
        X_te = X_all.iloc[end_tr:end_te]

        if scaling_method != "none":
            try:
                X_tr_scaled, X_te_scaled = _robust_scaling(X_tr, X_te, scaling_method)
            except Exception as e:
                logging.warning(f"Scaling failed, using raw features: {str(e)}")
                X_tr_scaled, X_te_scaled = X_tr, X_te
        else:
            X_tr_scaled, X_te_scaled = X_tr, X_te 

        # Reset indices for statsmodels
        y_tr_idx = y_tr.reset_index(drop=True)
        X_tr_idx = X_tr_scaled.reset_index(drop=True)
        X_te_idx = X_te_scaled.reset_index(drop=True)

        # Enhanced model selection
        best_ic = np.inf
        best_result = None
        best_order = None
        successful_models = []

        for order in order_grid:
            fit_result = _fit_arimax_robust(
                y_tr_idx, X_tr_idx, order, 
                validate_residuals=save_diagnostics
            )
            
            if fit_result is None:
                continue
            
            try:
                model_obj = fit_result['model']
                y_hat = model_obj.forecast(steps=len(y_te), exog=X_te_idx)
                
                # Calculate IC
                if use_aicc:
                    ic = _aicc(fit_result['llf'], model_obj.nobs, len(model_obj.params))
                else:
                    ic = model_obj.aic
                
                successful_models.append({
                    'order': order,
                    'prediction': y_hat,
                    'ic': ic,
                    'llf': fit_result['llf'],
                    'converged': fit_result['converged']
                })
                
                if ic < best_ic:
                    best_ic = ic
                    best_result = fit_result
                    best_order = order
                    
                all_convergence.append(fit_result['converged'])
                all_llf.append(fit_result['llf'])
                
            except Exception as e:
                logging.debug(f"Forecasting failed for order {order}: {str(e)}")
                continue

        # Handle case where no models succeeded
        if not successful_models:
            if verbose:
                print(f"Warning: All models failed for window {len(preds)}")
            # Naive forecast
            best_pred = pd.Series([y_tr.iloc[-1]] * len(y_te))
            best_order = "naive"
            converged = False
        else:
            # Get best prediction
            best_model_info = min(successful_models, key=lambda x: x['ic'])
            best_pred = best_model_info['prediction']
            best_order = best_model_info['order']
            converged = best_model_info['converged']
            
            # Optional ensemble
            if ensemble_method and len(successful_models) > 1:
                predictions = np.column_stack([m['prediction'] for m in successful_models[:5]])  # Top 5
                
                if ensemble_method == "mean":
                    best_pred = pd.Series(predictions.mean(axis=1))
                elif ensemble_method == "median":
                    best_pred = pd.Series(np.median(predictions, axis=1))

        # Store detailed results
        window_df = pd.DataFrame({
            "y_true": y_te.values,
            "y_pred": best_pred.values if hasattr(best_pred, 'values') else best_pred
        }, index=y_te.index)
        
        # Enhanced metadata
        window_df["window_id"] = len(preds)
        window_df["train_start"] = y_tr.index[0]
        window_df["train_end"] = y_tr.index[-1]
        window_df["test_start"] = y_te.index[0]
        window_df["test_end"] = y_te.index[-1]
        window_df["order"] = str(best_order)
        window_df["converged"] = converged
        window_df["n_successful_models"] = len(successful_models)
        window_df["train_length"] = len(y_tr)
        
        preds.append(window_df)

        # Window-level metrics
        w_rmse = _rmse(window_df["y_true"], window_df["y_pred"])
        w_mae = _mae(window_df["y_true"], window_df["y_pred"])
        w_hit = float((np.sign(window_df["y_true"]) == np.sign(window_df["y_pred"])).mean())
        
        window_rmses.append(w_rmse)
        
        # Enhanced window diagnostics
        win_rows.append({
            "window_id": len(preds) - 1,
            "train_start": y_tr.index[0],
            "train_end": y_tr.index[-1],
            "test_start": y_te.index[0],
            "test_end": y_te.index[-1],
            "order": str(best_order),
            "rmse": w_rmse,
            "mae": w_mae,
            "hit_rate": w_hit,
            "converged": converged,
            "n_successful_models": len(successful_models),
            "train_length": len(y_tr),
            "ic_value": best_ic if best_ic != np.inf else np.nan
        })

        # Progress reporting
        if verbose and (len(preds) % 10 == 0 or len(preds) <= 5):
            print(f"Window {len(preds):3d}: RMSE={w_rmse:.4f}, Models={len(successful_models)}, Order={best_order}")

        # Early stopping check
        if early_stopping_patience and len(window_rmses) >= 5:
            recent_rmse = np.mean(window_rmses[-5:])
            if recent_rmse < best_rolling_rmse:
                best_rolling_rmse = recent_rmse
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    if verbose:
                        print(f"Early stopping at window {len(preds)} (patience={early_stopping_patience})")
                    break

        # Advance window
        if expanding:
            i += step
            if max_train_len:
                current_train_len = min(current_train_len + step, max_train_len)
            else:
                current_train_len = min(current_train_len + step, n - test_len)
        else:
            i += step

    if not preds:
        raise ValueError("No windows were successfully processed")

    # Aggregate results
    per_point = pd.concat(preds, axis=0)

    # Enhanced overall metrics
    overall_rmse = _rmse(per_point["y_true"], per_point["y_pred"])
    overall_mae = _mae(per_point["y_true"], per_point["y_pred"])
    overall_hit = float((np.sign(per_point["y_true"]) == np.sign(per_point["y_pred"])).mean())
    
    # Financial metrics
    pred_sharpe = None
    max_dd = None
    if len(per_point["y_pred"]) > 1:
        if per_point["y_pred"].std() > 0:
            pred_sharpe = float(per_point["y_pred"].mean() / per_point["y_pred"].std() * np.sqrt(252))
        max_dd = _calculate_max_drawdown(per_point["y_pred"])
    
    # Model stability
    model_stability = float(np.std(window_rmses)) if len(window_rmses) > 1 else np.nan
    avg_successful = float(np.mean([row["n_successful_models"] for row in win_rows]))

    overall = WalkMetrics(
        rmse=overall_rmse,
        mae=overall_mae,
        hit_rate=overall_hit,
        n_windows=len(win_rows),
        sharpe_ratio=pred_sharpe,
        max_drawdown=max_dd,
        avg_successful_fits=avg_successful,
        model_stability=model_stability
    )

    per_window = pd.DataFrame(win_rows)
    
    # Model diagnostics
    diagnostics = None
    if save_diagnostics and all_convergence:
        diagnostics = ModelDiagnostics(
            convergence_rate=float(np.mean(all_convergence)),
            avg_log_likelihood=float(np.mean(all_llf)) if all_llf else np.nan,
        )
    
    return per_point, overall, per_window, diagnostics
        

def analyze_residuals(per_point: pd.DataFrame, plot: bool = False) -> Dict[str, float] :
    """
    Analyze prediction residuals for model validation 
    
    Returns various residuals statistics and tests.
    
    """

    residuals = per_point["y_true"] - per_point["y_pred"]

    analysis = {
        "mean_residual": float(residuals.mean()),
        "residual_std": float(residuals.std()),
        "residual_skew": float(residuals.skew()),
        "residual_kurt": float(residuals.kurtosis()),
        "ljung_box_pval": np.nan
    }

    # Ljung box test for residual autocorrelation

    try:
        if len(residuals) > 20:
            lb_test = acorr_ljungbox(residuals, lags=min(10, len(residuals)//4))
            analysis["ljung_box_pval"] = float(lb_test['lb_pvalue'].iloc[-1])
    except Exception:
        pass 

    return analysis

def generate_summary_report(overall: WalkMetrics,
                                per_window: pd.DataFrame,
                                diagnostics: Optional[ModelDiagnostics] = None) -> str:
        """ Generate a summary report"""

        report = f"""
    ARIMAX Walk-Forward Validation Summary

    Overall Performance
    - RMSE: {overall.rmse:.6f}
    - MAE: {overall.mae:.6f}
    - Hit Rate: {overall.hit_rate:.1%}
    - Windows: {overall.n_windows}
    - Model Stability (RMSE std): {overall.model_stability:.6f}

    Financial Metrics:
    - Sharpe Ratio: {overall.sharpe_ratio:.3f if overall.sharpe_ratio else 'N/A'}
    - Max Drawdown: {overall.max_drawdown:.1%}
    - Avg Successful Fits: {overall.avg_successful_fits:.1f}

    Window Analysis:
    - Best Window RMSE: {per_window['rmse'].min():.6f}
    - Worst Window RMSE: {per_window['rmse'].max():.6f}
    - Convergence Rate: {(per_window['converged'].mean()*100):.1f}%
    """
    
        if diagnostics:
            report += f"""
    Model Diagnostics:
    - Overall Convergence Rate: {diagnostics.convergence_rate:.1%}
    - Average Log-Likelihood: {diagnostics.avg_log_likelihood:.2f}
    """

    # Order frequency analysis
        order_counts = per_window['order'].value_counts()
        report += f"""
    Most Common Orders:
    {order_counts.head().to_string()}
    """
    
        return report