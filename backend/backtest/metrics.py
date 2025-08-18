import pandas as pd

def compute_metrics(returns: pd.Series):
    """
    Compute simple performance metrics.
    returns: a pd.Series of periodic returns (e.g. minute returns)
    """
    # total return
    total_ret = (1 + returns).prod() - 1 if len(returns) > 0 else 0.0

    # approximate annualised return and sharpe for minute data:
    # periods per day = 1440
    if len(returns) > 1:
        avg_ret = returns.mean()
        std_ret = returns.std(ddof=0) if returns.std(ddof=0) != 0 else 1e-8
        annual_factor = 1440  # minutes per day used as simple scaler
        sharpe = (avg_ret * annual_factor) / (std_ret * (annual_factor ** 0.5))
    else:
        sharpe = 0.0

    # max drawdown
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    dd = ((cum - peak) / peak).min() if len(cum) > 0 else 0.0

    return {
        "total_return": float(total_ret),
        "sharpe": float(sharpe),
        "max_drawdown": float(dd)
    }
