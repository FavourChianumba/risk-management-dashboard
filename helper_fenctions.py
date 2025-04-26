import pandas as pd
import numpy as np
from scipy import stats
import datetime as dt
from typing import List, Dict, Union, Tuple, Optional
from pathlib import Path

def calculate_returns(prices: pd.DataFrame, method: str = 'log') -> pd.DataFrame:
    """
    Calculate returns from price data.
    
    Parameters:
    -----------
    prices : pd.DataFrame
        DataFrame of price data
    method : str
        Method to use for calculating returns ('simple' or 'log')
    
    Returns:
    --------
    pd.DataFrame
        DataFrame of returns
    """
    if method.lower() == 'simple':
        returns = prices.pct_change().dropna()
    elif method.lower() == 'log':
        returns = np.log(prices / prices.shift(1)).dropna()
    else:
        raise ValueError("method must be 'simple' or 'log'")
    
    return returns

def calculate_historical_var(returns: pd.Series, 
                            confidence_level: float = 0.95, 
                            investment_value: float = 1000000) -> float:
    """
    Calculate historical Value-at-Risk.
    
    Parameters:
    -----------
    returns : pd.Series
        Series of returns
    confidence_level : float
        Confidence level (e.g., 0.95 for 95%)
    investment_value : float
        Investment value
    
    Returns:
    --------
    float
        Historical VaR value
    """
    var_percentile = 1 - confidence_level
    var_return = np.percentile(returns, var_percentile * 100)
    var_value = -var_return * investment_value
    
    return var_value

def calculate_parametric_var(returns: pd.Series, 
                           confidence_level: float = 0.95, 
                           investment_value: float = 1000000) -> float:
    """
    Calculate parametric Value-at-Risk assuming normal distribution.
    
    Parameters:
    -----------
    returns : pd.Series
        Series of returns
    confidence_level : float
        Confidence level (e.g., 0.95 for 95%)
    investment_value : float
        Investment value
    
    Returns:
    --------
    float
        Parametric VaR value
    """
    mu = returns.mean()
    sigma = returns.std()
    z_score = stats.norm.ppf(1 - confidence_level)
    var_return = mu + z_score * sigma
    var_value = -var_return * investment_value
    
    return var_value

def calculate_expected_shortfall(returns: pd.Series, 
                               confidence_level: float = 0.95, 
                               investment_value: float = 1000000) -> float:
    """
    Calculate Expected Shortfall (Conditional VaR).
    
    Parameters:
    -----------
    returns : pd.Series
        Series of returns
    confidence_level : float
        Confidence level (e.g., 0.95 for 95%)
    investment_value : float
        Investment value
    
    Returns:
    --------
    float
        Expected Shortfall value
    """
    var_percentile = 1 - confidence_level
    var_return = np.percentile(returns, var_percentile * 100)
    
    # Calculate average of returns beyond VaR
    tail_returns = returns[returns <= var_return]
    es_return = tail_returns.mean()
    es_value = -es_return * investment_value
    
    return es_value

def calculate_drawdown(returns: pd.Series) -> Tuple[float, pd.Series]:
    """
    Calculate maximum drawdown and drawdown series.
    
    Parameters:
    -----------
    returns : pd.Series
        Series of returns
    
    Returns:
    --------
    tuple
        Maximum drawdown value, drawdown series
    """
    # Calculate cumulative returns
    cumulative_returns = (1 + returns).cumprod()
    
    # Calculate running maximum
    running_max = cumulative_returns.cummax()
    
    # Calculate drawdown
    drawdown = (cumulative_returns / running_max) - 1
    
    # Calculate maximum drawdown
    max_drawdown = drawdown.min()
    
    return max_drawdown, drawdown

def calculate_risk_metrics(returns: pd.Series) -> Dict[str, float]:
    """
    Calculate common risk metrics.
    
    Parameters:
    -----------
    returns : pd.Series
        Series of returns
    
    Returns:
    --------
    dict
        Dictionary of risk metrics
    """
    # Calculate basic statistics
    mean_return = returns.mean()
    std_dev = returns.std()
    
    # Calculate annualized metrics (assuming daily data)
    ann_return = (1 + mean_return) ** 252 - 1
    ann_vol = std_dev * np.sqrt(252)
    
    # Calculate Sharpe ratio (assuming 0% risk-free rate)
    sharpe_ratio = ann_return / ann_vol if ann_vol != 0 else 0
    
    # Calculate maximum drawdown
    max_drawdown, _ = calculate_drawdown(returns)
    
    # Calculate skewness and kurtosis
    skewness = stats.skew(returns.dropna())
    kurtosis = stats.kurtosis(returns.dropna())
    
    # Calculate win rate
    win_rate = (returns > 0).sum() / len(returns)
    
    # Return metrics as dictionary
    metrics = {
        'mean_return': mean_return,
        'std_dev': std_dev,
        'ann_return': ann_return,
        'ann_vol': ann_vol,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'skewness': skewness,
        'kurtosis': kurtosis,
        'win_rate': win_rate
    }
    
    return metrics

def identify_extreme_events(returns: pd.Series, threshold: float = 3.0) -> pd.Series:
    """
    Identify extreme return events beyond a threshold number of standard deviations.
    
    Parameters:
    -----------
    returns : pd.Series
        Series of returns
    threshold : float
        Threshold in number of standard deviations
    
    Returns:
    --------
    pd.Series
        Series of extreme events with boolean values
    """
    mean = returns.mean()
    std = returns.std()
    
    # Identify extreme events
    extreme_events = abs(returns - mean) > (threshold * std)
    
    return extreme_events

def format_currency(value: float) -> str:
    """Format value as currency."""
    return f"${value:,.2f}"

def format_percentage(value: float) -> str:
    """Format value as percentage."""
    return f"{value:.2%}"

def date_range_to_str(date_range: List[dt.datetime]) -> str:
    """Convert date range to string representation."""
    if len(date_range) == 2:
        start, end = date_range
        return f"{start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}"
    else:
        return "Invalid date range"

def filter_by_date_range(df: pd.DataFrame, 
                        date_range: List[dt.datetime], 
                        date_column: str = None) -> pd.DataFrame:
    """
    Filter DataFrame by date range.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to filter
    date_range : list
        List of two dates [start_date, end_date]
    date_column : str, optional
        Name of date column (if not index)
    
    Returns:
    --------
    pd.DataFrame
        Filtered DataFrame
    """
    if len(date_range) != 2:
        return df
    
    start_date, end_date = date_range
    
    if date_column is None:
        # Filter by index
        return df.loc[start_date:end_date]
    else:
        # Filter by column
        mask = (df[date_column] >= start_date) & (df[date_column] <= end_date)
        return df[mask]

def risk_level_from_var(var_pct: float) -> str:
    """
    Determine risk level based on VaR percentage.
    
    Parameters:
    -----------
    var_pct : float
        VaR as percentage of investment value
    
    Returns:
    --------
    str
        Risk level (Low, Medium, High, Extreme)
    """
    if var_pct > 0.05:
        return "Extreme"
    elif var_pct > 0.03:
        return "High"
    elif var_pct > 0.01:
        return "Medium"
    else:
        return "Low"

def create_calendar_table(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Create a calendar table from a date index.
    
    Parameters:
    -----------
    dates : pd.DatetimeIndex
        DatetimeIndex of dates
    
    Returns:
    --------
    pd.DataFrame
        Calendar table with date dimensions
    """
    # Convert index to DataFrame
    calendar = pd.DataFrame({'Date': dates})
    
    # Extract date components
    calendar['Year'] = calendar['Date'].dt.year
    calendar['Month'] = calendar['Date'].dt.month
    calendar['MonthName'] = calendar['Date'].dt.strftime('%b')
    calendar['Quarter'] = calendar['Date'].dt.quarter
    calendar['Day'] = calendar['Date'].dt.day
    calendar['DayOfWeek'] = calendar['Date'].dt.dayofweek
    calendar['DayName'] = calendar['Date'].dt.strftime('%a')
    calendar['WeekOfYear'] = calendar['Date'].dt.isocalendar().week
    calendar['YearMonth'] = calendar['Date'].dt.strftime('%Y-%m')
    
    return calendar

def load_and_validate_data(data_dir: Path) -> Dict[str, pd.DataFrame]:
    """
    Load and validate all required data files.
    
    Parameters:
    -----------
    data_dir : Path
        Path to data directory
    
    Returns:
    --------
    dict
        Dictionary of DataFrames
    """
    # Define file paths
    files = {
        'portfolio_returns': data_dir / 'processed' / 'portfolio_returns.csv',
        'var_results': data_dir / 'results' / 'var_results.csv',
        'stress_test_results': data_dir / 'results' / 'stress_test_results.csv',
        'var_backtest_summary': data_dir / 'results' / 'var_backtest_summary.csv',
    }
    
    # Load data files
    data = {}
    missing_files = []
    
    for name, path in files.items():
        if path.exists():
            try:
                # Handle date parsing for portfolio returns
                if name == 'portfolio_returns':
                    df = pd.read_csv(path, parse_dates=['Date'], index_col='Date')
                    # Rename column if necessary
                    if '0' in df.columns:
                        df.columns = ['Return']
                else:
                    df = pd.read_csv(path)
                
                data[name] = df
            except Exception as e:
                print(f"Error loading {name}: {e}")
                missing_files.append(name)
        else:
            missing_files.append(name)
    
    # Report missing files
    if missing_files:
        print(f"Warning: The following files were not found: {', '.join(missing_files)}")
    
    return data