"""
Historical Value-at-Risk (VaR) implementation.
This module calculates VaR using the historical simulation method.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from scipy import stats
from datetime import datetime

# Get project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"

# Set up logger
logger = logging.getLogger(__name__)

def load_model_config():
    """
    Load model configuration from config file.
    
    Returns:
        dict: Model configuration for historical VaR
    """
    try:
        with open(CONFIG_DIR / "model_config.json", 'r') as f:
            model_config = json.load(f)
        
        return model_config['var_models']['historical']
    except Exception as e:
        logger.error(f"Error loading model configuration: {e}")
        # Return default configuration
        return {
            'confidence_levels': [0.90, 0.95, 0.99],
            'default_confidence': 0.95,
            'lookback_days': 252,
            'time_horizon': 1
        }

def calculate_historical_var(returns, confidence_level=None, investment_value=1000000, lookback_days=None):
    """
    Calculate Value-at-Risk using historical simulation method.
    
    Args:
        returns (pandas.Series or pandas.DataFrame): Historical returns data
        confidence_level (float, optional): Confidence level (e.g., 0.95 for 95% confidence)
        investment_value (float, optional): Current portfolio value
        lookback_days (int, optional): Number of days to look back. If None, uses all available data
        
    Returns:
        float: Value-at-Risk estimate
    """
    # Load configuration if parameters not provided
    config = load_model_config()
    
    if confidence_level is None:
        confidence_level = config['default_confidence']
    
    # If returns is a DataFrame with multiple columns, raise an error
    if isinstance(returns, pd.DataFrame) and returns.shape[1] > 1:
        raise ValueError("Please provide portfolio returns as a pandas Series, not a DataFrame with multiple columns")
    elif isinstance(returns, pd.DataFrame) and returns.shape[1] == 1:
        # Convert DataFrame with one column to Series
        returns = returns.iloc[:, 0]
    
    # Check if returns is empty
    if len(returns) == 0:
        raise ValueError("Returns data is empty")
    
    # Use specified lookback window if provided
    if lookback_days is not None:
        if lookback_days > len(returns):
            logger.warning(f"Requested lookback period ({lookback_days}) exceeds available data length ({len(returns)})")
        else:
            returns = returns.iloc[-lookback_days:]
    
    # Calculate the percentile corresponding to the VaR
    var_percentile = 1 - confidence_level
    
    # Get the return at the specified percentile
    var_return = np.percentile(returns, var_percentile * 100)
    
    # Convert to monetary value
    var_value = abs(var_return * investment_value)
    
    return var_value

def calculate_conditional_var(returns, confidence_level=None, investment_value=1000000, lookback_days=None):
    """
    Calculate Conditional Value-at-Risk (CVaR), also known as Expected Shortfall.
    CVaR is the expected loss given that the loss exceeds VaR.
    
    Args:
        returns (pandas.Series or pandas.DataFrame): Historical returns data
        confidence_level (float, optional): Confidence level (e.g., 0.95 for 95% confidence)
        investment_value (float, optional): Current portfolio value
        lookback_days (int, optional): Number of days to look back. If None, uses all available data
        
    Returns:
        float: Conditional Value-at-Risk estimate
    """
    # Load configuration if parameters not provided
    config = load_model_config()
    
    if confidence_level is None:
        confidence_level = config['default_confidence']
    
    # If returns is a DataFrame with multiple columns, raise an error
    if isinstance(returns, pd.DataFrame) and returns.shape[1] > 1:
        raise ValueError("Please provide portfolio returns as a pandas Series, not a DataFrame with multiple columns")
    elif isinstance(returns, pd.DataFrame) and returns.shape[1] == 1:
        # Convert DataFrame with one column to Series
        returns = returns.iloc[:, 0]
    
    # Check if returns is empty
    if len(returns) == 0:
        raise ValueError("Returns data is empty")
    
    # Use specified lookback window if provided
    if lookback_days is not None:
        if lookback_days > len(returns):
            logger.warning(f"Requested lookback period ({lookback_days}) exceeds available data length ({len(returns)})")
        else:
            returns = returns.iloc[-lookback_days:]
    
    # Calculate the percentile corresponding to the VaR
    var_percentile = 1 - confidence_level
    
    # Get the return at the specified percentile
    var_return = np.percentile(returns, var_percentile * 100)
    
    # Calculate CVaR as the mean of returns beyond VaR
    cvar_returns = returns[returns <= var_return]
    
    if len(cvar_returns) == 0:
        logger.warning("No returns found beyond VaR threshold. Using VaR as CVaR.")
        cvar_return = var_return
    else:
        cvar_return = cvar_returns.mean()
    
    # Convert to monetary value
    cvar_value = abs(cvar_return * investment_value)
    
    return cvar_value

def calculate_historical_var_time_scaled(returns, confidence_level=None, investment_value=1000000, 
                                        time_horizon=1, lookback_days=None):
    """
    Calculate Value-at-Risk for multiple time horizons using historical simulation method.
    VaR is scaled by the square root of time.
    
    Args:
        returns (pandas.Series): Historical returns data
        confidence_level (float, optional): Confidence level (e.g., 0.95 for 95% confidence)
        investment_value (float, optional): Current portfolio value
        time_horizon (int, optional): Time horizon in days
        lookback_days (int, optional): Number of days to look back. If None, uses all available data
        
    Returns:
        float: Value-at-Risk estimate for the specified time horizon
    """
    # Calculate single-day VaR
    one_day_var = calculate_historical_var(
        returns=returns,
        confidence_level=confidence_level,
        investment_value=investment_value,
        lookback_days=lookback_days
    )
    
    # Scale VaR by square root of time (assuming i.i.d. returns)
    time_scaled_var = one_day_var * np.sqrt(time_horizon)
    
    return time_scaled_var

def calculate_var_by_confidence(returns, confidence_levels=None, investment_value=1000000, lookback_days=None):
    """
    Calculate VaR for multiple confidence levels.
    
    Args:
        returns (pandas.Series): Historical returns data
        confidence_levels (list, optional): List of confidence levels
        investment_value (float, optional): Current portfolio value
        lookback_days (int, optional): Number of days to look back. If None, uses all available data
        
    Returns:
        pandas.DataFrame: VaR estimates for different confidence levels
    """
    # Load configuration if parameters not provided
    config = load_model_config()
    
    if confidence_levels is None:
        confidence_levels = config['confidence_levels']
    
    # Calculate VaR for each confidence level
    var_results = []
    
    for conf_level in confidence_levels:
        var_value = calculate_historical_var(
            returns=returns,
            confidence_level=conf_level,
            investment_value=investment_value,
            lookback_days=lookback_days
        )
        
        cvar_value = calculate_conditional_var(
            returns=returns,
            confidence_level=conf_level,
            investment_value=investment_value,
            lookback_days=lookback_days
        )
        
        var_results.append({
            'confidence_level': conf_level,
            'var_value': var_value,
            'cvar_value': cvar_value,
            'var_pct': var_value / investment_value * 100,
            'cvar_pct': cvar_value / investment_value * 100,
            'cvar_var_ratio': cvar_value / var_value
        })
    
    # Convert to DataFrame
    var_df = pd.DataFrame(var_results)
    
    return var_df

def calculate_rolling_var(returns, window_size=None, confidence_level=None, investment_value=1000000):
    """
    Calculate rolling VaR over time.
    
    Args:
        returns (pandas.Series): Historical returns data
        window_size (int, optional): Rolling window size
        confidence_level (float, optional): Confidence level (e.g., 0.95 for 95% confidence)
        investment_value (float, optional): Current portfolio value
        
    Returns:
        pandas.Series: Rolling VaR estimates
    """
    # Load configuration if parameters not provided
    config = load_model_config()
    
    if window_size is None:
        window_size = config['lookback_days']
    
    if confidence_level is None:
        confidence_level = config['default_confidence']
    
    # Check if we have enough data
    if len(returns) < window_size:
        logger.warning(f"Not enough data for rolling VaR. Required: {window_size}, Available: {len(returns)}")
        return pd.Series(index=returns.index)
    
    # Calculate the percentile corresponding to the VaR
    var_percentile = 1 - confidence_level
    
    # Calculate rolling VaR
    rolling_var = []
    rolling_dates = []
    
    for i in range(window_size, len(returns) + 1):
        window = returns.iloc[i-window_size:i]
        var_return = np.percentile(window, var_percentile * 100)
        var_value = abs(var_return * investment_value)
        
        rolling_var.append(var_value)
        rolling_dates.append(returns.index[i-1])
    
    # Create Series with rolling VaR values
    rolling_var_series = pd.Series(index=rolling_dates, data=rolling_var)
    
    return rolling_var_series

def analyze_var_exceedances(returns, confidence_level=None, lookback_days=None):
    """
    Analyze VaR exceedances (breaches) over time.
    
    Args:
        returns (pandas.Series): Historical returns data
        confidence_level (float, optional): Confidence level (e.g., 0.95 for 95% confidence)
        lookback_days (int, optional): Number of days for rolling VaR calculation
        
    Returns:
        dict: Dictionary with exceedance analysis results
    """
    # Load configuration if parameters not provided
    config = load_model_config()
    
    if confidence_level is None:
        confidence_level = config['default_confidence']
    
    if lookback_days is None:
        lookback_days = config['lookback_days']
    
    # Check if we have enough data
    if len(returns) <= lookback_days:
        logger.warning(f"Not enough data for exceedance analysis. Required: > {lookback_days}, Available: {len(returns)}")
        return {"error": "Not enough data for analysis"}
    
    # Calculate rolling VaR (in return space, not monetary)
    var_percentile = 1 - confidence_level
    rolling_var_returns = []
    
    for i in range(lookback_days, len(returns)):
        window = returns.iloc[i-lookback_days:i]
        var_return = np.percentile(window, var_percentile * 100)
        rolling_var_returns.append(var_return)
    
    # Create Series with rolling VaR thresholds
    var_thresholds = pd.Series(index=returns.index[lookback_days:], data=rolling_var_returns)
    
    # Identify exceedances
    actual_returns = returns.loc[var_thresholds.index]
    exceedances = actual_returns < var_thresholds
    
    # Calculate exceedance statistics
    n_exceedances = exceedances.sum()
    n_observations = len(exceedances)
    exceedance_rate = n_exceedances / n_observations
    expected_rate = 1 - confidence_level
    
    # Kupiec test (unconditional coverage)
    if n_exceedances > 0:
        kupiec_statistic = -2 * (
            n_observations * np.log(1 - expected_rate) + 
            n_exceedances * np.log(expected_rate) -
            (n_observations - n_exceedances) * np.log(1 - exceedance_rate) -
            n_exceedances * np.log(exceedance_rate)
        )
        kupiec_p_value = 1 - stats.chi2.cdf(kupiec_statistic, 1)
    else:
        kupiec_statistic = np.nan
        kupiec_p_value = np.nan
    
    # Identify consecutive exceedances (clusters)
    clusters = []
    current_cluster = 0
    
    for is_exceed in exceedances:
        if is_exceed:
            current_cluster += 1
        elif current_cluster > 0:
            clusters.append(current_cluster)
            current_cluster = 0
    
    # Add the last cluster if it exists
    if current_cluster > 0:
        clusters.append(current_cluster)
    
    # Calculate cluster statistics
    if clusters:
        max_cluster = max(clusters)
        avg_cluster = np.mean(clusters)
    else:
        max_cluster = 0
        avg_cluster = 0
    
    # Compile results
    results = {
        "confidence_level": confidence_level,
        "lookback_days": lookback_days,
        "n_observations": n_observations,
        "n_exceedances": n_exceedances,
        "exceedance_rate": exceedance_rate,
        "expected_rate": expected_rate,
        "exceedance_ratio": exceedance_rate / expected_rate,
        "kupiec_statistic": kupiec_statistic,
        "kupiec_p_value": kupiec_p_value,
        "kupiec_test_result": "Pass" if kupiec_p_value >= 0.05 else "Fail",
        "n_clusters": len(clusters),
        "max_cluster_size": max_cluster,
        "avg_cluster_size": avg_cluster,
        "exceedance_dates": actual_returns.index[exceedances],
        "var_thresholds": var_thresholds,
        "actual_returns": actual_returns
    }
    
    return results

def save_var_results(results, filename, results_dir=None):
    """
    Save VaR analysis results to file.
    
    Args:
        results (dict or pandas.DataFrame): VaR analysis results
        filename (str): Name of the file to save
        results_dir (str or Path, optional): Directory to save results. If None, uses default.
        
    Returns:
        Path: Path to the saved file
    """
    if results_dir is None:
        results_dir = PROJECT_ROOT / "data" / "results"
    else:
        results_dir = Path(results_dir)
    
    # Create directory if it doesn't exist
    results_dir.mkdir(exist_ok=True, parents=True)
    
    # Determine file extension and save format
    if filename.endswith('.csv'):
        if isinstance(results, pd.DataFrame):
            file_path = results_dir / filename
            results.to_csv(file_path)
        else:
            # Convert dict to DataFrame, then save
            file_path = results_dir / filename
            pd.DataFrame([results]).to_csv(file_path, index=False)
    elif filename.endswith('.json'):
        import json
        file_path = results_dir / filename
        with open(file_path, 'w') as f:
            if isinstance(results, pd.DataFrame):
                json.dump(results.to_dict(orient='records'), f, indent=4)
            else:
                json.dump(results, f, indent=4)
    else:
        # Default to CSV
        filename = filename + '.csv'
        file_path = results_dir / filename
        if isinstance(results, pd.DataFrame):
            results.to_csv(file_path)
        else:
            pd.DataFrame([results]).to_csv(file_path, index=False)
    
    logger.info(f"Results saved to {file_path}")
    
    return file_path

def generate_var_report(returns, confidence_levels=None, investment_value=1000000, lookback_days=None):
    """
    Generate a comprehensive VaR analysis report.
    
    Args:
        returns (pandas.Series): Historical returns data
        confidence_levels (list, optional): List of confidence levels
        investment_value (float, optional): Current portfolio value
        lookback_days (int, optional): Number of days to look back
        
    Returns:
        dict: Dictionary with comprehensive VaR analysis
    """
    # Load configuration if parameters not provided
    config = load_model_config()
    
    if confidence_levels is None:
        confidence_levels = config['confidence_levels']
    
    if lookback_days is None:
        lookback_days = config['lookback_days']
    
    # Calculate VaR for different confidence levels
    var_by_conf = calculate_var_by_confidence(
        returns=returns,
        confidence_levels=confidence_levels,
        investment_value=investment_value,
        lookback_days=lookback_days
    )
    
    # Calculate rolling VaR for default confidence level
    default_conf = config['default_confidence']
    rolling_var = calculate_rolling_var(
        returns=returns,
        window_size=lookback_days,
        confidence_level=default_conf,
        investment_value=investment_value
    )
    
    # Analyze VaR exceedances
    exceedance_analysis = analyze_var_exceedances(
        returns=returns,
        confidence_level=default_conf,
        lookback_days=lookback_days
    )
    
    # Calculate time-scaled VaR for different horizons
    time_horizons = [1, 5, 10, 20]  # days
    time_scaled_var = []
    
    for horizon in time_horizons:
        for conf_level in confidence_levels:
            var_value = calculate_historical_var_time_scaled(
                returns=returns,
                confidence_level=conf_level,
                investment_value=investment_value,
                time_horizon=horizon,
                lookback_days=lookback_days
            )
            
            time_scaled_var.append({
                'confidence_level': conf_level,
                'time_horizon': horizon,
                'var_value': var_value,
                'var_pct': var_value / investment_value * 100
            })
    
    time_scaled_var_df = pd.DataFrame(time_scaled_var)
    
    # Compile report
    report = {
        "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "var_by_confidence": var_by_conf.to_dict(orient='records'),
        "rolling_var": rolling_var.to_dict(),
        "exceedance_analysis": exceedance_analysis,
        "time_scaled_var": time_scaled_var_df.to_dict(orient='records'),
        "parameters": {
            "investment_value": investment_value,
            "lookback_days": lookback_days,
            "confidence_levels": confidence_levels,
            "time_horizons": time_horizons
        },
        "portfolio_stats": {
            "mean_return": returns.mean(),
            "std_dev": returns.std(),
            "min_return": returns.min(),
            "max_return": returns.max(),
            "skewness": returns.skew(),
            "kurtosis": returns.kurtosis()
        }
    }
    
    return report

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Example usage
    try:
        # Load portfolio returns
        data_dir = PROJECT_ROOT / "data" / "processed"
        returns_file = data_dir / "portfolio_returns.csv"
        
        if not returns_file.exists():
            logger.error("Portfolio returns file not found. Please run process_data.py first.")
            exit(1)
        
        # Load returns data
        portfolio_returns = pd.read_csv(returns_file, index_col=0, parse_dates=True)
        
        # Convert DataFrame to Series if needed
        if isinstance(portfolio_returns, pd.DataFrame) and portfolio_returns.shape[1] == 1:
            portfolio_returns = portfolio_returns.iloc[:, 0]
        
        # Calculate VaR
        logger.info("Calculating Historical VaR...")
        config = load_model_config()
        default_confidence = config['default_confidence']
        
        var_value = calculate_historical_var(
            returns=portfolio_returns,
            confidence_level=default_confidence,
            investment_value=1000000
        )
        
        logger.info(f"Historical VaR ({default_confidence*100}%): ${var_value:,.2f}")
        
        # Calculate VaR for multiple confidence levels
        logger.info("\nVaR by confidence level:")
        var_by_confidence = calculate_var_by_confidence(portfolio_returns)
        logger.info(var_by_confidence)
        
        # Calculate rolling VaR
        logger.info("\nCalculating rolling VaR...")
        rolling_var = calculate_rolling_var(
            returns=portfolio_returns,
            window_size=252,
            confidence_level=default_confidence
        )
        logger.info(f"Rolling VaR calculated for {len(rolling_var)} days")
        
        # Analyze VaR exceedances
        logger.info("\nAnalyzing VaR exceedances...")
        exceedance_analysis = analyze_var_exceedances(portfolio_returns)
        logger.info(f"Exceedance rate: {exceedance_analysis['exceedance_rate']:.2%} (Expected: {exceedance_analysis['expected_rate']:.2%})")
        logger.info(f"Kupiec test p-value: {exceedance_analysis['kupiec_p_value']:.4f} (Result: {exceedance_analysis['kupiec_test_result']})")
        
        # Generate comprehensive report
        logger.info("\nGenerating comprehensive VaR report...")
        report = generate_var_report(portfolio_returns)
        
        # Save results
        results_dir = PROJECT_ROOT / "data" / "results"
        save_var_results(var_by_confidence, "var_by_confidence.csv", results_dir)
        save_var_results(exceedance_analysis, "var_exceedance_analysis.json", results_dir)
        save_var_results(report, "var_comprehensive_report.json", results_dir)
        
        logger.info("Historical VaR analysis completed successfully.")
        
    except Exception as e:
        logger.error(f"Error in Historical VaR analysis: {e}", exc_info=True)