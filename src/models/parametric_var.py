"""
Parametric Value-at-Risk (VaR) implementation.
This module calculates VaR using parametric methods, primarily the normal distribution.
"""

import json
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import logging

# Get project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"

# Set up logger
logger = logging.getLogger(__name__)

def load_model_config():
    """
    Load model configuration from config file.
    
    Returns:
        dict: Model configuration for parametric VaR
    """
    try:
        with open(CONFIG_DIR / "model_config.json", 'r') as f:
            model_config = json.load(f)
        
        # Check if parametric config exists, otherwise use historical
        if 'parametric' in model_config['var_models']:
            return model_config['var_models']['parametric']
        else:
            logger.warning("Parametric VaR configuration not found in model_config.json. Using historical configuration.")
            return model_config['var_models']['historical']
    except Exception as e:
        logger.error(f"Error loading model configuration: {e}")
        # Return default configuration
        return {
            'confidence_levels': [0.90, 0.95, 0.99],
            'default_confidence': 0.95,
            'time_horizon': 1
        }

def calculate_parametric_var(returns, confidence_level=None, investment_value=1000000):
    """
    Calculate Value-at-Risk using the parametric method assuming a normal distribution.
    
    Args:
        returns (pandas.Series or pandas.DataFrame): Historical returns data
        confidence_level (float, optional): Confidence level (e.g., 0.95 for 95% confidence)
        investment_value (float, optional): Current portfolio value
        
    Returns:
        float: Value-at-Risk estimate
    """
    # Load configuration if parameters not provided
    if confidence_level is None:
        config = load_model_config()
        confidence_level = config['default_confidence']
    
    # If returns is a DataFrame (multiple assets), raise error
    if isinstance(returns, pd.DataFrame):
        if returns.shape[1] > 1:
            raise ValueError("Please provide portfolio returns as a pandas Series, not a DataFrame with multiple columns")
        else:
            # Convert DataFrame with one column to Series
            returns = returns.iloc[:, 0]
    
    # Calculate mean and standard deviation of returns
    mu = returns.mean()
    sigma = returns.std()
    
    # Calculate z-score for the given confidence level
    z_score = stats.norm.ppf(1 - confidence_level)
    
    # Calculate VaR
    var_return = mu + z_score * sigma
    
    # Convert to monetary value
    var_value = abs(var_return * investment_value)
    
    return var_value

def calculate_parametric_expected_shortfall(returns, confidence_level=None, investment_value=1000000):
    """
    Calculate Expected Shortfall (ES) using the parametric method assuming a normal distribution.
    ES is the expected loss given that we are in the tail beyond VaR.
    
    Args:
        returns (pandas.Series or pandas.DataFrame): Historical returns data
        confidence_level (float, optional): Confidence level (e.g., 0.95 for 95% confidence)
        investment_value (float, optional): Current portfolio value
        
    Returns:
        float: Expected Shortfall estimate
    """
    # Load configuration if parameters not provided
    if confidence_level is None:
        config = load_model_config()
        confidence_level = config['default_confidence']
    
    # If returns is a DataFrame (multiple assets), raise error
    if isinstance(returns, pd.DataFrame):
        if returns.shape[1] > 1:
            raise ValueError("Please provide portfolio returns as a pandas Series, not a DataFrame with multiple columns")
        else:
            # Convert DataFrame with one column to Series
            returns = returns.iloc[:, 0]
    
    # Calculate mean and standard deviation of returns
    mu = returns.mean()
    sigma = returns.std()
    
    # Calculate z-score for the given confidence level
    z_score = stats.norm.ppf(1 - confidence_level)
    
    # Calculate Expected Shortfall for normal distribution
    # ES = μ - σ * φ(z) / (1-α) where φ is the PDF of the standard normal distribution
    es_return = mu - sigma * stats.norm.pdf(z_score) / (1 - confidence_level)
    
    # Convert to monetary value
    es_value = abs(es_return * investment_value)
    
    return es_value

def calculate_parametric_var_t(returns, confidence_level=None, investment_value=1000000, degrees_of_freedom=None):
    """
    Calculate Value-at-Risk using the parametric method assuming a Student's t-distribution.
    This can better account for fat tails in the return distribution.
    
    Args:
        returns (pandas.Series or pandas.DataFrame): Historical returns data
        confidence_level (float, optional): Confidence level (e.g., 0.95 for 95% confidence)
        investment_value (float, optional): Current portfolio value
        degrees_of_freedom (int, optional): Degrees of freedom for t-distribution. If None, estimated from data.
        
    Returns:
        float: Value-at-Risk estimate
    """
    # Load configuration if parameters not provided
    if confidence_level is None:
        config = load_model_config()
        confidence_level = config['default_confidence']
    
    # If returns is a DataFrame (multiple assets), raise error
    if isinstance(returns, pd.DataFrame):
        if returns.shape[1] > 1:
            raise ValueError("Please provide portfolio returns as a pandas Series, not a DataFrame with multiple columns")
        else:
            # Convert DataFrame with one column to Series
            returns = returns.iloc[:, 0]
    
    # Calculate mean and standard deviation of returns
    mu = returns.mean()
    sigma = returns.std()
    
    # Estimate degrees of freedom if not provided
    if degrees_of_freedom is None:
        # Simple estimate based on kurtosis
        kurtosis = returns.kurtosis()
        if kurtosis <= 0:
            # If kurtosis is non-positive, use normal distribution
            return calculate_parametric_var(returns, confidence_level, investment_value)
        
        # For t-distribution, df = 4 + 6 / kurtosis
        degrees_of_freedom = max(3, min(30, 4 + 6 / kurtosis))
        logger.info(f"Estimated degrees of freedom: {degrees_of_freedom:.2f}")
    
    # Calculate t-value for the given confidence level
    t_value = stats.t.ppf(1 - confidence_level, degrees_of_freedom)
    
    # Calculate VaR
    var_return = mu + t_value * sigma
    
    # Convert to monetary value
    var_value = abs(var_return * investment_value)
    
    return var_value

def var_confidence_interval(returns, confidence_level=None, investment_value=1000000, alpha=0.05):
    """
    Calculate confidence interval for the VaR estimate.
    
    Args:
        returns (pandas.Series): Historical returns data
        confidence_level (float, optional): Confidence level for VaR (e.g., 0.95 for 95% VaR)
        investment_value (float, optional): Current portfolio value
        alpha (float, optional): Significance level for the confidence interval (e.g., 0.05 for 95% CI)
        
    Returns:
        tuple: Lower and upper bounds of the VaR confidence interval
    """
    # Load configuration if parameters not provided
    if confidence_level is None:
        config = load_model_config()
        confidence_level = config['default_confidence']
    
    # If returns is a DataFrame (multiple assets), raise error
    if isinstance(returns, pd.DataFrame):
        if returns.shape[1] > 1:
            raise ValueError("Please provide portfolio returns as a pandas Series, not a DataFrame with multiple columns")
        else:
            # Convert DataFrame with one column to Series
            returns = returns.iloc[:, 0]
    
    # Sort returns to identify the VaR quantile
    sorted_returns = returns.sort_values()
    n = len(sorted_returns)
    
    # Find the index for the VaR quantile
    q = 1 - confidence_level
    var_index = int(np.floor(q * n))
    
    # Calculate the standard error of the quantile estimator
    # Based on the variance of the order statistic
    se = np.sqrt((q * (1 - q)) / (n * stats.norm.pdf(stats.norm.ppf(q))**2))
    
    # Calculate the confidence interval indices
    lower_idx = max(0, int(np.floor((q - stats.norm.ppf(1 - alpha/2) * se) * n)))
    upper_idx = min(n - 1, int(np.ceil((q + stats.norm.ppf(1 - alpha/2) * se) * n)))
    
    # Get the returns at these indices
    lower_var_return = sorted_returns.iloc[upper_idx]  # Note: lower return = higher VaR
    upper_var_return = sorted_returns.iloc[lower_idx]  # Note: higher return = lower VaR
    
    # Convert to monetary value
    lower_var = abs(lower_var_return * investment_value)
    upper_var = abs(upper_var_return * investment_value)
    
    return (lower_var, upper_var)

def calculate_var_by_distribution(returns, confidence_levels=None, investment_value=1000000):
    """
    Calculate VaR using different distributional assumptions.
    
    Args:
        returns (pandas.Series): Historical returns data
        confidence_levels (list, optional): List of confidence levels
        investment_value (float, optional): Current portfolio value
        
    Returns:
        pandas.DataFrame: VaR estimates for different distributions and confidence levels
    """
    # Load configuration if parameters not provided
    config = load_model_config()
    
    if confidence_levels is None:
        confidence_levels = config['confidence_levels']
    
    # Calculate VaR for each confidence level and distribution
    var_results = []
    
    for conf_level in confidence_levels:
        # Normal distribution
        normal_var = calculate_parametric_var(
            returns=returns,
            confidence_level=conf_level,
            investment_value=investment_value
        )
        
        normal_es = calculate_parametric_expected_shortfall(
            returns=returns,
            confidence_level=conf_level,
            investment_value=investment_value
        )
        
        # t-distribution
        t_var = calculate_parametric_var_t(
            returns=returns,
            confidence_level=conf_level,
            investment_value=investment_value
        )
        
        # Historical (for comparison)
        from .historical_var import calculate_historical_var, calculate_conditional_var
        
        hist_var = calculate_historical_var(
            returns=returns,
            confidence_level=conf_level,
            investment_value=investment_value
        )
        
        hist_es = calculate_conditional_var(
            returns=returns,
            confidence_level=conf_level,
            investment_value=investment_value
        )
        
        var_results.append({
            'confidence_level': conf_level,
            'normal_var': normal_var,
            'normal_es': normal_es,
            't_var': t_var,
            'historical_var': hist_var,
            'historical_es': hist_es
        })
    
    # Convert to DataFrame
    var_df = pd.DataFrame(var_results)
    
    return var_df

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
        logger.info("Calculating Parametric VaR...")
        config = load_model_config()
        default_confidence = config['default_confidence']
        
        var_value = calculate_parametric_var(
            returns=portfolio_returns,
            confidence_level=default_confidence,
            investment_value=1000000
        )
        
        logger.info(f"Parametric VaR ({default_confidence*100}%): ${var_value:,.2f}")
        
        # Calculate Expected Shortfall
        es_value = calculate_parametric_expected_shortfall(
            returns=portfolio_returns,
            confidence_level=default_confidence,
            investment_value=1000000
        )
        
        logger.info(f"Parametric Expected Shortfall: ${es_value:,.2f}")
        
        # Calculate VaR using t-distribution
        t_var = calculate_parametric_var_t(
            returns=portfolio_returns,
            confidence_level=default_confidence,
            investment_value=1000000
        )
        
        logger.info(f"Parametric VaR (t-distribution): ${t_var:,.2f}")
        
        # Calculate VaR confidence interval
        var_ci = var_confidence_interval(
            returns=portfolio_returns,
            confidence_level=default_confidence,
            investment_value=1000000
        )
        
        logger.info(f"VaR 95% Confidence Interval: (${var_ci[0]:,.2f}, ${var_ci[1]:,.2f})")
        
        # Compare distributions
        var_comparison = calculate_var_by_distribution(portfolio_returns)
        logger.info("\nVaR by Distribution:")
        logger.info(var_comparison)
        
    except Exception as e:
        logger.error(f"Error calculating Parametric VaR: {e}", exc_info=True)