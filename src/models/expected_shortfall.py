"""
Expected Shortfall implementation.
This module calculates Expected Shortfall (ES), also known as Conditional VaR (CVaR).
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
        dict: Model configuration for Expected Shortfall
    """
    try:
        with open(CONFIG_DIR / "model_config.json", 'r') as f:
            model_config = json.load(f)
        
        # Check if expected_shortfall config exists, otherwise use historical
        if 'expected_shortfall' in model_config['var_models']:
            return model_config['var_models']['expected_shortfall']
        else:
            logger.warning("Expected Shortfall configuration not found in model_config.json. Using historical configuration.")
            return model_config['var_models']['historical']
    except Exception as e:
        logger.error(f"Error loading model configuration: {e}")
        # Return default configuration
        return {
            'confidence_levels': [0.90, 0.95, 0.99],
            'default_confidence': 0.95
        }

def calculate_expected_shortfall(returns, confidence_level=None, investment_value=1000000, method='historical'):
    """
    Calculate Expected Shortfall (ES), also known as Conditional VaR (CVaR).
    ES is the expected loss given that we are in the tail beyond VaR.
    
    Args:
        returns (pandas.Series or pandas.DataFrame): Historical returns data
        confidence_level (float, optional): Confidence level (e.g., 0.95 for 95% confidence)
        investment_value (float, optional): Current portfolio value
        method (str, optional): Method to use ('historical' or 'parametric')
        
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
    
    if method.lower() == 'historical':
        # Calculate the VaR threshold
        var_percentile = 1 - confidence_level
        var_threshold = np.percentile(returns, var_percentile * 100)
        
        # Calculate ES as the mean of returns beyond VaR
        tail_returns = returns[returns <= var_threshold]
        
        if len(tail_returns) == 0:
            logger.warning(f"No returns found beyond VaR threshold. Using VaR instead.")
            es_return = var_threshold
        else:
            es_return = tail_returns.mean()
        
        # Convert to monetary value
        es_value = abs(es_return * investment_value)
        
    elif method.lower() == 'parametric':
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
        
    else:
        raise ValueError("Method must be either 'historical' or 'parametric'")
    
    return es_value

def compare_var_and_es(returns, confidence_level=None, investment_value=1000000):
    """
    Compare VaR and ES calculated using different methods.
    
    Args:
        returns (pandas.Series): Historical returns data
        confidence_level (float, optional): Confidence level (e.g., 0.95 for 95% confidence)
        investment_value (float, optional): Current portfolio value
        
    Returns:
        pandas.DataFrame: VaR and ES estimates for different methods
    """
    # Load configuration if parameters not provided
    if confidence_level is None:
        config = load_model_config()
        confidence_level = config['default_confidence']
    
    # Import VaR functions
    from .historical_var import calculate_historical_var, calculate_conditional_var
    from .parametric_var import calculate_parametric_var, calculate_parametric_expected_shortfall
    
    # Calculate VaR and ES using different methods
    results = {
        'Method': ['Historical', 'Parametric'],
        'VaR': [
            calculate_historical_var(returns, confidence_level, investment_value),
            calculate_parametric_var(returns, confidence_level, investment_value)
        ],
        'ES': [
            calculate_conditional_var(returns, confidence_level, investment_value),
            calculate_parametric_expected_shortfall(returns, confidence_level, investment_value)
        ]
    }
    
    # Calculate ES/VaR ratios
    results['ES/VaR Ratio'] = [
        results['ES'][0] / results['VaR'][0],
        results['ES'][1] / results['VaR'][1]
    ]
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Add percentage of investment
    results_df['VaR %'] = results_df['VaR'] / investment_value * 100
    results_df['ES %'] = results_df['ES'] / investment_value * 100
    
    return results_df

def calculate_es_by_confidence(returns, confidence_levels=None, investment_value=1000000, method='historical'):
    """
    Calculate ES for multiple confidence levels.
    
    Args:
        returns (pandas.Series): Historical returns data
        confidence_levels (list, optional): List of confidence levels
        investment_value (float, optional): Current portfolio value
        method (str, optional): Method to use ('historical' or 'parametric')
        
    Returns:
        pandas.DataFrame: ES estimates for different confidence levels
    """
    # Load configuration if parameters not provided
    config = load_model_config()
    
    if confidence_levels is None:
        confidence_levels = config['confidence_levels']
    
    # Calculate ES for each confidence level
    es_results = []
    
    for conf_level in confidence_levels:
        es_value = calculate_expected_shortfall(
            returns=returns,
            confidence_level=conf_level,
            investment_value=investment_value,
            method=method
        )
        
        # Import VaR functions
        if method.lower() == 'historical':
            from .historical_var import calculate_historical_var
            var_value = calculate_historical_var(
                returns=returns,
                confidence_level=conf_level,
                investment_value=investment_value
            )
        else:
            from .parametric_var import calculate_parametric_var
            var_value = calculate_parametric_var(
                returns=returns,
                confidence_level=conf_level,
                investment_value=investment_value
            )
        
        es_results.append({
            'confidence_level': conf_level,
            'var_value': var_value,
            'es_value': es_value,
            'es_var_ratio': es_value / var_value,
            'var_pct': var_value / investment_value * 100,
            'es_pct': es_value / investment_value * 100
        })
    
    # Convert to DataFrame
    es_df = pd.DataFrame(es_results)
    
    return es_df

def calculate_es_contribution(returns, weights, confidence_level=None, investment_value=1000000):
    """
    Calculate the contribution of each asset to the portfolio Expected Shortfall.
    
    Args:
        returns (pandas.DataFrame): Historical returns data for all assets
        weights (list or ndarray): Portfolio weights
        confidence_level (float, optional): Confidence level (e.g., 0.95 for 95% confidence)
        investment_value (float, optional): Current portfolio value
        
    Returns:
        pandas.DataFrame: ES contribution for each asset
    """
    # Load configuration if parameters not provided
    if confidence_level is None:
        config = load_model_config()
        confidence_level = config['default_confidence']
    
    # Convert weights to numpy array if it's a list
    weights = np.array(weights)
    
    # Calculate portfolio returns
    portfolio_returns = returns.dot(weights)
    
    # Calculate VaR threshold
    var_percentile = 1 - confidence_level
    var_threshold = np.percentile(portfolio_returns, var_percentile * 100)
    
    # Identify the tail scenarios
    tail_scenarios = portfolio_returns <= var_threshold
    
    # Calculate ES
    if tail_scenarios.sum() == 0:
        logger.warning(f"No returns found beyond VaR threshold. Using all returns instead.")
        tail_scenarios = portfolio_returns <= portfolio_returns.mean()
    
    # Calculate the average loss in the tail scenarios
    es_portfolio = portfolio_returns[tail_scenarios].mean()
    
    # Calculate the contribution of each asset
    contributions = []
    
    for i, asset in enumerate(returns.columns):
        # Calculate the average contribution to losses in tail scenarios
        contribution = weights[i] * returns[asset][tail_scenarios].mean()
        
        # Calculate the percentage contribution
        pct_contribution = contribution / es_portfolio * 100
        
        contributions.append({
            'asset': asset,
            'weight': weights[i],
            'contribution': contribution,
            'contribution_monetary': abs(contribution * investment_value),
            'pct_contribution': pct_contribution
        })
    
    # Convert to DataFrame
    contributions_df = pd.DataFrame(contributions)
    
    # Sort by absolute contribution
    contributions_df = contributions_df.sort_values(by='contribution_monetary', ascending=False)
    
    return contributions_df

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
        
        # Calculate ES
        logger.info("Calculating Expected Shortfall...")
        config = load_model_config()
        default_confidence = config['default_confidence']
        
        # Calculate historical ES
        hist_es = calculate_expected_shortfall(
            returns=portfolio_returns,
            confidence_level=default_confidence,
            investment_value=1000000,
            method='historical'
        )
        
        logger.info(f"Historical Expected Shortfall ({default_confidence*100}%): ${hist_es:,.2f}")
        
        # Calculate parametric ES
        param_es = calculate_expected_shortfall(
            returns=portfolio_returns,
            confidence_level=default_confidence,
            investment_value=1000000,
            method='parametric'
        )
        
        logger.info(f"Parametric Expected Shortfall ({default_confidence*100}%): ${param_es:,.2f}")
        
        # Compare VaR and ES
        comparison = compare_var_and_es(
            returns=portfolio_returns,
            confidence_level=default_confidence,
            investment_value=1000000
        )
        
        logger.info("\nVaR and ES Comparison:")
        logger.info(comparison)
        
        # Calculate ES for different confidence levels
        es_by_confidence = calculate_es_by_confidence(
            returns=portfolio_returns,
            investment_value=1000000
        )
        
        logger.info("\nExpected Shortfall by Confidence Level:")
        logger.info(es_by_confidence)
        
    except Exception as e:
        logger.error(f"Error calculating Expected Shortfall: {e}", exc_info=True)