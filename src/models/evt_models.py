"""
Extreme Value Theory (EVT) models for tail risk assessment.
This module implements methods based on Extreme Value Theory for more accurate 
estimation of tail risks beyond traditional VaR.
"""

import numpy as np
import pandas as pd
from scipy import stats
import logging
from typing import Tuple, Union, Dict, List, Optional

# Set up logger
logger = logging.getLogger(__name__)

def fit_pot_model(returns: pd.Series, threshold: Optional[float] = None, threshold_method: str = 'percentile',
                 percentile: float = 0.05) -> Dict:
    """
    Fit a Peaks-Over-Threshold (POT) model to the return data.
    
    Args:
        returns (pandas.Series): Historical returns data (negative returns represent losses)
        threshold (float, optional): Threshold above which to consider returns as extreme
        threshold_method (str): Method to determine threshold ('percentile', 'fixed')
        percentile (float): Percentile to use if threshold_method is 'percentile'
        
    Returns:
        dict: Dictionary containing fitted GPD parameters
    """
    # Ensure we're working with negative returns (losses)
    losses = -returns
    
    # Determine threshold
    if threshold is None:
        if threshold_method == 'percentile':
            threshold = np.percentile(losses, 100 * (1 - percentile))
        else:
            # Default: use 95th percentile as threshold
            threshold = np.percentile(losses, 95)
    
    # Extract exceedances
    exceedances = losses[losses > threshold] - threshold
    
    if len(exceedances) < 10:
        logger.warning(f"Too few exceedances ({len(exceedances)}) for reliable EVT estimation.")
        return {
            'threshold': threshold,
            'exceedances_count': len(exceedances),
            'shape': np.nan,
            'scale': np.nan,
            'success': False
        }
    
    # Fit Generalized Pareto Distribution (GPD)
    try:
        # MLE estimation of GPD parameters
        shape, loc, scale = stats.genpareto.fit(exceedances, floc=0)  # Fix location at 0
        
        result = {
            'threshold': threshold,
            'exceedances_count': len(exceedances),
            'shape': shape,
            'scale': scale,
            'success': True
        }
        
        logger.info(f"Successfully fitted POT model with {len(exceedances)} exceedances.")
        logger.info(f"GPD parameters: shape={shape:.4f}, scale={scale:.4f}")
        
        return result
    
    except Exception as e:
        logger.error(f"Error fitting GPD distribution: {e}")
        return {
            'threshold': threshold,
            'exceedances_count': len(exceedances),
            'shape': np.nan,
            'scale': np.nan,
            'success': False
        }

def calculate_evt_var(returns: pd.Series, confidence_level: float = 0.99, 
                     investment_value: float = 1000000, pot_params: Optional[Dict] = None) -> float:
    """
    Calculate Value-at-Risk (VaR) using Extreme Value Theory (EVT).
    
    Args:
        returns (pandas.Series): Historical returns data
        confidence_level (float): Confidence level for VaR calculation
        investment_value (float): Current portfolio value
        pot_params (dict, optional): Pre-fitted POT model parameters
        
    Returns:
        float: EVT-based VaR estimate
    """
    n = len(returns)
    losses = -returns
    
    # Fit POT model if parameters not provided
    if pot_params is None or not pot_params.get('success', False):
        # Use the 90th percentile as threshold for standard POT model
        pot_params = fit_pot_model(returns, threshold_method='percentile', percentile=0.1)
    
    if not pot_params.get('success', False):
        logger.warning("POT model fitting failed. Falling back to historical VaR.")
        # Fall back to historical VaR
        var_percentile = 1 - confidence_level
        return abs(np.percentile(returns, var_percentile * 100) * investment_value)
    
    # Extract POT parameters
    threshold = pot_params['threshold']
    shape = pot_params['shape']
    scale = pot_params['scale']
    n_exceedances = pot_params['exceedances_count']
    
    # Calculate exceedance probability
    p_exceed = n_exceedances / n
    
    # Calculate VaR quantile
    p = confidence_level
    q = (1 - p) / p_exceed
    
    if shape == 0:  # Special case when shape is 0 (exponential distribution)
        var_loss = threshold + scale * np.log(q)
    else:
        var_loss = threshold + (scale / shape) * (q**shape - 1)
    
    # Convert to VaR
    var_value = var_loss * investment_value
    
    return var_value

def calculate_evt_es(returns: pd.Series, confidence_level: float = 0.99, 
                    investment_value: float = 1000000, pot_params: Optional[Dict] = None) -> float:
    """
    Calculate Expected Shortfall (ES) using Extreme Value Theory (EVT).
    
    Args:
        returns (pandas.Series): Historical returns data
        confidence_level (float): Confidence level for ES calculation
        investment_value (float): Current portfolio value
        pot_params (dict, optional): Pre-fitted POT model parameters
        
    Returns:
        float: EVT-based ES estimate
    """
    n = len(returns)
    losses = -returns
    
    # Fit POT model if parameters not provided
    if pot_params is None or not pot_params.get('success', False):
        # Use the 90th percentile as threshold for standard POT model
        pot_params = fit_pot_model(returns, threshold_method='percentile', percentile=0.1)
    
    if not pot_params.get('success', False):
        logger.warning("POT model fitting failed. Falling back to historical ES.")
        # Fall back to historical ES
        var_percentile = 1 - confidence_level
        var_threshold = np.percentile(returns, var_percentile * 100)
        es_value = returns[returns <= var_threshold].mean()
        return abs(es_value * investment_value)
    
    # Extract POT parameters
    threshold = pot_params['threshold']
    shape = pot_params['shape']
    scale = pot_params['scale']
    n_exceedances = pot_params['exceedances_count']
    
    # Calculate exceedance probability
    p_exceed = n_exceedances / n
    
    # Calculate VaR quantile
    p = confidence_level
    q = (1 - p) / p_exceed
    
    # Calculate VaR first (needed for ES)
    if shape == 0:  # Special case when shape is 0 (exponential distribution)
        var_loss = threshold + scale * np.log(q)
    else:
        var_loss = threshold + (scale / shape) * (q**shape - 1)
    
    # Calculate Expected Shortfall
    if shape >= 1:
        logger.warning("GPD shape parameter >= 1, ES is infinite. Falling back to historical ES.")
        # Fall back to historical ES
        var_percentile = 1 - confidence_level
        var_threshold = np.percentile(returns, var_percentile * 100)
        es_value = returns[returns <= var_threshold].mean()
        return abs(es_value * investment_value)
    elif shape == 0:
        es_loss = var_loss + scale
    else:
        es_loss = (var_loss + scale - shape * threshold) / (1 - shape)
    
    # Convert to ES
    es_value = es_loss * investment_value
    
    return es_value

def estimate_return_period(model_params: Dict, threshold_loss: float) -> float:
    """
    Estimate the return period (in days) for a loss exceeding a certain threshold.
    
    Args:
        model_params (dict): POT model parameters
        threshold_loss (float): Loss threshold to estimate return period for
        
    Returns:
        float: Estimated return period in days
    """
    # Extract POT parameters
    threshold = model_params['threshold']
    shape = model_params['shape']
    scale = model_params['scale']
    n_exceedances = model_params['exceedances_count']
    n_total = model_params.get('n_total', 252 * 10)  # Default: 10 years of data
    
    # Calculate exceedance probability
    p_exceed = n_exceedances / n_total
    
    # Check if loss is below threshold
    if threshold_loss <= threshold:
        logger.warning(f"Loss ({threshold_loss}) is below the threshold ({threshold}). Cannot estimate return period.")
        return np.nan
    
    # Calculate probability of exceeding the threshold loss
    if shape == 0:  # Special case
        p_exceed_loss = p_exceed * np.exp(-(threshold_loss - threshold) / scale)
    else:
        p_exceed_loss = p_exceed * (1 + shape * (threshold_loss - threshold) / scale) ** (-1 / shape)
    
    # Calculate return period
    if p_exceed_loss > 0:
        return_period = 1 / p_exceed_loss
    else:
        return_period = np.inf
    
    return return_period

def plot_threshold_selection(returns: pd.Series, thresholds: List[float]) -> Dict:
    """
    Analyze the stability of GPD parameters across different thresholds.
    This helps in selecting an appropriate threshold for the POT model.
    
    Args:
        returns (pandas.Series): Historical returns data
        thresholds (list): List of thresholds to analyze
        
    Returns:
        dict: Dictionary with threshold analysis results
    """
    results = {
        'thresholds': thresholds,
        'shape_params': [],
        'scale_params': [],
        'exceedances': []
    }
    
    for threshold in thresholds:
        pot_params = fit_pot_model(returns, threshold=threshold, threshold_method='fixed')
        
        results['shape_params'].append(pot_params['shape'])
        results['scale_params'].append(pot_params['scale'])
        results['exceedances'].append(pot_params['exceedances_count'])
    
    # Calculate modified scale parameter (scale - shape * threshold)
    modified_scale = [scale - shape * threshold for scale, shape, threshold in 
                      zip(results['scale_params'], results['shape_params'], thresholds)]
    
    results['modified_scale'] = modified_scale
    
    return results

def run_evt_analysis(returns: pd.Series, confidence_levels: List[float] = [0.95, 0.99, 0.999],
                    investment_value: float = 1000000) -> pd.DataFrame:
    """
    Run a comprehensive EVT analysis for multiple confidence levels.
    
    Args:
        returns (pandas.Series): Historical returns data
        confidence_levels (list): List of confidence levels to analyze
        investment_value (float): Current portfolio value
        
    Returns:
        pandas.DataFrame: DataFrame with EVT analysis results
    """
    # Fit POT model
    pot_params = fit_pot_model(returns)
    
    if not pot_params.get('success', False):
        logger.warning("POT model fitting failed. Analysis may not be reliable.")
    
    # Calculate VaR and ES for each confidence level
    results = []
    
    for conf_level in confidence_levels:
        # Calculate EVT VaR
        evt_var = calculate_evt_var(
            returns=returns,
            confidence_level=conf_level,
            investment_value=investment_value,
            pot_params=pot_params
        )
        
        # Calculate EVT ES
        evt_es = calculate_evt_es(
            returns=returns,
            confidence_level=conf_level,
            investment_value=investment_value,
            pot_params=pot_params
        )
        
        # Calculate return periods (in years, assuming 252 trading days)
        var_return_period = estimate_return_period(
            model_params={**pot_params, 'n_total': len(returns)},
            threshold_loss=evt_var / investment_value
        ) / 252  # Convert to years
        
        # Add to results
        results.append({
            'confidence_level': conf_level,
            'evt_var': evt_var,
            'evt_es': evt_es,
            'evt_var_pct': evt_var / investment_value * 100,
            'evt_es_pct': evt_es / investment_value * 100,
            'var_return_period_years': var_return_period
        })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    return results_df

if __name__ == "__main__":
    # Example usage
    import matplotlib.pyplot as plt
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Generate sample returns
    np.random.seed(42)
    n = 1000
    
    # Generate normal returns with occasional extreme values
    returns = np.random.normal(0.0005, 0.01, n)
    
    # Add some extreme values to simulate fat tails
    extreme_indices = np.random.choice(n, 20, replace=False)
    returns[extreme_indices] = np.random.normal(-0.05, 0.02, 20)  # Negative extreme values
    
    # Convert to Series
    returns_series = pd.Series(returns)
    
    # Run EVT analysis
    evt_results = run_evt_analysis(returns_series)
    print("EVT Analysis Results:")
    print(evt_results)
    
    # Analyze different thresholds
    thresholds = np.linspace(0.01, 0.05, 10)
    threshold_analysis = plot_threshold_selection(returns_series, thresholds)
    
    # Plot threshold analysis
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(threshold_analysis['thresholds'], threshold_analysis['shape_params'], 'o-')
    plt.title('Shape Parameter vs Threshold')
    plt.xlabel('Threshold')
    plt.ylabel('Shape Parameter')
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.plot(threshold_analysis['thresholds'], threshold_analysis['scale_params'], 'o-')
    plt.title('Scale Parameter vs Threshold')
    plt.xlabel('Threshold')
    plt.ylabel('Scale Parameter')
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    plt.plot(threshold_analysis['thresholds'], threshold_analysis['modified_scale'], 'o-')
    plt.title('Modified Scale Parameter vs Threshold')
    plt.xlabel('Threshold')
    plt.ylabel('Modified Scale')
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    plt.plot(threshold_analysis['thresholds'], threshold_analysis['exceedances'], 'o-')
    plt.title('Number of Exceedances vs Threshold')
    plt.xlabel('Threshold')
    plt.ylabel('Number of Exceedances')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()