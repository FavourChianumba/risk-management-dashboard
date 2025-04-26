"""
Multivariate simulation module for Value-at-Risk (VaR) calculations.
This module provides functions for simulating correlated asset returns and calculating VaR.
"""

import pandas as pd
import numpy as np
from scipy import stats
import logging

# Set up logger
logger = logging.getLogger(__name__)

def generate_correlated_returns(mean_returns, cov_matrix, n_simulations=10000, random_seed=None):
    """
    Generate correlated random returns for multiple assets.
    
    Args:
        mean_returns (numpy.ndarray): Vector of mean returns for each asset
        cov_matrix (numpy.ndarray): Covariance matrix of asset returns
        n_simulations (int): Number of simulations to generate
        random_seed (int, optional): Random seed for reproducibility
        
    Returns:
        numpy.ndarray: Simulated correlated returns (n_simulations x n_assets)
    """
    # Set random seed if specified
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Get number of assets
    n_assets = len(mean_returns)
    
    # Generate random normal samples
    random_samples = np.random.normal(0, 1, size=(n_simulations, n_assets))
    
    # Calculate Cholesky decomposition of covariance matrix
    try:
        cholesky_matrix = np.linalg.cholesky(cov_matrix)
    except np.linalg.LinAlgError:
        # If covariance matrix is not positive definite, make a small adjustment
        logger.warning("Covariance matrix is not positive definite. Adding small diagonal adjustment.")
        min_eigenval = np.min(np.linalg.eigvals(cov_matrix))
        if min_eigenval < 0:
            cov_matrix -= 1.1 * min_eigenval * np.eye(n_assets)
        cholesky_matrix = np.linalg.cholesky(cov_matrix)
    
    # Generate correlated returns using Cholesky decomposition
    correlated_returns = mean_returns + np.dot(random_samples, cholesky_matrix.T)
    
    return correlated_returns

def simulate_portfolio_returns(asset_returns, weights, n_simulations=10000, time_horizon=1, random_seed=None):
    """
    Simulate portfolio returns using multivariate normal distribution.
    
    Args:
        asset_returns (pandas.DataFrame): Historical asset returns
        weights (numpy.ndarray or list): Portfolio weights for each asset
        n_simulations (int): Number of simulations to generate
        time_horizon (int): Time horizon in days
        random_seed (int, optional): Random seed for reproducibility
        
    Returns:
        numpy.ndarray: Simulated portfolio returns
    """
    # Convert weights to numpy array
    weights = np.array(weights)
    
    # Check that weights add up to 1
    if not np.isclose(np.sum(weights), 1.0):
        logger.warning("Portfolio weights do not sum to 1. Normalizing weights.")
        weights = weights / np.sum(weights)
    
    # Extract means and covariance matrix from historical returns
    mean_returns = asset_returns.mean().values
    cov_matrix = asset_returns.cov().values
    
    # Scale means and covariance matrix for time horizon
    mean_returns_scaled = mean_returns * time_horizon
    cov_matrix_scaled = cov_matrix * time_horizon
    
    # Generate correlated asset returns
    simulated_asset_returns = generate_correlated_returns(
        mean_returns=mean_returns_scaled,
        cov_matrix=cov_matrix_scaled,
        n_simulations=n_simulations,
        random_seed=random_seed
    )
    
    # Calculate portfolio returns
    simulated_portfolio_returns = np.dot(simulated_asset_returns, weights)
    
    return simulated_portfolio_returns

def multivariate_monte_carlo_var(asset_returns, weights, confidence_level=0.95, 
                                investment_value=1000000, n_simulations=10000, 
                                time_horizon=1, random_seed=None):
    """
    Calculate Value-at-Risk using multivariate Monte Carlo simulation.
    
    Args:
        asset_returns (pandas.DataFrame): Historical asset returns
        weights (numpy.ndarray or list): Portfolio weights for each asset
        confidence_level (float): Confidence level (e.g., 0.95 for 95% confidence)
        investment_value (float): Current portfolio value
        n_simulations (int): Number of simulations to generate
        time_horizon (int): Time horizon in days
        random_seed (int, optional): Random seed for reproducibility
        
    Returns:
        float: Value-at-Risk estimate
    """
    # Simulate portfolio returns
    simulated_returns = simulate_portfolio_returns(
        asset_returns=asset_returns,
        weights=weights,
        n_simulations=n_simulations,
        time_horizon=time_horizon,
        random_seed=random_seed
    )
    
    # Calculate simulated portfolio values
    simulated_values = investment_value * (1 + simulated_returns)
    
    # Calculate potential losses
    potential_losses = investment_value - simulated_values
    
    # Calculate VaR at specified confidence level
    var_percentile = confidence_level
    var_value = np.percentile(potential_losses, var_percentile * 100)
    
    # Ensure VaR is positive (representing a loss)
    var_value = max(0, var_value)
    
    return var_value

def calculate_asset_var_contributions(asset_returns, weights, confidence_level=0.95, 
                                    investment_value=1000000, n_simulations=10000, 
                                    time_horizon=1, random_seed=None):
    """
    Calculate the contribution of each asset to the total portfolio VaR.
    
    Args:
        asset_returns (pandas.DataFrame): Historical asset returns
        weights (numpy.ndarray or list): Portfolio weights for each asset
        confidence_level (float): Confidence level (e.g., 0.95 for 95% confidence)
        investment_value (float): Current portfolio value
        n_simulations (int): Number of simulations to generate
        time_horizon (int): Time horizon in days
        random_seed (int, optional): Random seed for reproducibility
        
    Returns:
        pandas.DataFrame: VaR contribution of each asset
    """
    # Convert weights to numpy array
    weights = np.array(weights)
    
    # Extract asset names
    asset_names = asset_returns.columns
    
    # Extract means and covariance matrix from historical returns
    mean_returns = asset_returns.mean().values
    cov_matrix = asset_returns.cov().values
    
    # Scale means and covariance matrix for time horizon
    mean_returns_scaled = mean_returns * time_horizon
    cov_matrix_scaled = cov_matrix * time_horizon
    
    # Generate correlated asset returns
    simulated_asset_returns = generate_correlated_returns(
        mean_returns=mean_returns_scaled,
        cov_matrix=cov_matrix_scaled,
        n_simulations=n_simulations,
        random_seed=random_seed
    )
    
    # Calculate portfolio returns
    simulated_portfolio_returns = np.dot(simulated_asset_returns, weights)
    
    # Identify the scenarios beyond VaR threshold
    var_percentile = 1 - confidence_level
    var_threshold = np.percentile(simulated_portfolio_returns, var_percentile * 100)
    tail_scenarios = simulated_portfolio_returns <= var_threshold
    
    # Extract the tail scenarios
    tail_asset_returns = simulated_asset_returns[tail_scenarios]
    
    # Calculate the average loss contribution of each asset in the tail
    contributions = []
    
    for i, asset in enumerate(asset_names):
        # Calculate the average contribution to losses in tail scenarios
        asset_contribution = weights[i] * np.mean(tail_asset_returns[:, i])
        asset_value = weights[i] * investment_value
        
        # Calculate the percentage contribution
        pct_contribution = asset_contribution / np.mean(simulated_portfolio_returns[tail_scenarios]) * 100
        
        contributions.append({
            'asset': asset,
            'weight': weights[i],
            'value': asset_value,
            'mean_return': mean_returns[i],
            'contribution': asset_contribution,
            'contribution_value': abs(asset_contribution * investment_value),
            'pct_contribution': pct_contribution
        })
    
    # Convert to DataFrame
    contributions_df = pd.DataFrame(contributions)
    
    # Sort by absolute contribution
    contributions_df = contributions_df.sort_values(by='contribution_value', ascending=False)
    
    return contributions_df

def run_multivariate_stress_test(asset_returns, weights, stress_scenarios, investment_value=1000000):
    """
    Run multivariate stress tests on the portfolio.
    
    Args:
        asset_returns (pandas.DataFrame): Historical asset returns
        weights (numpy.ndarray or list): Portfolio weights for each asset
        stress_scenarios (dict): Dictionary mapping scenario names to stress parameters
        investment_value (float): Current portfolio value
        
    Returns:
        pandas.DataFrame: Stress test results
    """
    # Convert weights to numpy array
    weights = np.array(weights)
    
    # Extract asset names
    asset_names = asset_returns.columns
    
    # Calculate baseline values
    baseline_mean = np.dot(asset_returns.mean().values, weights)
    baseline_std = np.sqrt(np.dot(weights.T, np.dot(asset_returns.cov().values, weights)))
    
    # Run stress tests
    results = []
    
    for scenario_name, scenario_params in stress_scenarios.items():
        # Apply stress to mean returns
        if 'mean_shift' in scenario_params:
            mean_shift = np.array(scenario_params['mean_shift'])
            stressed_mean = asset_returns.mean().values + mean_shift
        else:
            stressed_mean = asset_returns.mean().values.copy()
        
        # Apply stress to volatility
        if 'vol_multiplier' in scenario_params:
            vol_multiplier = scenario_params['vol_multiplier']
            
            # Get correlation matrix
            corr_matrix = asset_returns.corr().values
            
            # Get standard deviations and apply multiplier
            std_devs = np.sqrt(np.diag(asset_returns.cov().values))
            stressed_std_devs = std_devs * vol_multiplier
            
            # Reconstruct covariance matrix
            std_matrix = np.diag(stressed_std_devs)
            stressed_cov = std_matrix @ corr_matrix @ std_matrix
        else:
            stressed_cov = asset_returns.cov().values.copy()
        
        # Apply stress to correlations
        if 'corr_shift' in scenario_params:
            corr_shift = scenario_params['corr_shift']
            
            # Get correlation matrix
            corr_matrix = asset_returns.corr().values
            
            # Apply shift (ensuring values stay in [-1, 1] range)
            np.fill_diagonal(corr_matrix, 1)  # Ensure diagonal remains 1
            stressed_corr = np.clip(corr_matrix + corr_shift, -1, 1)
            np.fill_diagonal(stressed_corr, 1)  # Ensure diagonal remains 1
            
            # Get standard deviations
            std_devs = np.sqrt(np.diag(stressed_cov))
            
            # Reconstruct covariance matrix
            std_matrix = np.diag(std_devs)
            stressed_cov = std_matrix @ stressed_corr @ std_matrix
        
        # Simulate returns under stress
        n_simulations = 10000
        stressed_returns = generate_correlated_returns(
            mean_returns=stressed_mean,
            cov_matrix=stressed_cov,
            n_simulations=n_simulations
        )
        
        # Calculate portfolio returns
        stressed_portfolio_returns = np.dot(stressed_returns, weights)
        
        # Calculate metrics
        mean_return = np.mean(stressed_portfolio_returns)
        std_dev = np.std(stressed_portfolio_returns)
        var_95 = np.percentile(stressed_portfolio_returns, 5)
        var_99 = np.percentile(stressed_portfolio_returns, 1)
        
        # Calculate portfolio values
        mean_value = investment_value * (1 + mean_return)
        var_95_value = investment_value * (1 + var_95)
        var_99_value = investment_value * (1 + var_99)
        
        # Calculate potential losses
        mean_loss = investment_value - mean_value
        var_95_loss = investment_value - var_95_value
        var_99_loss = investment_value - var_99_value
        
        # Add to results
        results.append({
            'scenario': scenario_name,
            'mean_return': mean_return,
            'std_dev': std_dev,
            'var_95': var_95,
            'var_99': var_99,
            'mean_loss': mean_loss,
            'var_95_loss': var_95_loss,
            'var_99_loss': var_99_loss,
            'mean_loss_pct': mean_loss / investment_value * 100,
            'var_95_loss_pct': var_95_loss / investment_value * 100,
            'var_99_loss_pct': var_99_loss / investment_value * 100
        })
    
    # Add baseline scenario
    results.append({
        'scenario': 'Baseline',
        'mean_return': baseline_mean,
        'std_dev': baseline_std,
        'var_95': stats.norm.ppf(0.05, loc=baseline_mean, scale=baseline_std),
        'var_99': stats.norm.ppf(0.01, loc=baseline_mean, scale=baseline_std),
        'mean_loss': -baseline_mean * investment_value,
        'var_95_loss': -stats.norm.ppf(0.05, loc=baseline_mean, scale=baseline_std) * investment_value,
        'var_99_loss': -stats.norm.ppf(0.01, loc=baseline_mean, scale=baseline_std) * investment_value,
        'mean_loss_pct': -baseline_mean * 100,
        'var_95_loss_pct': -stats.norm.ppf(0.05, loc=baseline_mean, scale=baseline_std) * 100,
        'var_99_loss_pct': -stats.norm.ppf(0.01, loc=baseline_mean, scale=baseline_std) * 100
    })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Sort with Baseline first, then by highest VaR95 loss
    results_df = pd.concat([
        results_df[results_df['scenario'] == 'Baseline'],
        results_df[results_df['scenario'] != 'Baseline'].sort_values(by='var_95_loss', ascending=False)
    ]).reset_index(drop=True)
    
    return results_df

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Example usage
    import os
    from pathlib import Path
    
    # Get project root directory
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    DATA_DIR = PROJECT_ROOT / "data"
    PROCESSED_DIR = DATA_DIR / "processed"
    
    try:
        # Load market returns
        market_returns_file = PROCESSED_DIR / "market_returns.csv"
        
        if not market_returns_file.exists():
            logger.error("Market returns file not found. Please run process_data.py first.")
            exit(1)
        
        # Load returns data
        market_returns = pd.read_csv(market_returns_file, index_col=0, parse_dates=True)
        
        # Handle MultiIndex columns if present
        if isinstance(market_returns.columns, pd.MultiIndex):
            # Get first level (e.g., 'TRDPRC_1' or 'Close')
            price_col = market_returns.columns.levels[0][0]
            assets = market_returns.columns.levels[1]
            market_returns = market_returns[price_col]
            logger.info(f"Using price column: {price_col}")
        
        # Example weights (equal weighting)
        n_assets = len(market_returns.columns)
        weights = np.ones(n_assets) / n_assets
        
        logger.info(f"Using equal weights for {n_assets} assets")
        
        # Run multivariate Monte Carlo simulation
        logger.info("Running multivariate Monte Carlo simulation...")
        var_value = multivariate_monte_carlo_var(
            asset_returns=market_returns,
            weights=weights,
            confidence_level=0.95,
            investment_value=1000000,
            n_simulations=10000,
            time_horizon=1
        )
        
        logger.info(f"Multivariate Monte Carlo VaR (95%): ${var_value:,.2f}")
        
        # Calculate asset contributions to VaR
        logger.info("Calculating asset contributions to VaR...")
        contributions = calculate_asset_var_contributions(
            asset_returns=market_returns,
            weights=weights,
            confidence_level=0.95,
            investment_value=1000000,
            n_simulations=10000
        )
        
        logger.info("Asset VaR contributions:")
        logger.info(contributions)
        
        # Define stress scenarios
        stress_scenarios = {
            "Severe Market Decline": {
                "mean_shift": np.ones(n_assets) * -0.05,  # 5% decline across all assets
                "vol_multiplier": 2.0  # Double volatility
            },
            "Interest Rate Shock": {
                "mean_shift": np.array([-0.02] * (n_assets-1) + [-0.10]),  # Larger impact on bonds
                "corr_shift": 0.2  # Increased correlations
            },
            "Sector Rotation": {
                "mean_shift": np.random.uniform(-0.05, 0.05, n_assets),  # Mixed impact
                "vol_multiplier": 1.5  # Increased volatility
            }
        }
        
        # Run stress tests
        logger.info("Running multivariate stress tests...")
        stress_results = run_multivariate_stress_test(
            asset_returns=market_returns,
            weights=weights,
            stress_scenarios=stress_scenarios,
            investment_value=1000000
        )
        
        logger.info("Stress test results:")
        logger.info(stress_results)
        
    except Exception as e:
        logger.error(f"Error in multivariate simulation: {e}", exc_info=True)