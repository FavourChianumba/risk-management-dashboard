"""
Copula models for modeling complex dependencies between assets.
This module provides functions for fitting and simulating from copula models.
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import minimize
import logging

# Set up logger
logger = logging.getLogger(__name__)

def to_uniform(data):
    """
    Transform data to uniform distribution using empirical CDF.
    
    Args:
        data (numpy.ndarray): Input data array
        
    Returns:
        numpy.ndarray: Transformed data in (0,1) range
    """
    n = len(data)
    return np.array([stats.rankdata(data[:, i], method='average') / (n + 1) for i in range(data.shape[1])]).T

def fit_gaussian_copula(data):
    """
    Fit a Gaussian copula to multivariate data.
    
    Args:
        data (pandas.DataFrame or numpy.ndarray): Multivariate data
        
    Returns:
        numpy.ndarray: Fitted correlation matrix
    """
    if isinstance(data, pd.DataFrame):
        data = data.values
    
    # Transform data to uniform marginals
    u_data = to_uniform(data)
    
    # Transform to standard normal
    z_data = stats.norm.ppf(u_data)
    
    # Calculate correlation matrix
    corr_matrix = np.corrcoef(z_data, rowvar=False)
    
    return corr_matrix

def fit_t_copula(data, nu_guess=4):
    """
    Fit a Student's t-copula to multivariate data.
    
    Args:
        data (pandas.DataFrame or numpy.ndarray): Multivariate data
        nu_guess (float): Initial guess for degrees of freedom
        
    Returns:
        tuple: (correlation matrix, degrees of freedom)
    """
    if isinstance(data, pd.DataFrame):
        data = data.values
    
    # Transform data to uniform marginals
    u_data = to_uniform(data)
    
    # Transform to standard normal
    z_data = stats.norm.ppf(u_data)
    
    # Calculate correlation matrix
    corr_matrix = np.corrcoef(z_data, rowvar=False)
    
    # Function to optimize for degrees of freedom
    def t_copula_neg_loglik(nu):
        try:
            # Transform to t-distribution
            t_data = stats.t.ppf(u_data, df=nu)
            
            # Compute negative log likelihood
            log_lik = 0
            n_samples, n_dim = t_data.shape
            
            for i in range(n_samples):
                x = t_data[i, :]
                log_lik -= stats.multivariate_t.logpdf(x, loc=np.zeros(n_dim), shape=corr_matrix, df=nu)
            
            return log_lik
        except Exception as e:
            logger.warning(f"Error in t-copula likelihood: {e}")
            return 1e10  # Large value for error cases
    
    # Optimize for degrees of freedom (constrained to be > 2)
    result = minimize(t_copula_neg_loglik, x0=nu_guess, bounds=[(2.1, 100)])
    nu = result.x[0]
    
    logger.info(f"Fitted t-copula with {nu:.2f} degrees of freedom")
    
    return corr_matrix, nu

def simulate_from_gaussian_copula(corr_matrix, n_samples, random_seed=None):
    """
    Simulate from a Gaussian copula with the given correlation matrix.
    
    Args:
        corr_matrix (numpy.ndarray): Correlation matrix
        n_samples (int): Number of samples to generate
        random_seed (int, optional): Random seed for reproducibility
        
    Returns:
        numpy.ndarray: Simulated uniform samples from the copula
    """
    # Set random seed if specified
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Get number of dimensions
    n_dim = corr_matrix.shape[0]
    
    # Generate multivariate normal samples
    mvn_samples = np.random.multivariate_normal(mean=np.zeros(n_dim), cov=corr_matrix, size=n_samples)
    
    # Transform to uniform using normal CDF
    u_samples = stats.norm.cdf(mvn_samples)
    
    return u_samples

def simulate_from_t_copula(corr_matrix, nu, n_samples, random_seed=None):
    """
    Simulate from a Student's t-copula with the given correlation matrix and degrees of freedom.
    
    Args:
        corr_matrix (numpy.ndarray): Correlation matrix
        nu (float): Degrees of freedom
        n_samples (int): Number of samples to generate
        random_seed (int, optional): Random seed for reproducibility
        
    Returns:
        numpy.ndarray: Simulated uniform samples from the copula
    """
    # Set random seed if specified
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Get number of dimensions
    n_dim = corr_matrix.shape[0]
    
    # Generate multivariate t samples
    mvt_samples = np.zeros((n_samples, n_dim))
    
    # Generate chi-squared samples for t-distribution
    chi_samples = np.random.chisquare(df=nu, size=n_samples) / nu
    
    # Generate multivariate normal samples
    mvn_samples = np.random.multivariate_normal(mean=np.zeros(n_dim), cov=corr_matrix, size=n_samples)
    
    # Combine to get multivariate t samples
    for i in range(n_samples):
        mvt_samples[i, :] = mvn_samples[i, :] / np.sqrt(chi_samples[i])
    
    # Transform to uniform using t CDF
    u_samples = np.zeros_like(mvt_samples)
    for i in range(n_dim):
        u_samples[:, i] = stats.t.cdf(mvt_samples[:, i], df=nu)
    
    return u_samples

def fit_copula(returns, copula_type='gaussian'):
    """
    Fit a copula model to asset returns.
    
    Args:
        returns (pandas.DataFrame): Asset returns
        copula_type (str): Type of copula ('gaussian' or 't')
        
    Returns:
        dict: Fitted copula parameters
    """
    # Handle MultiIndex columns if present
    if isinstance(returns.columns, pd.MultiIndex):
        # Get first level (e.g., 'TRDPRC_1' or 'Close')
        price_col = returns.columns.levels[0][0]
        assets = returns.columns.levels[1]
        data = returns[price_col].values
        asset_names = list(assets)
    else:
        data = returns.values
        asset_names = list(returns.columns)
    
    # Fit selected copula
    if copula_type.lower() == 'gaussian':
        corr_matrix = fit_gaussian_copula(data)
        return {
            'type': 'gaussian',
            'corr_matrix': corr_matrix,
            'asset_names': asset_names
        }
    elif copula_type.lower() == 't':
        corr_matrix, nu = fit_t_copula(data)
        return {
            'type': 't',
            'corr_matrix': corr_matrix,
            'nu': nu,
            'asset_names': asset_names
        }
    else:
        raise ValueError(f"Unsupported copula type: {copula_type}. Use 'gaussian' or 't'.")

def simulate_from_copula(copula_params, marginal_params, n_samples=10000, random_seed=None):
    """
    Simulate asset returns from a fitted copula with specified marginal distributions.
    
    Args:
        copula_params (dict): Fitted copula parameters
        marginal_params (dict): Marginal distribution parameters for each asset
        n_samples (int): Number of samples to generate
        random_seed (int, optional): Random seed for reproducibility
        
    Returns:
        pandas.DataFrame: Simulated asset returns
    """
    # Generate uniform samples from copula
    if copula_params['type'] == 'gaussian':
        u_samples = simulate_from_gaussian_copula(
            corr_matrix=copula_params['corr_matrix'],
            n_samples=n_samples,
            random_seed=random_seed
        )
    elif copula_params['type'] == 't':
        u_samples = simulate_from_t_copula(
            corr_matrix=copula_params['corr_matrix'],
            nu=copula_params['nu'],
            n_samples=n_samples,
            random_seed=random_seed
        )
    else:
        raise ValueError(f"Unsupported copula type: {copula_params['type']}")
    
    # Transform uniform samples to target marginal distributions
    asset_names = copula_params['asset_names']
    n_assets = len(asset_names)
    simulated_returns = np.zeros_like(u_samples)
    
    for i in range(n_assets):
        asset = asset_names[i]
        marginal = marginal_params[asset]
        
        # Apply inverse CDF of the specified distribution
        if marginal['type'] == 'normal':
            simulated_returns[:, i] = stats.norm.ppf(
                u_samples[:, i],
                loc=marginal['mean'],
                scale=marginal['std']
            )
        elif marginal['type'] == 't':
            simulated_returns[:, i] = stats.t.ppf(
                u_samples[:, i],
                df=marginal['df'],
                loc=marginal['loc'],
                scale=marginal['scale']
            )
        elif marginal['type'] == 'empirical':
            # Use empirical CDF (interpolate between observed values)
            sorted_data = np.sort(marginal['data'])
            simulated_returns[:, i] = np.interp(
                u_samples[:, i] * len(sorted_data),
                np.arange(len(sorted_data)),
                sorted_data
            )
        else:
            raise ValueError(f"Unsupported marginal distribution type: {marginal['type']}")
    
    # Create DataFrame with simulated returns
    simulated_df = pd.DataFrame(simulated_returns, columns=asset_names)
    
    return simulated_df

def estimate_marginal_distributions(returns):
    """
    Estimate parameters for marginal distributions of asset returns.
    
    Args:
        returns (pandas.DataFrame): Asset returns
        
    Returns:
        dict: Marginal distribution parameters for each asset
    """
    # Handle MultiIndex columns if present
    if isinstance(returns.columns, pd.MultiIndex):
        # Get first level (e.g., 'TRDPRC_1' or 'Close')
        price_col = returns.columns.levels[0][0]
        assets = returns.columns.levels[1]
        data = returns[price_col]
    else:
        data = returns
        assets = returns.columns
    
    # Estimate parameters for each asset
    marginal_params = {}
    
    for asset in assets:
        asset_returns = data[asset].dropna().values
        
        # Fit normal distribution
        norm_params = stats.norm.fit(asset_returns)
        
        # Fit t distribution
        t_params = stats.t.fit(asset_returns)
        
        # Perform Shapiro-Wilk test for normality
        shapiro_test = stats.shapiro(asset_returns)
        
        # Determine best distribution based on fit
        if shapiro_test.pvalue > 0.05:
            # Data appears normally distributed
            marginal_params[asset] = {
                'type': 'normal',
                'mean': norm_params[0],
                'std': norm_params[1],
                'shapiro_pvalue': shapiro_test.pvalue
            }
        else:
            # Data has heavy tails, use t distribution
            marginal_params[asset] = {
                'type': 't',
                'df': t_params[0],
                'loc': t_params[1],
                'scale': t_params[2],
                'shapiro_pvalue': shapiro_test.pvalue
            }
        
        # Also store empirical distribution (for non-parametric approach)
        marginal_params[asset+'_empirical'] = {
            'type': 'empirical',
            'data': asset_returns
        }
    
    return marginal_params

def copula_monte_carlo_var(returns, weights, confidence_level=0.95, investment_value=1000000,
                         n_simulations=10000, time_horizon=1, copula_type='t', random_seed=None):
    """
    Calculate Value-at-Risk using copula-based Monte Carlo simulation.
    
    Args:
        returns (pandas.DataFrame): Asset returns
        weights (numpy.ndarray or list): Portfolio weights
        confidence_level (float): Confidence level (e.g., 0.95 for 95% confidence)
        investment_value (float): Current portfolio value
        n_simulations (int): Number of simulations
        time_horizon (int): Time horizon in days
        copula_type (str): Type of copula ('gaussian' or 't')
        random_seed (int, optional): Random seed for reproducibility
        
    Returns:
        float: Value-at-Risk estimate
    """
    # Convert weights to numpy array
    weights = np.array(weights)
    
    # Fit copula
    copula_params = fit_copula(returns, copula_type=copula_type)
    
    # Estimate marginal distributions
    marginal_params = estimate_marginal_distributions(returns)
    
    # Scale marginal parameters for time horizon
    asset_names = copula_params['asset_names']
    for asset in asset_names:
        if marginal_params[asset]['type'] == 'normal':
            marginal_params[asset]['mean'] *= time_horizon
            marginal_params[asset]['std'] *= np.sqrt(time_horizon)
        elif marginal_params[asset]['type'] == 't':
            marginal_params[asset]['loc'] *= time_horizon
            marginal_params[asset]['scale'] *= np.sqrt(time_horizon)
    
    # Simulate returns
    simulated_returns = simulate_from_copula(
        copula_params=copula_params,
        marginal_params=marginal_params,
        n_samples=n_simulations,
        random_seed=random_seed
    )
    
    # Calculate portfolio returns
    portfolio_returns = np.dot(simulated_returns.values, weights)
    
    # Calculate simulated portfolio values
    simulated_values = investment_value * (1 + portfolio_returns)
    
    # Calculate potential losses
    potential_losses = investment_value - simulated_values
    
    # Calculate VaR at specified confidence level
    var_percentile = confidence_level
    var_value = np.percentile(potential_losses, var_percentile * 100)
    
    # Ensure VaR is positive (representing a loss)
    var_value = max(0, var_value)
    
    return var_value

def calculate_tail_dependence(returns, quantile=0.05, method='empirical'):
    """
    Calculate tail dependence coefficients between assets.
    
    Args:
        returns (pandas.DataFrame): Asset returns
        quantile (float): Quantile defining the tail (e.g., 0.05 for 5% tail)
        method (str): Method for calculation ('empirical' or 'theoretical')
        
    Returns:
        pandas.DataFrame: Matrix of tail dependence coefficients
    """
    # Handle MultiIndex columns if present
    if isinstance(returns.columns, pd.MultiIndex):
        # Get first level (e.g., 'TRDPRC_1' or 'Close')
        price_col = returns.columns.levels[0][0]
        assets = returns.columns.levels[1]
        data = returns[price_col]
    else:
        data = returns
        assets = returns.columns
    
    n_assets = len(assets)
    tail_dependence = np.zeros((n_assets, n_assets))
    
    if method == 'empirical':
        # Transform to uniform marginals
        u_data = pd.DataFrame(index=data.index, columns=data.columns)
        for col in data.columns:
            u_data[col] = data[col].rank(method='average') / (len(data) + 1)
        
        # Calculate empirical tail dependence
        for i in range(n_assets):
            for j in range(n_assets):
                if i == j:
                    tail_dependence[i, j] = 1.0
                else:
                    # Lower tail dependence: P(U1 <= q | U2 <= q) as q → 0
                    asset1 = assets[i]
                    asset2 = assets[j]
                    joint_prob = np.mean((u_data[asset1] <= quantile) & (u_data[asset2] <= quantile))
                    tail_dependence[i, j] = joint_prob / quantile
    
    elif method == 'theoretical':
        # Fit a t-copula
        corr_matrix, nu = fit_t_copula(data.values)
        
        # Calculate theoretical tail dependence for t-copula
        # Lower tail dependence: 2 * t_{nu+1}(-sqrt((nu+1)*(1-rho)/(1+rho)))
        for i in range(n_assets):
            for j in range(n_assets):
                if i == j:
                    tail_dependence[i, j] = 1.0
                else:
                    rho = corr_matrix[i, j]
                    if abs(rho) > 0.9999:  # Handle perfect correlation
                        tail_dependence[i, j] = 1.0
                    else:
                        tail_dependence[i, j] = 2 * stats.t.cdf(
                            -np.sqrt((nu+1) * (1-rho) / (1+rho)),
                            df=nu+1
                        )
    else:
        raise ValueError(f"Unsupported method: {method}. Use 'empirical' or 'theoretical'.")
    
    # Create DataFrame
    tail_dependence_df = pd.DataFrame(tail_dependence, index=assets, columns=assets)
    
    return tail_dependence_df

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
        
        # Fit Gaussian copula
        logger.info("Fitting Gaussian copula...")
        gaussian_params = fit_copula(market_returns, copula_type='gaussian')
        logger.info(f"Gaussian copula fitted with correlation matrix shape: {gaussian_params['corr_matrix'].shape}")
        
        # Fit t copula
        logger.info("Fitting t copula...")
        t_params = fit_copula(market_returns, copula_type='t')
        logger.info(f"t copula fitted with nu = {t_params['nu']:.2f}")
        
        # Estimate marginal distributions
        logger.info("Estimating marginal distributions...")
        marginal_params = estimate_marginal_distributions(market_returns)
        
        # Example weights (equal weighting)
        n_assets = len(market_returns.columns)
        weights = np.ones(n_assets) / n_assets
        
        logger.info(f"Using equal weights for {n_assets} assets")
        
        # Calculate VaR using Gaussian copula
        logger.info("Calculating VaR using Gaussian copula...")
        gaussian_var = copula_monte_carlo_var(
            returns=market_returns,
            weights=weights,
            confidence_level=0.95,
            investment_value=1000000,
            n_simulations=10000,
            copula_type='gaussian'
        )
        
        logger.info(f"Gaussian copula VaR (95%): ${gaussian_var:,.2f}")
        
        # Calculate VaR using t copula
        logger.info("Calculating VaR using t copula...")
        t_var = copula_monte_carlo_var(
            returns=market_returns,
            weights=weights,
            confidence_level=0.95,
            investment_value=1000000,
            n_simulations=10000,
            copula_type='t'
        )
        
        logger.info(f"t copula VaR (95%): ${t_var:,.2f}")
        
        # Calculate tail dependence
        logger.info("Calculating tail dependence...")
        tail_dep = calculate_tail_dependence(market_returns, quantile=0.05)
        logger.info("Tail dependence matrix:")
        logger.info(tail_dep)
        
    except Exception as e:
        logger.error(f"Error in copula modeling: {e}", exc_info=True)