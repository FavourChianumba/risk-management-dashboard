"""
Risk factor simulation module for advanced VaR calculations.
This module implements methods for identifying and simulating risk factors.
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
import statsmodels.api as sm
import logging

# Set up logger
logger = logging.getLogger(__name__)

def identify_risk_factors(returns, method='pca', n_factors=None, variance_explained=0.95):
    """
    Identify risk factors from asset returns.
    
    Args:
        returns (pandas.DataFrame): Asset returns
        method (str): Method for factor identification ('pca' or 'manual')
        n_factors (int, optional): Number of factors to extract
        variance_explained (float, optional): Minimum variance to be explained by factors
        
    Returns:
        dict: Dictionary containing risk factor model
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
    
    # Drop any rows with missing values
    data = data.dropna()
    
    if method.lower() == 'pca':
        # Determine number of factors
        if n_factors is None:
            # Perform PCA with full components
            pca = PCA()
            pca.fit(data)
            
            # Determine number of factors needed to explain specified variance
            explained_variance_ratio = pca.explained_variance_ratio_
            cumulative_variance = np.cumsum(explained_variance_ratio)
            n_factors = np.argmax(cumulative_variance >= variance_explained) + 1
            
            logger.info(f"Selected {n_factors} PCA factors to explain {variance_explained*100:.1f}% of variance")
        
        # Perform PCA with selected number of factors
        pca = PCA(n_components=n_factors)
        factor_loadings = pca.fit_transform(data)
        
        # Calculate factor scores and loadings
        loadings = pca.components_.T
        explained_variance = pca.explained_variance_ratio_
        
        # Create factor returns DataFrame
        factor_returns = pd.DataFrame(
            factor_loadings,
            index=data.index,
            columns=[f'Factor_{i+1}' for i in range(n_factors)]
        )
        
        # Create loadings DataFrame
        loadings_df = pd.DataFrame(
            loadings,
            index=data.columns,
            columns=[f'Factor_{i+1}' for i in range(n_factors)]
        )
        
        # Fit linear models for each asset
        factor_model = {}
        for asset in data.columns:
            model = sm.OLS(data[asset], sm.add_constant(factor_returns)).fit()
            factor_model[asset] = {
                'alpha': model.params[0],
                'betas': model.params[1:].values,
                'r_squared': model.rsquared,
                'residual_std': np.std(model.resid)
            }
        
        # Return factor model
        return {
            'method': 'pca',
            'n_factors': n_factors,
            'explained_variance': explained_variance,
            'factor_returns': factor_returns,
            'loadings': loadings_df,
            'asset_models': factor_model
        }
    
    elif method.lower() == 'manual':
        # For manual method, the caller should specify the factors in the return data
        if n_factors is None:
            logger.warning("n_factors should be specified for manual method. Using all columns as factors.")
            factor_cols = data.columns
        else:
            # Use the first n_factors columns as factors
            factor_cols = data.columns[:n_factors]
        
        # Extract factor returns
        factor_returns = data[factor_cols]
        
        # Fit linear models for each asset
        factor_model = {}
        for asset in data.columns:
            if asset in factor_cols:
                # If the asset is also a factor, use identity mapping
                factor_model[asset] = {
                    'alpha': 0.0,
                    'betas': np.zeros(len(factor_cols)),
                    'r_squared': 1.0,
                    'residual_std': 0.0
                }
                factor_model[asset]['betas'][list(factor_cols).index(asset)] = 1.0
            else:
                # Otherwise, fit a linear model
                model = sm.OLS(data[asset], sm.add_constant(factor_returns)).fit()
                factor_model[asset] = {
                    'alpha': model.params[0],
                    'betas': model.params[1:].values,
                    'r_squared': model.rsquared,
                    'residual_std': np.std(model.resid)
                }
        
        # Return factor model
        return {
            'method': 'manual',
            'n_factors': len(factor_cols),
            'factor_returns': factor_returns,
            'loadings': pd.DataFrame(
                index=data.columns,
                columns=factor_cols,
                data=np.eye(len(data.columns), len(factor_cols))
            ),
            'asset_models': factor_model
        }
    
    else:
        raise ValueError(f"Unsupported method: {method}. Use 'pca' or 'manual'.")

def fit_risk_factor_model(returns, macro_data=None, method='pca', n_factors=None):
    """
    Fit a risk factor model to asset returns, optionally including macroeconomic factors.
    
    Args:
        returns (pandas.DataFrame): Asset returns
        macro_data (pandas.DataFrame, optional): Macroeconomic factor data
        method (str): Method for factor identification ('pca', 'manual', or 'macro')
        n_factors (int, optional): Number of factors to extract
        
    Returns:
        dict: Dictionary containing risk factor model
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
    
    # Drop any rows with missing values
    data = data.dropna()
    
    if method.lower() == 'macro' and macro_data is not None:
        # Align dates between returns and macro data
        aligned_dates = data.index.intersection(macro_data.index)
        if len(aligned_dates) == 0:
            raise ValueError("No common dates between returns and macro data")
        
        data = data.loc[aligned_dates]
        macro_data = macro_data.loc[aligned_dates]
        
        # Use macroeconomic variables as factors
        factor_returns = macro_data
        
        # Fit linear models for each asset
        factor_model = {}
        for asset in data.columns:
            model = sm.OLS(data[asset], sm.add_constant(factor_returns)).fit()
            factor_model[asset] = {
                'alpha': model.params[0],
                'betas': model.params[1:].values,
                'r_squared': model.rsquared,
                'residual_std': np.std(model.resid)
            }
        
        # Return factor model
        return {
            'method': 'macro',
            'n_factors': len(factor_returns.columns),
            'factor_returns': factor_returns,
            'asset_models': factor_model
        }
    else:
        # Use PCA or manual method
        return identify_risk_factors(returns, method=method, n_factors=n_factors)

def simulate_risk_factors(factor_model, n_simulations=10000, time_horizon=1, random_seed=None):
    """
    Simulate risk factor returns based on fitted model.
    
    Args:
        factor_model (dict): Fitted risk factor model
        n_simulations (int): Number of simulations
        time_horizon (int): Time horizon in days
        random_seed (int, optional): Random seed for reproducibility
        
    Returns:
        pandas.DataFrame: Simulated factor returns
    """
    # Set random seed if specified
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Extract factor returns from model
    factor_returns = factor_model['factor_returns']
    n_factors = factor_model['n_factors']
    
    # Calculate mean and covariance of factor returns
    factor_mean = factor_returns.mean().values * time_horizon
    factor_cov = factor_returns.cov().values * time_horizon
    
    # Generate simulated factor returns
    simulated_factors = np.random.multivariate_normal(
        mean=factor_mean,
        cov=factor_cov,
        size=n_simulations
    )
    
    # Create DataFrame with simulated factor returns
    simulated_df = pd.DataFrame(
        simulated_factors,
        columns=factor_returns.columns
    )
    
    return simulated_df

def predict_asset_returns(factor_model, simulated_factors):
    """
    Predict asset returns from simulated factor returns.
    
    Args:
        factor_model (dict): Fitted risk factor model
        simulated_factors (pandas.DataFrame): Simulated factor returns
        
    Returns:
        pandas.DataFrame: Simulated asset returns
    """
    # Get asset models
    asset_models = factor_model['asset_models']
    asset_names = list(asset_models.keys())
    
    # Initialize array for simulated asset returns
    n_simulations = len(simulated_factors)
    n_assets = len(asset_names)
    simulated_returns = np.zeros((n_simulations, n_assets))
    
    # Predict returns for each asset
    for i, asset in enumerate(asset_names):
        model = asset_models[asset]
        
        # Calculate systematic return
        systematic_return = model['alpha'] + np.dot(simulated_factors.values, model['betas'])
        
        # Add idiosyncratic return (residual)
        if model['residual_std'] > 0:
            idiosyncratic_return = np.random.normal(0, model['residual_std'], n_simulations)
            asset_return = systematic_return + idiosyncratic_return
        else:
            asset_return = systematic_return
        
        simulated_returns[:, i] = asset_return
    
    # Create DataFrame with simulated asset returns
    simulated_df = pd.DataFrame(
        simulated_returns,
        columns=asset_names
    )
    
    return simulated_df

def risk_factor_monte_carlo_var(returns, weights, confidence_level=0.95, investment_value=1000000,
                              n_simulations=10000, time_horizon=1, method='pca', n_factors=None,
                              macro_data=None, random_seed=None):
    """
    Calculate Value-at-Risk using risk factor Monte Carlo simulation.
    
    Args:
        returns (pandas.DataFrame): Asset returns
        weights (numpy.ndarray or list): Portfolio weights
        confidence_level (float): Confidence level (e.g., 0.95 for 95% confidence)
        investment_value (float): Current portfolio value
        n_simulations (int): Number of simulations
        time_horizon (int): Time horizon in days
        method (str): Method for factor identification ('pca', 'manual', or 'macro')
        n_factors (int, optional): Number of factors to extract
        macro_data (pandas.DataFrame, optional): Macroeconomic factor data
        random_seed (int, optional): Random seed for reproducibility
        
    Returns:
        float: Value-at-Risk estimate
    """
    # Convert weights to numpy array
    weights = np.array(weights)
    
    # Fit factor model
    factor_model = fit_risk_factor_model(
        returns=returns,
        macro_data=macro_data,
        method=method,
        n_factors=n_factors
    )
    
    # Simulate factor returns
    simulated_factors = simulate_risk_factors(
        factor_model=factor_model,
        n_simulations=n_simulations,
        time_horizon=time_horizon,
        random_seed=random_seed
    )
    
    # Predict asset returns from simulated factors
    simulated_returns = predict_asset_returns(
        factor_model=factor_model,
        simulated_factors=simulated_factors
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

def decompose_var_by_factor(factor_model, simulated_factors, simulated_returns, weights, 
                          confidence_level=0.95, investment_value=1000000):
    """
    Decompose VaR by risk factor contribution.
    
    Args:
        factor_model (dict): Fitted risk factor model
        simulated_factors (pandas.DataFrame): Simulated factor returns
        simulated_returns (pandas.DataFrame): Simulated asset returns
        weights (numpy.ndarray or list): Portfolio weights
        confidence_level (float): Confidence level (e.g., 0.95 for 95% confidence)
        investment_value (float): Current portfolio value
        
    Returns:
        pandas.DataFrame: VaR contribution by risk factor
    """
    # Convert weights to numpy array
    weights = np.array(weights)
    
    # Calculate portfolio returns
    portfolio_returns = np.dot(simulated_returns.values, weights)
    
    # Identify tail scenarios
    var_percentile = 1 - confidence_level
    var_threshold = np.percentile(portfolio_returns, var_percentile * 100)
    tail_scenarios = portfolio_returns <= var_threshold
    
    # Extract tail factor returns
    tail_factors = simulated_factors.iloc[tail_scenarios]
    
    # Calculate factor contribution to tail losses
    factor_contributions = []
    factor_names = simulated_factors.columns
    
    for factor in factor_names:
        # Calculate correlation with portfolio returns in the tail
        factor_return = tail_factors[factor]
        correlation = np.corrcoef(factor_return, portfolio_returns[tail_scenarios])[0, 1]
        
        # Calculate Marginal VaR (sensitivity of VaR to factor)
        # This is a simplified approach using correlation
        marginal_var = correlation * np.std(factor_return) / np.std(portfolio_returns[tail_scenarios])
        
        # Calculate Component VaR (contribution to total VaR)
        component_var = marginal_var * np.mean(factor_return) / np.mean(portfolio_returns[tail_scenarios])
        
        factor_contributions.append({
            'factor': factor,
            'mean_return': np.mean(factor_return),
            'std_dev': np.std(factor_return),
            'correlation': correlation,
            'marginal_var': marginal_var,
            'component_var': component_var,
            'component_value': abs(component_var * investment_value)
        })
    
    # Convert to DataFrame
    factor_contributions_df = pd.DataFrame(factor_contributions)
    
    # Sort by component value
    factor_contributions_df = factor_contributions_df.sort_values(by='component_value', ascending=False)
    
    return factor_contributions_df

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
        
        # Load macro data if available
        macro_file = PROCESSED_DIR / "macro_data.csv"
        if macro_file.exists():
            macro_data = pd.read_csv(macro_file, index_col=0, parse_dates=True)
            logger.info(f"Loaded macro data with shape {macro_data.shape}")
        else:
            macro_data = None
            logger.info("No macro data available")
        
        # Identify risk factors using PCA
        logger.info("Identifying risk factors using PCA...")
        pca_model = identify_risk_factors(market_returns, method='pca')
        logger.info(f"Identified {pca_model['n_factors']} PCA factors")
        logger.info(f"Explained variance: {pca_model['explained_variance']}")
        
        # Example weights (equal weighting)
        n_assets = len(market_returns.columns)
        weights = np.ones(n_assets) / n_assets
        
        logger.info(f"Using equal weights for {n_assets} assets")
        
        # Calculate VaR using PCA risk factors
        logger.info("Calculating VaR using PCA risk factors...")
        pca_var = risk_factor_monte_carlo_var(
            returns=market_returns,
            weights=weights,
            confidence_level=0.95,
            investment_value=1000000,
            n_simulations=10000,
            method='pca'
        )
        
        logger.info(f"PCA risk factor VaR (95%): ${pca_var:,.2f}")
        
        # If macro data is available, calculate VaR using macro factors
        if macro_data is not None:
            logger.info("Calculating VaR using macro risk factors...")
            macro_var = risk_factor_monte_carlo_var(
                returns=market_returns,
                weights=weights,
                confidence_level=0.95,
                investment_value=1000000,
                n_simulations=10000,
                method='macro',
                macro_data=macro_data
            )
            
            logger.info(f"Macro risk factor VaR (95%): ${macro_var:,.2f}")
        
    except Exception as e:
        logger.error(f"Error in risk factor simulation: {e}", exc_info=True)