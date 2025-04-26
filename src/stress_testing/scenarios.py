"""
Stress test scenario definitions and implementations.
This module provides functions for defining and running stress test scenarios.
"""

import pandas as pd
import numpy as np
from scipy import stats
import logging

# Set up logger
logger = logging.getLogger(__name__)

def historical_scenario(returns, crisis_period, start_date=None, end_date=None):
    """
    Apply a historical stress scenario based on a specific crisis period.
    
    Args:
        returns (pandas.DataFrame): Asset returns data
        crisis_period (str or dict): Crisis period name or dictionary with start_date and end_date
        start_date (str, optional): Start date of crisis period (if not provided in crisis_period)
        end_date (str, optional): End date of crisis period (if not provided in crisis_period)
        
    Returns:
        pandas.DataFrame: Returns during the crisis period
    """
    # If crisis_period is a string, look up in predefined crisis periods
    if isinstance(crisis_period, str):
        from ..data.retrieve_data import load_crisis_periods
        crisis_periods = load_crisis_periods()
        
        # Find the specified crisis period
        crisis_df = crisis_periods[crisis_periods['name'] == crisis_period]
        if len(crisis_df) == 0:
            raise ValueError(f"Crisis period '{crisis_period}' not found")
        
        # Extract start and end dates
        start_date = crisis_df.iloc[0]['start_date']
        end_date = crisis_df.iloc[0]['end_date']
    else:
        # Extract dates from provided dictionary
        start_date = crisis_period.get('start_date', start_date)
        end_date = crisis_period.get('end_date', end_date)
    
    # Validate dates
    if start_date is None or end_date is None:
        raise ValueError("Start date and end date must be provided")
    
    # Convert to pandas datetime if they are strings
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date)
    
    # Extract returns during the crisis period
    crisis_returns = returns.loc[start_date:end_date]
    
    if len(crisis_returns) == 0:
        logger.warning(f"No returns data found for period {start_date} to {end_date}")
    else:
        logger.info(f"Extracted {len(crisis_returns)} days of returns for period {start_date} to {end_date}")
    
    return crisis_returns

def synthetic_scenario(returns, shock_params, correlation_adjustment=None, volatility_adjustment=None):
    """
    Generate a synthetic stress scenario by applying shocks to asset returns.
    
    Args:
        returns (pandas.DataFrame): Asset returns data
        shock_params (dict): Dictionary mapping asset names to shock magnitudes (in percentage points)
        correlation_adjustment (float, optional): Adjustment to correlation matrix (between -1 and 1)
        volatility_adjustment (float, optional): Multiplier for volatility
        
    Returns:
        pandas.DataFrame: Simulated returns under the stress scenario
    """
    # Calculate mean and covariance of historical returns
    mu = returns.mean()
    cov = returns.cov()
    
    # Apply shocks to mean returns
    shocked_mu = mu.copy()
    for asset, shock in shock_params.items():
        if asset in shocked_mu.index:
            shocked_mu[asset] += shock / 100  # Convert percentage points to decimal
        else:
            logger.warning(f"Asset {asset} not found in returns data")
    
    # Apply correlation adjustment if provided
    if correlation_adjustment is not None:
        # Get correlation matrix
        corr = returns.corr()
        
        # Apply adjustment (ensuring values stay in [-1, 1] range)
        adjusted_corr = corr.copy()
        for i in range(len(corr)):
            for j in range(i+1, len(corr)):
                new_corr = corr.iloc[i, j] + correlation_adjustment
                adjusted_corr.iloc[i, j] = max(-1, min(1, new_corr))
                adjusted_corr.iloc[j, i] = adjusted_corr.iloc[i, j]  # Maintain symmetry
        
        # Convert back to covariance matrix
        std = np.sqrt(np.diag(cov))
        shocked_cov = adjusted_corr.values * np.outer(std, std)
        shocked_cov = pd.DataFrame(shocked_cov, index=cov.index, columns=cov.columns)
    else:
        shocked_cov = cov
    
    # Apply volatility adjustment if provided
    if volatility_adjustment is not None:
        # Scale variances (diagonal elements)
        for i in range(len(shocked_cov)):
            shocked_cov.iloc[i, i] *= volatility_adjustment
    
    # Simulate returns
    n_simulations = 1000
    n_assets = len(returns.columns)
    
    # Generate random normal samples
    np.random.seed(42)  # For reproducibility
    random_samples = np.random.normal(0, 1, size=(n_simulations, n_assets))
    
    # Calculate Cholesky decomposition of covariance matrix
    try:
        cholesky = np.linalg.cholesky(shocked_cov)
    except np.linalg.LinAlgError:
        # If covariance matrix is not positive definite, make a small adjustment
        logger.warning("Covariance matrix is not positive definite. Adding small diagonal adjustment.")
        min_eigenval = np.min(np.linalg.eigvals(shocked_cov))
        if min_eigenval < 0:
            shocked_cov = shocked_cov - 1.1 * min_eigenval * np.eye(n_assets)
        cholesky = np.linalg.cholesky(shocked_cov)
    
    # Generate simulated returns
    simulated_returns = shocked_mu.values + np.dot(random_samples, cholesky.T)
    
    # Create DataFrame
    simulated_df = pd.DataFrame(simulated_returns, columns=returns.columns)
    
    return simulated_df

def macro_shock_scenario(returns, macro_data=None, macro_shocks=None, sensitivity_model=None):
    """
    Apply macroeconomic shocks to asset returns based on estimated sensitivities.
    
    Args:
        returns (pandas.DataFrame): Asset returns data
        macro_data (pandas.DataFrame, optional): Macroeconomic factor data
        macro_shocks (dict): Dictionary mapping factor names to shock magnitudes
        sensitivity_model (dict, optional): Pre-calculated sensitivity model
        
    Returns:
        pandas.DataFrame: Simulated returns under the macro shock scenario
    """
    # If sensitivity model not provided, estimate from data
    if sensitivity_model is None and macro_data is not None:
        sensitivity_model = estimate_macro_sensitivities(returns, macro_data)
    
    if sensitivity_model is None:
        raise ValueError("Either sensitivity_model or macro_data must be provided")
    
    # Apply macro shocks
    shocked_returns = []
    
    for asset in returns.columns:
        if asset in sensitivity_model:
            # Get asset sensitivities
            asset_model = sensitivity_model[asset]
            
            # Calculate expected return under shock
            expected_return = asset_model['alpha']
            
            # Add factor contributions
            for factor, shock in macro_shocks.items():
                if factor in asset_model['betas']:
                    expected_return += asset_model['betas'][factor] * shock
            
            # Simulate returns with shock
            n_simulations = 1000
            residual_std = asset_model.get('residual_std', returns[asset].std())
            shocked_asset_returns = np.random.normal(expected_return, residual_std, n_simulations)
            
            shocked_returns.append(shocked_asset_returns)
        else:
            # If asset not in model, use historical distribution
            n_simulations = 1000
            mu = returns[asset].mean()
            sigma = returns[asset].std()
            shocked_asset_returns = np.random.normal(mu, sigma, n_simulations)
            
            shocked_returns.append(shocked_asset_returns)
    
    # Combine into DataFrame
    shocked_returns = np.column_stack(shocked_returns)
    shocked_df = pd.DataFrame(shocked_returns, columns=returns.columns)
    
    return shocked_df

def estimate_macro_sensitivities(returns, macro_data):
    """
    Estimate asset sensitivities to macroeconomic factors.
    
    Args:
        returns (pandas.DataFrame): Asset returns data
        macro_data (pandas.DataFrame): Macroeconomic factor data
        
    Returns:
        dict: Dictionary mapping asset names to sensitivity models
    """
    import statsmodels.api as sm
    
    # Align dates
    aligned_dates = returns.index.intersection(macro_data.index)
    returns_aligned = returns.loc[aligned_dates]
    macro_aligned = macro_data.loc[aligned_dates]
    
    # Estimate sensitivities for each asset
    sensitivity_model = {}
    
    for asset in returns.columns:
        # Add constant
        X = sm.add_constant(macro_aligned)
        y = returns_aligned[asset]
        
        # Fit linear model
        try:
            model = sm.OLS(y, X).fit()
            
            # Store parameters
            sensitivity_model[asset] = {
                'alpha': model.params[0],
                'betas': {factor: model.params[i+1] for i, factor in enumerate(macro_aligned.columns)},
                'r_squared': model.rsquared,
                'residual_std': model.resid.std()
            }
        except Exception as e:
            logger.warning(f"Error estimating sensitivities for {asset}: {e}")
    
    return sensitivity_model

def liquidity_stress_scenario(returns, liquidity_params):
    """
    Simulate a liquidity stress scenario where correlations increase and volatility spikes.
    
    Args:
        returns (pandas.DataFrame): Asset returns data
        liquidity_params (dict): Parameters for liquidity stress, including:
            - volatility_multiplier: Multiplier for volatility
            - correlation_increase: Increase in correlations
            - mean_return_shock: Shock to mean returns (negative for stress)
        
    Returns:
        pandas.DataFrame: Simulated returns under the liquidity stress scenario
    """
    # Extract parameters
    vol_mult = liquidity_params.get('volatility_multiplier', 2.0)
    corr_increase = liquidity_params.get('correlation_increase', 0.2)
    mean_shock = liquidity_params.get('mean_return_shock', -0.01)  # -1% daily return
    
    # Create shock dictionary for all assets
    shock_params = {asset: mean_shock * 100 for asset in returns.columns}  # Convert to percentage points
    
    # Generate synthetic scenario with increased correlations and volatility
    stressed_returns = synthetic_scenario(
        returns=returns,
        shock_params=shock_params,
        correlation_adjustment=corr_increase,
        volatility_adjustment=vol_mult
    )
    
    return stressed_returns

def multiple_scenario_analysis(returns, scenarios, investment_value=1000000, confidence_level=0.95):
    """
    Run multiple stress scenarios and compare results.
    
    Args:
        returns (pandas.DataFrame): Asset returns data
        scenarios (dict): Dictionary mapping scenario names to scenario parameters
        investment_value (float): Current portfolio value
        confidence_level (float): Confidence level for VaR calculation
        
    Returns:
        pandas.DataFrame: Results of scenario analysis
    """
    from ..models.historical_var import calculate_historical_var, calculate_conditional_var
    
    # Run each scenario
    results = []
    
    # Baseline (historical) VaR
    baseline_var = calculate_historical_var(
        returns=returns,
        confidence_level=confidence_level,
        investment_value=investment_value
    )
    
    baseline_es = calculate_conditional_var(
        returns=returns,
        confidence_level=confidence_level,
        investment_value=investment_value
    )
    
    results.append({
        'scenario': 'Baseline',
        'var_value': baseline_var,
        'es_value': baseline_es,
        'var_pct': baseline_var / investment_value * 100,
        'es_pct': baseline_es / investment_value * 100,
        'max_loss': returns.min().min() * investment_value * -1,
        'max_loss_pct': returns.min().min() * -100
    })
    
    # Run each stress scenario
    for scenario_name, scenario_params in scenarios.items():
        scenario_type = scenario_params.get('type', 'historical')
        
        if scenario_type == 'historical':
            # Historical scenario
            scenario_returns = historical_scenario(
                returns=returns,
                crisis_period=scenario_params.get('crisis_period'),
                start_date=scenario_params.get('start_date'),
                end_date=scenario_params.get('end_date')
            )
        elif scenario_type == 'synthetic':
            # Synthetic scenario
            scenario_returns = synthetic_scenario(
                returns=returns,
                shock_params=scenario_params.get('shock_params', {}),
                correlation_adjustment=scenario_params.get('correlation_adjustment'),
                volatility_adjustment=scenario_params.get('volatility_adjustment')
            )
        elif scenario_type == 'macro':
            # Macro shock scenario
            scenario_returns = macro_shock_scenario(
                returns=returns,
                macro_data=scenario_params.get('macro_data'),
                macro_shocks=scenario_params.get('macro_shocks', {}),
                sensitivity_model=scenario_params.get('sensitivity_model')
            )
        elif scenario_type == 'liquidity':
            # Liquidity stress scenario
            scenario_returns = liquidity_stress_scenario(
                returns=returns,
                liquidity_params=scenario_params.get('liquidity_params', {})
            )
        else:
            logger.warning(f"Unknown scenario type: {scenario_type}")
            continue

         # Skip if no data for this scenario period
        if len(scenario_returns) == 0 or (hasattr(scenario_returns, 'empty') and scenario_returns.empty):
            print(f"Skipping {scenario_name}: No returns data available for this period")
            continue
        
        # Calculate VaR and ES for this scenario
        scenario_var = calculate_historical_var(
            returns=scenario_returns,
            confidence_level=confidence_level,
            investment_value=investment_value
        )
        
        scenario_es = calculate_conditional_var(
            returns=scenario_returns,
            confidence_level=confidence_level,
            investment_value=investment_value
        )
        
        # Calculate maximum loss
        max_loss = scenario_returns.min().min() * investment_value * -1
        
        # Add to results
        results.append({
            'scenario': scenario_name,
            'var_value': scenario_var,
            'es_value': scenario_es,
            'var_pct': scenario_var / investment_value * 100,
            'es_pct': scenario_es / investment_value * 100,
            'max_loss': max_loss,
            'max_loss_pct': scenario_returns.min().min() * -100
        })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    return results_df

def run_predefined_scenarios(returns, investment_value=1000000, confidence_level=0.95):
    """
    Run a set of predefined stress scenarios.
    
    Args:
        returns (pandas.DataFrame): Asset returns data
        investment_value (float): Current portfolio value
        confidence_level (float): Confidence level for VaR calculation
        
    Returns:
        pandas.DataFrame: Results of scenario analysis
    """
    # Load crisis periods
    from ..data.retrieve_data import load_crisis_periods
    crisis_periods = load_crisis_periods()
    
    # Define scenarios
    scenarios = {}
    
    # Add historical crisis scenarios
    for _, crisis in crisis_periods.iterrows():
        scenarios[crisis['name']] = {
            'type': 'historical',
            'crisis_period': crisis['name']
        }
    
    # Add synthetic scenarios
    scenarios['Severe Rate Hike'] = {
        'type': 'synthetic',
        'shock_params': {asset: -20 for asset in returns.columns},  # -20% shock to all assets
        'volatility_adjustment': 1.5  # 50% increase in volatility
    }
    
    scenarios['Equity Market Crash'] = {
        'type': 'synthetic',
        'shock_params': {asset: -40 for asset in returns.columns if 'bond' not in asset.lower()},  # -40% shock to equities
        'correlation_adjustment': 0.3  # Increase in correlations
    }
    
    scenarios['Liquidity Crisis'] = {
        'type': 'liquidity',
        'liquidity_params': {
            'volatility_multiplier': 2.5,
            'correlation_increase': 0.4,
            'mean_return_shock': -0.02  # -2% daily return
        }
    }
    
    # Run scenarios
    results = multiple_scenario_analysis(
        returns=returns,
        scenarios=scenarios,
        investment_value=investment_value,
        confidence_level=confidence_level
    )
    
    return results