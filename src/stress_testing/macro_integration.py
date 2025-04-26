"""
Macroeconomic factor integration module for stress testing.
This module provides functionality for incorporating macroeconomic factors into stress tests.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
import logging

# Set up logger
logger = logging.getLogger(__name__)

def load_macro_factors(macro_data=None):
    """
    Load macroeconomic factors from file or use provided data.
    
    Args:
        macro_data (pandas.DataFrame, optional): Provided macroeconomic data
        
    Returns:
        pandas.DataFrame: Macroeconomic factor data
    """
    if macro_data is not None:
        return macro_data
    
    # Load macro data from processed directory
    try:
        from pathlib import Path
        project_root = Path(__file__).resolve().parents[2]
        processed_dir = project_root / "data" / "processed"
        
        macro_file = processed_dir / "macro_data.csv"
        if macro_file.exists():
            macro_data = pd.read_csv(macro_file, index_col=0, parse_dates=True)
            logger.info(f"Loaded macro data with shape {macro_data.shape}")
            return macro_data
        else:
            logger.warning(f"Macro data file not found at {macro_file}")
            return None
    except Exception as e:
        logger.error(f"Error loading macro data: {e}")
        return None

def estimate_factor_model(returns, macro_data):
    """
    Estimate a factor model relating asset returns to macroeconomic factors.
    
    Args:
        returns (pandas.DataFrame): Asset returns
        macro_data (pandas.DataFrame): Macroeconomic factor data
        
    Returns:
        dict: Factor model with coefficients and statistics
    """
    # Align dates
    aligned_dates = returns.index.intersection(macro_data.index)
    if len(aligned_dates) == 0:
        raise ValueError("No common dates between returns and macro data")
    
    returns_aligned = returns.loc[aligned_dates]
    macro_aligned = macro_data.loc[aligned_dates]
    
    # Create factor model for each asset
    factor_model = {}
    
    for asset in returns.columns:
        # Add constant for regression
        X = sm.add_constant(macro_aligned)
        y = returns_aligned[asset]
        
        # Fit linear model
        try:
            model = sm.OLS(y, X).fit()
            
            # Extract coefficients and statistics
            factor_model[asset] = {
                'alpha': model.params[0],
                'betas': {factor: model.params[i+1] for i, factor in enumerate(macro_aligned.columns)},
                'r_squared': model.rsquared,
                'p_values': {factor: model.pvalues[i+1] for i, factor in enumerate(macro_aligned.columns)},
                'residual_std': model.resid.std()
            }
            
            logger.info(f"Factor model for {asset}: R² = {model.rsquared:.4f}")
        except Exception as e:
            logger.warning(f"Error estimating factor model for {asset}: {e}")
    
    return factor_model

def simulate_macro_scenario(factor_model, macro_shocks, n_simulations=1000, random_seed=None):
    """
    Simulate asset returns based on a macroeconomic scenario.
    
    Args:
        factor_model (dict): Estimated factor model
        macro_shocks (dict): Dictionary mapping factor names to shocked values
        n_simulations (int): Number of simulations to generate
        random_seed (int, optional): Random seed for reproducibility
        
    Returns:
        pandas.DataFrame: Simulated asset returns under the scenario
    """
    # Set random seed if specified
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Generate simulations for each asset
    simulated_returns = {}
    
    for asset, model in factor_model.items():
        # Calculate expected return under scenario
        expected_return = model['alpha']
        
        for factor, shock in macro_shocks.items():
            if factor in model['betas']:
                expected_return += model['betas'][factor] * shock
        
        # Generate random residuals
        residuals = np.random.normal(0, model['residual_std'], n_simulations)
        
        # Calculate simulated returns
        simulated_returns[asset] = expected_return + residuals
    
    # Create DataFrame
    return pd.DataFrame(simulated_returns)

def create_macro_stress_scenarios(macro_data=None):
    """
    Create predefined macroeconomic stress scenarios.
    
    Args:
        macro_data (pandas.DataFrame, optional): Macroeconomic factor data
        
    Returns:
        dict: Dictionary of stress scenarios
    """
    # Load macro data if not provided
    if macro_data is None:
        macro_data = load_macro_factors()
        
    if macro_data is None:
        logger.warning("No macro data available for creating scenarios")
        return {}
    
    # Calculate historical statistics
    macro_stats = {
        'mean': macro_data.mean(),
        'std': macro_data.std(),
        'min': macro_data.min(),
        'max': macro_data.max()
    }
    
    # Create stress scenarios
    scenarios = {}
    
    # 1. Fed Funds Rate Shock
    if 'DFF' in macro_data.columns:
        ff_shock = macro_stats['std']['DFF'] * 3  # 3 standard deviation shock
        scenarios['Fed Rate Shock'] = {
            'description': 'Rapid increase in Fed Funds Rate',
            'shocks': {
                'DFF': macro_stats['mean']['DFF'] + ff_shock
            }
        }
    
    # 2. VIX Spike
    if 'VIXCLS' in macro_data.columns:
        vix_shock = macro_stats['std']['VIXCLS'] * 4  # 4 standard deviation shock
        scenarios['VIX Spike'] = {
            'description': 'Dramatic increase in market volatility',
            'shocks': {
                'VIXCLS': macro_stats['mean']['VIXCLS'] + vix_shock
            }
        }
    
    # 3. Credit Spread Widening
    if 'TEDRATE' in macro_data.columns:
        ted_shock = macro_stats['std']['TEDRATE'] * 3  # 3 standard deviation shock
        scenarios['Credit Stress'] = {
            'description': 'Widening credit spreads and banking stress',
            'shocks': {
                'TEDRATE': macro_stats['mean']['TEDRATE'] + ted_shock
            }
        }
    
    # 4. Combined Stress (financial crisis-like)
    combined_shocks = {}
    for col in macro_data.columns:
        # Determine direction of shock based on historical crisis behavior
        if col in ['VIXCLS', 'TEDRATE']:
            # These typically increase in a crisis
            combined_shocks[col] = macro_stats['mean'][col] + macro_stats['std'][col] * 3
        elif col in ['DFF']:
            # Fed funds rate typically falls in a crisis (policy response)
            combined_shocks[col] = max(0, macro_stats['mean'][col] - macro_stats['std'][col])
    
    scenarios['Financial Crisis'] = {
        'description': 'Combined stress similar to 2008 financial crisis',
        'shocks': combined_shocks
    }
    
    # 5. Historical worst-case scenario
    historical_worst = {}
    for col in macro_data.columns:
        # Determine worst case (can be min or max, depending on the factor)
        if col in ['VIXCLS', 'TEDRATE']:
            # Higher is worse for these
            historical_worst[col] = macro_stats['max'][col]
        elif col in ['DFF']:
            # Can be worst as either high or low, use min for now
            historical_worst[col] = macro_stats['min'][col]
    
    scenarios['Historical Worst'] = {
        'description': 'Combination of worst historical factor values',
        'shocks': historical_worst
    }
    
    return scenarios

def run_macro_scenario_analysis(returns, macro_data, scenarios=None, investment_value=1000000, confidence_level=0.95):
    """
    Run a comprehensive macroeconomic scenario analysis.
    
    Args:
        returns (pandas.DataFrame): Asset returns
        macro_data (pandas.DataFrame): Macroeconomic factor data
        scenarios (dict, optional): Dictionary of scenarios to run
        investment_value (float): Current portfolio value
        confidence_level (float): Confidence level for VaR calculation
        
    Returns:
        pandas.DataFrame: Results of scenario analysis
    """
def run_macro_scenario_analysis(returns, macro_data, scenarios, investment_value, confidence_level):
    from ..models.historical_var import calculate_historical_var
    
    # Estimate factor model
    factor_model = estimate_factor_model(returns, macro_data)
    
    # Create scenarios if not provided
    if scenarios is None:
        scenarios = create_macro_stress_scenarios(macro_data)
    
    # Create portfolio returns if multiple columns are provided
    if isinstance(returns, pd.DataFrame) and returns.shape[1] > 1:
        # Use equal weights for portfolio construction
        weights = np.ones(returns.shape[1]) / returns.shape[1]
        portfolio_returns = pd.Series(returns.dot(weights), index=returns.index)
    elif isinstance(returns, pd.DataFrame) and returns.shape[1] == 1:
        # If DataFrame with one column, convert to Series
        portfolio_returns = returns.iloc[:, 0]
    else:
        # Already a Series or other format
        portfolio_returns = returns
    
    # Run scenarios
    results = []
    
    # Baseline (historical) VaR
    baseline_var = calculate_historical_var(
        returns=portfolio_returns,
        confidence_level=confidence_level,
        investment_value=investment_value
    )
    
    results.append({
        'scenario': 'Baseline',
        'description': 'Historical VaR',
        'var_value': baseline_var,
        'var_pct': baseline_var / investment_value * 100
    })
    
    # Run each scenario
    for scenario_name, scenario in scenarios.items():
        # Simulate returns under scenario
        sim_returns = simulate_macro_scenario(
            factor_model=factor_model,
            macro_shocks=scenario['shocks'],
            n_simulations=10000,
            random_seed=42
        )
        
        # Convert simulated returns to portfolio returns if needed
        if isinstance(sim_returns, pd.DataFrame) and sim_returns.shape[1] > 1:
            weights = np.ones(sim_returns.shape[1]) / sim_returns.shape[1]
            portfolio_sim_returns = pd.Series(sim_returns.dot(weights), index=sim_returns.index)
        elif isinstance(sim_returns, pd.DataFrame) and sim_returns.shape[1] == 1:
            portfolio_sim_returns = sim_returns.iloc[:, 0]
        else:
            portfolio_sim_returns = sim_returns
        
        # Calculate VaR under scenario
        scenario_var = calculate_historical_var(
            returns=portfolio_sim_returns,
            confidence_level=confidence_level,
            investment_value=investment_value
        )
        
        # Calculate maximum loss
        max_loss = sim_returns.min().min() * investment_value * -1
        
        # Add to results
        results.append({
            'scenario': scenario_name,
            'description': scenario['description'],
            'var_value': scenario_var,
            'var_pct': scenario_var / investment_value * 100,
            'var_change': scenario_var - baseline_var,
            'var_change_pct': (scenario_var - baseline_var) / baseline_var * 100,
            'max_loss': max_loss,
            'max_loss_pct': sim_returns.min().min() * -100
        })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    return results_df

def visualize_macro_scenarios(scenario_results):
    """
    Create visualizations for macro scenario analysis.
    
    Args:
        scenario_results (pandas.DataFrame): Results of scenario analysis
        
    Returns:
        matplotlib.figure.Figure: Figure containing the visualization
    """
    import matplotlib.pyplot as plt
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Sort by VaR value
    sorted_results = scenario_results.sort_values(by='var_value', ascending=False)
    
    # Plot VaR values
    bar_plot = ax1.barh(sorted_results['scenario'], sorted_results['var_value'], color='skyblue')
    ax1.set_title('Value-at-Risk by Scenario')
    ax1.set_xlabel('VaR ($)')
    ax1.set_ylabel('Scenario')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(sorted_results['var_value']):
        ax1.text(v + v*0.01, i, f"${v:,.0f}", va='center')
    
    # Plot VaR change percentage
    if 'var_change_pct' in sorted_results.columns:
        # Exclude baseline (which has no change)
        change_data = sorted_results[sorted_results['scenario'] != 'Baseline']
        change_data = change_data.sort_values(by='var_change_pct', ascending=False)
        
        colors = ['red' if x > 0 else 'green' for x in change_data['var_change_pct']]
        ax2.barh(change_data['scenario'], change_data['var_change_pct'], color=colors)
        ax2.set_title('VaR Change from Baseline (%)')
        ax2.set_xlabel('Change (%)')
        ax2.set_ylabel('Scenario')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(change_data['var_change_pct']):
            ax2.text(v + v*0.01 if v > 0 else v - abs(v)*0.05, i, f"{v:+.1f}%", va='center')
    
    plt.tight_layout()
    
    return fig