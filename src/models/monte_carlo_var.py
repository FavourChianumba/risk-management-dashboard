"""
Monte Carlo Value-at-Risk (VaR) implementation.
This module calculates VaR using Monte Carlo simulation.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path

# Get project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"

def load_model_config():
    """
    Load model configuration from config file.
    
    Returns:
        dict: Model configuration for Monte Carlo VaR
    """
    with open(CONFIG_DIR / "model_config.json", 'r') as f:
        model_config = json.load(f)
    
    return model_config['var_models']['monte_carlo']

def monte_carlo_var(returns, confidence_level=None, investment_value=1000000, 
                    n_simulations=None, time_horizon=1, random_seed=None):
    """
    Calculate Value-at-Risk using Monte Carlo simulation.
    
    Args:
        returns (pandas.Series): Historical returns data
        confidence_level (float, optional): Confidence level (e.g., 0.95 for 95% confidence)
        investment_value (float, optional): Current portfolio value
        n_simulations (int, optional): Number of Monte Carlo simulations
        time_horizon (int, optional): Time horizon in days
        random_seed (int, optional): Random seed for reproducibility
        
    Returns:
        float: Value-at-Risk estimate
    """
    # Load configuration if parameters not provided
    config = load_model_config()
    
    if confidence_level is None:
        confidence_level = config['default_confidence']
    
    if n_simulations is None:
        n_simulations = config['n_simulations']
    
    if random_seed is None and 'random_seed' in config:
        random_seed = config['random_seed']
    
    # Set random seed if specified
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Calculate mean and standard deviation of returns
    mu = returns.mean()
    sigma = returns.std()
    
    # Generate random returns using a normal distribution
    # This assumes returns follow a normal distribution
    random_returns = np.random.normal(
        loc=mu * time_horizon,
        scale=sigma * np.sqrt(time_horizon),
        size=n_simulations
    )
    
    # Calculate simulated portfolio values
    simulated_values = investment_value * (1 + random_returns)
    
    # Calculate potential losses
    potential_losses = investment_value - simulated_values
    
    # Calculate VaR at specified confidence level
    var_percentile = confidence_level
    var_value = np.percentile(potential_losses, var_percentile * 100)
    
    # Ensure VaR is positive (representing a loss)
    var_value = max(0, var_value)
    
    return var_value

def monte_carlo_expected_shortfall(returns, confidence_level=None, investment_value=1000000, 
                                 n_simulations=None, time_horizon=1, random_seed=None):
    """
    Calculate Expected Shortfall (CVaR) using Monte Carlo simulation.
    
    Args:
        returns (pandas.Series): Historical returns data
        confidence_level (float, optional): Confidence level (e.g., 0.95 for 95% confidence)
        investment_value (float, optional): Current portfolio value
        n_simulations (int, optional): Number of Monte Carlo simulations
        time_horizon (int, optional): Time horizon in days
        random_seed (int, optional): Random seed for reproducibility
        
    Returns:
        float: Expected Shortfall estimate
    """
    # Load configuration if parameters not provided
    config = load_model_config()
    
    if confidence_level is None:
        confidence_level = config['default_confidence']
    
    if n_simulations is None:
        n_simulations = config['n_simulations']
    
    if random_seed is None and 'random_seed' in config:
        random_seed = config['random_seed']
    
    # Set random seed if specified
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Calculate mean and standard deviation of returns
    mu = returns.mean()
    sigma = returns.std()
    
    # Generate random returns using a normal distribution
    random_returns = np.random.normal(
        loc=mu * time_horizon,
        scale=sigma * np.sqrt(time_horizon),
        size=n_simulations
    )
    
    # Calculate simulated portfolio values
    simulated_values = investment_value * (1 + random_returns)
    
    # Calculate potential losses
    potential_losses = investment_value - simulated_values
    
    # Calculate VaR at specified confidence level
    var_percentile = confidence_level
    var_value = np.percentile(potential_losses, var_percentile * 100)
    
    # Calculate Expected Shortfall (average loss beyond VaR)
    losses_beyond_var = potential_losses[potential_losses >= var_value]
    
    # If no losses beyond VaR, return VaR as the Expected Shortfall
    if len(losses_beyond_var) == 0:
        return var_value
    
    expected_shortfall = losses_beyond_var.mean()
    
    return expected_shortfall

def run_monte_carlo_scenarios(returns, confidence_levels=None, investment_value=1000000, 
                             n_simulations=None, time_horizons=None, random_seed=None):
    """
    Run Monte Carlo VaR scenarios for multiple confidence levels and time horizons.
    
    Args:
        returns (pandas.Series): Historical returns data
        confidence_levels (list, optional): List of confidence levels
        investment_value (float, optional): Current portfolio value
        n_simulations (int, optional): Number of Monte Carlo simulations
        time_horizons (list, optional): List of time horizons in days
        random_seed (int, optional): Random seed for reproducibility
        
    Returns:
        pandas.DataFrame: VaR and Expected Shortfall estimates for different scenarios
    """
    # Load configuration if parameters not provided
    config = load_model_config()
    
    if confidence_levels is None:
        confidence_levels = config['confidence_levels']
    
    if n_simulations is None:
        n_simulations = config['n_simulations']
    
    if time_horizons is None:
        time_horizons = [1, 5, 10, 20]
    
    # Generate scenarios
    scenarios = []
    
    for confidence in confidence_levels:
        for horizon in time_horizons:
            # Calculate VaR
            var_value = monte_carlo_var(
                returns=returns,
                confidence_level=confidence,
                investment_value=investment_value,
                n_simulations=n_simulations,
                time_horizon=horizon,
                random_seed=random_seed
            )
            
            # Calculate Expected Shortfall
            es_value = monte_carlo_expected_shortfall(
                returns=returns,
                confidence_level=confidence,
                investment_value=investment_value,
                n_simulations=n_simulations,
                time_horizon=horizon,
                random_seed=random_seed
            )
            
            # Add scenario to results
            scenarios.append({
                'confidence_level': confidence,
                'time_horizon': horizon,
                'var_value': var_value,
                'es_value': es_value
            })
    
    # Convert to DataFrame
    scenarios_df = pd.DataFrame(scenarios)
    
    return scenarios_df

if __name__ == "__main__":
    # Example usage
    try:
        # Load portfolio returns
        data_dir = PROJECT_ROOT / "data" / "processed"
        returns_file = data_dir / "portfolio_returns.csv"
        
        if not returns_file.exists():
            print("Portfolio returns file not found. Please run process_data.py first.")
            exit(1)
        
        # Load returns data
        portfolio_returns = pd.read_csv(returns_file, index_col=0, parse_dates=True)
        portfolio_returns = portfolio_returns.iloc[:, 0]  # Convert DataFrame to Series
        
        # Calculate Monte Carlo VaR
        print("Calculating Monte Carlo VaR...")
        config = load_model_config()
        default_confidence = config['default_confidence']
        
        var_value = monte_carlo_var(
            returns=portfolio_returns,
            confidence_level=default_confidence,
            investment_value=1000000
        )
        
        print(f"Monte Carlo VaR ({default_confidence*100}%): ${var_value:,.2f}")
        
        # Run Monte Carlo scenarios
        print("\nRunning Monte Carlo scenarios...")
        scenarios = run_monte_carlo_scenarios(portfolio_returns)
        print(scenarios)
        
    except Exception as e:
        print(f"Error calculating Monte Carlo VaR: {e}")