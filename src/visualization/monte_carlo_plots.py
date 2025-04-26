"""
Visualization utilities for Monte Carlo simulations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import logging

# Set up logger
logger = logging.getLogger(__name__)

def plot_simulation_paths(simulated_returns, n_paths=100, investment_value=1000000, title="Monte Carlo Simulation Paths"):
    """
    Plot simulation paths for portfolio values.
    
    Args:
        simulated_returns (numpy.ndarray): Simulated portfolio returns
        n_paths (int): Number of paths to display
        investment_value (float): Initial investment value
        title (str): Plot title
        
    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    try:
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Calculate cumulative returns
        if n_paths < len(simulated_returns):
            # Select a random subset of paths
            indices = np.random.choice(len(simulated_returns), n_paths, replace=False)
            paths_to_plot = simulated_returns[indices]
        else:
            paths_to_plot = simulated_returns
        
        # Calculate cumulative portfolio values
        cum_values = investment_value * np.cumprod(1 + paths_to_plot, axis=1)
        
        # Plot each path
        time_steps = cum_values.shape[1]
        x = np.arange(time_steps)
        
        for i in range(len(paths_to_plot)):
            ax.plot(x, cum_values[i], linewidth=0.8, alpha=0.5)
        
        # Plot the mean path
        mean_path = np.mean(cum_values, axis=0)
        ax.plot(x, mean_path, 'r-', linewidth=2, label="Mean Path")
        
        # Add 5th and 95th percentiles
        percentile_5 = np.percentile(cum_values, 5, axis=0)
        percentile_95 = np.percentile(cum_values, 95, axis=0)
        
        ax.plot(x, percentile_5, 'k--', linewidth=1.5, label="5th Percentile")
        ax.plot(x, percentile_95, 'k--', linewidth=1.5, label="95th Percentile")
        
        # Fill the area between percentiles
        ax.fill_between(x, percentile_5, percentile_95, color='blue', alpha=0.1)
        
        # Add horizontal line at initial investment
        ax.axhline(y=investment_value, color='black', linestyle='-', linewidth=1, alpha=0.3)
        
        # Customize plot
        ax.set_title(title)
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Portfolio Value ($)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating simulation paths plot: {e}")
        return None

def plot_simulated_distribution(simulated_returns, investment_value=1000000, confidence_level=0.95, 
                             title="Simulated Return Distribution"):
    """
    Plot the distribution of simulated portfolio returns.
    
    Args:
        simulated_returns (numpy.ndarray): Simulated portfolio returns
        investment_value (float): Initial investment value
        confidence_level (float): Confidence level for VaR calculation
        title (str): Plot title
        
    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    try:
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Calculate portfolio values
        if simulated_returns.ndim > 1 and simulated_returns.shape[1] > 1:
            # For multi-period simulations, take the final values
            final_returns = simulated_returns[:, -1]
        else:
            # For single-period simulations
            final_returns = simulated_returns
        
        # Calculate simulated portfolio values
        simulated_values = investment_value * (1 + final_returns)
        
        # Calculate potential losses
        potential_losses = investment_value - simulated_values
        
        # Calculate VaR
        var_percentile = confidence_level
        var_value = np.percentile(potential_losses, var_percentile * 100)
        
        # Plot distribution of potential losses
        sns.histplot(potential_losses, bins=50, kde=True, alpha=0.6, ax=ax)
        
        # Add vertical line for VaR
        ax.axvline(x=var_value, color='red', linestyle='--', 
                   label=f"VaR ({confidence_level*100}%): ${var_value:,.2f}")
        
        # Add vertical line for 0 (breakeven)
        ax.axvline(x=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        
        # Shade the area beyond VaR
        x_vals = np.linspace(var_value, potential_losses.max(), 1000)
        y_vals = ax.get_ylim()[1] * stats.kde.gaussian_kde(potential_losses)(x_vals)
        ax.fill_between(x_vals, y_vals, alpha=0.3, color='red', 
                         label=f"Expected Shortfall Region")
        
        # Calculate Expected Shortfall
        es_value = potential_losses[potential_losses >= var_value].mean()
        ax.axvline(x=es_value, color='darkred', linestyle=':', 
                   label=f"Expected Shortfall: ${es_value:,.2f}")
        
        # Customize plot
        ax.set_title(title)
        ax.set_xlabel("Potential Loss ($)")
        ax.set_ylabel("Frequency")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating simulated distribution plot: {e}")
        return None

def plot_monte_carlo_comparison(simulated_returns_dict, investment_value=1000000, confidence_level=0.95,
                             title="Monte Carlo Methods Comparison"):
    """
    Compare different Monte Carlo simulation methods.
    
    Args:
        simulated_returns_dict (dict): Dictionary mapping method names to simulated returns
        investment_value (float): Initial investment value
        confidence_level (float): Confidence level for VaR calculation
        title (str): Plot title
        
    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    try:
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        
        # Calculate VaR and ES for each method
        results = []
        colors = plt.cm.tab10(np.linspace(0, 1, len(simulated_returns_dict)))
        
        for i, (method, returns) in enumerate(simulated_returns_dict.items()):
            # Calculate final returns
            if returns.ndim > 1 and returns.shape[1] > 1:
                final_returns = returns[:, -1]
            else:
                final_returns = returns
            
            # Calculate potential losses
            simulated_values = investment_value * (1 + final_returns)
            potential_losses = investment_value - simulated_values
            
            # Calculate VaR and ES
            var_percentile = confidence_level
            var_value = np.percentile(potential_losses, var_percentile * 100)
            es_value = potential_losses[potential_losses >= var_value].mean()
            
            results.append({
                'method': method,
                'var': var_value,
                'es': es_value,
                'var_pct': var_value / investment_value * 100,
                'es_pct': es_value / investment_value * 100,
                'mean': np.mean(potential_losses),
                'std': np.std(potential_losses),
                'skew': stats.skew(potential_losses),
                'kurt': stats.kurtosis(potential_losses)
            })
            
            # Plot distribution of potential losses
            sns.histplot(potential_losses, bins=30, kde=True, alpha=0.6, 
                         ax=axes[0], color=colors[i], label=method)
        
        # Add vertical lines for VaR values on first plot
        for i, result in enumerate(results):
            axes[0].axvline(x=result['var'], color=colors[i], linestyle='--', alpha=0.7)
        
        # Customize first plot
        axes[0].set_title("Loss Distributions by Method")
        axes[0].set_xlabel("Potential Loss ($)")
        axes[0].set_ylabel("Density")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Create bar chart for VaR and ES comparison
        methods = [r['method'] for r in results]
        var_values = [r['var'] for r in results]
        es_values = [r['es'] for r in results]
        
        x = np.arange(len(methods))
        width = 0.35
        
        axes[1].bar(x - width/2, var_values, width, label='VaR', alpha=0.7)
        axes[1].bar(x + width/2, es_values, width, label='ES', alpha=0.7)
        
        # Customize second plot
        axes[1].set_title(f"VaR and ES by Method ({confidence_level*100}% Confidence)")
        axes[1].set_xlabel("Method")
        axes[1].set_ylabel("Value ($)")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(methods)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(var_values):
            axes[1].text(i - width/2, v + 0.02*max(var_values), f"${v:,.0f}", 
                         ha='center', va='bottom', fontsize=9)
        
        for i, v in enumerate(es_values):
            axes[1].text(i + width/2, v + 0.02*max(es_values), f"${v:,.0f}", 
                         ha='center', va='bottom', fontsize=9)
        
        # Set overall title
        fig.suptitle(title, fontsize=16)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating Monte Carlo comparison plot: {e}")
        return None

def plot_copula_dependencies(returns, copula_type='t', n_samples=1000, figsize=(15, 12)):
    """
    Plot copula dependencies between asset returns.
    
    Args:
        returns (pandas.DataFrame): Asset returns
        copula_type (str): Type of copula ('gaussian' or 't')
        n_samples (int): Number of samples to generate
        figsize (tuple): Figure size
        
    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    from ..models.copula_models import fit_copula, simulate_from_copula, estimate_marginal_distributions
    
    try:
        # Fit copula
        copula_params = fit_copula(returns, copula_type=copula_type)
        
        # Estimate marginal distributions
        marginal_params = estimate_marginal_distributions(returns)
        
        # Simulate from copula
        simulated_returns = simulate_from_copula(
            copula_params=copula_params,
            marginal_params=marginal_params,
            n_samples=n_samples,
            random_seed=42
        )
        
        # Get asset names
        asset_names = copula_params['asset_names']
        n_assets = len(asset_names)
        
        # Create a grid of scatterplots
        fig, axes = plt.subplots(n_assets, n_assets, figsize=figsize)
        
        # Plot scatterplots
        for i in range(n_assets):
            for j in range(n_assets):
                ax = axes[i, j]
                
                if i == j:
                    # Histogram on diagonal
                    ax.hist(simulated_returns[asset_names[i]], bins=20, alpha=0.7, color='blue')
                    ax.hist(returns[asset_names[i]], bins=20, alpha=0.5, color='red')
                    ax.set_title(asset_names[i])
                else:
                    # Scatterplot on off-diagonal
                    ax.scatter(
                        simulated_returns[asset_names[j]], 
                        simulated_returns[asset_names[i]], 
                        alpha=0.5, s=5, color='blue'
                    )
                    ax.scatter(
                        returns[asset_names[j]], 
                        returns[asset_names[i]], 
                        alpha=0.5, s=5, color='red'
                    )
                
                # Remove ticks for cleaner look
                ax.set_xticks([])
                ax.set_yticks([])
                
                # Add axes labels on the edge
                if i == n_assets - 1:
                    ax.set_xlabel(asset_names[j])
                if j == 0:
                    ax.set_ylabel(asset_names[i])
        
        # Add legend to the first plot
        import matplotlib.patches as mpatches
        blue_patch = mpatches.Patch(color='blue', alpha=0.7, label='Simulated')
        red_patch = mpatches.Patch(color='red', alpha=0.5, label='Historical')
        axes[0, 0].legend(handles=[blue_patch, red_patch])
        
        # Set title
        plt.suptitle(f"{copula_type.capitalize()} Copula Dependencies", fontsize=16)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating copula dependencies plot: {e}")
        return None

def plot_factor_contributions(factor_contributions, title="Risk Factor Contributions to VaR"):
    """
    Plot contributions of risk factors to VaR.
    
    Args:
        factor_contributions (pandas.DataFrame): DataFrame with factor contributions
        title (str): Plot title
        
    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    try:
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
        
        # Sort by absolute contribution (largest first)
        sorted_df = factor_contributions.sort_values(by='component_value', ascending=True)
        
        # Plot contribution in monetary terms
        sorted_df.plot(x='factor', y='component_value', kind='barh', ax=ax1, color='skyblue')
        ax1.set_title('VaR Contribution in Monetary Terms')
        ax1.set_xlabel('Contribution ($)')
        ax1.set_ylabel('Risk Factor')
        ax1.grid(True, alpha=0.3)
        
        # Plot correlation with returns in tail
        sorted_df.plot(x='factor', y='correlation', kind='barh', ax=ax2, color='salmon')
        ax2.set_title('Correlation with Portfolio Returns in Tail')
        ax2.set_xlabel('Correlation')
        ax2.set_ylabel('Risk Factor')
        ax2.grid(True, alpha=0.3)
        
        # Set overall title
        fig.suptitle(title, fontsize=16)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating factor contributions plot: {e}")
        return None

def plot_monte_carlo_var_by_time_horizon(var_results, title="Monte Carlo VaR by Time Horizon"):
    """
    Plot Monte Carlo VaR estimates across different time horizons.
    
    Args:
        var_results (pandas.DataFrame): DataFrame with VaR results by time horizon
        title (str): Plot title
        
    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    try:
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot VaR by time horizon for each confidence level
        confidence_levels = var_results['confidence_level'].unique()
        time_horizons = var_results['time_horizon'].unique()
        
        for conf in confidence_levels:
            conf_results = var_results[var_results['confidence_level'] == conf]
            ax.plot(conf_results['time_horizon'], conf_results['var_value'], 
                    'o-', label=f"{conf*100:.0f}% Confidence")
        
        # Calculate theoretical growth based on square root of time rule
        if len(time_horizons) > 1:
            base_var = var_results[var_results['time_horizon'] == time_horizons[0]]['var_value'].values[0]
            sqrt_time = base_var * np.sqrt(time_horizons / time_horizons[0])
            ax.plot(time_horizons, sqrt_time, 'k--', label='Square Root of Time Rule')
        
        # Customize plot
        ax.set_title(title)
        ax.set_xlabel('Time Horizon (Days)')
        ax.set_ylabel('VaR ($)')
        ax.set_xticks(time_horizons)
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating VaR by time horizon plot: {e}")
        return None

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Example usage
    import os
    from pathlib import Path
    import numpy as np
    
    # Get project root directory
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    PLOTS_DIR = PROJECT_ROOT / "plots"
    PLOTS_DIR.mkdir(exist_ok=True)
    
    # Generate some example simulated returns
    np.random.seed(42)
    n_simulations = 10000
    n_periods = 20
    
    # Method 1: Normal distribution
    mu1, sigma1 = 0.0005, 0.01
    normal_returns = np.random.normal(mu1, sigma1, (n_simulations, n_periods))
    
    # Method 2: t-distribution (heavier tails)
    df = 5
    mu2, sigma2 = 0.0005, 0.01
    t_returns = np.random.standard_t(df, (n_simulations, n_periods)) * sigma2 / np.sqrt(df/(df-2)) + mu2
    
    # Method 3: Skewed distribution
    skew_param = -0.5
    mu3, sigma3 = 0.0005, 0.01
    skewed_returns = np.random.normal(mu3, sigma3, (n_simulations, n_periods))
    skewed_returns = skewed_returns + skew_param * np.abs(skewed_returns)
    
    # Create a dictionary of return simulations
    simulated_returns_dict = {
        'Normal': normal_returns,
        'Student-t': t_returns,
        'Skewed': skewed_returns
    }
    
    # Plot simulation paths
    logger.info("Creating simulation paths plot...")
    fig1 = plot_simulation_paths(normal_returns, n_paths=100, investment_value=1000000)
    if fig1:
        fig1.savefig(PLOTS_DIR / "simulation_paths.png")
        logger.info(f"Saved simulation paths plot to {PLOTS_DIR / 'simulation_paths.png'}")
    
    # Plot simulated distribution
    logger.info("Creating simulated distribution plot...")
    fig2 = plot_simulated_distribution(normal_returns, investment_value=1000000)
    if fig2:
        fig2.savefig(PLOTS_DIR / "simulated_distribution.png")
        logger.info(f"Saved simulated distribution plot to {PLOTS_DIR / 'simulated_distribution.png'}")
    
    # Plot Monte Carlo comparison
    logger.info("Creating Monte Carlo comparison plot...")
    fig3 = plot_monte_carlo_comparison(simulated_returns_dict, investment_value=1000000)
    if fig3:
        fig3.savefig(PLOTS_DIR / "monte_carlo_comparison.png")
        logger.info(f"Saved Monte Carlo comparison plot to {PLOTS_DIR / 'monte_carlo_comparison.png'}")