"""
Visualization utilities for Value-at-Risk (VaR) analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import logging

# Set up logger
logger = logging.getLogger(__name__)

def plot_var_distributions(returns, confidence_level=0.95, title="Return Distribution with VaR"):
    """
    Plot return distribution with VaR thresholds from different methodologies.
    
    Args:
        returns (pandas.Series): Portfolio returns
        confidence_level (float): Confidence level for VaR calculation
        title (str): Plot title
        
    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    # Import VaR functions
    from ..models.historical_var import calculate_historical_var
    from ..models.parametric_var import calculate_parametric_var
    
    try:
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Calculate VaR thresholds in return space
        # Historical VaR
        var_percentile = 1 - confidence_level
        hist_var_return = np.percentile(returns, var_percentile * 100)
        
        # Parametric VaR (normal distribution)
        mu = returns.mean()
        sigma = returns.std()
        z_score = stats.norm.ppf(1 - confidence_level)
        param_var_return = mu + z_score * sigma
        
        # Plot histogram of returns
        sns.histplot(returns, bins=50, kde=True, alpha=0.6, ax=ax, label="Return Distribution")
        
        # Add VaR lines
        ax.axvline(x=hist_var_return, color='red', linestyle='--', 
                   label=f"Historical VaR ({confidence_level*100}%): {hist_var_return:.2%}")
        ax.axvline(x=param_var_return, color='blue', linestyle='--', 
                   label=f"Parametric VaR ({confidence_level*100}%): {param_var_return:.2%}")
        
        # Add zero line
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # Customize plot
        ax.set_title(title)
        ax.set_xlabel("Return")
        ax.set_ylabel("Frequency")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating VaR distribution plot: {e}")
        return None

def plot_var_comparison(returns, confidence_levels=None, investment_value=1000000):
    """
    Plot VaR and ES comparison across different methodologies and confidence levels.
    
    Args:
        returns (pandas.Series): Portfolio returns
        confidence_levels (list): List of confidence levels to compare
        investment_value (float): Current portfolio value
        
    Returns:
        tuple: Tuple containing the two generated figures
    """
    # Import VaR functions
    from ..models.historical_var import calculate_historical_var, calculate_conditional_var
    from ..models.parametric_var import calculate_parametric_var, calculate_parametric_expected_shortfall
    
    try:
        # Set default confidence levels if not provided
        if confidence_levels is None:
            confidence_levels = [0.90, 0.95, 0.99]
        
        # Calculate VaR and ES for different methodologies and confidence levels
        results = []
        
        for conf in confidence_levels:
            # Historical VaR and ES
            hist_var = calculate_historical_var(
                returns=returns,
                confidence_level=conf,
                investment_value=investment_value
            )
            
            hist_es = calculate_conditional_var(
                returns=returns,
                confidence_level=conf,
                investment_value=investment_value
            )
            
            # Parametric VaR and ES
            param_var = calculate_parametric_var(
                returns=returns,
                confidence_level=conf,
                investment_value=investment_value
            )
            
            param_es = calculate_parametric_expected_shortfall(
                returns=returns,
                confidence_level=conf,
                investment_value=investment_value
            )
            
            results.append({
                'confidence_level': conf,
                'historical_var': hist_var,
                'historical_es': hist_es,
                'parametric_var': param_var,
                'parametric_es': param_es,
                'hist_es_var_ratio': hist_es / hist_var,
                'param_es_var_ratio': param_es / param_var
            })
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Create first figure - VaR and ES by methodology
        fig1, axes1 = plt.subplots(1, 2, figsize=(15, 6))
        
        # Prepare data for plotting
        conf_levels = results_df['confidence_level'] * 100
        x = np.arange(len(conf_levels))
        width = 0.35
        
        # Plot VaR comparison
        axes1[0].bar(x - width/2, results_df['historical_var'], width, label='Historical VaR')
        axes1[0].bar(x + width/2, results_df['parametric_var'], width, label='Parametric VaR')
        axes1[0].set_title('VaR by Methodology')
        axes1[0].set_xlabel('Confidence Level (%)')
        axes1[0].set_ylabel('Value ($)')
        axes1[0].set_xticks(x)
        axes1[0].set_xticklabels([f"{cl:.0f}%" for cl in conf_levels])
        axes1[0].legend()
        axes1[0].grid(True, alpha=0.3)
        
        # Plot ES comparison
        axes1[1].bar(x - width/2, results_df['historical_es'], width, label='Historical ES')
        axes1[1].bar(x + width/2, results_df['parametric_es'], width, label='Parametric ES')
        axes1[1].set_title('Expected Shortfall by Methodology')
        axes1[1].set_xlabel('Confidence Level (%)')
        axes1[1].set_ylabel('Value ($)')
        axes1[1].set_xticks(x)
        axes1[1].set_xticklabels([f"{cl:.0f}%" for cl in conf_levels])
        axes1[1].legend()
        axes1[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Create second figure - ES/VaR ratio
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        
        # Plot ES/VaR ratios
        ax2.plot(conf_levels, results_df['hist_es_var_ratio'], 'o-', label='Historical ES/VaR Ratio')
        ax2.plot(conf_levels, results_df['param_es_var_ratio'], 's-', label='Parametric ES/VaR Ratio')
        
        ax2.set_title('ES/VaR Ratio by Confidence Level')
        ax2.set_xlabel('Confidence Level (%)')
        ax2.set_ylabel('Ratio')
        ax2.grid(True)
        ax2.legend()
        
        plt.tight_layout()
        
        return fig1, fig2
        
    except Exception as e:
        logger.error(f"Error creating VaR comparison plot: {e}")
        return None, None

def plot_return_with_var(returns, confidence_level=0.95, lookback_window=None):
    """
    Plot portfolio returns with VaR thresholds and highlight breaches.
    
    Args:
        returns (pandas.Series): Portfolio returns
        confidence_level (float): Confidence level for VaR calculation
        lookback_window (int): Lookback window for rolling VaR calculation
        
    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    try:
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Plot portfolio returns
        ax.plot(returns.index, returns, linewidth=1, alpha=0.7, label='Portfolio Returns')
        
        # Calculate static VaR thresholds
        var_percentile = 1 - confidence_level
        hist_var_return = np.percentile(returns, var_percentile * 100)
        
        mu = returns.mean()
        sigma = returns.std()
        z_score = stats.norm.ppf(1 - confidence_level)
        param_var_return = mu + z_score * sigma
        
        # Add horizontal lines for static VaR thresholds
        ax.axhline(y=hist_var_return, color='red', linestyle='--', 
                   label=f'Historical VaR ({confidence_level*100}%): {hist_var_return:.2%}')
        ax.axhline(y=param_var_return, color='blue', linestyle='--', 
                   label=f'Parametric VaR ({confidence_level*100}%): {param_var_return:.2%}')
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # If lookback window provided, calculate rolling VaR
        if lookback_window is not None and lookback_window < len(returns):
            # Create rolling window
            rolling_var = []
            rolling_dates = []
            
            for i in range(lookback_window, len(returns)):
                window = returns.iloc[i-lookback_window:i]
                rolling_var.append(np.percentile(window, var_percentile * 100))
                rolling_dates.append(returns.index[i])
            
            # Plot rolling VaR
            ax.plot(rolling_dates, rolling_var, color='green', linestyle='-', 
                    label=f'Rolling Historical VaR ({lookback_window} days)')
        
        # Highlight VaR breaches
        breaches = returns[returns < hist_var_return]
        ax.scatter(breaches.index, breaches, color='red', s=50, 
                   label=f'VaR Breaches: {len(breaches)} ({len(breaches)/len(returns):.2%})')
        
        # Customize plot
        ax.set_title(f'Portfolio Returns with {confidence_level*100}% VaR Thresholds')
        ax.set_xlabel('Date')
        ax.set_ylabel('Return')
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating returns with VaR plot: {e}")
        return None

def plot_es_contribution(contributions_df, title="Expected Shortfall Contribution by Asset"):
    """
    Plot the contribution of each asset to the portfolio Expected Shortfall.
    
    Args:
        contributions_df (pandas.DataFrame): DataFrame with ES contributions
        title (str): Plot title
        
    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    try:
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
        
        # Sort by absolute contribution (largest first)
        sorted_df = contributions_df.sort_values(by='contribution_monetary', ascending=True)
        
        # Plot contribution in monetary terms
        sorted_df.plot(x='asset', y='contribution_monetary', kind='barh', ax=ax1, color='skyblue')
        ax1.set_title('ES Contribution in Monetary Terms')
        ax1.set_xlabel('Contribution ($)')
        ax1.set_ylabel('Asset')
        ax1.grid(True, alpha=0.3)
        
        # Plot percentage contribution
        sorted_df.plot(x='asset', y='pct_contribution', kind='barh', ax=ax2, color='salmon')
        ax2.set_title('ES Contribution in Percentage')
        ax2.set_xlabel('Contribution (%)')
        ax2.set_ylabel('Asset')
        ax2.grid(True, alpha=0.3)
        
        # Set overall title
        fig.suptitle(title, fontsize=16)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating ES contribution plot: {e}")
        return None

def plot_var_heatmap(results_df, metric='historical_var', title=None):
    """
    Create a heatmap of VaR or ES values across different confidence levels and time horizons.
    
    Args:
        results_df (pandas.DataFrame): DataFrame with VaR/ES results
        metric (str): The metric to plot ('historical_var', 'historical_es', etc.)
        title (str): Plot title
        
    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    try:
        # Create pivot table for heatmap
        pivot_data = results_df.pivot(index='time_horizon', columns='confidence_level', values=metric)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create heatmap
        sns.heatmap(pivot_data, annot=True, fmt=".0f", cmap="YlOrRd", ax=ax)
        
        # Set labels
        ax.set_xlabel('Confidence Level')
        ax.set_ylabel('Time Horizon (Days)')
        
        # Format x-axis labels as percentages
        x_labels = [f"{cl*100:.0f}%" for cl in pivot_data.columns]
        ax.set_xticklabels(x_labels)
        
        # Set title
        if title is None:
            metric_name = metric.replace('_', ' ').title()
            title = f"{metric_name} by Confidence Level and Time Horizon"
        ax.set_title(title)
        
        plt.tight_layout()
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating VaR heatmap: {e}")
        return None

def plot_var_surface(results_df, metric='historical_var', title=None):
    """
    Create a 3D surface plot of VaR or ES values across different confidence levels and time horizons.
    
    Args:
        results_df (pandas.DataFrame): DataFrame with VaR/ES results
        metric (str): The metric to plot ('historical_var', 'historical_es', etc.)
        title (str): Plot title
        
    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    try:
        # Import 3D plotting
        from mpl_toolkits.mplot3d import Axes3D
        
        # Create pivot table for surface plot
        pivot_data = results_df.pivot(index='time_horizon', columns='confidence_level', values=metric)
        
        # Create mesh grid
        confidence_levels = pivot_data.columns
        time_horizons = pivot_data.index
        X, Y = np.meshgrid(confidence_levels, time_horizons)
        Z = pivot_data.values
        
        # Create figure
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create surface plot
        surf = ax.plot_surface(X * 100, Y, Z, cmap='viridis', edgecolor='none', alpha=0.8)
        
        # Add color bar
        cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        cbar.set_label('Value ($)')
        
        # Set labels
        ax.set_xlabel('Confidence Level (%)')
        ax.set_ylabel('Time Horizon (Days)')
        ax.set_zlabel('Value ($)')
        
        # Set title
        if title is None:
            metric_name = metric.replace('_', ' ').title()
            title = f"{metric_name} by Confidence Level and Time Horizon"
        ax.set_title(title)
        
        plt.tight_layout()
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating VaR surface plot: {e}")
        return None

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Example usage
    try:
        import os
        from pathlib import Path
        
        # Get project root directory
        PROJECT_ROOT = Path(__file__).resolve().parents[2]
        DATA_DIR = PROJECT_ROOT / "data"
        PROCESSED_DIR = DATA_DIR / "processed"
        
        # Load portfolio returns
        returns_file = PROCESSED_DIR / "portfolio_returns.csv"
        
        if not returns_file.exists():
            logger.error("Portfolio returns file not found. Please run process_data.py first.")
            exit(1)
        
        # Load returns data
        portfolio_returns = pd.read_csv(returns_file, index_col=0, parse_dates=True)
        
        # Convert DataFrame to Series if needed
        if isinstance(portfolio_returns, pd.DataFrame) and portfolio_returns.shape[1] == 1:
            portfolio_returns = portfolio_returns.iloc[:, 0]
        
        # Create output directory for plots
        plots_dir = PROJECT_ROOT / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        # Plot VaR distributions
        logger.info("Creating VaR distribution plot...")
        fig1 = plot_var_distributions(portfolio_returns)
        if fig1:
            fig1.savefig(plots_dir / "var_distributions.png")
            logger.info(f"Saved VaR distribution plot to {plots_dir / 'var_distributions.png'}")
        
        # Plot VaR comparison
        logger.info("Creating VaR comparison plots...")
        fig2, fig3 = plot_var_comparison(portfolio_returns)
        if fig2:
            fig2.savefig(plots_dir / "var_methodology_comparison.png")
            logger.info(f"Saved VaR methodology comparison plot to {plots_dir / 'var_methodology_comparison.png'}")
        if fig3:
            fig3.savefig(plots_dir / "es_var_ratio.png")
            logger.info(f"Saved ES/VaR ratio plot to {plots_dir / 'es_var_ratio.png'}")
        
        # Plot returns with VaR
        logger.info("Creating returns with VaR plot...")
        fig4 = plot_return_with_var(portfolio_returns, lookback_window=252)
        if fig4:
            fig4.savefig(plots_dir / "returns_with_var.png")
            logger.info(f"Saved returns with VaR plot to {plots_dir / 'returns_with_var.png'}")
        
    except Exception as e:
        logger.error(f"Error in example usage: {e}")