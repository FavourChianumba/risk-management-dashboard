"""
Utilities for data quality assessment and visualization.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import logging

# Set up logger
logger = logging.getLogger(__name__)

def analyze_data_quality(df, title="Dataset"):
    """
    Analyze data quality and log results.
    
    Args:
        df (pandas.DataFrame): Data to analyze
        title (str): Title for the analysis
        
    Returns:
        dict: Dictionary containing quality metrics
    """
    logger.info(f"Analyzing data quality for {title}")
    
    # Initialize quality metrics
    quality_metrics = {
        'title': title,
        'shape': df.shape,
        'missing_values': 0,
        'missing_percentage': 0.0,
        'outliers': 0,
        'outlier_percentage': 0.0
    }
    
    if df.empty:
        logger.warning(f"Empty DataFrame, skipping analysis for {title}")
        return quality_metrics
    
    # Basic info
    logger.info(f"Shape: {df.shape}")
    logger.info(f"Date range: {df.index.min()} to {df.index.max()}")
    logger.info(f"Total days: {len(df)}")
    
    # Calendar coverage
    date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
    missing_days = len(date_range) - len(df)
    logger.info(f"Calendar days in range: {len(date_range)}")
    logger.info(f"Missing days: {missing_days} ({missing_days / len(date_range):.2%})")
    
    quality_metrics['date_range_start'] = df.index.min()
    quality_metrics['date_range_end'] = df.index.max()
    quality_metrics['total_days'] = len(df)
    quality_metrics['calendar_days'] = len(date_range)
    quality_metrics['missing_days'] = missing_days
    quality_metrics['missing_days_percentage'] = missing_days / len(date_range) if len(date_range) > 0 else 0
    
    # Missing values
    total_cells = df.shape[0] * df.shape[1]
    missing_cells = df.isna().sum().sum()
    logger.info(f"Missing values: {missing_cells} ({missing_cells / total_cells:.2%})")
    
    quality_metrics['missing_values'] = missing_cells
    quality_metrics['missing_percentage'] = missing_cells / total_cells if total_cells > 0 else 0
    
    # Missing values by column
    missing_by_column = {}
    
    if isinstance(df.columns, pd.MultiIndex):
        # For MultiIndex columns, check by level 0 (data type) and level 1 (asset)
        for level0 in df.columns.levels[0]:
            sub_df = df[level0]
            missing = sub_df.isna().sum().sum()
            total = sub_df.shape[0] * sub_df.shape[1]
            missing_by_column[f"{level0}"] = {'missing': missing, 'percentage': missing / total if total > 0 else 0}
            logger.info(f"  - {level0}: {missing} ({missing / total:.2%})")
        
        # Check for price column
        price_col = None
        for col in ['TRDPRC_1', 'Close', 'Adj Close']:
            if col in df.columns.levels[0]:
                price_col = col
                break
        
        if price_col:
            for asset in df[price_col].columns:
                missing = df[price_col][asset].isna().sum()
                missing_by_column[f"{price_col}_{asset}"] = {'missing': missing, 'percentage': missing / len(df)}
                if missing > 0:
                    logger.info(f"  - {asset} ({price_col}): {missing} ({missing / len(df):.2%})")
    else:
        # For regular DataFrame
        for col in df.columns:
            missing = df[col].isna().sum()
            missing_by_column[col] = {'missing': missing, 'percentage': missing / len(df)}
            if missing > 0:
                logger.info(f"  - {col}: {missing} ({missing / len(df):.2%})")
    
    quality_metrics['missing_by_column'] = missing_by_column
    
    # Detect outliers
    outlier_metrics = {}
    outlier_count = 0
    
    if isinstance(df.columns, pd.MultiIndex):
        # For price data, check daily returns for outliers
        if price_col:
            for asset in df[price_col].columns:
                returns = df[price_col][asset].pct_change().dropna()
                
                # Identify outliers as returns beyond 3 standard deviations
                mean = returns.mean()
                std = returns.std()
                outliers = returns[abs(returns - mean) > 3 * std]
                
                outlier_metrics[f"{asset}"] = {
                    'count': len(outliers),
                    'percentage': len(outliers) / len(returns) if len(returns) > 0 else 0
                }
                
                outlier_count += len(outliers)
                
                if len(outliers) > 0:
                    logger.info(f"  - {asset}: {len(outliers)} outliers ({len(outliers) / len(returns):.2%})")
    else:
        # For numeric columns, detect outliers
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            values = df[col].dropna()
            
            # Calculate z-scores
            z_scores = np.abs((values - values.mean()) / values.std())
            outliers = z_scores[z_scores > 3]
            
            outlier_metrics[col] = {
                'count': len(outliers),
                'percentage': len(outliers) / len(values) if len(values) > 0 else 0
            }
            
            outlier_count += len(outliers)
            
            if len(outliers) > 0:
                logger.info(f"  - {col}: {len(outliers)} outliers ({len(outliers) / len(values):.2%})")
    
    quality_metrics['outliers'] = outlier_count
    quality_metrics['outlier_metrics'] = outlier_metrics
    
    # Calculate total number of data points for outlier percentage
    total_data_points = sum(len(df[col].dropna()) for col in df.columns)
    quality_metrics['outlier_percentage'] = outlier_count / total_data_points if total_data_points > 0 else 0
    
    return quality_metrics

def plot_missing_values(df, title="Missing Values"):
    """
    Create a heatmap visualization of missing values.
    
    Args:
        df (pandas.DataFrame): Data to visualize
        title (str): Title for the plot
        
    Returns:
        matplotlib.figure.Figure: Figure object containing the plot
    """
    if df.empty:
        logger.warning(f"Empty DataFrame, skipping {title} plot")
        return None
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Handle MultiIndex DataFrame
    if isinstance(df.columns, pd.MultiIndex):
        # Determine which price column to use
        price_col = None
        for col in ['TRDPRC_1', 'Close', 'Adj Close']:
            if col in df.columns.levels[0]:
                price_col = col
                break
        
        if price_col:
            # Focus on price data
            missing_data = df[price_col].isna()
            
            # Create a heatmap of missing values
            sns.heatmap(missing_data, cmap='viridis', cbar_kws={'label': 'Missing'}, ax=ax)
            ax.set_title(f"Missing Values in {title} - {price_col}")
            ax.set_xlabel("Asset")
            ax.set_ylabel("Date")
        else:
            logger.warning(f"No suitable price columns found in {title}")
            return None
    else:
        # For regular DataFrame
        missing_data = df.isna()
        
        # Create a heatmap of missing values
        sns.heatmap(missing_data, cmap='viridis', cbar_kws={'label': 'Missing'}, ax=ax)
        ax.set_title(f"Missing Values in {title}")
        ax.set_xlabel("Column")
        ax.set_ylabel("Date")
    
    plt.tight_layout()
    
    return fig

def plot_return_distributions(returns, title="Return Distributions"):
    """
    Create a plot of return distributions.
    
    Args:
        returns (pandas.DataFrame): Return data to visualize
        title (str): Title for the plot
        
    Returns:
        matplotlib.figure.Figure: Figure object containing the plot
    """
    if returns.empty:
        logger.warning(f"Empty DataFrame, skipping {title} plot")
        return None
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Handle MultiIndex DataFrame
    if isinstance(returns.columns, pd.MultiIndex):
        for asset in returns.columns.levels[1]:
            asset_returns = returns.xs(asset, axis=1, level=1)
            if not asset_returns.empty:
                sns.kdeplot(asset_returns.iloc[:, 0], label=asset, ax=ax)
    else:
        # For regular DataFrame
        for col in returns.columns:
            sns.kdeplot(returns[col], label=col, ax=ax)
    
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel("Return")
    ax.set_ylabel("Density")
    ax.grid(True)
    ax.legend()
    
    plt.tight_layout()
    
    return fig

def check_normality(returns):
    """
    Check if return data follows a normal distribution.
    
    Args:
        returns (pandas.Series or pandas.DataFrame): Return data to check
        
    Returns:
        dict: Dictionary containing normality test results
    """
    if isinstance(returns, pd.DataFrame):
        if returns.shape[1] == 1:
            # If it's a DataFrame with one column, convert to Series
            returns = returns.iloc[:, 0]
        else:
            logger.warning("Multiple columns provided, normality test will be performed on each column")
            results = {}
            for col in returns.columns:
                results[col] = check_normality(returns[col])
            return results
    
    # Calculate descriptive statistics
    stats_results = {
        'mean': returns.mean(),
        'std_dev': returns.std(),
        'skewness': returns.skew(),
        'kurtosis': returns.kurtosis(),
        'min': returns.min(),
        'max': returns.max()
    }
    
    # Perform Shapiro-Wilk test for normality
    shapiro_test = stats.shapiro(returns)
    stats_results['shapiro_statistic'] = shapiro_test[0]
    stats_results['shapiro_p_value'] = shapiro_test[1]
    stats_results['is_normal'] = shapiro_test[1] > 0.05
    
    # Perform D'Agostino-Pearson test for normality
    k2_test = stats.normaltest(returns)
    stats_results['dagostino_statistic'] = k2_test[0]
    stats_results['dagostino_p_value'] = k2_test[1]
    
    # Perform Kolmogorov-Smirnov test
    ks_test = stats.kstest(returns, 'norm', args=(returns.mean(), returns.std()))
    stats_results['ks_statistic'] = ks_test[0]
    stats_results['ks_p_value'] = ks_test[1]
    
    # Log results
    logger.info(f"Normality test results:")
    logger.info(f"  - Mean: {stats_results['mean']:.6f}")
    logger.info(f"  - Std Dev: {stats_results['std_dev']:.6f}")
    logger.info(f"  - Skewness: {stats_results['skewness']:.4f}")
    logger.info(f"  - Kurtosis: {stats_results['kurtosis']:.4f}")
    logger.info(f"  - Shapiro-Wilk p-value: {stats_results['shapiro_p_value']:.6f}")
    logger.info(f"  - D'Agostino-Pearson p-value: {stats_results['dagostino_p_value']:.6f}")
    logger.info(f"  - Kolmogorov-Smirnov p-value: {stats_results['ks_p_value']:.6f}")
    
    if stats_results['is_normal']:
        logger.info("Data appears to be normally distributed (Shapiro-Wilk p > 0.05)")
    else:
        logger.info("Data does not appear to be normally distributed (Shapiro-Wilk p <= 0.05)")
    
    return stats_results

def plot_normality_tests(returns, title="Return Distribution"):
    """
    Create plots for assessing normality of return data.
    
    Args:
        returns (pandas.Series): Return data to visualize
        title (str): Title for the plot
        
    Returns:
        tuple: Tuple containing two figure objects (histogram, QQ plot)
    """
    if isinstance(returns, pd.DataFrame):
        if returns.shape[1] == 1:
            # If it's a DataFrame with one column, convert to Series
            returns = returns.iloc[:, 0]
        else:
            logger.warning("Multiple columns provided, using first column for normality plot")
            returns = returns.iloc[:, 0]
    
    # First figure: Histogram with normal distribution overlay
    fig1, ax1 = plt.subplots(figsize=(12, 8))
    
    # Plot histogram of returns
    sns.histplot(returns, kde=True, stat="density", label="Actual Returns", ax=ax1)
    
    # Plot normal distribution with same mean and std
    x = np.linspace(returns.min(), returns.max(), 1000)
    y = stats.norm.pdf(x, returns.mean(), returns.std())
    ax1.plot(x, y, 'r--', label="Normal Distribution")
    
    ax1.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    ax1.set_title(f"{title} vs. Normal Distribution")
    ax1.set_xlabel("Return")
    ax1.set_ylabel("Density")
    ax1.grid(True)
    ax1.legend()
    
    # Second figure: Q-Q plot
    fig2, ax2 = plt.subplots(figsize=(10, 10))
    stats.probplot(returns, dist="norm", plot=ax2)
    ax2.set_title(f"Q-Q Plot of {title}")
    ax2.grid(True)
    
    plt.tight_layout()
    
    return fig1, fig2

def calculate_portfolio_metrics(returns):
    """
    Calculate key performance metrics for a portfolio.
    
    Args:
        returns (pandas.Series): Portfolio returns
        
    Returns:
        dict: Dictionary containing portfolio metrics
    """
    if returns.empty:
        logger.warning("Empty returns data, cannot calculate portfolio metrics")
        return {}
    
    # Calculate metrics
    metrics = {
        'mean_return': returns.mean(),
        'std_dev': returns.std(),
        'annualized_return': returns.mean() * 252,  # Assuming daily returns
        'annualized_volatility': returns.std() * np.sqrt(252),  # Assuming daily returns
        'sharpe_ratio': (returns.mean() / returns.std()) * np.sqrt(252),  # Assuming daily returns and 0% risk-free rate
        'skewness': returns.skew(),
        'kurtosis': returns.kurtosis(),
        'min_return': returns.min(),
        'max_return': returns.max(),
        'median_return': returns.median(),
        'negative_returns_pct': (returns < 0).mean() * 100,
        'positive_returns_pct': (returns > 0).mean() * 100
    }
    
    # Calculate drawdowns
    cumulative_returns = (1 + returns).cumprod()
    running_max = cumulative_returns.cummax()
    drawdowns = (cumulative_returns / running_max) - 1
    
    metrics['max_drawdown'] = drawdowns.min()
    metrics['max_drawdown_date'] = drawdowns.idxmin()
    
    # Calculate Value at Risk (VaR) at different confidence levels
    for conf_level in [0.95, 0.99]:
        metrics[f'var_{int(conf_level*100)}'] = np.percentile(returns, 100 * (1 - conf_level))
    
    # Log results
    logger.info(f"Portfolio metrics:")
    logger.info(f"  - Mean daily return: {metrics['mean_return']:.6f}")
    logger.info(f"  - Daily volatility: {metrics['std_dev']:.6f}")
    logger.info(f"  - Annualized return: {metrics['annualized_return']:.4f}")
    logger.info(f"  - Annualized volatility: {metrics['annualized_volatility']:.4f}")
    logger.info(f"  - Sharpe ratio: {metrics['sharpe_ratio']:.4f}")
    logger.info(f"  - Maximum drawdown: {metrics['max_drawdown']:.4f} on {metrics['max_drawdown_date']}")
    logger.info(f"  - VaR (95%): {metrics['var_95']:.6f}")
    logger.info(f"  - VaR (99%): {metrics['var_99']:.6f}")
    
    return metrics