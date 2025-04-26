"""
Backtesting module for Value-at-Risk (VaR) model validation.
This module provides functions to backtest VaR models against historical data.
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import logging
from datetime import timedelta

# Set up logger
logger = logging.getLogger(__name__)

def calculate_var_breaches(returns, var_values, significance_level=0.05):
    """
    Calculate VaR breaches (exceptions) over a historical period.
    
    Args:
        returns (pandas.Series): Historical returns
        var_values (pandas.Series): VaR predictions for each date
        significance_level (float): Significance level (alpha) for VaR
        
    Returns:
        pandas.Series: Boolean series indicating breaches
    """
    # Ensure the sign convention is consistent (VaR as a positive value)
    expected_breach_rate = significance_level
    
    # Calculate breaches (where actual loss exceeds VaR)
    breaches = (-returns) > var_values
    
    return breaches

def backtest_var_model(returns, var_model, window_size=252, step_size=1, 
                     confidence_level=0.95, investment_value=1000000, **model_params):
    """
    Perform a rolling window backtest of a VaR model.
    
    Args:
        returns (pandas.Series): Historical returns
        var_model (callable): VaR calculation function
        window_size (int): Size of the rolling window
        step_size (int): Number of days to move forward between calculations
        confidence_level (float): Confidence level for VaR calculation
        investment_value (float): Portfolio value for VaR calculation
        **model_params: Additional parameters for the VaR model
        
    Returns:
        pandas.DataFrame: Backtest results with VaR predictions and breaches
    """
    if len(returns) <= window_size:
        raise ValueError(f"Not enough data for backtesting. Need more than {window_size} observations.")
    
    # Initialize results DataFrame
    results = []
    
    # Calculate significance level from confidence level
    significance_level = 1 - confidence_level
    
    # Perform rolling window backtest
    for i in range(window_size, len(returns), step_size):
        # Extract training window
        train_window = returns.iloc[i-window_size:i]
        
        # Get test date (the day after the training window)
        if i < len(returns):
            test_date = returns.index[i]
            actual_return = returns.iloc[i]
        else:
            break
        
        # Calculate VaR using the training window
        try:
            var_prediction = var_model(
                returns=train_window,
                confidence_level=confidence_level,
                investment_value=investment_value,
                **model_params
            )
        except Exception as e:
            logger.warning(f"Error calculating VaR for date {test_date}: {e}")
            continue
        
        # Calculate VaR as a percentage of investment
        var_pct = var_prediction / investment_value
        
        # Convert VaR to return space (VaR as a negative return)
        var_return = var_pct
        
        # Record results
        results.append({
            'date': test_date,
            'actual_return': actual_return,
            'var_dollar': var_prediction,
            'var_pct': var_pct,
            'var_return': var_return,
            'breach': (-actual_return) > var_return  # True if actual loss exceeds VaR
        })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    results_df.set_index('date', inplace=True)
    
    # Calculate breach rate
    breach_rate = results_df['breach'].mean()
    expected_breach_rate = significance_level
    
    logger.info(f"Backtest completed with {len(results_df)} VaR predictions")
    logger.info(f"Breach rate: {breach_rate:.2%} (Expected: {expected_breach_rate:.2%})")
    
    return results_df

def kupiec_test(breaches, n_observations, significance_level=0.05, confidence_level=0.95):
    """
    Perform Kupiec's proportion of failures (POF) test to assess VaR model accuracy.
    
    Args:
        breaches (pandas.Series or array-like): Boolean series of VaR breaches
        n_observations (int): Total number of observations
        significance_level (float): Significance level (alpha) for the test
        confidence_level (float): Confidence level used for VaR calculation
        
    Returns:
        dict: Results of Kupiec's test
    """
    # Calculate number of breaches
    n_breaches = np.sum(breaches)
    
    # Expected breach rate
    expected_breach_rate = 1 - confidence_level
    
    # Observed breach rate
    observed_breach_rate = n_breaches / n_observations
    
    # Calculate test statistic
    if n_breaches == 0:
        # Avoid log(0) error
        test_statistic = -2 * n_observations * np.log(1 - expected_breach_rate)
    elif n_breaches == n_observations:
        # Avoid log(0) error
        test_statistic = -2 * n_observations * np.log(expected_breach_rate)
    else:
        # Standard calculation
        test_statistic = -2 * np.log(
            (1 - expected_breach_rate) ** (n_observations - n_breaches) * 
            expected_breach_rate ** n_breaches
        ) + 2 * np.log(
            (1 - observed_breach_rate) ** (n_observations - n_breaches) * 
            observed_breach_rate ** n_breaches
        )
    
    # Calculate p-value (chi-squared with 1 degree of freedom)
    p_value = 1 - stats.chi2.cdf(test_statistic, 1)
    
    # Test result
    reject_h0 = p_value < significance_level
    
    # Create result dictionary
    result = {
        'n_observations': n_observations,
        'n_breaches': n_breaches,
        'expected_breach_rate': expected_breach_rate,
        'observed_breach_rate': observed_breach_rate,
        'test_statistic': test_statistic,
        'p_value': p_value,
        'reject_null': reject_h0,
        'result': "Reject" if reject_h0 else "Fail to Reject"
    }
    
    return result

def christoffersen_test(breaches, significance_level=0.05):
    """
    Perform Christoffersen's independence test to check for clustering of VaR breaches.
    
    Args:
        breaches (pandas.Series or array-like): Boolean series of VaR breaches
        significance_level (float): Significance level (alpha) for the test
        
    Returns:
        dict: Results of Christoffersen's test
    """
    # Convert to numpy array if not already
    breaches = np.asarray(breaches)
    
    # Count transitions
    n00 = np.sum((breaches[:-1] == 0) & (breaches[1:] == 0))
    n01 = np.sum((breaches[:-1] == 0) & (breaches[1:] == 1))
    n10 = np.sum((breaches[:-1] == 1) & (breaches[1:] == 0))
    n11 = np.sum((breaches[:-1] == 1) & (breaches[1:] == 1))
    
    # Calculate transition probabilities
    p01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0
    p11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0
    p = (n01 + n11) / (n00 + n01 + n10 + n11)
    
    # Calculate test statistic
    if p01 == 0 or p11 == 0 or p == 0 or p == 1:
        # Handle degenerate cases
        test_statistic = 0
        p_value = 1
    else:
        # Standard calculation
        test_statistic = -2 * np.log(
            (1 - p) ** (n00 + n10) * p ** (n01 + n11)
        ) + 2 * np.log(
            (1 - p01) ** n00 * p01 ** n01 * (1 - p11) ** n10 * p11 ** n11
        )
        
        # Calculate p-value (chi-squared with 1 degree of freedom)
        p_value = 1 - stats.chi2.cdf(test_statistic, 1)
    
    # Test result
    reject_h0 = p_value < significance_level
    
    # Create result dictionary
    result = {
        'transitions': {
            'n00': n00,
            'n01': n01,
            'n10': n10,
            'n11': n11
        },
        'probabilities': {
            'p01': p01,
            'p11': p11,
            'p': p
        },
        'test_statistic': test_statistic,
        'p_value': p_value,
        'reject_null': reject_h0,
        'result': "Reject (Clustered Breaches)" if reject_h0 else "Fail to Reject (Independent Breaches)"
    }
    
    return result

def combined_var_test(breaches, n_observations, confidence_level=0.95, significance_level=0.05):
    """
    Perform combined Kupiec and Christoffersen tests.
    
    Args:
        breaches (pandas.Series or array-like): Boolean series of VaR breaches
        n_observations (int): Total number of observations
        confidence_level (float): Confidence level used for VaR calculation
        significance_level (float): Significance level (alpha) for the test
        
    Returns:
        dict: Results of the combined test
    """
    # Perform Kupiec test
    kupiec_result = kupiec_test(
        breaches=breaches,
        n_observations=n_observations,
        significance_level=significance_level,
        confidence_level=confidence_level
    )
    
    # Perform Christoffersen test
    christoffersen_result = christoffersen_test(
        breaches=breaches,
        significance_level=significance_level
    )
    
    # Calculate combined test statistic
    combined_statistic = kupiec_result['test_statistic'] + christoffersen_result['test_statistic']
    
    # Calculate p-value (chi-squared with 2 degrees of freedom)
    p_value = 1 - stats.chi2.cdf(combined_statistic, 2)
    
    # Test result
    reject_h0 = p_value < significance_level
    
    # Overall assessment
    if kupiec_result['reject_null'] and christoffersen_result['reject_null']:
        assessment = "Model fails both coverage and independence tests"
    elif kupiec_result['reject_null']:
        assessment = "Model fails coverage test but passes independence test"
    elif christoffersen_result['reject_null']:
        assessment = "Model passes coverage test but fails independence test"
    else:
        assessment = "Model passes both coverage and independence tests"
    
    # Create result dictionary
    result = {
        'kupiec_test': kupiec_result,
        'christoffersen_test': christoffersen_result,
        'combined_statistic': combined_statistic,
        'p_value': p_value,
        'reject_null': reject_h0,
        'result': "Reject" if reject_h0 else "Fail to Reject",
        'assessment': assessment
    }
    
    return result

def compare_var_models(returns, var_models, window_size=252, confidence_level=0.95, 
                     investment_value=1000000, significance_level=0.05):
    """
    Compare multiple VaR models using backtesting.
    
    Args:
        returns (pandas.Series): Historical returns
        var_models (dict): Dictionary mapping model names to VaR functions
        window_size (int): Size of the rolling window
        confidence_level (float): Confidence level for VaR calculation
        investment_value (float): Portfolio value for VaR calculation
        significance_level (float): Significance level for statistical tests
        
    Returns:
        dict: Comparison results for each model
    """
    comparison_results = {}
    
    for model_name, var_func in var_models.items():
        logger.info(f"Backtesting {model_name}...")
        
        # Perform backtest
        backtest_results = backtest_var_model(
            returns=returns,
            var_model=var_func,
            window_size=window_size,
            confidence_level=confidence_level,
            investment_value=investment_value
        )
        
        # Calculate test statistics
        test_results = combined_var_test(
            breaches=backtest_results['breach'],
            n_observations=len(backtest_results),
            confidence_level=confidence_level,
            significance_level=significance_level
        )
        
        # Store results
        comparison_results[model_name] = {
            'backtest_results': backtest_results,
            'test_results': test_results,
            'breach_rate': backtest_results['breach'].mean(),
            'expected_breach_rate': 1 - confidence_level,
            'avg_var': backtest_results['var_pct'].mean(),
            'max_var': backtest_results['var_pct'].max()
        }
    
    return comparison_results

def plot_backtest_results(backtest_results, confidence_level=0.95, title=None):
    """
    Plot backtest results with VaR predictions and breaches.
    
    Args:
        backtest_results (pandas.DataFrame): Results from backtest_var_model
        confidence_level (float): Confidence level used for VaR calculation
        title (str, optional): Plot title
        
    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot actual returns and VaR
    ax1.plot(backtest_results.index, backtest_results['actual_return'] * 100, 'b-', 
             alpha=0.5, label='Actual Returns')
    ax1.plot(backtest_results.index, -backtest_results['var_return'] * 100, 'r--', 
             label=f'VaR ({confidence_level*100:.0f}%)')
    
    # Highlight breaches
    breach_dates = backtest_results.index[backtest_results['breach']]
    breach_returns = backtest_results.loc[breach_dates, 'actual_return'] * 100
    ax1.scatter(breach_dates, breach_returns, color='red', s=50, label='VaR Breaches')
    
    # Set title and labels
    if title:
        ax1.set_title(title)
    else:
        ax1.set_title(f'VaR Backtest Results ({confidence_level*100:.0f}% Confidence Level)')
    
    ax1.set_ylabel('Return (%)')
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot breach indicators in bottom panel
    ax2.fill_between(backtest_results.index, 0, 1, 
                     where=backtest_results['breach'], 
                     color='red', alpha=0.3, label='Breach')
    
    # Add expected breach rate line
    expected_breach_rate = 1 - confidence_level
    ax2.axhline(y=0.5, color='blue', linestyle='--', 
                label=f'Expected Breach Rate: {expected_breach_rate:.1%}')
    
    # Calculate breach rate
    breach_rate = backtest_results['breach'].mean()
    ax2.text(backtest_results.index[-1], 0.5, 
             f'Actual: {breach_rate:.1%}', 
             ha='right', va='bottom', color='blue')
    
    ax2.set_yticks([])
    ax2.set_xlabel('Date')
    ax2.set_title('VaR Breaches')
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    
    return fig

def plot_breach_clustering(backtest_results, title=None):
    """
    Plot breach clusters to visualize independence/clustering of VaR exceptions.
    
    Args:
        backtest_results (pandas.DataFrame): Results from backtest_var_model
        title (str, optional): Plot title
        
    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Get breach data
    breaches = backtest_results['breach']
    breach_dates = backtest_results.index[breaches]
    
    # Calculate consecutive breaches
    consecutive_breaches = []
    current_run = 0
    
    for date, is_breach in breaches.items():
        if is_breach:
            current_run += 1
        elif current_run > 0:
            consecutive_breaches.append((date - timedelta(days=current_run), current_run))
            current_run = 0
    
    # Add the last run if it exists
    if current_run > 0:
        consecutive_breaches.append((breach_dates[-1], current_run))
    
    # Plot breach timeline
    ax.vlines(breach_dates, 0, 1, color='red', alpha=0.7, label='VaR Breaches')
    
    # Highlight consecutive breaches
    for start_date, run_length in consecutive_breaches:
        if run_length > 1:
            # Plot consecutive breach
            ax.plot(
                [start_date, start_date + timedelta(days=run_length-1)],
                [0.5, 0.5],
                color='darkred',
                linewidth=4,
                alpha=0.7
            )
            
            # Add label
            ax.text(
                start_date + timedelta(days=(run_length-1)/2),
                0.6,
                f"{run_length}",
                ha='center',
                va='bottom',
                color='darkred'
            )
    
    # Set title and labels
    if title:
        ax.set_title(title)
    else:
        ax.set_title('VaR Breach Clustering Analysis')
    
    ax.set_xlabel('Date')
    ax.set_yticks([])
    ax.set_ylim(0, 1)
    
    # Add breach count
    ax.text(
        0.02, 0.95,
        f"Total Breaches: {len(breach_dates)}",
        transform=ax.transAxes,
        ha='left',
        va='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
    )
    
    # Calculate clustering statistics
    n_consecutive = sum(1 for _, run in consecutive_breaches if run > 1)
    max_consecutive = max((run for _, run in consecutive_breaches), default=0)
    
    # Add clustering statistics
    ax.text(
        0.02, 0.85,
        f"Consecutive Breach Runs: {n_consecutive}\nLongest Run: {max_consecutive}",
        transform=ax.transAxes,
        ha='left',
        va='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
    )
    
    plt.tight_layout()
    
    return fig

def create_backtest_summary(comparison_results, confidence_level=0.95):
    """
    Create a summary DataFrame of backtest results for multiple models.
    
    Args:
        comparison_results (dict): Output from compare_var_models
        confidence_level (float): Confidence level used for VaR calculation
        
    Returns:
        pandas.DataFrame: Summary of backtest results
    """
    # Expected breach rate
    expected_breach_rate = 1 - confidence_level
    
    # Create summary data
    summary_data = []
    
    for model_name, results in comparison_results.items():
        backtest = results['backtest_results']
        tests = results['test_results']
        
        # Calculate breach statistics
        n_breaches = backtest['breach'].sum()
        breach_rate = backtest['breach'].mean()
        
        # Add model summary
        summary_data.append({
            'Model': model_name,
            'Avg VaR (%)': backtest['var_pct'].mean() * 100,
            'Max VaR (%)': backtest['var_pct'].max() * 100,
            'Breaches': n_breaches,
            'Breach Rate (%)': breach_rate * 100,
            'Expected (%)': expected_breach_rate * 100,
            'Breach Ratio': breach_rate / expected_breach_rate,
            'Kupiec p-value': tests['kupiec_test']['p_value'],
            'Kupiec Test': tests['kupiec_test']['result'],
            'Christoffersen p-value': tests['christoffersen_test']['p_value'],
            'Christoffersen Test': tests['christoffersen_test']['result'],
            'Combined Test': tests['result'],
            'Assessment': tests['assessment']
        })
    
    # Create DataFrame
    summary_df = pd.DataFrame(summary_data)
    
    # Round numeric columns
    numeric_cols = ['Avg VaR (%)', 'Max VaR (%)', 'Breach Rate (%)', 
                   'Expected (%)', 'Breach Ratio', 'Kupiec p-value', 
                   'Christoffersen p-value']
    
    summary_df[numeric_cols] = summary_df[numeric_cols].round(2)
    
    return summary_df