"""
Unit tests for the historical VaR implementation.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add the src directory to the Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.models.historical_var import (
    calculate_historical_var,
    calculate_conditional_var,
    calculate_var_by_confidence
)

@pytest.fixture
def sample_returns():
    """Create sample returns data for testing."""
    np.random.seed(42)  # For reproducibility
    # Generate 1000 random returns with a normal distribution
    returns = np.random.normal(0.0005, 0.01, 1000)
    return pd.Series(returns)

def test_historical_var_calculation(sample_returns):
    """Test the basic historical VaR calculation."""
    # Calculate VaR at 95% confidence level
    var_95 = calculate_historical_var(
        returns=sample_returns,
        confidence_level=0.95,
        investment_value=1000000
    )
    
    # VaR should be a positive number
    assert var_95 > 0
    
    # Test with different confidence level
    var_99 = calculate_historical_var(
        returns=sample_returns,
        confidence_level=0.99,
        investment_value=1000000
    )
    
    # Higher confidence level should yield higher VaR
    assert var_99 > var_95

def test_conditional_var_calculation(sample_returns):
    """Test the conditional VaR (Expected Shortfall) calculation."""
    # Calculate CVaR at 95% confidence level
    cvar_95 = calculate_conditional_var(
        returns=sample_returns,
        confidence_level=0.95,
        investment_value=1000000
    )
    
    # Calculate regular VaR at 95% confidence level
    var_95 = calculate_historical_var(
        returns=sample_returns,
        confidence_level=0.95,
        investment_value=1000000
    )
    
    # CVaR should be greater than or equal to VaR
    assert cvar_95 >= var_95

def test_var_by_confidence(sample_returns):
    """Test VaR calculation for multiple confidence levels."""
    # Calculate VaR for multiple confidence levels
    confidence_levels = [0.90, 0.95, 0.99]
    var_results = calculate_var_by_confidence(
        returns=sample_returns,
        confidence_levels=confidence_levels,
        investment_value=1000000
    )
    
    # Check that we get results for all confidence levels
    assert len(var_results) == len(confidence_levels)
    
    # Check that VaR increases with confidence level
    assert var_results['var_value'].is_monotonic_increasing
    
    # Check that CVaR increases with confidence level
    assert var_results['cvar_value'].is_monotonic_increasing

def test_var_with_different_investment(sample_returns):
    """Test that VaR scales proportionally with investment size."""
    # Calculate VaR with different investment sizes
    var_1m = calculate_historical_var(
        returns=sample_returns,
        confidence_level=0.95,
        investment_value=1000000
    )
    
    var_2m = calculate_historical_var(
        returns=sample_returns,
        confidence_level=0.95,
        investment_value=2000000
    )
    
    # VaR should double when investment doubles
    assert abs(var_2m - 2*var_1m) < 0.01  # Allow for small rounding errors

def test_var_with_lookback(sample_returns):
    """Test VaR calculation with different lookback periods."""
    # Calculate VaR with different lookback periods
    var_full = calculate_historical_var(
        returns=sample_returns,
        confidence_level=0.95,
        investment_value=1000000
    )
    
    var_500 = calculate_historical_var(
        returns=sample_returns,
        confidence_level=0.95,
        investment_value=1000000,
        lookback_days=500
    )
    
    # Values should be different when using a different lookback period
    assert var_full != var_500

if __name__ == "__main__":
    pytest.main(["-xvs", __file__])