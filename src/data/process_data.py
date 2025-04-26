"""
Data processing module for the risk management dashboard.
This module handles cleaning, transforming, and preparing data for analysis.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import logging
import warnings

# Get project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"

# Set up logger
logger = logging.getLogger(__name__)

def load_data_config():
    """
    Load data processing configuration from config file.
    
    Returns:
        dict: Data processing configuration
    """
    try:
        with open(CONFIG_DIR / "data_config.json", 'r') as f:
            data_config = json.load(f)
        
        return data_config['data_processing']
    except Exception as e:
        logger.error(f"Error loading data config: {e}")
        # Return default configuration
        return {
            'returns_method': 'log',
            'fill_method': 'ffill',
            'outlier_threshold': 3
        }

def clean_data(df, config=None, handle_outliers=True):
    """
    Clean and preprocess market data.
    
    Args:
        df (pandas.DataFrame): Raw market data
        config (dict, optional): Data processing configuration. If None, loads from config file.
        handle_outliers (bool, optional): Whether to remove outliers
        
    Returns:
        pandas.DataFrame: Cleaned market data
    """
    if config is None:
        config = load_data_config()
    
    # Create a copy to avoid modifying the original dataframe
    cleaned_df = df.copy()
    
    # Check if data is empty
    if cleaned_df.empty:
        logger.warning("Empty DataFrame provided for cleaning")
        return cleaned_df
    
    # Handle missing values
    fill_method = config.get('fill_method', 'ffill')
    logger.info(f"Filling missing values using method: {fill_method}")
    
    # First forward fill, then backward fill for any remaining NaNs at the beginning
    cleaned_df = cleaned_df.fillna(method=fill_method)
    cleaned_df = cleaned_df.fillna(method='bfill')
    
    # Check if any NaNs remain
    if cleaned_df.isna().any().any():
        logger.warning(f"Data still contains {cleaned_df.isna().sum().sum()} missing values after filling")
        # Drop rows with missing values
        old_shape = cleaned_df.shape
        cleaned_df = cleaned_df.dropna()
        logger.info(f"Dropped {old_shape[0] - cleaned_df.shape[0]} rows with missing values")
    
    # Remove outliers if specified
    if handle_outliers and 'outlier_threshold' in config:
        threshold = config['outlier_threshold']
        logger.info(f"Removing outliers with z-score threshold: {threshold}")
        
        # Calculate z-scores for each column
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Check if we have multi-level columns (e.g., from LSEG data)
            if isinstance(cleaned_df.columns, pd.MultiIndex):
                # Handle each column group separately
                for level0 in cleaned_df.columns.levels[0]:
                    # Only apply to numeric columns like prices
                    if level0 in ['TRDPRC_1', 'HIGH_1', 'LOW_1', 'OPEN_PRC', 'Adj Close', 'Close', 'High', 'Low', 'Open']:
                        sub_df = cleaned_df[level0]
                        z_scores = stats.zscore(sub_df, nan_policy='omit')
                        outliers = (np.abs(z_scores) > threshold).any(axis=1)
                        cleaned_df = cleaned_df.loc[~outliers]
            else:
                # For simple DataFrame, just calculate z-scores directly
                numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    z_scores = stats.zscore(cleaned_df[numeric_cols], nan_policy='omit')
                    outliers = (np.abs(z_scores) > threshold).any(axis=1)
                    cleaned_df = cleaned_df.loc[~outliers]
    
    # Save processed data
    processed_dir = DATA_DIR / "processed"
    processed_dir.mkdir(exist_ok=True, parents=True)
    
    # Generate filename based on original DataFrame
    if hasattr(df, 'columns') and isinstance(df.columns, pd.MultiIndex):
        # This is likely market data
        filename = "cleaned_market_data.csv"
    else:
        # This is likely macro data
        filename = "cleaned_macro_data.csv"
    
    cleaned_df.to_csv(processed_dir / filename)
    logger.info(f"Cleaned data saved to {processed_dir / filename}")
    
    return cleaned_df

def calculate_returns(df, method='simple', column=None):
    """
    Calculate returns from price data.
    
    Args:
        df (pandas.DataFrame): Dataframe containing price data
        method (str): Method for calculating returns ('simple' or 'log')
        column (str, optional): Column name for price data. If None, tries to detect.
        
    Returns:
        pandas.DataFrame: DataFrame containing calculated returns
    """
    # Create a copy to avoid modifying the original dataframe
    data = df.copy()
    
    # Handle empty DataFrame
    if data.empty:
        logger.warning("Empty DataFrame provided for return calculation")
        return data
    
    # Detect the appropriate column if not specified
    if column is None:
        # Check if we have multi-level columns (e.g., from LSEG data)
        if isinstance(data.columns, pd.MultiIndex):
            # Try to find a price column
            price_columns = ['TRDPRC_1', 'Close', 'Adj Close']
            for col in price_columns:
                if col in data.columns.levels[0]:
                    column = col
                    break
            
            if column is None:
                logger.warning("Could not detect price column in multi-level DataFrame")
                # Use the first level 0 column
                column = data.columns.levels[0][0]
        else:
            # For simple DataFrame, assume the data is already prices
            column = None
    
    # Extract price data
    if column is not None and isinstance(data.columns, pd.MultiIndex):
        prices = data[column]
    else:
        prices = data
    
    logger.info(f"Calculating {method} returns")
    
    # Calculate returns based on specified method
    if method.lower() == 'simple':
        returns = prices.pct_change().dropna()
    elif method.lower() == 'log':
        returns = np.log(prices / prices.shift(1)).dropna()
    else:
        raise ValueError("Method must be either 'simple' or 'log'")
    
    # Save processed returns
    processed_dir = DATA_DIR / "processed"
    processed_dir.mkdir(exist_ok=True, parents=True)
    returns.to_csv(processed_dir / f"{method}_returns.csv")
    logger.info(f"{method.capitalize()} returns saved to {processed_dir / f'{method}_returns.csv'}")
    
    return returns

def create_portfolio(returns_df, weights=None, asset_names=None):
    """
    Create a portfolio from individual asset returns.
    
    Args:
        returns_df (pandas.DataFrame): DataFrame containing asset returns
        weights (dict or list, optional): Asset weights. If None, equal weights are used.
        asset_names (list, optional): List of asset names to include in portfolio
        
    Returns:
        pandas.Series: Portfolio returns
    """
    # If weights not provided, use equal weighting
    if weights is None:
        n_assets = returns_df.shape[1]
        weights = [1/n_assets] * n_assets
        logger.info(f"Using equal weights (1/{n_assets}) for all assets")
    
    # If asset_names provided, filter the returns DataFrame
    if asset_names is not None:
        # Check if we have multi-level columns
        if isinstance(returns_df.columns, pd.MultiIndex):
            level1_values = returns_df.columns.get_level_values(1)
            matching_cols = [col for col in returns_df.columns if col[1] in asset_names]
            
            if not matching_cols:
                logger.warning(f"No matching assets found for {asset_names}")
                matching_cols = returns_df.columns
            
            returns_df = returns_df[matching_cols]
        else:
            matching_cols = [col for col in returns_df.columns if col in asset_names]
            
            if not matching_cols:
                logger.warning(f"No matching assets found for {asset_names}")
                matching_cols = returns_df.columns
            
            returns_df = returns_df[matching_cols]
    
    # If weights is a dictionary, convert to list in the same order as columns
    if isinstance(weights, dict):
        # Check if we have multi-level columns
        if isinstance(returns_df.columns, pd.MultiIndex):
            weights_list = []
            for col in returns_df.columns:
                asset = col[1]  # Assuming asset name is in level 1
                weights_list.append(weights.get(asset, 0))
        else:
            weights_list = [weights.get(col, 0) for col in returns_df.columns]
        
        weights = weights_list
    
    # Normalize weights to sum to 1
    weights = np.array(weights) / sum(weights)
    
    # Display weights for logging
    weight_info = {}
    for i, col in enumerate(returns_df.columns):
        if isinstance(col, tuple):
            asset = col[1]  # For multi-level columns
        else:
            asset = col
        weight_info[asset] = weights[i]
    
    logger.info(f"Portfolio weights: {weight_info}")
    
    # Calculate portfolio returns
    portfolio_returns = returns_df.dot(weights)
    
    # Convert to Series if it's not already
    if isinstance(portfolio_returns, pd.DataFrame):
        portfolio_returns = portfolio_returns.iloc[:, 0]
    
    # Save processed portfolio returns
    processed_dir = DATA_DIR / "processed"
    processed_dir.mkdir(exist_ok=True, parents=True)
    portfolio_returns.to_csv(processed_dir / "portfolio_returns.csv")
    logger.info(f"Portfolio returns saved to {processed_dir / 'portfolio_returns.csv'}")
    
    return portfolio_returns

def align_data(market_data, macro_data):
    """
    Align market data and macroeconomic data to common dates.
    
    Args:
        market_data (pandas.DataFrame): Market data
        macro_data (pandas.DataFrame): Macroeconomic data
        
    Returns:
        tuple: Aligned market data and macro data
    """
    # Check if either DataFrame is empty
    if market_data.empty or macro_data.empty:
        logger.warning("One or both DataFrames are empty, cannot align")
        return market_data, macro_data
    
    # Make a copy to avoid modifying the original
    market_data_copy = market_data.copy()
    
    # Clean up index if needed
    if isinstance(market_data_copy.index, pd.Index) and not pd.api.types.is_datetime64_any_dtype(market_data_copy.index):
        # Check for problematic values
        if 'Date' in market_data_copy.index:
            # Drop 'Date' row
            market_data_copy = market_data_copy.drop('Date')
        
        # Try to convert index to datetime
        try:
            market_data_copy.index = pd.to_datetime(market_data_copy.index, errors='coerce')
            # Drop rows where conversion failed
            market_data_copy = market_data_copy.loc[~market_data_copy.index.isna()]
        except Exception as e:
            logger.error(f"Error converting market data index to datetime: {e}")
            # Create a new numeric index
            market_data_copy = market_data_copy.reset_index(drop=True)
            logger.info("Created new numeric index for market data")
            return market_data_copy, macro_data  # Return early as we can't align
    
    # Ensure macro_data index is datetime
    if not pd.api.types.is_datetime64_any_dtype(macro_data.index):
        try:
            macro_data.index = pd.to_datetime(macro_data.index)
        except Exception as e:
            logger.error(f"Error converting macro data index to datetime: {e}")
            return market_data_copy, macro_data  # Return early as we can't align
    
    # Get common dates
    common_dates = market_data_copy.index.intersection(macro_data.index)
    
    if len(common_dates) == 0:
        logger.warning("No common dates found between market and macro data")
        return market_data_copy, macro_data
    
    # Align data
    aligned_market = market_data_copy.loc[common_dates]
    aligned_macro = macro_data.loc[common_dates]
    
    logger.info(f"Aligned data on {len(common_dates)} common dates")
    
    # Save aligned data
    processed_dir = DATA_DIR / "processed"
    processed_dir.mkdir(exist_ok=True, parents=True)
    
    aligned_market.to_csv(processed_dir / "aligned_market_data.csv")
    aligned_macro.to_csv(processed_dir / "aligned_macro_data.csv")
    
    logger.info(f"Aligned data saved to {processed_dir}")
    
    return aligned_market, aligned_macro

def process_data_pipeline(market_data=None, macro_data=None, config=None):
    """
    Run the complete data processing pipeline.
    
    Args:
        market_data (pandas.DataFrame, optional): Raw market data. If None, tries to load from file.
        macro_data (pandas.DataFrame, optional): Raw macro data. If None, tries to load from file.
        config (dict, optional): Data processing configuration
        
    Returns:
        tuple: (processed_market_data, processed_macro_data, market_returns, portfolio_returns)
    """
    # Set up logging if not already configured
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO)
    
    # Load configuration if not provided
    if config is None:
        config = load_data_config()
    
    # If data not provided, try to load from files
    if market_data is None or macro_data is None:
        raw_dir = DATA_DIR / "raw"
        
        if not raw_dir.exists():
            logger.error(f"Raw data directory not found: {raw_dir}")
            return None, None, None, None
        
        # Load market data
        if market_data is None:
            market_files = list(raw_dir.glob("*data_*.csv"))
            
            if market_files:
                latest_market_file = max(market_files, key=lambda f: f.stat().st_mtime)
                logger.info(f"Loading market data from {latest_market_file}")
                
                try:
                    market_data = pd.read_csv(latest_market_file, index_col=0, parse_dates=True)
                    
                    # Check if we need to convert to MultiIndex
                    if '_' in market_data.columns[0]:
                        # This is likely a flattened MultiIndex
                        market_data.columns = pd.MultiIndex.from_tuples(
                            [tuple(col.split('_', 1)) for col in market_data.columns]
                        )
                except Exception as e:
                    logger.error(f"Error loading market data: {e}")
                    market_data = pd.DataFrame()
            else:
                logger.warning("No market data files found")
                market_data = pd.DataFrame()
        
        # Load macro data
        if macro_data is None:
            fred_files = list(raw_dir.glob("fred_data_*.csv"))
            
            if fred_files:
                latest_fred_file = max(fred_files, key=lambda f: f.stat().st_mtime)
                logger.info(f"Loading macro data from {latest_fred_file}")
                
                try:
                    macro_data = pd.read_csv(latest_fred_file, index_col=0, parse_dates=True)
                except Exception as e:
                    logger.error(f"Error loading macro data: {e}")
                    macro_data = pd.DataFrame()
            else:
                logger.warning("No macro data files found")
                macro_data = pd.DataFrame()
    
    # Clean data
    logger.info("Cleaning market data...")
    cleaned_market = clean_data(market_data, config=config)
    
    logger.info("Cleaning macro data...")
    cleaned_macro = clean_data(macro_data, config=config)
    
    # Align data
    logger.info("Aligning market and macro data...")
    aligned_market, aligned_macro = align_data(cleaned_market, cleaned_macro)
    
    # Calculate returns
    returns_method = config.get('returns_method', 'log')
    logger.info(f"Calculating {returns_method} returns...")
    market_returns = calculate_returns(aligned_market, method=returns_method)
    
    # Create portfolio
    logger.info("Creating portfolio...")
    portfolio_returns = create_portfolio(market_returns)
    
    logger.info("Data processing pipeline completed successfully")
    
    return aligned_market, aligned_macro, market_returns, portfolio_returns

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Example usage
    try:
        # Run the complete data processing pipeline
        logger.info("Running data processing pipeline...")
        processed_market, processed_macro, market_returns, portfolio_returns = process_data_pipeline()
        
        if portfolio_returns is not None:
            logger.info(f"Portfolio returns shape: {portfolio_returns.shape}")
            logger.info(f"Portfolio mean return: {portfolio_returns.mean():.6f}")
            logger.info(f"Portfolio volatility: {portfolio_returns.std():.6f}")
        else:
            logger.error("Failed to create portfolio returns")
        
    except Exception as e:
        logger.error(f"Error in data processing pipeline: {e}", exc_info=True)