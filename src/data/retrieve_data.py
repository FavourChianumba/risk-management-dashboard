"""
Data retrieval module for the risk management dashboard.
This module handles fetching market data from Yahoo Finance, Alpha Vantage, and macroeconomic data from FRED.
"""

import os
import json
import pandas as pd
import numpy as np
import datetime as dt
from pathlib import Path
import logging
import time
from functools import wraps
import requests

# Try to import API libraries
try:
    import yfinance as yf
except ImportError:
    yf = None

try:
    from fredapi import Fred
except ImportError:
    Fred = None

try:
    import yahooquery as yq
except ImportError:
    yq = None

# Get project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"

# Set up logger
logger = logging.getLogger(__name__)

# Rate limiting decorator for API calls
def rate_limit(calls_limit=5, period=60):
    """Decorator to rate limit API calls"""
    calls = []
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Check if we need to wait
            now = time.time()
            # Remove old calls
            while calls and calls[0] < now - period:
                calls.pop(0)
            # If we've made too many calls, wait
            if len(calls) >= calls_limit:
                sleep_time = period - (now - calls[0])
                if sleep_time > 0:
                    logger.info(f"Rate limiting: waiting {sleep_time:.2f} seconds")
                    time.sleep(sleep_time)
            # Add current call timestamp
            calls.append(time.time())
            # Make the API call
            return func(*args, **kwargs)
        return wrapper
    return decorator

def load_api_keys():
    """
    Load API keys from the configuration file.
    
    Returns:
        dict: Dictionary containing API keys
    """
    # First check if we have a .env file
    try:
        from dotenv import load_dotenv
        load_dotenv(PROJECT_ROOT / ".env")
        
        # Get API keys from environment variables
        alpha_vantage_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        polygon_key = os.getenv('POLYGON_API_KEY')
        fred_key = os.getenv('FRED_API_KEY')
        
        # If we have credentials from environment variables, return them
        if alpha_vantage_key or polygon_key or fred_key:
            return {
                'alpha_vantage': {'api_key': alpha_vantage_key},
                'polygon': {'api_key': polygon_key},
                'fred': {'api_key': fred_key}
            }
    except ImportError:
        pass  # If python-dotenv is not installed, continue to the JSON method
    
    # Fall back to the JSON file
    api_keys_path = CONFIG_DIR / "api_keys.json"
    
    if not api_keys_path.exists():
        # Try to create from example if available
        example_path = CONFIG_DIR / "api_keys.json.example"
        if example_path.exists():
            logger.warning(
                f"API keys file not found at {api_keys_path}. "
                f"Creating from example template. Please edit with your credentials."
            )
            with open(example_path, 'r') as f:
                example_content = f.read()
            
            with open(api_keys_path, 'w') as f:
                f.write(example_content)
        else:
            raise FileNotFoundError(
                f"API keys file not found at {api_keys_path}. "
                "Please create this file from the example template."
            )
    
    with open(api_keys_path, 'r') as f:
        api_keys = json.load(f)
    
    return api_keys

def get_yahoo_finance_data(tickers, start_date, end_date):
    """
    Retrieve market data from Yahoo Finance for the specified tickers.
    
    Args:
        tickers (list): List of ticker symbols
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format
        
    Returns:
        pandas.DataFrame: DataFrame containing the requested market data
    """
    if yf is None:
        raise ImportError("yfinance package is not installed. Please install it with 'pip install yfinance'.")
    
    # Map common ticker symbols to Yahoo Finance format
    ticker_map = {
        "SPX": "^GSPC",  # S&P 500
        "DJIA": "^DJI",   # Dow Jones Industrial Average
        "FTSE": "^FTSE",  # FTSE 100
        "GDAXI": "^GDAXI",  # DAX
        "FCHI": "^FCHI",  # CAC 40
        "N225": "^N225",  # Nikkei 225
        "HSI": "^HSI",    # Hang Seng
        "US10YT=RR": "^TNX",  # 10-Year Treasury Yield
        "US2YT=RR": "^IRX",   # 2-Year Treasury Yield
        "US30YT=RR": "^TYX"   # 30-Year Treasury Yield
    }
    
    # Map tickers
    yahoo_tickers = [ticker_map.get(ticker, ticker) for ticker in tickers]
    
    logger.info(f"Retrieving data from Yahoo Finance for {yahoo_tickers}")
    
    try:
        # Download data from Yahoo Finance
        data = yf.download(
            tickers=yahoo_tickers,
            start=start_date,
            end=end_date,
            progress=False
        )
        
        # Handle single ticker case (yfinance returns different format)
        if len(yahoo_tickers) == 1:
            data.columns = pd.MultiIndex.from_product([data.columns, yahoo_tickers])
        
        # Create a mapping from Yahoo format to standard format
        column_mapping = {
            'Adj Close': 'TRDPRC_1',
            'High': 'HIGH_1',
            'Low': 'LOW_1',
            'Open': 'OPEN_PRC',
            'Volume': 'VOLUME'
        }
        
        # Rename columns to match standard format
        data = data.rename(columns=column_mapping, level=0)
        
        # Rename tickers back to original format
        reverse_map = {v: k for k, v in ticker_map.items()}
        renamed_columns = []
        
        for col in data.columns:
            metric, ticker = col
            original_ticker = reverse_map.get(ticker, ticker)
            renamed_columns.append((metric, original_ticker))
        
        data.columns = pd.MultiIndex.from_tuples(renamed_columns)
        
        # Save data to CSV
        raw_data_dir = DATA_DIR / "raw"
        raw_data_dir.mkdir(exist_ok=True, parents=True)
        
        filename = f"yahoo_data_{dt.datetime.now().strftime('%Y%m%d')}.csv"
        data.to_csv(raw_data_dir / filename)
        
        logger.info(f"Yahoo Finance data saved to {raw_data_dir / filename}")
        
        return data
        
    except Exception as e:
        logger.error(f"Error retrieving data from Yahoo Finance: {e}")
        raise

def get_yahoo_query_data(tickers, start_date, end_date):
    """
    Retrieve market data using yahooquery for the specified tickers.
    
    Args:
        tickers (list): List of ticker symbols
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format
        
    Returns:
        pandas.DataFrame: DataFrame containing the requested market data
    """
    if yq is None:
        raise ImportError("yahooquery package is not installed. Please install it with 'pip install yahooquery'.")
    
    # Map common ticker symbols to Yahoo Finance format
    ticker_map = {
        "SPX": "^GSPC",  # S&P 500
        "DJIA": "^DJI",   # Dow Jones Industrial Average
        "FTSE": "^FTSE",  # FTSE 100
        "GDAXI": "^GDAXI",  # DAX
        "FCHI": "^FCHI",  # CAC 40
        "N225": "^N225",  # Nikkei 225
        "HSI": "^HSI",    # Hang Seng
        "US10YT=RR": "^TNX",  # 10-Year Treasury Yield
        "US2YT=RR": "^IRX",   # 2-Year Treasury Yield
        "US30YT=RR": "^TYX"   # 30-Year Treasury Yield
    }
    
    # Map tickers
    yahoo_tickers = [ticker_map.get(ticker, ticker) for ticker in tickers]
    
    logger.info(f"Retrieving data using yahooquery for {yahoo_tickers}")
    
    try:
        # Initialize yahooquery Ticker object
        ticker_obj = yq.Ticker(yahoo_tickers)
        
        # Download historical data
        data = ticker_obj.history(start=start_date, end=end_date)
        
        # Process the data
        if isinstance(data.index, pd.MultiIndex):
            # Create an empty DataFrame for results
            processed_data = pd.DataFrame()
            
            # Process each asset
            for ticker in yahoo_tickers:
                # Filter data for this ticker
                ticker_data = data.xs(ticker, level='symbol')
                
                if not ticker_data.empty:
                    # Get the original ticker name
                    original_ticker = next((k for k, v in ticker_map.items() if v == ticker), ticker)
                    
                    # Add data to the result DataFrame
                    if processed_data.empty:
                        processed_data = pd.DataFrame(index=ticker_data.index)
                    
                    # Map column names
                    processed_data[('TRDPRC_1', original_ticker)] = ticker_data['close']
                    processed_data[('HIGH_1', original_ticker)] = ticker_data['high']
                    processed_data[('LOW_1', original_ticker)] = ticker_data['low']
                    processed_data[('OPEN_PRC', original_ticker)] = ticker_data['open']
                    processed_data[('VOLUME', original_ticker)] = ticker_data['volume']
                    
                    logger.info(f"Successfully processed data for {original_ticker}")
                else:
                    logger.warning(f"No data available for {ticker}")
            
            # Make sure the index is sorted
            processed_data = processed_data.sort_index()
            
            # Save data to CSV
            raw_data_dir = DATA_DIR / "raw"
            raw_data_dir.mkdir(exist_ok=True, parents=True)
            
            filename = f"yahooquery_data_{dt.datetime.now().strftime('%Y%m%d')}.csv"
            processed_data.to_csv(raw_data_dir / filename)
            
            logger.info(f"Yahoo Query data saved to {raw_data_dir / filename}")
            
            return processed_data
        else:
            logger.warning("Unexpected data format from yahooquery")
            return None
        
    except Exception as e:
        logger.error(f"Error retrieving data from yahooquery: {e}")
        return None

@rate_limit(calls_limit=5, period=60)
def get_alpha_vantage_data(tickers, start_date, end_date, api_key):
    """
    Retrieve market data from Alpha Vantage for the specified tickers.
    
    Args:
        tickers (list): List of ticker symbols
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format
        api_key (str): Alpha Vantage API key
        
    Returns:
        pandas.DataFrame: DataFrame containing the requested market data
    """
    if not api_key:
        logger.warning("Alpha Vantage API key not provided")
        return None
    
    logger.info(f"Retrieving data from Alpha Vantage for {tickers}")
    
    # Create empty DataFrame with MultiIndex columns
    data = pd.DataFrame()
    base_url = "https://www.alphavantage.co/query"
    
    # Convert date strings to datetime objects for filtering
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    for ticker in tickers:
        try:
            # Define API parameters
            params = {
                "function": "TIME_SERIES_DAILY",
                "symbol": ticker,
                "outputsize": "full",
                "datatype": "json",
                "apikey": api_key
            }
            
            # Make API request
            response = requests.get(base_url, params=params, timeout=30)
            response.raise_for_status()  # Raise exception for 4XX/5XX responses
            
            # Parse JSON response
            result = response.json()
            
            # Check for error messages
            if "Error Message" in result:
                logger.warning(f"Alpha Vantage error for {ticker}: {result['Error Message']}")
                continue
                
            # Check for information messages (including premium endpoint messages)
            if "Information" in result:
                logger.warning(f"Alpha Vantage information for {ticker}: {result['Information']}")
                continue
            
            # Parse time series data
            if "Time Series (Daily)" in result:
                time_series = result["Time Series (Daily)"]
                
                # Convert to DataFrame
                ticker_df = pd.DataFrame.from_dict(time_series, orient="index")
                ticker_df.index = pd.to_datetime(ticker_df.index)
                ticker_df = ticker_df.sort_index()
                
                # Filter by date range
                ticker_df = ticker_df[(ticker_df.index >= start_dt) & (ticker_df.index <= end_dt)]
                
                # Rename columns to match standard format
                column_mapping = {
                    "1. open": ("OPEN_PRC", ticker),
                    "2. high": ("HIGH_1", ticker),
                    "3. low": ("LOW_1", ticker),
                    "4. close": ("TRDPRC_1", ticker),
                    "5. volume": ("VOLUME", ticker)
                }
                
                # Convert to MultiIndex columns
                ticker_df = ticker_df.rename(columns=column_mapping)
                
                # Merge with main DataFrame
                if data.empty:
                    data = ticker_df
                else:
                    data = data.join(ticker_df, how="outer")
                
                logger.info(f"Successfully retrieved data for {ticker}")
            else:
                logger.warning(f"No time series data found for {ticker}")
                continue
                
        except Exception as e:
            logger.error(f"Error retrieving Alpha Vantage data for {ticker}: {e}")
            continue
    
    # Save data to CSV if not empty
    if not data.empty:
        raw_data_dir = DATA_DIR / "raw"
        raw_data_dir.mkdir(exist_ok=True, parents=True)
        
        filename = f"alpha_vantage_data_{dt.datetime.now().strftime('%Y%m%d')}.csv"
        data.to_csv(raw_data_dir / filename)
        
        logger.info(f"Alpha Vantage data saved to {raw_data_dir / filename}")
    
    return data

def get_market_data(tickers, start_date, end_date):
    """
    Retrieve market data using available data sources with fallback mechanism.
    
    Args:
        tickers (list): List of ticker symbols
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format
        
    Returns:
        pandas.DataFrame: DataFrame containing the requested market data
    """
    # Load API credentials
    api_keys = load_api_keys()
    
    # Get data priority from environment or use default
    data_priority = os.environ.get("DATA_SOURCE_PRIORITY", "yahoo,alphavantage")
    data_sources = data_priority.lower().split(",")
    
    # Try each data source in order of priority
    for source in data_sources:
        try:
            if source == "yahoo":
                if yf is not None:
                    logger.info("Attempting to retrieve data from Yahoo Finance...")
                    return get_yahoo_finance_data(tickers, start_date, end_date)
                else:
                    logger.warning("yfinance not installed, skipping Yahoo Finance data source")
            
            elif source == "yahooquery":
                if yq is not None:
                    logger.info("Attempting to retrieve data using yahooquery...")
                    data = get_yahoo_query_data(tickers, start_date, end_date)
                    if data is not None and not data.empty:
                        return data
                    logger.warning("yahooquery retrieval failed or returned empty data")
                else:
                    logger.warning("yahooquery not installed, skipping yahooquery data source")
            
            elif source == "alphavantage":
                if api_keys.get('alpha_vantage', {}).get('api_key'):
                    logger.info("Attempting to retrieve data from Alpha Vantage...")
                    data = get_alpha_vantage_data(
                        tickers, 
                        start_date, 
                        end_date, 
                        api_keys['alpha_vantage']['api_key']
                    )
                    if data is not None and not data.empty:
                        return data
                    logger.warning("Alpha Vantage retrieval failed or returned empty data")
                else:
                    logger.warning("Alpha Vantage API key not available, skipping Alpha Vantage data source")
            
            else:
                logger.warning(f"Unknown data source: {source}")
                
        except Exception as e:
            logger.error(f"Error retrieving data from {source}: {e}")
    
    # If all data sources fail, raise exception
    raise RuntimeError("All data sources failed to retrieve market data")

def get_fred_data(series_ids, start_date, end_date):
    """
    Retrieve macroeconomic data from FRED for the specified series IDs.
    
    Args:
        series_ids (list): List of FRED series IDs
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format
        
    Returns:
        pandas.DataFrame: DataFrame containing the requested macroeconomic data
    """
    # Load API credentials
    api_keys = load_api_keys()
    
    # Create empty DataFrame to store results
    data = pd.DataFrame()
    
    # Try using fredapi first
    if Fred is not None and api_keys.get('fred', {}).get('api_key'):
        try:
            # Initialize FRED client
            fred = Fred(api_key=api_keys['fred']['api_key'])
            
            # Convert string dates to datetime objects
            start_dt = dt.datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = dt.datetime.strptime(end_date, '%Y-%m-%d')
            
            # Get data for each series
            for series_id in series_ids:
                try:
                    series_data = fred.get_series(series_id, start_dt, end_dt)
                    if not series_data.empty:
                        data[series_id] = series_data
                        logger.info(f"Retrieved FRED data for {series_id}")
                    else:
                        logger.warning(f"No data returned for FRED series {series_id}")
                except Exception as e:
                    logger.error(f"Error retrieving FRED data for {series_id}: {e}")
            
        except Exception as e:
            logger.error(f"Error using FRED API: {e}")
    
    # If fredapi fails or is not available, try pandas_datareader
    if data.empty:
        try:
            import pandas_datareader as pdr
            
            logger.info("Attempting to retrieve FRED data using pandas_datareader...")
            
            for series_id in series_ids:
                try:
                    series_data = pdr.get_data_fred(series_id, start_date, end_date)
                    if not series_data.empty:
                        if data.empty:
                            data = series_data
                        else:
                            data = data.join(series_data, how="outer")
                        logger.info(f"Retrieved FRED data for {series_id} using pandas_datareader")
                    else:
                        logger.warning(f"No data returned for FRED series {series_id}")
                except Exception as e:
                    logger.error(f"Error retrieving FRED data for {series_id}: {e}")
            
        except ImportError:
            logger.error("pandas_datareader is not installed. Install with 'pip install pandas_datareader'")
        except Exception as e:
            logger.error(f"Error using pandas_datareader for FRED: {e}")
    
    # Fill missing values with forward fill and then backward fill
    if not data.empty:
        data = data.fillna(method='ffill').fillna(method='bfill')
        
        # Save data to CSV
        raw_data_dir = DATA_DIR / "raw"
        raw_data_dir.mkdir(exist_ok=True, parents=True)
        
        filename = f"fred_data_{dt.datetime.now().strftime('%Y%m%d')}.csv"
        data.to_csv(raw_data_dir / filename)
        
        logger.info(f"FRED data saved to {raw_data_dir / filename}")
    
    return data

def load_crisis_periods():
    """
    Load pre-defined crisis periods from external data.
    
    Returns:
        pandas.DataFrame: DataFrame containing crisis period definitions
    """
    crisis_file = DATA_DIR / "external" / "crisis_periods.csv"
    
    if not crisis_file.exists():
        logger.warning(f"Crisis periods file not found at {crisis_file}.")
        
        # Create default crisis periods if file doesn't exist
        default_crisis_periods = [
            {
                'name': 'Global Financial Crisis',
                'start_date': '2008-09-01',
                'end_date': '2009-03-31',
                'description': 'Severe economic downturn following the collapse of Lehman Brothers'
            },
            {
                'name': 'COVID-19 Crash',
                'start_date': '2020-02-20',
                'end_date': '2020-03-23',
                'description': 'Rapid market decline due to the COVID-19 pandemic'
            },
            {
                'name': '2022 Rate Hikes',
                'start_date': '2022-01-01',
                'end_date': '2022-12-31',
                'description': 'Market decline due to aggressive Fed rate hikes to combat inflation'
            }
        ]
        
        # Create DataFrame
        crisis_df = pd.DataFrame(default_crisis_periods)
        
        # Convert dates to datetime
        crisis_df['start_date'] = pd.to_datetime(crisis_df['start_date'])
        crisis_df['end_date'] = pd.to_datetime(crisis_df['end_date'])
        
        # Calculate duration in days
        crisis_df['duration_days'] = (crisis_df['end_date'] - crisis_df['start_date']).dt.days
        
        # Save to CSV
        DATA_DIR.mkdir(exist_ok=True, parents=True)
        (DATA_DIR / "external").mkdir(exist_ok=True, parents=True)
        crisis_df.to_csv(crisis_file, index=False)
        
        logger.info(f"Created default crisis periods at {crisis_file}")
        
        return crisis_df
    
    # Load crisis periods from file
    crisis_periods = pd.read_csv(crisis_file, parse_dates=['start_date', 'end_date'])
    logger.info(f"Loaded crisis periods from {crisis_file}")
    
    return crisis_periods

def get_data_for_date_range(start_date, end_date):
    """
    Comprehensive function to retrieve all necessary data for a date range.
    
    Args:
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format
        
    Returns:
        tuple: (market_data, macro_data, crisis_periods)
    """
    logger.info(f"Retrieving data for period: {start_date} to {end_date}")
    
    # Load data configuration
    try:
        with open(CONFIG_DIR / "data_config.json", 'r') as f:
            data_config = json.load(f)
        
        config = data_config['data_retrieval']
    except Exception as e:
        logger.error(f"Error loading data configuration: {e}")
        raise
    
    # Get equity and bond tickers
    equity_tickers = [item['ticker'] for item in config['equity_indices']]
    bond_tickers = [item['ticker'] for item in config['bond_indices']]
    all_tickers = equity_tickers + bond_tickers
    
    # Get macro indicators
    macro_indicators = [item['series_id'] for item in config['macro_indicators']]
    
    # Retrieve market data with fallback mechanism
    market_data = get_market_data(
        tickers=all_tickers,
        start_date=start_date,
        end_date=end_date
    )
    
    # Retrieve macroeconomic data
    try:
        macro_data = get_fred_data(
            series_ids=macro_indicators,
            start_date=start_date,
            end_date=end_date
        )
    except Exception as e:
        logger.error(f"Error retrieving macroeconomic data: {e}")
        macro_data = pd.DataFrame()
    
    # Load crisis periods
    crisis_periods = load_crisis_periods()
    
    return market_data, macro_data, crisis_periods

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Example usage
    try:
        # Load data configuration
        with open(CONFIG_DIR / "data_config.json", 'r') as f:
            data_config = json.load(f)
        
        # Extract configuration parameters
        config = data_config['data_retrieval']
        start_date = config['start_date']
        end_date = config['end_date']
        
        # Get all data in one call
        print(f"Retrieving all data for period: {start_date} to {end_date}")
        market_data, macro_data, crisis_periods = get_data_for_date_range(
            start_date=start_date,
            end_date=end_date
        )
        
        print("Data retrieval completed successfully.")
        print(f"Market data shape: {market_data.shape}")
        print(f"Macro data shape: {macro_data.shape}")
        print(f"Crisis periods: {len(crisis_periods)}")
        
    except Exception as e:
        print(f"Error retrieving data: {e}")