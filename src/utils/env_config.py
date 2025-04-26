"""
Environment configuration utilities.
"""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv

from .logger import setup_logger

logger = setup_logger("env_config", level=logging.INFO)

def load_env_vars(env_file=None):
    """
    Load environment variables from .env file.
    
    Args:
        env_file (str, optional): Path to .env file. If None, searches in project root.
        
    Returns:
        bool: True if successful, False otherwise
    """
    # If no env_file is specified, look in project root
    if env_file is None:
        project_root = Path(__file__).resolve().parents[2]
        env_file = project_root / ".env"
    
    # Check if .env file exists
    if not Path(env_file).exists():
        logger.warning(f".env file not found at {env_file}")
        logger.info("Using environment variables from the system environment")
        return False
    
    # Load environment variables from .env file
    load_dotenv(env_file)
    logger.info(f"Loaded environment variables from {env_file}")
    return True

def get_api_credentials():
    """
    Get API credentials from environment variables.
    
    Returns:
        dict: Dictionary of API credentials
    """
    # Ensure environment variables are loaded
    load_env_vars()
    
    # LSEG/Refinitiv credentials
    lseg_credentials = {
        'app_key': os.getenv('LSEG_APP_KEY'),
        'username': os.getenv('LSEG_USERNAME'),
        'password': os.getenv('LSEG_PASSWORD')
    }
    
    # FRED API key
    fred_credentials = {
        'api_key': os.getenv('FRED_API_KEY')
    }
    
    # Check if credentials are available
    if not all(lseg_credentials.values()):
        logger.warning("One or more LSEG/Refinitiv credentials are missing")
    
    if not fred_credentials['api_key']:
        logger.warning("FRED API key is missing")
    
    return {
        'lseg': lseg_credentials,
        'fred': fred_credentials
    }

def get_log_level():
    """
    Get the log level from environment variables.
    
    Returns:
        int: Logging level
    """
    log_level_str = os.getenv('LOG_LEVEL', 'INFO')
    
    # Convert string to logging level
    log_levels = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    
    return log_levels.get(log_level_str.upper(), logging.INFO)

def get_data_path():
    """
    Get the data storage path from environment variables.
    
    Returns:
        Path: Path to data storage directory
    """
    project_root = Path(__file__).resolve().parents[2]
    data_path = os.getenv('DATA_STORAGE_PATH', 'data')
    
    # If data_path is a relative path, make it absolute
    if not os.path.isabs(data_path):
        data_path = project_root / data_path
    
    return Path(data_path)

# Load environment variables when module is imported
load_env_vars()