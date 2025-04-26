"""
Helper functions for the risk management dashboard.
"""

import os
import json
import datetime as dt
from pathlib import Path

def validate_date_format(date_str):
    """
    Validate if a string is in the correct date format (YYYY-MM-DD).
    
    Args:
        date_str (str): Date string to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
    try:
        dt.datetime.strptime(date_str, '%Y-%m-%d')
        return True
    except ValueError:
        return False

def create_directory_if_not_exists(directory_path):
    """
    Create a directory if it doesn't exist.
    
    Args:
        directory_path (str or Path): Directory path to create
        
    Returns:
        Path: Path to the created/existing directory
    """
    path = Path(directory_path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def load_json_config(config_path):
    """
    Load and parse a JSON configuration file.
    
    Args:
        config_path (str or Path): Path to the configuration file
        
    Returns:
        dict: Parsed configuration
    """
    path = Path(config_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    
    with open(path, 'r') as f:
        return json.load(f)

def timestamp_filename(base_name, file_ext, include_time=False):
    """
    Create a timestamped filename.
    
    Args:
        base_name (str): Base filename
        file_ext (str): File extension (without the dot)
        include_time (bool, optional): Whether to include time in the timestamp
        
    Returns:
        str: Timestamped filename
    """
    if include_time:
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    else:
        timestamp = dt.datetime.now().strftime("%Y%m%d")
    
    return f"{base_name}_{timestamp}.{file_ext}"

def get_project_root():
    """
    Get the project root directory.
    
    Returns:
        Path: Path to project root
    """
    return Path(__file__).resolve().parents[2]

def get_config_dir():
    """
    Get the configuration directory.
    
    Returns:
        Path: Path to config directory
    """
    return get_project_root() / "configs"

def get_data_dir():
    """
    Get the data directory.
    
    Returns:
        Path: Path to data directory
    """
    return get_project_root() / "data"