"""
File utility functions for handling data files.
"""

import os
import pandas as pd
import json
from pathlib import Path
from datetime import datetime

from .logger import get_default_logger

logger = get_default_logger()

def get_latest_file(directory, pattern="*", sort_by_name=False):
    """
    Get the latest file in a directory matching the specified pattern.
    
    Args:
        directory (str or Path): Directory to search in
        pattern (str, optional): File pattern to match
        sort_by_name (bool, optional): Sort by filename instead of modification time
        
    Returns:
        Path: Path to the latest file, or None if no files found
    """
    directory = Path(directory)
    if not directory.exists() or not directory.is_dir():
        logger.error(f"Directory not found: {directory}")
        return None
    
    files = list(directory.glob(pattern))
    if not files:
        logger.warning(f"No files found matching pattern '{pattern}' in {directory}")
        return None
    
    if sort_by_name:
        return max(files, key=lambda p: p.name)
    else:
        return max(files, key=lambda p: p.stat().st_mtime)

def save_dataframe(df, file_path, index=True, include_timestamp=True):
    """
    Save a DataFrame to a CSV file.
    
    Args:
        df (pandas.DataFrame): DataFrame to save
        file_path (str or Path): File path to save to
        index (bool, optional): Whether to include the index in the output
        include_timestamp (bool, optional): Whether to include a timestamp in the filename
        
    Returns:
        Path: Path to the saved file
    """
    file_path = Path(file_path)
    
    # Create directory if it doesn't exist
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Add timestamp to filename if requested
    if include_timestamp:
        stem = file_path.stem
        suffix = file_path.suffix
        timestamp = datetime.now().strftime("%Y%m%d")
        file_path = file_path.parent / f"{stem}_{timestamp}{suffix}"
    
    # Save the DataFrame
    df.to_csv(file_path, index=index)
    logger.info(f"DataFrame saved to {file_path} ({len(df)} rows)")
    
    return file_path

def load_dataframe(file_path, **kwargs):
    """
    Load a DataFrame from a CSV file.
    
    Args:
        file_path (str or Path): File path to load from
        **kwargs: Additional arguments to pass to pd.read_csv()
        
    Returns:
        pandas.DataFrame: Loaded DataFrame
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Load the DataFrame
    df = pd.read_csv(file_path, **kwargs)
    logger.info(f"DataFrame loaded from {file_path} ({len(df)} rows)")
    
    return df

def save_json(data, file_path, pretty=True):
    """
    Save data to a JSON file.
    
    Args:
        data (dict): Data to save
        file_path (str or Path): File path to save to
        pretty (bool, optional): Whether to format the JSON with indentation
        
    Returns:
        Path: Path to the saved file
    """
    file_path = Path(file_path)
    
    # Create directory if it doesn't exist
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save the data
    with open(file_path, 'w') as f:
        if pretty:
            json.dump(data, f, indent=4)
        else:
            json.dump(data, f)
    
    logger.info(f"JSON data saved to {file_path}")
    
    return file_path

def load_json(file_path):
    """
    Load data from a JSON file.
    
    Args:
        file_path (str or Path): File path to load from
        
    Returns:
        dict: Loaded data
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Load the data
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    logger.info(f"JSON data loaded from {file_path}")
    
    return data