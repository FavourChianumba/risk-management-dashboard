#!/usr/bin/env python
"""
Project initialization script for Risk Management Dashboard.
This script sets up the required directory structure and configuration files.
"""

import os
import sys
import json
import shutil
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

from src.utils.logger import setup_logger

# Set up logger
logger = setup_logger("setup_project")

def create_directory_structure():
    """Create the standard directory structure for the project"""
    logger.info("Creating directory structure...")
    
    # Define directories to create
    directories = [
        "data/raw",
        "data/processed",
        "data/external",
        "data/results",
        "configs",
        "logs",
        "notebooks",
        "src/data",
        "src/models",
        "src/visualization",
        "src/stress_testing",
        "src/utils",
        "tests"
    ]
    
    # Create each directory
    for directory in directories:
        dir_path = project_root / directory
        dir_path.mkdir(exist_ok=True, parents=True)
        logger.info(f"Created directory: {dir_path}")
    
    logger.info("Directory structure created successfully.")

def create_config_files():
    """Create default configuration files if they don't exist"""
    logger.info("Creating configuration files...")
    
    config_dir = project_root / "configs"
    
    # Create data_config.json if it doesn't exist
    data_config_path = config_dir / "data_config.json"
    if not data_config_path.exists():
        data_config = {
            "data_retrieval": {
                "start_date": "2014-01-01",
                "end_date": "2024-01-01",
                "lookback_years": 10,
                "equity_indices": [
                    {"ticker": "SPX", "description": "S&P 500", "weight": 0.6}
                ],
                "bond_indices": [
                    {"ticker": "US10YT=RR", "description": "10Y US Treasury", "weight": 0.4}
                ],
                "macro_indicators": [
                    {"series_id": "TEDRATE", "description": "TED Spread", "frequency": "D"},
                    {"series_id": "VIXCLS", "description": "VIX", "frequency": "D"},
                    {"series_id": "DFF", "description": "Fed Funds Rate", "frequency": "D"}
                ]
            },
            "data_processing": {
                "returns_method": "log",
                "fill_method": "ffill",
                "outlier_threshold": 3
            }
        }
        
        with open(data_config_path, "w") as f:
            json.dump(data_config, f, indent=4)
        logger.info(f"Created default data configuration: {data_config_path}")
    
    # Create model_config.json if it doesn't exist
    model_config_path = config_dir / "model_config.json"
    if not model_config_path.exists():
        model_config = {
            "var_models": {
                "historical": {
                    "default_confidence": 0.95,
                    "confidence_levels": [0.9, 0.95, 0.99],
                    "lookback_days": 252
                },
                "parametric": {
                    "default_confidence": 0.95,
                    "confidence_levels": [0.9, 0.95, 0.99],
                    "distribution": "normal"
                },
                "monte_carlo": {
                    "default_confidence": 0.95,
                    "confidence_levels": [0.9, 0.95, 0.99],
                    "n_simulations": 10000,
                    "time_horizons": [1, 5, 10, 20]
                }
            },
            "backtesting": {
                "window_size": 252,
                "step_size": 1,
                "significance_level": 0.05
            },
            "stress_testing": {
                "historical_scenarios": ["Global Financial Crisis", "COVID-19 Crash"],
                "synthetic_scenarios": {
                    "severe_market_drop": {
                        "equity_shock": -0.25,
                        "bond_shock": -0.10,
                        "volatility_multiplier": 2.0
                    },
                    "rate_shock": {
                        "equity_shock": -0.15,
                        "bond_shock": -0.20,
                        "volatility_multiplier": 1.5
                    }
                }
            }
        }
        
        with open(model_config_path, "w") as f:
            json.dump(model_config, f, indent=4)
        logger.info(f"Created default model configuration: {model_config_path}")
    
    # Create API keys template if it doesn't exist
    api_keys_template = config_dir / "api_keys.json.example"
    api_keys_path = config_dir / "api_keys.json"
    
    if not api_keys_template.exists():
        api_keys = {
            "alpha_vantage": {
                "api_key": "YOUR_ALPHA_VANTAGE_API_KEY_HERE"
            },
            "polygon": {
                "api_key": "YOUR_POLYGON_API_KEY_HERE"
            },
            "fred": {
                "api_key": "YOUR_FRED_API_KEY_HERE"
            }
        }
        
        with open(api_keys_template, "w") as f:
            json.dump(api_keys, f, indent=4)
        logger.info(f"Created API keys template: {api_keys_template}")
    
    if not api_keys_path.exists():
        shutil.copy(api_keys_template, api_keys_path)
        logger.info(f"Created API keys file from template: {api_keys_path}")
    
    logger.info("Configuration files created successfully.")

def create_env_template():
    """Create .env template file if it doesn't exist"""
    logger.info("Creating .env template...")
    
    env_template = project_root / ".env.example"
    env_file = project_root / ".env"
    
    env_template_content = """# ============================
# API Keys
# ============================
FRED_API_KEY=your_fred_api_key_here
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key_here
POLYGON_API_KEY=your_polygon_key_here

# ============================
# Data Source Configuration
# ============================
# Data Source Priority (comma-separated)
# Options: yahoo, yahooquery, alphavantage, polygon
DATA_SOURCE_PRIORITY=yahoo,yahooquery,alphavantage,polygon

# API Timeout Settings
YFINANCE_TIMEOUT=30

# ============================
# Data Processing Parameters
# ============================
# Maximum percentage of missing values allowed
MAX_MISSING_PCT=10.0

# Z-score threshold for outliers
OUTLIER_THRESHOLD=4.0

# Method for calculating returns (log or simple)
RETURNS_METHOD=log

# ============================
# Risk Parameters
# ============================
# Annual risk-free rate (3.5%)
RISK_FREE_RATE=0.035

# Default VaR confidence level (95%)
DEFAULT_CONFIDENCE_LEVEL=0.95

# Default investment amount ($1M)
DEFAULT_INVESTMENT_VALUE=1000000

# ============================
# Model Parameters
# ============================
# Historical VaR lookback window (1 year)
HISTORICAL_WINDOW=252

# Number of Monte Carlo simulations
MONTE_CARLO_SIMULATIONS=10000

# ============================
# Backtesting Parameters
# ============================
# Backtesting window size
BACKTEST_WINDOW=252

# Backtesting step size
BACKTEST_STEP=1

# ============================
# Logging Settings
# ============================
# Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
LOG_LEVEL=INFO
"""
    
    # Create .env.example
    with open(env_template, "w") as f:
        f.write(env_template_content)
    logger.info(f"Created .env template: {env_template}")
    
    # Create .env if it doesn't exist
    if not env_file.exists():
        shutil.copy(env_template, env_file)
        logger.info(f"Created .env file from template: {env_file}")
    
    logger.info(".env template created successfully.")

def create_init_files():
    """Create __init__.py files for all Python packages"""
    logger.info("Creating __init__.py files...")
    
    # Find all directories in src
    src_dir = project_root / "src"
    package_dirs = [src_dir]
    
    for dirpath, dirnames, _ in os.walk(src_dir):
        dirpath = Path(dirpath)
        for dirname in dirnames:
            package_dirs.append(dirpath / dirname)
    
    # Create __init__.py in each directory
    for package_dir in package_dirs:
        init_file = package_dir / "__init__.py"
        if not init_file.exists():
            with open(init_file, "w") as f:
                f.write(f'"""\n{package_dir.name} package.\n"""\n')
            logger.info(f"Created __init__.py: {init_file}")
    
    logger.info("__init__.py files created successfully.")

def create_crisis_periods():
    """Create default crisis periods file"""
    logger.info("Creating default crisis periods file...")
    
    crisis_file = project_root / "data" / "external" / "crisis_periods.csv"
    
    if not crisis_file.exists():
        # Define default crisis periods
        crisis_periods = [
            "name,start_date,end_date,description,duration_days",
            "Global Financial Crisis,2008-09-01,2009-03-31,Severe economic downturn following the collapse of Lehman Brothers,211",
            "COVID-19 Crash,2020-02-20,2020-03-23,Rapid market decline due to the COVID-19 pandemic,32",
            "2022 Rate Hikes,2022-01-01,2022-12-31,Market decline due to aggressive Fed rate hikes to combat inflation,364",
            "Dot-com Bubble,2000-03-10,2002-10-09,Collapse of technology stocks following excessive speculation,943",
            "2013 Taper Tantrum,2013-05-22,2013-09-05,Market reaction to Fed announcement of QE tapering,106",
            "2018 Q4 Selloff,2018-10-01,2018-12-24,Rapid market decline due to Fed rate hikes and trade tensions,84"
        ]
        
        # Create directory if it doesn't exist
        (project_root / "data" / "external").mkdir(exist_ok=True, parents=True)
        
        # Write the file
        with open(crisis_file, "w") as f:
            f.write("\n".join(crisis_periods))
        
        logger.info(f"Created default crisis periods file: {crisis_file}")
    
    logger.info("Crisis periods file created successfully.")

def main():
    """Main function to set up the project"""
    logger.info("Starting project setup...")
    
    # Create directory structure
    create_directory_structure()
    
    # Create configuration files
    create_config_files()
    
    # Create .env template
    create_env_template()
    
    # Create __init__.py files
    create_init_files()
    
    # Create crisis periods file
    create_crisis_periods()
    
    logger.info("Project setup completed successfully.")
    logger.info("Next steps:")
    logger.info("1. Edit configs/api_keys.json with your API keys")
    logger.info("2. Edit .env with your environment-specific settings")
    logger.info("3. Run the data retrieval notebook to fetch initial data")

if __name__ == "__main__":
    main()