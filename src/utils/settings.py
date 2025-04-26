"""
Settings module for the risk management dashboard.
This module centralizes configuration settings loaded from environment variables.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Try to load .env file if it exists
env_path = Path('.env')
if env_path.exists():
    load_dotenv(dotenv_path=env_path)

class Settings:
    """Global settings for the risk management dashboard project"""
    
    def __init__(self):
        # Project directories
        self.PROJECT_ROOT = Path(__file__).resolve().parents[1]
        self.CONFIG_DIR = self.PROJECT_ROOT / "configs"
        self.DATA_DIR = self.PROJECT_ROOT / "data"
        self.DATA_RAW_DIR = self.DATA_DIR / "raw"
        self.DATA_PROCESSED_DIR = self.DATA_DIR / "processed"
        self.RESULTS_DIR = self.DATA_DIR / "results"
        self.LOG_DIR = self.PROJECT_ROOT / "logs"
        
        # Ensure directories exist
        self.DATA_RAW_DIR.mkdir(exist_ok=True, parents=True)
        self.DATA_PROCESSED_DIR.mkdir(exist_ok=True, parents=True)
        self.RESULTS_DIR.mkdir(exist_ok=True, parents=True)
        self.LOG_DIR.mkdir(exist_ok=True, parents=True)
        
        # API configurations
        self.YFINANCE_TIMEOUT = 30  # Seconds
        self.FRED_API_KEY = ""
        self.ALPHA_VANTAGE_API_KEY = ""
        self.POLYGON_API_KEY = ""
        
        # Data source priorities
        self.DATA_SOURCE_PRIORITY = ["yahoo", "yahooquery", "alphavantage", "polygon"]
        
        # Risk parameters
        self.RISK_FREE_RATE = 0.035  # Annual risk-free rate (3.5%)
        self.DEFAULT_CONFIDENCE_LEVEL = 0.95  # Default VaR confidence level
        self.DEFAULT_INVESTMENT_VALUE = 1000000  # Default investment value ($1M)
        
        # Data cleaning parameters
        self.MAX_MISSING_PCT = 10.0  # Maximum percentage of missing values allowed
        self.OUTLIER_THRESHOLD = 4.0  # Z-score threshold for outliers
        self.RETURNS_METHOD = "log"  # Method for calculating returns (log or simple)
        
        # VaR model parameters
        self.HISTORICAL_WINDOW = 252  # Historical VaR lookback window (1 year)
        self.MONTE_CARLO_SIMULATIONS = 10000  # Number of Monte Carlo simulations
        self.TIME_HORIZONS = [1, 5, 10, 20]  # VaR time horizons in days
        self.CONFIDENCE_LEVELS = [0.90, 0.95, 0.99]  # VaR confidence levels
        
        # Backtesting parameters
        self.BACKTEST_WINDOW = 252  # Backtesting window size
        self.BACKTEST_STEP = 1  # Backtesting step size
        
        # Stress testing parameters
        self.STRESS_SHOCK_LEVELS = {
            'severe': 3.0,  # Severe shock (3 standard deviations)
            'moderate': 2.0,  # Moderate shock (2 standard deviations) 
            'mild': 1.0  # Mild shock (1 standard deviation)
        }
        
        # Logging settings
        self.LOG_LEVEL = "INFO"
        self.LOG_FILE = self.LOG_DIR / "risk_dashboard.log"
        
        # Load environment variables
        self._load_from_env()
    
    def _load_from_env(self):
        """Load settings from environment variables"""
        # API keys
        self.FRED_API_KEY = os.environ.get('FRED_API_KEY', '')
        self.ALPHA_VANTAGE_API_KEY = os.environ.get('ALPHA_VANTAGE_API_KEY', '')
        self.POLYGON_API_KEY = os.environ.get('POLYGON_API_KEY', '')
        
        # Override data source priority if set
        if os.environ.get('DATA_SOURCE_PRIORITY'):
            self.DATA_SOURCE_PRIORITY = os.environ.get('DATA_SOURCE_PRIORITY').lower().split(',')
        
        # Risk and model parameters
        if os.environ.get('RISK_FREE_RATE'):
            self.RISK_FREE_RATE = float(os.environ.get('RISK_FREE_RATE'))
        
        if os.environ.get('DEFAULT_CONFIDENCE_LEVEL'):
            self.DEFAULT_CONFIDENCE_LEVEL = float(os.environ.get('DEFAULT_CONFIDENCE_LEVEL'))
        
        if os.environ.get('RETURNS_METHOD'):
            self.RETURNS_METHOD = os.environ.get('RETURNS_METHOD').lower()
        
        if os.environ.get('HISTORICAL_WINDOW'):
            self.HISTORICAL_WINDOW = int(os.environ.get('HISTORICAL_WINDOW'))
        
        if os.environ.get('MONTE_CARLO_SIMULATIONS'):
            self.MONTE_CARLO_SIMULATIONS = int(os.environ.get('MONTE_CARLO_SIMULATIONS'))
        
        # Data cleaning parameters
        if os.environ.get('MAX_MISSING_PCT'):
            self.MAX_MISSING_PCT = float(os.environ.get('MAX_MISSING_PCT'))
        
        if os.environ.get('OUTLIER_THRESHOLD'):
            self.OUTLIER_THRESHOLD = float(os.environ.get('OUTLIER_THRESHOLD'))
        
        # Logging settings
        if os.environ.get('LOG_LEVEL'):
            self.LOG_LEVEL = os.environ.get('LOG_LEVEL').upper()
        
        # Timeout settings
        if os.environ.get('YFINANCE_TIMEOUT'):
            self.YFINANCE_TIMEOUT = int(os.environ.get('YFINANCE_TIMEOUT'))

settings = Settings()