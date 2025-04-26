# Risk Management Dashboard

A comprehensive risk management solution focused on Value-at-Risk (VaR) calculations and stress testing for investment portfolios. I've implemented multiple VaR methodologies, stress scenario simulations, dependency modeling with copulas, and dynamic correlation analysis in an interactive Streamlit dashboard.

## About & Motivation

I created this project as part of my journey to become a quantitative analyst. With a background in software engineering and a growing interest in financial markets, I wanted to build something that would help me understand the mathematical foundations of risk management while creating a practical tool.

My goal was to implement the theoretical concepts from books like Hull's "Risk Management and Financial Institutions" and McNeil's "Quantitative Risk Management" in a real-world application. By coding these models from scratch, I've gained a much deeper understanding of how financial institutions measure and manage risk.

This project has also served as a portfolio piece to demonstrate my skills in Python, financial modeling, and data visualization to potential employers in the quantitative finance field.

## Overview

This risk management dashboard allows portfolio managers and risk analysts to:

- Calculate portfolio VaR using multiple methodologies (Historical, Parametric, Monte Carlo)
- Model complex dependencies between assets with copula-based approaches
- Simulate crisis scenarios with macroeconomic factor integration
- Visualize asset correlations under stress conditions
- Perform backtesting with statistical tests to validate model accuracy

## Project Structure

```
risk_management_dashboard/
│
├── data/                         # Data storage directory
│   ├── raw/                      # Raw, unprocessed data files
│   ├── processed/                # Cleaned and processed data
│   ├── results/                  # Risk model output and metrics
│   └── external/                 # External reference data (crisis periods, etc.)
│
├── notebooks/                    # Jupyter notebooks for development and analysis
│   ├── 01_data_retrieval.ipynb   # Data retrieval from various sources
│   ├── 02_data_cleaning.ipynb    # Data cleaning and preprocessing
│   ├── 03_var_modeling.ipynb     # VaR model implementation
│   ├── 04_monte_carlo_sim.ipynb  # Monte Carlo simulations
│   ├── 05_stress_testing.ipynb   # Stress test scenarios
│   └── 06_backtesting.ipynb      # Model validation and backtesting
│
├── src/                          # Source code modules
│   ├── data/                     # Data processing modules
│   │   ├── retrieve_data.py      # Functions to retrieve market data
│   │   └── process_data.py       # Data processing utilities
│   │
│   ├── models/                   # Risk model implementations
│   │   ├── historical_var.py     # Historical VaR implementation
│   │   ├── parametric_var.py     # Parametric VaR implementation
│   │   ├── monte_carlo_var.py    # Monte Carlo VaR implementation
│   │   ├── expected_shortfall.py # Expected Shortfall calculations
│   │   ├── copula_models.py      # Copula models for dependencies
│   │   ├── evt_models.py         # Extreme Value Theory models
│   │   ├── multivariate_sim.py   # Multivariate simulation utilities
│   │   ├── risk_factor_sim.py    # Risk factor identification and simulation
│   │   └── backtesting.py        # VaR backtesting framework
│   │
│   ├── stress_testing/           # Stress testing functionality
│   │   ├── scenarios.py          # Predefined stress scenarios
│   │   └── macro_integration.py  # Macroeconomic factor integration
│   │
│   ├── visualization/            # Visualization utilities
│   │   ├── monte_carlo_plots.py  # Visualizations for Monte Carlo simulations
│   │   ├── var_plots.py          # Visualizations for VaR analysis
│   │   └── streamlit_prep.py     # Data preparation for Streamlit
│   │
│   └── utils/                    # Utility functions
│       ├── logger.py             # Logging configuration
│       ├── settings.py           # Global settings
│       ├── helpers.py            # General helper functions
│       └── file_utils.py         # File handling utilities
│
├── configs/                      # Configuration files
│   ├── data_config.json          # Data retrieval configuration
│   ├── model_config.json         # Model parameters
│   └── api_keys.json.example     # Template for API keys (gitignored)
│
├── tests/                        # Unit and integration tests
│   ├── test_historical_var.py    # Tests for Historical VaR
│   ├── test_monte_carlo_var.py   # Tests for Monte Carlo VaR
│   └── test_stress_testing.py    # Tests for stress testing
│
├── scripts/                      # Utility scripts
│   ├── data_refresh.py           # Script to refresh data
│   └── run_backtests.py          # Script to run backtests
│
├── logs/                         # Log files
│   └── risk_dashboard_*.log      # Application logs
│
├── app.py                        # Main Streamlit dashboard application
├── helper_functions.py           # Helper functions for the Streamlit app
│
├── requirements.txt              # Python dependencies
├── requirements-dev.txt          # Development dependencies
├── .env.example                  # Environment variables template
├── setup.py                      # Package installation script
├── setup_project.py              # Project initialization script
├── .gitignore                    # Git ignore file
└── README.md                     # Project README
```

## Installation

### Prerequisites

- Python 3.9+
- Streamlit (installed via requirements.txt)
- FRED API key (optional for macroeconomic data)
- Alpha Vantage API key (optional for enhanced market data)
- Polygon API key (optional as additional data source)

### Setup

1. Clone the repository:
   ```
   git clone https://github.com/your-username/risk-management-dashboard.git
   cd risk-management-dashboard
   ```

2. Create a virtual environment and install dependencies:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Run the project initialization script to create the directory structure and configuration files:
   ```
   python setup_project.py
   ```

4. Set up API credentials:
   - Edit the `.env` file with your API keys
   - Or edit `configs/api_keys.json` directly

## Data Sources

I've implemented a multi-source data retrieval system with fallback mechanisms to ensure reliability:

1. **Yahoo Finance** (via yfinance): Primary source for market data, no API key required
2. **Yahoo Finance** (via yahooquery): Alternative source with additional data capabilities 
3. **Alpha Vantage**: Financial market data with more extensive API (requires free API key)
4. **FRED** (Federal Reserve Economic Data): Macroeconomic indicators (requires free API key)
5. **Polygon.io**: Alternative market data source (requires API key)

The system tries each data source in the specified order until successful retrieval.

### Getting API Keys

- **FRED API**: Register at [https://fred.stlouisfed.org/docs/api/api_key.html](https://fred.stlouisfed.org/docs/api/api_key.html)
- **Alpha Vantage API**: Register at [https://www.alphavantage.co/support/#api-key](https://www.alphavantage.co/support/#api-key)
- **Polygon.io API**: Register at [https://polygon.io/dashboard/signup](https://polygon.io/dashboard/signup)

### Adding Credentials to .env File

After obtaining your credentials, add them to your `.env` file:

```
# API Keys
FRED_API_KEY=your_fred_api_key_here
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key_here
POLYGON_API_KEY=your_polygon_key_here

# Data Source Priority
DATA_SOURCE_PRIORITY=yahoo,yahooquery,alphavantage,polygon

# Risk Parameters
RISK_FREE_RATE=0.035

# Logging Level
LOG_LEVEL=INFO
```

## Running the Project End-to-End

Follow these sequential steps to run the entire project pipeline from data retrieval to visualization:

### Step 1: Environment Setup

1. Clone the repository and set up your environment:
   ```bash
   git clone https://github.com/your-username/risk-management-dashboard.git
   cd risk-management-dashboard
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Run the setup script to create necessary directories and configuration files:
   ```bash
   python setup_project.py
   ```

3. Edit the `.env` file with your API credentials.

### Step 2: Data Pipeline Execution

1. **Data Retrieval** - Run notebook 01 to retrieve market and macro data:
   ```bash
   jupyter notebook notebooks/01_data_retrieval.ipynb
   ```
   - Execute all cells
   - Verify that raw data files are created in `data/raw/`
   - Also check that crisis periods are created in `data/external/`

2. **Data Cleaning** - Run notebook 02 to process the raw data:
   ```bash
   jupyter notebook notebooks/02_data_cleaning.ipynb
   ```
   - Execute all cells
   - Verify processed data files are created in `data/processed/`
   - Key files: `market_data.csv`, `portfolio_returns.csv`

### Step 3: Risk Model Calculations

3. **VaR Modeling** - Run notebook 03 to calculate VaR metrics:
   ```bash
   jupyter notebook notebooks/03_var_modeling.ipynb
   ```
   - Execute all cells
   - Verify results are saved in `data/results/`
   - Key files: `var_results.csv`, `var_comparison.csv`

4. **Monte Carlo Simulations** - Run notebook 04 for simulation-based analysis:
   ```bash
   jupyter notebook notebooks/04_monte_carlo_sim.ipynb
   ```
   - Execute all cells
   - Check for Monte Carlo results in `data/results/`
   - Key files: `monte_carlo_comparison.csv`, `monte_carlo_scenarios.csv`

5. **Stress Testing** - Run notebook 05 to perform stress test scenarios:
   ```bash
   jupyter notebook notebooks/05_stress_testing.ipynb
   ```
   - Execute all cells
   - Verify stress test results in `data/results/`
   - Key files: `stress_test_results.csv`, `historical_scenario_results.csv`

6. **Backtesting** - Run notebook 06 to validate model accuracy:
   ```bash
   jupyter notebook notebooks/06_backtesting.ipynb
   ```
   - Execute all cells
   - Check for backtesting results in `data/results/`
   - Key files: `var_backtest_summary.csv`, plots saved in `data/results/plots/`

### Step 4: Visualization with Streamlit

1. Launch the Streamlit application:
   ```bash
   streamlit run app.py
   ```

2. The dashboard will open in your default web browser, typically at http://localhost:8501

3. Use the sidebar filters to:
   - Select date ranges
   - Choose confidence levels
   - Set time horizons for analysis

4. Navigate through the different tabs to explore:
   - Executive Summary
   - VaR Analysis
   - Stress Testing
   - Model Validation
   - Return Analysis

## Challenges I Overcame

During development, I encountered several notable challenges:

### 1. Data Integration Issues

I initially attempted to use Refinitiv API for market data but faced compatibility issues with my Python environment. After numerous attempts to make it work, I pivoted to using Yahoo Finance as my primary data source with Alpha Vantage as backup. This required me to refactor my data retrieval module to handle multiple sources and implement a failover mechanism.

### 2. Monte Carlo Performance

My initial Monte Carlo simulation code was extremely slow, taking over 20 minutes to run 10,000 simulations. I had to optimize the code by:
- Vectorizing calculations using NumPy instead of pandas
- Implementing parallel processing with Python's multiprocessing module
- Optimizing memory usage by using generators where appropriate

These optimizations reduced runtime to under 2 minutes, making the simulations practical for interactive use.

### 3. Copula Implementation Challenges

Implementing t-copulas proved particularly challenging. The mathematical foundation was complex, and available Python libraries had inconsistent implementations. I spent weeks studying the theory and comparing different approaches before creating my own implementation. The breakthrough came when I realized I needed to use the multivariate t-distribution's degrees of freedom parameter to properly model tail dependence.

### 4. Streamlit Memory Issues

When deploying the dashboard, I discovered that Streamlit would crash when loading large datasets or running simulations. To solve this, I had to:
- Restructure the application to calculate results in advance and store them as CSV files
- Implement caching for expensive functions using `@st.cache_data`
- Optimize data loading to use lazy evaluation where possible
- Add pagination for large result sets

### 5. Backtesting Statistical Validity

During backtesting, I initially observed extremely poor performance for all models. After investigation, I discovered I was making a fundamental methodological error in my rolling window implementation, where I was using data from "the future" to predict "the past." Once I corrected this look-ahead bias, the model performance became much more realistic.

## Key Insights from Model Comparison

Through this project, I discovered several interesting insights about portfolio risk measurement:

### VaR Methodology Hierarchy

My analysis of VaR at 95% confidence shows a clear pattern:
- Historical VaR provides the most conservative estimate ($20,928)
- Parametric/Gaussian approaches cluster in the middle ($21,700-$22,400)
- More sophisticated models incorporating dependencies show higher estimates ($24,500-$25,400)

This progression demonstrates how simple models might underestimate risk by missing complex dependencies.

### Risk Factor Analysis

My PCA analysis reveals the portfolio risk separates cleanly into two orthogonal factors:
- Factor 1: Heavily loaded on interest rates (0.9925 weight on US10YT=RR)
- Factor 2: Heavily loaded on equity (0.9925 weight on SPX)

This clean separation indicates minimal overlap between equity and interest rate risks in the portfolio.

### Asymmetric Risk Contribution

Despite equal weights in the portfolio (50/50), I discovered risk contribution is highly asymmetric:
- Treasury yields (US10YT=RR) contribute 83% of portfolio VaR
- Equity index (SPX) contributes only 17% of portfolio VaR

This suggests that Treasury yield volatility is driving most of the portfolio risk, which was surprising but reflects current market conditions.

### Tail Dependency Effects

I found moderate tail dependence (0.28) between equities and rates explains why:
- t-Copula VaR ($22,207) exceeds Gaussian Copula VaR ($21,699)
- Macro factor models produce the highest VaR estimates ($25,382)

These findings demonstrate why simple VaR methods may underestimate portfolio risk during market stress, when correlations and volatilities behave differently than in normal market conditions.

## Stress Testing

I implemented various stress testing methodologies:

### Historical Scenario Analysis

Replays historical crisis periods:
- Global Financial Crisis (2008)
- COVID-19 Market Crash (2020)
- Dot-com Bubble (2000-2002)
- Various Fed Rate Hike Cycles

```python
from src.stress_testing.scenarios import historical_scenario

crisis_returns = historical_scenario(
    returns=market_returns,
    crisis_period="Global Financial Crisis"
)
```

### Synthetic Scenario Generation

Creates hypothetical extreme scenarios:
- Fed Funds rate increases by 200bp with simultaneous 20% equity drop
- Global credit spread widening with liquidity contraction
- Stagflation scenario (high inflation, low growth)

```python
from src.stress_testing.scenarios import synthetic_scenario

shock_scenario = synthetic_scenario(
    returns=market_returns,
    shock_params={asset: -25 for asset in market_returns.columns},  # 25% drop for all assets
    correlation_adjustment=0.2,  # Increased correlations
    volatility_adjustment=1.5    # 50% higher volatility
)
```

### Macro Integration

Incorporates macroeconomic factor modeling:

```python
from src.stress_testing.macro_integration import run_macro_scenario_analysis

macro_results = run_macro_scenario_analysis(
    returns=market_returns,
    macro_data=macro_data,
    investment_value=1000000,
    confidence_level=0.95
)
```

## Key Stress Testing Results

My comprehensive stress testing framework reveals how the portfolio would perform under various extreme scenarios:

### Historical Scenario Impact

Analyzing actual market crises within our data range (2014-2022) shows:
- **COVID-19 Crash (2020)**: Most severe historical event with VaR of 4.81% ($48,062) - more than double the baseline risk
- **2018 Q4 Selloff**: Moderate impact with VaR of 2.28% ($22,835)
- **2022 Rate Hikes**: Lesser impact than expected with VaR of 1.99% ($19,887), slightly below baseline

Note: Earlier crises like the Global Financial Crisis (2008) and Dot-com Bubble (2000-2002) fall outside the data range I used.

### Synthetic Scenario Analysis

My hypothetical stress scenarios reveal potential vulnerabilities not captured in historical data:
- **Liquidity Crisis**: Most severe scenario overall with VaR of 5.27% ($52,724) and potential maximum loss of 8.97%
- **Severe Market Crash**: Significant impact with VaR of 2.93% ($29,252)
- **Rate Shock**: Surprisingly moderate impact with VaR of 2.06% ($20,639)

These synthetic scenarios demonstrate that liquidity stress represents a greater risk to the portfolio than direct market price shocks.

### Macroeconomic Scenario Findings

The factor-based scenarios I created highlight sensitivity to economic conditions:
- **Historical Worst**: Combined extreme factor values create VaR of 3.58% ($35,782)
- **Financial Crisis**: Systemic stress scenario shows VaR of 3.23% ($32,349)
- **VIX Spike**: Volatility shock produces VaR of 2.92% ($29,152), 28.1% above baseline
- **Fed Rate Shock**: Interestingly shows 34.1% decrease in VaR to 1.50% ($15,003)

The negative relationship between Fed rate shocks and portfolio VaR suggests potential hedging benefits from interest rate increases, likely due to the portfolio's significant exposure to US Treasury instruments.

## Backtesting Results

My rigorous model validation through backtesting reveals important insights about VaR model performance:

### Model Accuracy

#### Historical VaR
- **Breach rate**: 5.65% vs expected 5.00%
- **Kupiec test p-value**: 0.2265 (passes coverage test)
- **Christoffersen test p-value**: 0.0237 (fails independence test)
- **Average VaR**: 2.03% of portfolio value

#### Parametric VaR
- **Breach rate**: 5.47% vs expected 5.00%
- Similar statistical profile to Historical VaR
- **Average VaR**: 2.11% of portfolio value
- Slightly more conservative than Historical approach

#### Monte Carlo VaR
- **Breach rate**: 8.05%
- Fails coverage test but passes independence test
- Limited sample size (348 observations vs 1736 for others)
- **Average VaR**: 1.96% of portfolio value

### Breach Clustering Analysis

My analysis of Historical VaR shows evidence of breach clustering with:
- 98 total breaches across the testing period
- 10 consecutive breach runs
- Maximum run length of 3 consecutive days
- Transition probability from non-breach to breach: 5.25%
- Transition probability from breach to breach: 11.34%

This clustering indicates that when a breach occurs, there's a higher probability (11.34% vs 5.25%) of another breach the following day, violating the independence assumption of VaR models.

### Market Regime Dependency

My analysis reveals significant differences in model performance across market regimes:

#### Normal Market Conditions (75.3% of sample period)
- Both Historical and Parametric VaR show 4.25% breach rates
- Well below the expected 5.00% (conservative)
- Average Historical VaR: 1.83% of portfolio value

#### Stressed Market Conditions (24.7% of sample period)
- Historical VaR breach rate rises to 9.18% (2.16x normal rate)
- Parametric VaR breach rate rises to 8.57% (2.02x normal rate)
- Average Historical VaR increases to 2.56% (1.40x normal level)

The significant increase in breach rates during stressed periods indicates that both models underestimate tail risk during market turbulence.

## Future Enhancements

I plan to extend this project in several directions:

1. **Extreme Value Theory Integration**: I want to implement EVT models (Peaks Over Threshold and Block Maxima) to better model tail risk.

2. **Machine Learning for Risk Detection**: I'm exploring using LSTM networks to predict VaR breaches and regime shifts in the market.

3. **Real-Time Data Integration**: Planning to add websocket connections to market data providers for live risk monitoring.

4. **Portfolio Optimization Module**: Working on adding a module that suggests position adjustments to optimize the risk-return profile.

5. **Option Greeks Integration**: I want to incorporate options risk metrics (delta, gamma, vega) for a more complete risk picture.

If you're interested in collaborating on any of these enhancements, please reach out!

## Customization

### Adding Custom Assets

To add custom assets, modify the equity and bond indices in `configs/data_config.json`:

```json
"equity_indices": [
    {"ticker": "SPY", "description": "S&P 500 ETF", "weight": 0.4},
    {"ticker": "QQQ", "description": "Nasdaq 100 ETF", "weight": 0.2},
    {"ticker": "AAPL", "description": "Apple Inc.", "weight": 0.1}
]
```

### Adding Stress Scenarios

Define custom stress scenarios in `configs/model_config.json`:

```json
"synthetic_scenarios": {
    "tech_crash": {
        "equity_shock": -0.30,
        "bond_shock": -0.05,
        "volatility_multiplier": 2.5
    }
}
```

## Development

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_historical_var.py

# Run with verbosity
pytest -v tests/
```

### Logging

The application uses a centralized logging system:

```python
from src.utils.logger import setup_logger

logger = setup_logger(__name__)
logger.info("This is an information message")
logger.warning("This is a warning message")
logger.error("This is an error message")
```

Log files are stored in the `logs/` directory with timestamps.

## What I Learned

This project has been an incredible learning journey for me. I've gained deep insights into:

1. **Financial Risk Mathematics**: Implemented complex mathematical concepts like copulas, extreme value theory, and multivariate simulations.

2. **Market Dynamics**: Developed a much better understanding of how market factors interrelate, especially during crisis periods.

3. **Software Engineering Best Practices**: Created a modular, testable code architecture with proper separation of concerns.

4. **Data Visualization**: Learned how to effectively present complex risk metrics in an intuitive, interactive dashboard.

5. **Statistical Validation**: Implemented proper statistical tests to validate model accuracy and identify weaknesses.

I'm now applying these skills in my quantitative finance studies and hope to leverage this experience in a quant analyst role.

## References

- Hull, J. C. (2018). Risk Management and Financial Institutions. Wiley.
- McNeil, A. J., Frey, R., & Embrechts, P. (2015). Quantitative Risk Management: Concepts, Techniques and Tools. Princeton University Press.
- Basel Committee on Banking Supervision (2019). Minimum capital requirements for market risk.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

Feel free to contact me with any questions or suggestions for improvement. I'm always looking to connect with fellow quants and developers interested in financial risk management.