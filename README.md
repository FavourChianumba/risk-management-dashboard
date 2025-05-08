# Risk Management Dashboard

A comprehensive risk management solution focused on Value-at-Risk (VaR) calculations and stress testing for investment portfolios, featuring multiple methodologies, stress scenario simulations, and dependency modeling with copulas in an interactive Streamlit dashboard. **[Try the live dashboard here](https://my-risk-management-dashboard.streamlit.app/)**.

## Table of Contents
- [About & Motivation](#about--motivation)
- [Features](#features)
- [Installation](#installation)
- [Running the Dashboard](#running-the-dashboard)
- [Project Structure](#project-structure)
- [Key Insights](#key-insights)
- [Challenges Overcome](#challenges-overcome)
- [Future Enhancements](#future-enhancements)
- [What I Learned](#what-i-learned)
- [References](#references)

## About & Motivation

I created this project as part of my journey to become a quantitative analyst. With a background in software engineering and a growing interest in financial markets, I wanted to build something that would help me understand the mathematical foundations of risk management while creating a practical tool.

My goal was to implement the theoretical concepts from books like Hull's "Risk Management and Financial Institutions" and McNeil's "Quantitative Risk Management" in a real-world application. By coding these models from scratch, I've gained a much deeper understanding of how financial institutions measure and manage risk.

## Features

- **Multiple VaR Methodologies**: Historical, Parametric, Monte Carlo approaches
- **Dependency Modeling**: Copula-based approaches for capturing complex asset relationships
- **Comprehensive Stress Testing**: Historical scenarios, synthetic scenarios, and macro factor integration
- **Statistical Validation**: Backtesting with Kupiec and Christoffersen tests
- **Interactive Visualization**: Streamlit dashboard with multiple analytical views
- **Multi-Source Data Retrieval**: Automated data collection with fallback mechanisms

## Installation

### Prerequisites

- Python 3.9+
- Streamlit (installed via requirements.txt)
- FRED API key (optional for macroeconomic data)
- Alpha Vantage API key (optional for enhanced market data)

### Setup

```bash
# Clone the repository
git clone https://github.com/your-username/risk-management-dashboard.git
cd risk-management-dashboard

# Create a virtual environment and install dependencies
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Initialize project structure
python setup_project.py

# Set up API credentials (optional)
# Edit .env or configs/api_keys.json with your API keys
```

## Running the Dashboard

### Live Demo

**Check out the live dashboard**: [https://my-risk-management-dashboard.streamlit.app/](https://my-risk-management-dashboard.streamlit.app/)

### Local Setup

#### Data Pipeline

```bash
# 1. Data Retrieval
jupyter notebook notebooks/01_data_retrieval.ipynb

# 2. Data Cleaning & Preprocessing
jupyter notebook notebooks/02_data_cleaning.ipynb

# 3. Risk Model Calculations
jupyter notebook notebooks/03_var_modeling.ipynb
jupyter notebook notebooks/04_monte_carlo_sim.ipynb
jupyter notebook notebooks/05_stress_testing.ipynb
jupyter notebook notebooks/06_backtesting.ipynb
```

#### Launch the Dashboard

```bash
# Start the Streamlit app
streamlit run app.py
```

The local dashboard will open in your browser at http://localhost:8501, where you can:
- Select date ranges and confidence levels
- Compare different VaR methodologies
- Analyze stress test scenarios
- Review backtesting results
- Explore return distributions

## Project Structure

```
risk_management_dashboard/
├── data/                         # Data storage directory
├── notebooks/                    # Analysis notebooks
├── src/                          # Source code modules
│   ├── data/                     # Data processing
│   ├── models/                   # Risk models
│   ├── stress_testing/           # Stress scenarios
│   ├── visualization/            # Visual components
│   └── utils/                    # Utilities
├── configs/                      # Configuration files
├── tests/                        # Unit tests
├── app.py                        # Streamlit application
└── helper_functions.py           # Dashboard helper functions
```

## Key Insights

My analysis revealed several important findings about portfolio risk measurement:

- **VaR Methodology Comparison**: More sophisticated models incorporating dependencies produced higher VaR estimates ($24,500-$25,400) than traditional methods ($20,900-$22,400), suggesting simple models may underestimate risk.

- **Risk Factor Separation**: PCA analysis showed the portfolio risk separates cleanly into two orthogonal factors: interest rates and equity exposure.

- **Asymmetric Risk Contribution**: Despite equal portfolio weights (50/50), Treasury yield volatility contributed 83% of total VaR versus only 17% from equity exposure.

- **Crisis Response**: The COVID-19 crash produced the most severe historical VaR (4.81%), while synthetic liquidity crises showed the highest potential impact (5.27%).

## Challenges Overcome

- **Monte Carlo Performance**: Optimized simulation code by vectorizing calculations, implementing parallel processing, and using generators, reducing runtime from 20+ minutes to under 2 minutes.

- **Copula Implementation**: Created custom t-copula implementation after extensive research when existing libraries proved inconsistent.

- **Streamlit Memory Issues**: Restructured the application with advanced caching and lazy evaluation to prevent dashboard crashes with large datasets.

## Future Enhancements

1. **Extreme Value Theory Integration**: Implementing Peaks Over Threshold models for better tail risk estimation
2. **Machine Learning for Risk Detection**: Using LSTM networks to predict VaR breaches
3. **Real-Time Data Integration**: Adding websocket connections for live risk monitoring
4. **Portfolio Optimization Module**: Suggesting position adjustments to optimize risk-return profiles

## What I Learned

This project provided deep insights into:

1. **Financial Risk Mathematics**: Implemented complex concepts like copulas, extreme value theory, and multivariate simulations
2. **Market Dynamics**: Developed understanding of interrelationships between market factors during crises
3. **Software Engineering**: Created a modular, testable architecture with proper separation of concerns
4. **Interactive Data Visualization**: Learned techniques for presenting complex risk metrics intuitively

## References

- Hull, J. C. (2018). Risk Management and Financial Institutions. Wiley.
- McNeil, A. J., Frey, R., & Embrechts, P. (2015). Quantitative Risk Management: Concepts, Techniques and Tools. Princeton University Press.
- Basel Committee on Banking Supervision (2019). Minimum capital requirements for market risk.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
