from setuptools import setup, find_packages

setup(
    name="risk_management_dashboard",
    version="0.1.0",
    description="A comprehensive risk management solution with VaR calculations and stress testing",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.23.0",
        "scipy>=1.9.0",
        "scikit-learn>=1.1.0",
        "matplotlib>=3.6.0",
        "seaborn>=0.12.0",
        "plotly>=5.10.0",
        "yfinance>=0.1.87",
        "fredapi>=0.5.0",
        "pyfolio>=0.9.2",
        "refinitiv-data>=1.0.0",
    ],
    python_requires=">=3.9",
)