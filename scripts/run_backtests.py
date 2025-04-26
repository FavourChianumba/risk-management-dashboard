#!/usr/bin/env python
"""
Script to run backtests for VaR models.
This script tests the accuracy of various VaR models against historical data.
"""

import os
import sys
import argparse
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging

# Add project root to the path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

# Import models
from src.models.historical_var import calculate_historical_var
from src.models.parametric_var import calculate_parametric_var
from src.models.monte_carlo_var import monte_carlo_var
from src.models.expected_shortfall import calculate_expected_shortfall
from src.models.backtesting import (
    backtest_var_model,
    kupiec_test,
    christoffersen_test,
    combined_var_test,
    compare_var_models,
    plot_backtest_results,
    plot_breach_clustering,
    create_backtest_summary
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(project_root / "logs" / "backtests.log")
    ]
)

logger = logging.getLogger(__name__)

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run backtests for VaR models")
    
    parser.add_argument(
        "--model",
        type=str,
        choices=["historical_var", "parametric_var", "monte_carlo_var", "all"],
        default="all",
        help="VaR model to backtest"
    )
    
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="Confidence level for VaR calculation"
    )
    
    parser.add_argument(
        "--window",
        type=int,
        default=252,
        help="Rolling window size (in trading days)"
    )
    
    parser.add_argument(
        "--investment",
        type=int,
        default=1000000,
        help="Investment value for VaR calculation"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=str(project_root / "data" / "results" / "backtest_results.csv"),
        help="Output file for backtest results"
    )
    
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Generate and save backtest plots"
    )
    
    return parser.parse_args()

def load_portfolio_returns():
    """Load portfolio returns from processed data."""
    # Path to portfolio returns
    returns_file = project_root / "data" / "processed" / "portfolio_returns.csv"
    
    if not returns_file.exists():
        logger.error(f"Portfolio returns file not found at {returns_file}")
        sys.exit(1)
    
    # Load portfolio returns
    returns = pd.read_csv(returns_file, index_col=0, parse_dates=True)
    
    # Convert to Series if it's a DataFrame with one column
    if isinstance(returns, pd.DataFrame) and returns.shape[1] == 1:
        returns = returns.iloc[:, 0]
    
    logger.info(f"Loaded portfolio returns from {returns_file} ({len(returns)} observations)")
    
    return returns

def main():
    """Main function to run backtests."""
    # Parse command-line arguments
    args = parse_args()
    
    # Load portfolio returns
    portfolio_returns = load_portfolio_returns()
    
    # Set up models for backtesting
    var_models = {}
    
    if args.model == "historical_var" or args.model == "all":
        var_models["Historical VaR"] = calculate_historical_var
    
    if args.model == "parametric_var" or args.model == "all":
        var_models["Parametric VaR"] = calculate_parametric_var
    
    if args.model == "monte_carlo_var" or args.model == "all":
        var_models["Monte Carlo VaR"] = monte_carlo_var
    
    # Run backtests
    logger.info(f"Running backtests with window size: {args.window}, confidence level: {args.confidence}")
    
    comparison_results = compare_var_models(
        returns=portfolio_returns,
        var_models=var_models,
        window_size=args.window,
        confidence_level=args.confidence,
        investment_value=args.investment
    )
    
    # Create summary of backtest results
    summary = create_backtest_summary(comparison_results, confidence_level=args.confidence)
    
    # Display summary
    logger.info("\nBacktest Summary:")
    print(summary.to_string())
    
    # Save results
    output_file = Path(args.output)
    output_file.parent.mkdir(exist_ok=True, parents=True)
    summary.to_csv(output_file)
    logger.info(f"Saved backtest summary to {output_file}")
    
    # Generate plots if requested
    if args.plots:
        plots_dir = project_root / "data" / "results" / "plots"
        plots_dir.mkdir(exist_ok=True, parents=True)
        
        for model_name, results in comparison_results.items():
            # Plot backtest results
            fig1 = plot_backtest_results(
                backtest_results=results['backtest_results'],
                confidence_level=args.confidence,
                title=f"{model_name} Backtest Results ({args.confidence*100:.0f}% Confidence)"
            )
            
            fig1_file = plots_dir / f"{model_name.lower().replace(' ', '_')}_backtest.png"
            fig1.savefig(fig1_file, dpi=300, bbox_inches='tight')
            logger.info(f"Saved backtest plot to {fig1_file}")
            
            # Plot breach clustering
            fig2 = plot_breach_clustering(
                backtest_results=results['backtest_results'],
                title=f"{model_name} Breach Clustering"
            )
            
            fig2_file = plots_dir / f"{model_name.lower().replace(' ', '_')}_breaches.png"
            fig2.savefig(fig2_file, dpi=300, bbox_inches='tight')
            logger.info(f"Saved breach clustering plot to {fig2_file}")
            
            plt.close('all')  # Close all figures to save memory
    
    return 0

if __name__ == "__main__":
    sys.exit(main())