#!/bin/bash
# Script to create the complete directory structure for the risk management dashboard project

# Create main project directory if not already there
mkdir -p risk_management_dashboard

cd risk_management_dashboard

# Create data directories
mkdir -p data/raw
mkdir -p data/processed
mkdir -p data/external

# Create source code directories
mkdir -p src/data
mkdir -p src/models
mkdir -p src/stress_testing
mkdir -p src/visualization
mkdir -p src/utils

# Create notebook directory
mkdir -p notebooks

# Create tests directory
mkdir -p tests

# Create configuration directory
mkdir -p configs

# Create scripts directory
mkdir -p scripts

# Create documentation directory
mkdir -p docs

# Create tableau directory
mkdir -p tableau/templates

# Create logs directory
mkdir -p logs

# Create empty .gitkeep files for empty directories
touch data/raw/.gitkeep
touch data/processed/.gitkeep
touch data/external/.gitkeep
touch logs/.gitkeep
touch tableau/templates/.gitkeep

# Create empty __init__.py files for Python packages
touch src/__init__.py
touch src/data/__init__.py
touch src/models/__init__.py
touch src/stress_testing/__init__.py
touch src/visualization/__init__.py
touch src/utils/__init__.py
touch tests/__init__.py

echo "Directory structure created successfully."
echo "Next steps:"
echo "1. Copy your source files into the appropriate directories"
echo "2. Set up your virtual environment and install dependencies"
echo "3. Configure your API credentials in configs/api_keys.json"