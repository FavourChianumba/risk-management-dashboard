#!/bin/bash
# Script to create .gitkeep files in empty directories

# Create .gitkeep files in data directories
mkdir -p data/raw
mkdir -p data/processed
mkdir -p data/external
touch data/raw/.gitkeep
touch data/processed/.gitkeep
touch data/external/.gitkeep

# Create logs directory
mkdir -p logs
touch logs/.gitkeep

# Create tableau directory
mkdir -p tableau/templates
touch tableau/templates/.gitkeep

# Make sure notebook directory exists
mkdir -p notebooks

echo "Created .gitkeep files in empty directories"