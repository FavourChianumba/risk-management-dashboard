import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
from scipy import stats
import scipy.stats as scipy_stats
from datetime import datetime, timedelta
import os
from pathlib import Path
import base64
from PIL import Image
import io
import streamlit.components.v1 as components


# Set page configuration
st.set_page_config(
    page_title="Risk Management Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dashboard styling - updated with improved colors and visual elements
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1a237e;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e0e0e0;
    }
    .sub-header {
        font-size: 1.75rem;
        font-weight: 600;
        color: #283593;
        margin-bottom: 0.75rem;
        padding-bottom: 0.25rem;
        border-bottom: 1px solid #e0e0e0;
    }
    .metric-card {
        background-color: #f5f7fa;
        border-radius: 0.75rem;
        padding: 1.25rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        border-left: 4px solid #1a237e;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0, 0, 0, 0.1);
    }
    .metric-card h3 {
        color: #455a64;
        font-size: 1.1rem;
        margin-bottom: 0.5rem;
    }
    .risk-high {
        color: #d32f2f;
        font-weight: 600;
    }
    .risk-medium {
        color: #f57c00;
        font-weight: 600;
    }
    .risk-low {
        color: #2e7d32;
        font-weight: 600;
    }
    .info-text {
        font-size: 0.9rem;
        color: #546e7a;
        line-height: 1.5;
        margin-bottom: 1.5rem;
        padding: 0.75rem;
        background-color: #f5f7fa;
        border-radius: 0.5rem;
        border-left: 3px solid #90a4ae;
    }
    .stTabs {
        background-color: #ffffff;
        padding: 0.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background-color: #f5f7fa;
        padding: 0.25rem;
        border-radius: 0.5rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f5f7fa;
        border-radius: 0.5rem;
        gap: 1px;
        padding: 0.75rem 1.25rem;
        font-weight: 500;
        color: #455a64;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff;
        border-bottom: 3px solid #1a237e;
        color: #1a237e;
        font-weight: 600;
    }
    .data-stat {
        display: inline-flex;
        align-items: center;
        margin-right: 1rem;
        padding: 0.25rem 0.75rem;
        background-color: #f5f7fa;
        border-radius: 1rem;
        font-size: 0.85rem;
    }
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: help;
        color: #1a237e;
    }
    .tooltip:hover::after {
        content: attr(data-tooltip);
        position: absolute;
        bottom: 125%;
        left: 50%;
        transform: translateX(-50%);
        background-color: #455a64;
        color: white;
        padding: 0.5rem 0.75rem;
        border-radius: 0.25rem;
        white-space: nowrap;
        z-index: 100;
        font-size: 0.75rem;
    }
    .badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        border-radius: 1rem;
        font-size: 0.75rem;
        font-weight: 600;
        margin-left: 0.5rem;
    }
    .badge-high {
        background-color: #ffebee;
        color: #d32f2f;
    }
    .badge-medium {
        background-color: #fff3e0;
        color: #f57c00;
    }
    .badge-low {
        background-color: #e8f5e9;
        color: #2e7d32;
    }
    .data-card {
        background-color: #ffffff;
        border-radius: 0.75rem;
        padding: 1.25rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        margin-bottom: 1rem;
    }
    .data-table {
        width: 100%;
        border-collapse: collapse;
    }
    .data-table th {
        background-color: #f5f7fa;
        padding: 0.5rem 0.75rem;
        text-align: left;
        font-weight: 600;
        color: #455a64;
        border-bottom: 2px solid #e0e0e0;
    }
    .data-table td {
        padding: 0.5rem 0.75rem;
        border-bottom: 1px solid #f0f0f0;
    }
    footer {
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #e0e0e0;
        text-align: center;
        color: #78909c;
        font-size: 0.8rem;
    }
    .stProgress > div > div > div > div {
        background-color: #1a237e;
    }
    .stAlert {
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions - enhanced with additional utilities
def format_currency(value, precision=2):
    """Format value as currency with configurable precision."""
    if pd.isna(value):
        return "N/A"
    return f"${value:,.{precision}f}"

def format_percentage(value, precision=2):
    """Format value as percentage with configurable precision."""
    if pd.isna(value):
        return "N/A"
    
    # Check if value is already in percentage form
    if abs(value) > 1 and abs(value) < 100:
        value = value / 100  # Convert to decimal form
    return f"{value:.{precision}%}"

def risk_level_color(var_pct):
    """Get color based on VaR percentage."""
    if var_pct > 0.04:
        return "risk-high"
    elif var_pct > 0.02:
        return "risk-medium"
    else:
        return "risk-low"

def risk_level_text(var_pct):
    """Get risk level text based on VaR percentage."""
    if var_pct > 0.04:
        return "High"
    elif var_pct > 0.02:
        return "Medium"
    else:
        return "Low"

def get_arrow_emoji(value):
    """Return arrow emoji based on value."""
    if value > 0:
        return "↑"
    elif value < 0:
        return "↓"
    else:
        return "→"

def create_tooltip(text, tooltip_text):
    """Create an HTML tooltip."""
    return f'<span class="tooltip" data-tooltip="{tooltip_text}">{text} ℹ️</span>'

def create_badge(text, level):
    """Create a styled badge."""
    return f'<span class="badge badge-{level}">{text}</span>'

def create_gauge_chart(value, min_val=0, max_val=100, thresholds=None):
    """Create a gauge chart with thresholds."""
    if thresholds is None:
        thresholds = [
            {'range': [0, 50], 'color': "#DC2626"},
            {'range': [50, 80], 'color': "#F59E0B"},
            {'range': [80, 100], 'color': "#10B981"}
        ]
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        gauge={
            'axis': {'range': [min_val, max_val], 'tickwidth': 1},
            'bar': {'color': "#283593"},
            'steps': thresholds,
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 80
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=30, b=20)
    )
    
    return fig

# Image loading function
def get_base64_image(image_path):
    """Load and encode image to base64 for inline HTML display."""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except Exception:
        # Return a default placeholder icon if the file doesn't exist
        img = Image.new('RGB', (200, 200), color = (26, 35, 126))
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()

# Data loading - enhanced with better error handling and data validation
@st.cache_data(ttl=3600)
def load_data():
    """Load all data files with enhanced error handling and validation."""
    # Initialize empty dataframes as fallbacks
    portfolio_returns = pd.DataFrame(columns=['Return'])
    var_results = pd.DataFrame(columns=['confidence_level', 'time_horizon', 'historical_var', 
                                       'historical_es', 'parametric_var', 'parametric_es', 
                                       'historical_pct', 'parametric_pct'])
    stress_test_results = pd.DataFrame(columns=['scenario', 'var_value', 'var_pct', 'scenario_type'])
    var_backtest_summary = pd.DataFrame(columns=['Model', 'Avg VaR (%)', 'Max VaR (%)', 'Breaches', 
                                                'Breach Rate (%)', 'Expected (%)', 'Breach Ratio'])
    
    # Track data loading status
    data_status = {
        'portfolio_returns': {'status': 'failed', 'message': ''},
        'var_results': {'status': 'failed', 'message': ''},
        'stress_test_results': {'status': 'failed', 'message': ''},
        'var_backtest_summary': {'status': 'failed', 'message': ''}
    }
    
    try:
        # Define base directory relative to the current file
        base_dir = Path(__file__).parent
        
        # Define paths to data files
        portfolio_returns_path = base_dir / "data" / "processed" / "portfolio_returns.csv"
        var_results_path = base_dir / "data" / "results" / "var_results.csv"
        stress_test_results_path = base_dir / "data" / "results" / "stress_test_results.csv"
        var_backtest_summary_path = base_dir / "data" / "results" / "var_backtest_summary.csv"
        
        # Load each file with individual try-except blocks
        try:
            portfolio_returns = pd.read_csv(portfolio_returns_path, parse_dates=['Date'], index_col='Date')
            # Rename column for clarity if needed
            if len(portfolio_returns.columns) == 1 and portfolio_returns.columns[0] != 'Return':
                portfolio_returns.columns = ['Return']
            data_status['portfolio_returns'] = {'status': 'success', 'message': ''}
        except Exception as e:
            data_status['portfolio_returns'] = {'status': 'failed', 'message': str(e)}
        
        try:
            var_results = pd.read_csv(var_results_path)
            data_status['var_results'] = {'status': 'success', 'message': ''}
        except Exception as e:
            data_status['var_results'] = {'status': 'failed', 'message': str(e)}
        
        try:
            stress_test_results = pd.read_csv(stress_test_results_path)
            # Add risk level if not present
            if 'risk_level' not in stress_test_results.columns:
                stress_test_results['risk_level'] = stress_test_results['var_pct'].apply(risk_level_text)
            data_status['stress_test_results'] = {'status': 'success', 'message': ''}
        except Exception as e:
            data_status['stress_test_results'] = {'status': 'failed', 'message': str(e)}
        
        try:
            var_backtest_summary = pd.read_csv(var_backtest_summary_path)
            data_status['var_backtest_summary'] = {'status': 'success', 'message': ''}
        except Exception as e:
            data_status['var_backtest_summary'] = {'status': 'failed', 'message': str(e)}
    
    except Exception as e:
        st.error(f"Global error in data loading: {e}")
    
    return portfolio_returns, var_results, stress_test_results, var_backtest_summary, data_status

# Load data
portfolio_returns, var_results, stress_test_results, var_backtest_summary, data_status = load_data()

# Create a calendar/date table for better filtering
@st.cache_data
def create_calendar(portfolio_returns):
    """Create a calendar table from portfolio returns dates."""
    if portfolio_returns.empty:
        return pd.DataFrame(columns=['Date', 'Year', 'Month', 'MonthName', 'Quarter'])
    
    dates = portfolio_returns.index.to_frame(index=False)
    dates.columns = ['Date']
    dates['Year'] = dates['Date'].dt.year
    dates['Month'] = dates['Date'].dt.month
    dates['MonthName'] = dates['Date'].dt.strftime('%b')
    dates['Quarter'] = dates['Date'].dt.quarter
    dates['YearMonth'] = dates['Date'].dt.strftime('%Y-%m')
    dates['WeekDay'] = dates['Date'].dt.day_name()
    dates['WeekOfYear'] = dates['Date'].dt.isocalendar().week
    return dates

calendar = create_calendar(portfolio_returns)

# Sidebar - Enhanced with better organization and visual appeal
st.sidebar.markdown("""
<div style="text-align: center; padding-bottom: 1rem;">
    <h1 style="color: #1a237e; font-size: 1.75rem; margin-bottom: 0.5rem;">Risk Management Dashboard</h1>
    <p style="color: #546e7a; font-size: 0.9rem;">Comprehensive portfolio risk analysis and monitoring</p>
</div>
""", unsafe_allow_html=True)

# Try to load logo, fallback to a placeholder if not available
logo_base64 = get_base64_image("logo.png")
st.sidebar.markdown(f"""
<div style="text-align: center; margin-bottom: 2rem;">
    <img src="data:image/png;base64,{logo_base64}" width="180">
</div>
""", unsafe_allow_html=True)

# Add data quality indicators
st.sidebar.markdown("""
<h3 style="color: #455a64; font-size: 1.1rem; margin-bottom: 0.75rem; border-bottom: 1px solid #e0e0e0; padding-bottom: 0.5rem;">
    Data Quality
</h3>
""", unsafe_allow_html=True)

for dataset, status in data_status.items():
    icon = "✅" if status['status'] == 'success' else "❌"
    color = "#2e7d32" if status['status'] == 'success' else "#d32f2f"
    st.sidebar.markdown(f"""
    <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
        <span style="color: #455a64;">{dataset.replace('_', ' ').title()}</span>
        <span style="color: {color};">{icon}</span>
    </div>
    """, unsafe_allow_html=True)

# Divider
st.sidebar.markdown("<hr style='margin: 1.5rem 0; opacity: 0.3;'>", unsafe_allow_html=True)

# Date range filter - Enhanced with better defaults and indicators
st.sidebar.markdown("""
<h3 style="color: #455a64; font-size: 1.1rem; margin-bottom: 0.75rem;">
    Time Period
</h3>
""", unsafe_allow_html=True)

if not calendar.empty:
    min_date = calendar['Date'].min()
    max_date = calendar['Date'].max()
    
    # Default to last year of data
    default_start = max_date - timedelta(days=365) if min_date < max_date - timedelta(days=365) else min_date
    
    date_range = st.sidebar.date_input(
        "Date Range",
        value=(default_start, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    if len(date_range) == 2:
        start_date, end_date = date_range
        # Filter portfolio returns by date
        filtered_returns = portfolio_returns.loc[start_date:end_date]
        
        # Show data range stats
        days_selected = (end_date - start_date).days
        st.sidebar.markdown(f"""
        <div style="font-size: 0.85rem; color: #546e7a; margin-top: 0.5rem;">
            <span class="data-stat">📅 {days_selected} days</span>
            <span class="data-stat">📊 {len(filtered_returns)} data points</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        filtered_returns = portfolio_returns
else:
    filtered_returns = portfolio_returns

# Divider
st.sidebar.markdown("<hr style='margin: 1.5rem 0; opacity: 0.3;'>", unsafe_allow_html=True)

# Risk parameters - Enhanced with better organization and explanations
st.sidebar.markdown("""
<h3 style="color: #455a64; font-size: 1.1rem; margin-bottom: 0.75rem;">
    Risk Parameters
</h3>
""", unsafe_allow_html=True)

# Confidence level filter with better explanation
confidence_options = sorted(var_results['confidence_level'].unique()) if not var_results.empty else [0.95]
confidence_level = st.sidebar.selectbox(
    "Confidence Level",
    options=confidence_options,
    format_func=lambda x: f"{x:.0%}",
    index=1 if len(confidence_options) > 1 else 0,
    help="The probability that losses won't exceed the VaR estimate over the specified time horizon"
)

# Time horizon filter with better explanation
horizon_options = sorted(var_results['time_horizon'].unique()) if not var_results.empty else [1]
time_horizon = st.sidebar.selectbox(
    "Time Horizon (Days)",
    options=horizon_options,
    index=0,
    help="The time period over which the VaR is calculated"
)

# Filter VaR results based on selections
filtered_var_results = var_results[
    (var_results['confidence_level'] == confidence_level) & 
    (var_results['time_horizon'] == time_horizon)
] if not var_results.empty else pd.DataFrame()

# Display effective VaR threshold
if not filtered_var_results.empty:
    var_threshold = filtered_var_results['historical_pct'].values[0]
    threshold_color = risk_level_color(var_threshold)
    st.sidebar.markdown(f"""
    <div style="margin-top: 1rem; padding: 0.75rem; background-color: #f5f7fa; border-radius: 0.5rem; border-left: 3px solid #1a237e;">
        <p style="margin: 0; font-size: 0.9rem; color: #455a64;">Effective VaR Threshold:</p>
        <p style="margin: 0; font-size: 1.1rem; font-weight: 600;" class="{threshold_color}">{format_percentage(var_threshold)}</p>
    </div>
    """, unsafe_allow_html=True)

# Advanced sidebar options in expander
with st.sidebar.expander("Advanced Options"):
    st.markdown("""
    These settings control additional aspects of the risk analysis:
    """, unsafe_allow_html=True)
    
    # Example: Risk factor sensitivity
    risk_factor_sensitivity = st.slider(
        "Risk Factor Sensitivity",
        min_value=0.5,
        max_value=2.0,
        value=1.0,
        step=0.1,
        help="Multiplier for risk factor sensitivities in stress testing"
    )
    
    # Example: Correlation regime
    correlation_regime = st.selectbox(
        "Correlation Regime",
        options=["Normal", "Stress", "Crisis"],
        index=0,
        help="Correlation assumptions for risk calculations"
    )
    
    # Example: Display options
    show_tooltips = st.checkbox("Show Detailed Tooltips", value=True)
    show_technical_metrics = st.checkbox("Show Technical Metrics", value=False)

# Divider
st.sidebar.markdown("<hr style='margin: 1.5rem 0; opacity: 0.3;'>", unsafe_allow_html=True)

# Footer in sidebar
st.sidebar.markdown("""
<div style="text-align: center; margin-top: 2rem; font-size: 0.8rem; color: #78909c;">
    <p>Dashboard v1.5.0</p>
    <p>Last data update: April 26, 2025</p>
</div>
""", unsafe_allow_html=True)

# Navigation tabs - Enhanced with clearer labels and organization
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Executive Summary", 
    "📈 VaR Analysis", 
    "⚠️ Stress Testing", 
    "✓ Model Validation",
    "📉 Return Analysis"
])

# 1. Executive Summary - Enhanced with better organization and visual hierarchy
with tab1:
    st.markdown("<h1 class='main-header'>Risk Management Executive Summary</h1>", unsafe_allow_html=True)
    
    # Add timestamp and last update information
    last_update = datetime.now().strftime("%B %d, %Y at %H:%M")
    st.markdown(f"""
    <div style="margin-bottom: 1.5rem; display: flex; justify-content: space-between; align-items: center;">
        <span style="color: #546e7a; font-size: 0.9rem;">
            <strong>Analysis Period:</strong> {start_date.strftime('%b %d, %Y')} to {end_date.strftime('%b %d, %Y')}
        </span>
        <span style="color: #546e7a; font-size: 0.9rem;">
            <strong>Last Dashboard Update:</strong> {last_update}
        </span>
    </div>
    """, unsafe_allow_html=True)
    
    # Key metrics - Enhanced with better visual styling and information
    if not filtered_var_results.empty:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            hist_var = filtered_var_results['historical_var'].values[0]
            hist_pct = filtered_var_results['historical_pct'].values[0]
            risk_class = risk_level_text(hist_pct)
            risk_color = risk_level_color(hist_pct)
            
            # Better metric card with clearer information
            st.markdown(f"""
            <div class='metric-card'>
                <h3>Historical VaR ({confidence_level:.0%}, {time_horizon}d)</h3>
                <p style='font-size: 1.8rem; font-weight: 600; margin-bottom: 0.25rem;'>{format_currency(hist_var)}</p>
                <p style='margin: 0;'>
                    <span>({format_percentage(hist_pct)})</span> 
                    <span class='{risk_color}'>{risk_class} Risk</span>
                </p>
                <p style='margin-top: 0.5rem; font-size: 0.8rem; color: #546e7a;'>
                    Maximum expected loss with {confidence_level:.0%} confidence
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            param_var = filtered_var_results['parametric_var'].values[0]
            param_pct = filtered_var_results['parametric_pct'].values[0]
            param_risk_class = risk_level_text(param_pct)
            param_risk_color = risk_level_color(param_pct)
            
            # Comparison with Historical VaR
            diff = param_var - hist_var
            diff_pct = (param_var / hist_var - 1) if hist_var != 0 else 0
            diff_text = f"({format_percentage(diff_pct)} vs Historical)" if hist_var != 0 else ""
            
            st.markdown(f"""
            <div class='metric-card'>
                <h3>Parametric VaR ({confidence_level:.0%}, {time_horizon}d)</h3>
                <p style='font-size: 1.8rem; font-weight: 600; margin-bottom: 0.25rem;'>{format_currency(param_var)}</p>
                <p style='margin: 0;'>
                    <span>({format_percentage(param_pct)})</span>
                    <span class='{param_risk_color}'>{diff_text}</span>
                </p>
                <p style='margin-top: 0.5rem; font-size: 0.8rem; color: #546e7a;'>
                    Normal distribution-based estimate
                </p>
            </div>
            """, unsafe_allow_html=True)
        investment_value = 1000000
        
        with col3:
            hist_es = filtered_var_results['historical_es'].values[0]
            es_pct = hist_es / investment_value if 'investment_value' in locals() else hist_es / 1000000
            es_ratio = hist_es / hist_var
            es_risk_class = risk_level_text(es_pct)
            es_risk_color = risk_level_color(es_pct)
            
            st.markdown(f"""
            <div class='metric-card'>
                <h3>Expected Shortfall (CVaR)</h3>
                <p style='font-size: 1.8rem; font-weight: 600; margin-bottom: 0.25rem;'>{format_currency(hist_es)}</p>
                <p style='margin: 0;'>
                    <span>ES/VaR Ratio: {es_ratio:.2f}x</span>
                </p>
                <p style='margin-top: 0.5rem; font-size: 0.8rem; color: #546e7a;'>
                    Average loss when VaR is exceeded
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            # Get worst stress scenario
            if not stress_test_results.empty:
                # Exclude baseline
                non_baseline = stress_test_results[stress_test_results['scenario'] != 'Baseline']
                if not non_baseline.empty:
                    worst_scenario = non_baseline.sort_values('var_value', ascending=False).iloc[0]
                    worst_risk_color = risk_level_color(worst_scenario['var_pct'])
                    
                    st.markdown(f"""
                    <div class='metric-card'>
                        <h3>Worst Stress Scenario</h3>
                        <p style='font-size: 1.5rem; font-weight: 600; margin-bottom: 0.25rem;'>{worst_scenario['scenario']}</p>
                        <p style='margin: 0;'>
                            <span>{format_currency(worst_scenario['var_value'])}</span>
                            <span class='{worst_risk_color}'>({format_percentage(worst_scenario['var_pct'])})</span>
                        </p>
                        <p style='margin-top: 0.5rem; font-size: 0.8rem; color: #546e7a;'>
                            {worst_scenario['scenario_type']} scenario
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
    
    # Portfolio performance - Enhanced with better visualization and information
    st.markdown("<h2 class='sub-header'>Portfolio Performance</h2>", unsafe_allow_html=True)
    
    if not filtered_returns.empty:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Create cumulative returns
            cum_returns = (1 + filtered_returns).cumprod() - 1
            
            # Plot cumulative returns with VaR threshold
            fig = go.Figure()
            
            # Add cumulative returns line
            fig.add_trace(go.Scatter(
                x=cum_returns.index,
                y=cum_returns['Return'],
                mode='lines',
                name='Cumulative Return',
                line=dict(color='#283593', width=2.5)
            ))
            
            # Add VaR threshold if available
            if not filtered_var_results.empty:
                var_threshold = -filtered_var_results['historical_pct'].values[0]
                fig.add_trace(go.Scatter(
                    x=[cum_returns.index.min(), cum_returns.index.max()],
                    y=[var_threshold, var_threshold],
                    mode='lines',
                    name=f'{confidence_level:.0%} VaR Threshold',
                    line=dict(color='#d32f2f', width=2, dash='dash')
                ))
            
            # Add horizontal line at 0
            fig.add_shape(
                type="line",
                x0=cum_returns.index.min(),
                y0=0,
                x1=cum_returns.index.max(),
                y1=0,
                line=dict(color="#455a64", width=1, dash="dot"),
            )
            
            # Either define it or create a placeholder empty DataFrame
            try:
                # Check if key_events already exists
                if 'key_events' not in locals():
                    # Option 1: Create an empty DataFrame with the expected structure
                    key_events = pd.DataFrame(columns=['date', 'description'])
                    
                    # Option 2 (Alternative): You could populate it with some sample events
                    # key_events = pd.DataFrame([
                    #     {'date': pd.Timestamp('2020-03-23'), 'description': 'COVID-19 Market Bottom'},
                    #     {'date': pd.Timestamp('2022-01-03'), 'description': 'Fed Rate Hike Cycle Begins'}
                    # ])
            except Exception as e:
                # Fallback in case of any issues
                key_events = pd.DataFrame(columns=['date', 'description'])


            # Highlight key events if they exist in the dataframe
            if 'key_events' in locals() and not key_events.empty:
                for _, event in key_events.iterrows():
                    if event['date'] in cum_returns.index:
                        fig.add_annotation(
                            x=event['date'],
                            y=cum_returns.loc[event['date'], 'Return'],
                            text=event['description'],
                            showarrow=True,
                            arrowhead=1,
                            ax=0,
                            ay=-40
                        )
            
            fig.update_layout(
                title='Cumulative Portfolio Return',
                xaxis_title='Date',
                yaxis_title='Return',
                yaxis_tickformat='.1%',
                hovermode='x unified',
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                height=400,
                margin=dict(l=20, r=20, t=50, b=20),
                plot_bgcolor='#ffffff',
                paper_bgcolor='#ffffff',
                font=dict(color='#455a64')
            )
            
            # Add range selector
            fig.update_xaxes(
                rangeslider_visible=True,
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(count=1, label="YTD", step="year", stepmode="todate"),
                        dict(count=1, label="1y", step="year", stepmode="backward"),
                        dict(step="all")
                    ])
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Return statistics with enhanced formatting and explanations
            cum_final = cum_returns['Return'].iloc[-1]
            ann_return = (1 + filtered_returns['Return'].mean()) ** 252 - 1  # Assuming daily data
            ann_vol = filtered_returns['Return'].std() * np.sqrt(252)  # Assuming daily data
            sharpe = ann_return / ann_vol if ann_vol != 0 else 0
            
            # Calculate max drawdown
            rolling_max = cum_returns.cummax()
            drawdown = (cum_returns / rolling_max) - 1
            if isinstance(drawdown, pd.DataFrame):
                max_drawdown = drawdown.min().min()  # Get the global minimum across all columns
            else:
                max_drawdown = drawdown.min() if not drawdown.empty else 0
            
            # Calculate VaR ratio (risk-adjusted measure)
            var_ratio = filtered_var_results['historical_pct'].values[0] / ann_vol if not filtered_var_results.empty else np.nan
            
            stats = pd.DataFrame({
                'Metric': [
                    'Total Return',
                    'Annualized Return',
                    'Annualized Volatility',
                    'Sharpe Ratio',
                    'Max Drawdown',
                    'VaR Ratio'
                ],
                'Value': [
                    cum_final,
                    ann_return,
                    ann_vol,
                    sharpe,
                    max_drawdown,
                    var_ratio
                ],
                'Tooltip': [
                    'Cumulative return over the selected period',
                    'Return expressed on an annual basis',
                    'Standard deviation of returns on an annual basis',
                    'Risk-adjusted return (assuming 0% risk-free rate)',
                    'Maximum peak-to-trough decline',
                    'VaR as a percentage of annualized volatility'
                ]
            })
            
            # Format values with improved visuals
            styles = []
            for i, row in stats.iterrows():
                if row['Metric'] in ['Total Return', 'Annualized Return']:
                    value_str = format_percentage(row['Value'])
                    style = 'color: #2e7d32;' if row['Value'] > 0 else 'color: #d32f2f;'
                elif row['Metric'] in ['Annualized Volatility', 'Max Drawdown']:
                    value_str = format_percentage(row['Value'])
                    style = 'color: #d32f2f;'
                elif row['Metric'] == 'Sharpe Ratio':
                    value_str = f"{row['Value']:.2f}"
                    style = 'color: #2e7d32;' if row['Value'] > 1 else 'color: #f57c00;' if row['Value'] > 0 else 'color: #d32f2f;'
                elif row['Metric'] == 'VaR Ratio':
                    if np.isnan(row['Value']):
                        value_str = "N/A"
                        style = 'color: #78909c;'
                    else:
                        value_str = f"{row['Value']:.2f}"
                        style = 'color: #2e7d32;' if row['Value'] < 0.5 else 'color: #f57c00;' if row['Value'] < 1 else 'color: #d32f2f;'
                else:
                    value_str = str(row['Value'])
                    style = ''
                
                styles.append((row['Metric'], value_str, style, row['Tooltip']))
            
            # Create HTML table with tooltips
            stat_html = "<table class='data-table' style='margin-top: 0;'>"
            stat_html += "<tr><th>Metric</th><th>Value</th></tr>"
            
            for metric, value, style, tooltip in styles:
                stat_html += f"""
                <tr>
                    <td><span class="tooltip" data-tooltip="{tooltip}">{metric} ℹ️</span></td>
                    <td style="{style}">{value}</td>
                </tr>
                """
            
            stat_html += "</table>"
            
            components.html(stat_html, height=200, scrolling=False)
            
            # Add chart of rolling metrics
            window = min(60, len(filtered_returns) // 2)  # 60-day window or half the data length
            if window > 10:  # Only show if we have enough data
                rolling_vol = filtered_returns['Return'].rolling(window=window).std() * np.sqrt(252)
                rolling_sharpe = (filtered_returns['Return'].rolling(window=window).mean() * 252) / rolling_vol
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=rolling_vol.index,
                    y=rolling_vol,
                    mode='lines',
                    name='Rolling Volatility',
                    line=dict(color='#f57c00', width=2)
                ))
                
                fig.add_trace(go.Scatter(
                    x=rolling_sharpe.index,
                    y=rolling_sharpe,
                    mode='lines',
                    name='Rolling Sharpe',
                    line=dict(color='#2e7d32', width=2),
                    yaxis="y2"
                ))
                
                fig.update_layout(
                    title=f'{window}-Day Rolling Metrics',
                    xaxis_title='Date',
                    yaxis=dict(
                        title='Annualized Volatility',
                        title_font=dict(color='#f57c00'),
                        tickfont=dict(color='#f57c00'),
                        tickformat='.1%'
                    ),
                    yaxis2=dict(
                        title='Sharpe Ratio',
                        title_font=dict(color='#2e7d32'),
                        tickfont=dict(color='#2e7d32'),
                        anchor="x",
                        overlaying="y",
                        side="right"
                    ),
                    height=200,
                    margin=dict(l=10, r=10, t=40, b=10),
                    showlegend=False,
                    plot_bgcolor='#ffffff',
                    paper_bgcolor='#ffffff'
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    # Model performance summary - Enhanced with better visuals and information
    st.markdown("<h2 class='sub-header'>Model Performance Summary</h2>", unsafe_allow_html=True)
    
    if not var_backtest_summary.empty:
        # Enhanced model comparison with visual indicators
        model_comparison = var_backtest_summary.copy()
        
        # Calculate model accuracy score based on breach ratio
        if 'Breach Ratio' in model_comparison.columns:
            model_comparison['Accuracy Score'] = model_comparison['Breach Ratio'].apply(
                lambda ratio: 100 - min(100, abs(ratio - 1) * 100)
            )
        
        # Create visual display with enhanced elements
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Style the dataframe with custom formatting
            st.dataframe(
                model_comparison,
                use_container_width=True,
                column_config={
                    "Model": st.column_config.TextColumn("Model"),
                    "Avg VaR (%)": st.column_config.NumberColumn("Avg VaR (%)", format="%.2f%%"),
                    "Max VaR (%)": st.column_config.NumberColumn("Max VaR (%)", format="%.2f%%"),
                    "Breach Rate (%)": st.column_config.NumberColumn("Breach Rate (%)", format="%.2f%%"),
                    "Expected (%)": st.column_config.NumberColumn("Expected (%)", format="%.2f%%"),
                    "Breach Ratio": st.column_config.NumberColumn(
                        "Breach Ratio",
                        format="%.2f",
                        help="Ratio of actual to expected breaches. Ideal value is 1.0"
                    ),
                    "Assessment": st.column_config.TextColumn("Assessment"),
                    "Accuracy Score": st.column_config.ProgressColumn(
                        "Accuracy Score",
                        format="%.0f%%",
                        min_value=0,
                        max_value=100
                    ),
                },
                hide_index=True
            )
        
        with col2:
            # Display best model recommendation
            if 'Accuracy Score' in model_comparison.columns:
                best_model = model_comparison.loc[model_comparison['Accuracy Score'].idxmax()]
                
                st.markdown(f"""
                <div class='metric-card' style="height: 90%;">
                    <h3>Recommended Model</h3>
                    <p style='font-size: 1.5rem; font-weight: 600; margin-bottom: 0.5rem;'>{best_model['Model']}</p>
                    <p style='margin: 0;'>Accuracy: <span style='color: #2e7d32;'>{best_model['Accuracy Score']:.0f}%</span></p>
                    <p style='margin: 0;'>Assessment: <span style='color: #2e7d32;'>{best_model['Assessment']}</span></p>
                    <p style='margin-top: 1rem; font-size: 0.85rem; color: #546e7a;'>
                        This model provides the most accurate risk estimates based on backtesting results.
                    </p>
                </div>
                """, unsafe_allow_html=True)

    # Add key risk insights section
    st.markdown("<h2 class='sub-header'>Key Risk Insights</h2>", unsafe_allow_html=True)
    
    # Create three-column layout for insights
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Risk trend insight
        if not filtered_returns.empty and len(filtered_returns) > 30:
            # Calculate trend in volatility
            recent_window = 30
            recent_vol = filtered_returns['Return'].iloc[-recent_window:].std() * np.sqrt(252)
            previous_vol = filtered_returns['Return'].iloc[-2*recent_window:-recent_window].std() * np.sqrt(252)
            vol_change = (recent_vol / previous_vol) - 1
            
            trend_icon = "⬆️" if vol_change > 0.1 else "⬇️" if vol_change < -0.1 else "➡️"
            trend_color = "#d32f2f" if vol_change > 0.1 else "#2e7d32" if vol_change < -0.1 else "#f57c00"
            
            st.markdown(f"""
            <div class='data-card'>
                <h3 style='margin-top: 0;'>Volatility Trend</h3>
                <p style='font-size: 1.5rem; font-weight: 600; color: {trend_color};'>
                    {trend_icon} {format_percentage(vol_change)}
                </p>
                <p style='color: #546e7a; font-size: 0.9rem; margin-bottom: 0;'>
                    {'Increasing volatility indicates heightened risk' if vol_change > 0 else 
                     'Decreasing volatility indicates stabilizing markets' if vol_change < 0 else
                     'Stable volatility environment'}
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        # Correlation risk insight
        if not filtered_returns.empty:
            st.markdown(f"""
            <div class='data-card'>
                <h3 style='margin-top: 0;'>Tail Risk Assessment</h3>
                <p style='font-size: 1.2rem; font-weight: 600;'>
                    {create_badge("Moderate", "medium")}
                </p>
                <p style='color: #546e7a; font-size: 0.9rem; margin-bottom: 0;'>
                    ES/VaR ratio of {es_ratio:.2f}x indicates {'significant' if es_ratio > 1.5 else 'moderate' if es_ratio > 1.3 else 'normal'} tail risk
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    with col3:
        # Model reliability insight
        if not var_backtest_summary.empty:
            avg_accuracy = model_comparison['Accuracy Score'].mean() if 'Accuracy Score' in model_comparison.columns else 0
            reliability_level = "High" if avg_accuracy > 85 else "Medium" if avg_accuracy > 70 else "Low"
            reliability_color = "low" if reliability_level == "High" else "medium" if reliability_level == "Medium" else "high"
            
            st.markdown(f"""
            <div class='data-card'>
                <h3 style='margin-top: 0;'>Model Reliability</h3>
                <p style='font-size: 1.2rem; font-weight: 600;'>
                    {create_badge(reliability_level, reliability_color)}
                </p>
                <p style='color: #546e7a; font-size: 0.9rem; margin-bottom: 0;'>
                    Average model accuracy of {avg_accuracy:.0f}% based on backtesting results
                </p>
            </div>
            """, unsafe_allow_html=True)

# 2. VaR Analysis - Enhanced with better explanations and visualizations
with tab2:
    st.markdown("<h1 class='main-header'>Value-at-Risk (VaR) Analysis</h1>", unsafe_allow_html=True)
    
    if not filtered_var_results.empty:
        # Enhanced explanation with interactive elements
        with st.expander("Understanding Value-at-Risk (VaR)", expanded=False):
            st.markdown("""
            ### What is Value-at-Risk?
            
            **Value-at-Risk (VaR)** is a statistical measure that quantifies the potential loss in value of a portfolio 
            over a defined period for a given confidence interval. It answers the question:
            
            > "What is the maximum amount we expect to lose over X days, Y% of the time?"
            
            ### Key Components:
            
            - **Confidence Level**: The probability that the loss won't exceed the VaR estimate
            - **Time Horizon**: The period over which the VaR is calculated
            - **VaR Methodologies**: Different approaches to calculating VaR
            
            ### VaR Methodologies Compared:
            
            | Method | Description | Strengths | Limitations |
            |--------|-------------|-----------|-------------|
            | **Historical** | Uses actual past returns to estimate future losses | No distribution assumptions, captures actual extreme events | Limited by historical data, may miss future scenarios |
            | **Parametric** | Assumes returns follow a normal distribution | Simple, requires less data | Underestimates tail risk, assumes normality |
            | **Monte Carlo** | Simulates many possible market scenarios | Flexible, can incorporate various risk factors | Computationally intensive, sensitive to assumptions |
            
            ### Expected Shortfall (ES):
            
            Also known as Conditional VaR (CVaR), ES measures the average loss beyond the VaR threshold, 
            providing insight into the severity of losses in the tail of the distribution.
            """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <p class='info-text'>
            Value-at-Risk (VaR) estimates the maximum expected loss of a portfolio over a specified time horizon at a given 
            confidence level. This analysis compares different VaR methodologies and shows how VaR scales with different 
            time horizons and confidence levels.
        </p>
        """, unsafe_allow_html=True)
        
        st.markdown("<h2 class='sub-header'>VaR Methodology Comparison</h2>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # Create enhanced comparison bar chart
            fig = go.Figure()
            
            # Add bars for VaR values with improved styling
            methods = ['Historical VaR', 'Parametric VaR', 'Historical ES', 'Parametric ES']
            values = [
                filtered_var_results['historical_var'].values[0],
                filtered_var_results['parametric_var'].values[0],
                filtered_var_results['historical_es'].values[0],
                filtered_var_results['parametric_es'].values[0]
            ]
            
            # Define improved color scheme
            colors = ['#283593', '#00838f', '#283593', '#00838f']
            opacity = [1, 1, 0.6, 0.6]
            
            for i, (method, value) in enumerate(zip(methods, values)):
                fig.add_trace(go.Bar(
                    x=[method],
                    y=[value],
                    name=method,
                    marker_color=colors[i],
                    marker_opacity=opacity[i],
                    text=[f"${value:,.2f}"],
                    textposition='outside',
                    hovertemplate='%{x}: %{y:$,.2f}<extra></extra>'
                ))
            
            # Add ES/VaR ratio annotation
            hist_ratio = values[2] / values[0]
            param_ratio = values[3] / values[1]
            
            fig.add_annotation(
                x='Historical ES',
                y=values[2] * 1.05,
                text=f"ES/VaR: {hist_ratio:.2f}x",
                showarrow=False,
                font=dict(size=10, color="#455a64")
            )
            
            fig.add_annotation(
                x='Parametric ES',
                y=values[3] * 1.05,
                text=f"ES/VaR: {param_ratio:.2f}x",
                showarrow=False,
                font=dict(size=10, color="#455a64")
            )
            
            fig.update_layout(
                title=f"Risk Metrics at {confidence_level:.0%} Confidence Level ({time_horizon}-Day Horizon)",
                xaxis_title="Methodology",
                yaxis_title="Value ($)",
                barmode='group',
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                height=400,
                margin=dict(l=20, r=20, t=50, b=20),
                plot_bgcolor='#ffffff',
                paper_bgcolor='#ffffff',
                font=dict(color='#455a64')
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add insight callout below chart
            diff_pct = (values[1] / values[0] - 1) * 100
            insight_text = (
                f"Parametric VaR is {abs(diff_pct):.1f}% {'higher' if diff_pct > 0 else 'lower'} than Historical VaR, "
                f"suggesting the return distribution {'has lighter tails than' if diff_pct > 0 else 'has heavier tails than' if diff_pct < 0 else 'closely matches'} "
                f"a normal distribution."
            )
            
            st.markdown(f"""
            <div style="padding: 0.75rem; background-color: #e8f0fe; border-radius: 0.5rem; border-left: 3px solid #4285f4; margin-top: 1rem;">
                <p style="margin: 0; color: #455a64; font-size: 0.9rem;">
                    <strong>Insight:</strong> {insight_text}
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Create an enhanced comparison table with more context
            comparison_data = {
                'Metric': [
                    'Historical VaR',
                    'Parametric VaR',
                    'Difference',
                    'Historical ES',
                    'Parametric ES',
                    'ES/VaR Ratio (Hist)',
                    'ES/VaR Ratio (Param)'
                ],
                'Value': [
                    filtered_var_results['historical_var'].values[0],
                    filtered_var_results['parametric_var'].values[0],
                    filtered_var_results['historical_var'].values[0] - filtered_var_results['parametric_var'].values[0],
                    filtered_var_results['historical_es'].values[0],
                    filtered_var_results['parametric_es'].values[0],
                    filtered_var_results['historical_es'].values[0] / filtered_var_results['historical_var'].values[0],
                    filtered_var_results['parametric_es'].values[0] / filtered_var_results['parametric_var'].values[0]
                ],
                'Context': [
                    'Based on empirical distribution',
                    'Assumes normal distribution',
                    'Difference between methods',
                    'Average loss beyond VaR',
                    'Parametric tail estimate',
                    'Tail severity indicator',
                    'Parametric tail severity'
                ]
            }
            
            comparison_df = pd.DataFrame(comparison_data)
            
            # Format values with enhanced styling
            comparison_html = "<table class='data-table'>"
            comparison_html += "<tr><th>Metric</th><th>Value</th><th>Context</th></tr>"
            
            for i, row in comparison_df.iterrows():
                # Format value based on metric type
                if 'Ratio' in row['Metric']:
                    formatted_value = f"{row['Value']:.2f}x"
                    style = "color: #d32f2f;" if row['Value'] > 1.5 else "color: #f57c00;" if row['Value'] > 1.3 else "color: #2e7d32;"
                elif 'Difference' == row['Metric']:
                    formatted_value = f"${row['Value']:,.2f}"
                    style = "color: #d32f2f;" if row['Value'] > 0 else "color: #2e7d32;" if row['Value'] < 0 else ""
                else:
                    formatted_value = f"${row['Value']:,.2f}"
                    style = ""
                
                comparison_html += f"""
                <tr>
                    <td>{row['Metric']}</td>
                    <td style="{style}">{formatted_value}</td>
                    <td style="font-size: 0.85rem; color: #546e7a;">{row['Context']}</td>
                </tr>
                """
            
            comparison_html += "</table>"
            
            components.html(comparison_html, height=300, scrolling=False)
            
            # Add interpretation guide
            with st.expander("Interpretation Guide"):
                st.markdown("""
                - **VaR Difference**: Significant differences between Historical and Parametric VaR indicate non-normal return distributions.
                - **ES/VaR Ratio**: Ratios > 1.3 indicate significant tail risk. Higher ratios mean more severe potential losses beyond VaR.
                - **Method Selection**: Historical VaR typically captures actual market behavior better but requires more historical data.
                """, unsafe_allow_html=True)
        
        # Time horizon scaling with enhanced visualization and explanation
        st.markdown("<h2 class='sub-header'>VaR Scaling by Time Horizon</h2>", unsafe_allow_html=True)
        
        # Filter var_results for the selected confidence level but all time horizons
        horizon_data = var_results[var_results['confidence_level'] == confidence_level].copy()
        
        if not horizon_data.empty:
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # Enhanced plot with annotations and styling
                fig = go.Figure()
                
                # Add lines for each VaR method with improved styling
                fig.add_trace(go.Scatter(
                    x=horizon_data['time_horizon'],
                    y=horizon_data['historical_var'],
                    mode='lines+markers',
                    name='Historical VaR',
                    line=dict(color='#283593', width=3),
                    marker=dict(size=8)
                ))
                
                fig.add_trace(go.Scatter(
                    x=horizon_data['time_horizon'],
                    y=horizon_data['parametric_var'],
                    mode='lines+markers',
                    name='Parametric VaR',
                    line=dict(color='#00838f', width=3),
                    marker=dict(size=8)
                ))
                
                # Add theoretical sqrt(t) scaling line for comparison
                if len(horizon_data) > 1:
                    base_var = horizon_data.loc[horizon_data['time_horizon'] == 1, 'historical_var'].values[0]
                    sqrt_scaling = [base_var * np.sqrt(t) for t in horizon_data['time_horizon']]
                    
                    fig.add_trace(go.Scatter(
                        x=horizon_data['time_horizon'],
                        y=sqrt_scaling,
                        mode='lines',
                        name='√t Scaling',
                        line=dict(color='#4caf50', width=2, dash='dash')
                    ))
                
                # Add annotations explaining the implications
                max_horizon = horizon_data['time_horizon'].max()
                fig.add_annotation(
                    x=max_horizon,
                    y=horizon_data.loc[horizon_data['time_horizon'] == max_horizon, 'historical_var'].values[0],
                    text=f"{time_horizon}-day VaR: ${horizon_data.loc[horizon_data['time_horizon'] == max_horizon, 'historical_var'].values[0]:,.0f}",
                    showarrow=True,
                    arrowhead=1,
                    ax=30,
                    ay=-30,
                    font=dict(size=10, color="#455a64")
                )
                
                fig.update_layout(
                    title=f"VaR Scaling with Time Horizon ({confidence_level:.0%} Confidence)",
                    xaxis_title="Time Horizon (Days)",
                    yaxis_title="Value ($)",
                    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                    height=400,
                    margin=dict(l=20, r=20, t=50, b=20),
                    plot_bgcolor='#ffffff',
                    paper_bgcolor='#ffffff',
                    font=dict(color='#455a64'),
                    hovermode='x unified'
                )
                
                # Format x-axis to only show integer values
                fig.update_xaxes(tickmode='array', tickvals=list(horizon_data['time_horizon']))
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Add a data table showing the scaling factors
                horizon_data['Historical Scale'] = horizon_data['historical_var'] / horizon_data.loc[horizon_data['time_horizon'] == 1, 'historical_var'].values[0]
                horizon_data['Parametric Scale'] = horizon_data['parametric_var'] / horizon_data.loc[horizon_data['time_horizon'] == 1, 'parametric_var'].values[0]
                horizon_data['Theoretical Scale'] = np.sqrt(horizon_data['time_horizon'])
                
                scale_data = horizon_data[['time_horizon', 'Historical Scale', 'Parametric Scale', 'Theoretical Scale']]
                
                # Create an HTML table with better formatting
                scale_html = "<table class='data-table'>"
                scale_html += "<tr><th>Horizon</th><th>Hist. Scale</th><th>Param. Scale</th><th>√t Scale</th></tr>"
                
                for _, row in scale_data.iterrows():
                    # Highlight deviations from sqrt(t) rule
                    hist_diff = row['Historical Scale'] / row['Theoretical Scale'] - 1
                    param_diff = row['Parametric Scale'] / row['Theoretical Scale'] - 1
                    
                    hist_style = "" if abs(hist_diff) < 0.05 else "color: #d32f2f;" if hist_diff > 0 else "color: #2e7d32;"
                    param_style = "" if abs(param_diff) < 0.05 else "color: #d32f2f;" if param_diff > 0 else "color: #2e7d32;"
                    
                    scale_html += f"""
                    <tr>
                        <td>{int(row['time_horizon'])} days</td>
                        <td style="{hist_style}">{row['Historical Scale']:.2f}x</td>
                        <td style="{param_style}">{row['Parametric Scale']:.2f}x</td>
                        <td>{row['Theoretical Scale']:.2f}x</td>
                    </tr>
                    """
                
                scale_html += "</table>"
                
                components.html(scale_html, height=170, scrolling=False)
                
                # Add scaling explanation
                st.markdown("""
                <p style="font-size: 0.85rem; color: #546e7a; margin-top: 1rem;">
                    <strong>Scaling Factor</strong>: Under IID assumptions, VaR should scale with the square root of time (√t).
                    Deviations suggest:
                </p>
                <ul style="font-size: 0.85rem; color: #546e7a; margin: 0; padding-left: 1.5rem;">
                    <li>Lower scaling: Return mean-reversion</li>
                    <li>Higher scaling: Return momentum</li>
                </ul>
                """, unsafe_allow_html=True)
            
            # Add an expandable section with more detailed explanations
            with st.expander("About the Square Root of Time Rule"):
                st.markdown("""
                ### The Square Root of Time Rule
                
                The square root of time rule is commonly used to scale VaR from one time horizon to another:
                
                $$\text{VaR}(t \text{ days}) = \text{VaR}(1 \text{ day}) \times \sqrt{t}$$
                
                This approximation relies on these assumptions:
                
                1. Returns are independently and identically distributed (i.i.d.)
                2. Returns have zero mean or the mean effect is negligible over short horizons
                3. Portfolio composition remains constant
                
                #### Interpretation of Deviations:
                
                - **Actual scaling > √t scaling**: Suggests momentum effects or volatility clustering
                - **Actual scaling < √t scaling**: Suggests mean-reversion effects in the time series
                
                #### Regulatory Context:
                
                Many regulatory frameworks, including Basel III, allow for the use of the square root of time rule to 
                scale VaR from a 1-day to a 10-day horizon.
                """, unsafe_allow_html=True)
        else:
            st.warning("No VaR data available for different time horizons with the selected confidence level.")
            
        # VaR methodology deep dive
        st.markdown("<h2 class='sub-header'>VaR Methodology Deep Dive</h2>", unsafe_allow_html=True)
        
        # Add methodology selector with tabs
        method_tab1, method_tab2, method_tab3 = st.tabs(["Historical VaR", "Parametric VaR", "Expected Shortfall"])
        
        with method_tab1:
            st.markdown("""
            ### Historical Value-at-Risk
            
            Historical VaR uses actual historical returns to estimate potential future losses without assuming a specific probability distribution.
            
            #### Calculation Method:
            1. Collect historical return data for the portfolio
            2. Sort returns from worst to best
            3. Find the return value at the specified percentile (e.g., 5th percentile for 95% confidence level)
            4. Multiply this return value by the portfolio value to get VaR in monetary terms
            
            #### Strengths:
            - Uses actual historical data without distribution assumptions
            - Captures actual extreme events that occurred in the past
            - Preserves the historical correlations between assets
            
            #### Limitations:
            - Limited by available historical data (may miss potential future events)
            - Gives equal weight to all historical observations, regardless of when they occurred
            - May not reflect current market conditions if they differ significantly from historical periods
            """, unsafe_allow_html=True)
            
            # Add visualization example if returns data is available
            if not filtered_returns.empty:
                # Calculate percentile return for the display
                var_percentile = 1 - confidence_level
                hist_var_return = np.percentile(filtered_returns['Return'], var_percentile * 100)
                
                fig = go.Figure()
                
                # Add histogram of returns
                fig.add_trace(go.Histogram(
                    x=filtered_returns['Return'],
                    histnorm='probability',
                    nbinsx=30,
                    marker_color='rgba(40, 53, 147, 0.6)',
                    name='Return Distribution'
                ))
                
                # Add line for VaR threshold
                fig.add_shape(
                    type="line",
                    x0=hist_var_return, y0=0,
                    x1=hist_var_return, y1=0.15,
                    line=dict(color="#d32f2f", width=2, dash="dash"),
                )
                
                # Add area for tail
                fig.add_trace(go.Histogram(
                    x=filtered_returns[filtered_returns['Return'] <= hist_var_return]['Return'],
                    histnorm='probability',
                    nbinsx=15,
                    marker_color='rgba(211, 47, 47, 0.8)',
                    name=f'VaR Region ({confidence_level*100:.0f}%)'
                ))
                
                fig.add_annotation(
                    x=hist_var_return,
                    y=0.16,
                    text=f"VaR: {hist_var_return:.2%}",
                    showarrow=True,
                    arrowhead=1,
                    ax=-40,
                    ay=-20,
                    font=dict(color="#d32f2f")
                )
                
                fig.update_layout(
                    title="Historical VaR Visualization",
                    xaxis_title="Return",
                    yaxis_title="Probability",
                    barmode='overlay',
                    height=300,
                    margin=dict(l=20, r=20, t=50, b=20),
                    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                    xaxis=dict(tickformat='.1%'),
                    plot_bgcolor='#ffffff',
                    paper_bgcolor='#ffffff'
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        with method_tab2:
            st.markdown("""
            ### Parametric Value-at-Risk
            
            Parametric VaR assumes returns follow a specific probability distribution (typically normal) and uses the statistical properties of that distribution to calculate VaR.
            
            #### Calculation Method:
            1. Calculate the mean (μ) and standard deviation (σ) of historical returns
            2. Find the z-score for the specified confidence level (e.g., -1.645 for 95% confidence)
            3. Calculate VaR as: VaR = -(μ + z × σ) × portfolio value
            
            #### Strengths:
            - Simple to calculate and explain
            - Requires less historical data than non-parametric methods
            - Easy to decompose into risk factor contributions
            
            #### Limitations:
            - Assumes returns follow a normal distribution, which often underestimates tail risk
            - May not capture the fat tails and skewness typically seen in financial returns
            - Less accurate during periods of market stress when normality assumptions tend to break down
            """, unsafe_allow_html=True)
            
            # Add visualization example if returns data is available
            if not filtered_returns.empty:
                # Calculate normal distribution parameters
                mu = filtered_returns['Return'].mean()
                sigma = filtered_returns['Return'].std()
                
                # Calculate VaR based on normal distribution
                z_score = scipy_stats.norm.ppf(1 - confidence_level)
                param_var_return = mu + z_score * sigma
                
                # Generate x values for normal distribution curve
                x = np.linspace(mu - 4*sigma, mu + 4*sigma, 1000)
                y = scipy_stats.norm.pdf(x, mu, sigma)
                
                fig = go.Figure()
                
                # Add histogram of returns
                fig.add_trace(go.Histogram(
                    x=filtered_returns['Return'],
                    histnorm='probability density',
                    nbinsx=30,
                    marker_color='rgba(40, 53, 147, 0.3)',
                    name='Actual Returns'
                ))
                
                # Add normal distribution curve
                fig.add_trace(go.Scatter(
                    x=x,
                    y=y,
                    mode='lines',
                    line=dict(color='#00838f', width=2),
                    name='Normal Distribution'
                ))
                
                # Shade VaR region
                x_tail = [x for x in x if x <= param_var_return]
                y_tail = [scipy_stats.norm.pdf(val, mu, sigma) for val in x_tail]
                
                fig.add_trace(go.Scatter(
                    x=x_tail,
                    y=y_tail,
                    fill='tozeroy',
                    fillcolor='rgba(211, 47, 47, 0.2)',
                    line=dict(color='rgba(0,0,0,0)'),
                    name=f'VaR Region ({confidence_level*100:.0f}%)'
                ))
                
                # Add VaR line
                fig.add_shape(
                    type="line",
                    x0=param_var_return, y0=0,
                    x1=param_var_return, y1=scipy_stats.norm.pdf(param_var_return, mu, sigma) * 1.1,
                    line=dict(color="#d32f2f", width=2, dash="dash"),
                )
                
                fig.add_annotation(
                    x=param_var_return,
                    y=scipy_stats.norm.pdf(param_var_return, mu, sigma) * 1.2,
                    text=f"VaR: {param_var_return:.2%}",
                    showarrow=True,
                    arrowhead=1,
                    ax=-40,
                    ay=-20,
                    font=dict(color="#d32f2f")
                )
                
                fig.update_layout(
                    title="Parametric VaR Visualization",
                    xaxis_title="Return",
                    yaxis_title="Probability Density",
                    height=300,
                    margin=dict(l=20, r=20, t=50, b=20),
                    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                    xaxis=dict(tickformat='.1%'),
                    plot_bgcolor='#ffffff',
                    paper_bgcolor='#ffffff'
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        with method_tab3:
            st.markdown("""
            ### Expected Shortfall (ES) / Conditional VaR
            
            Expected Shortfall (ES), also known as Conditional VaR (CVaR) or Average VaR, measures the average loss beyond the VaR threshold, providing insight into the severity of potential losses in the tail of the distribution.
            
            #### Calculation Method:
            1. Calculate VaR at the specified confidence level
            2. Identify all returns that are worse than VaR (in the tail)
            3. Calculate the average of these tail returns
            4. Multiply this average by the portfolio value to get ES in monetary terms
            
            #### Strengths:
            - Provides information about the size of losses beyond VaR
            - Considered a coherent risk measure (unlike VaR)
            - Better captures tail risk and gives a more conservative estimate
            
            #### Limitations:
            - More difficult to backtest than VaR (requires more tail events)
            - More sensitive to extreme outliers in the data
            - May be less intuitive to interpret than VaR
            
            #### Regulatory Significance:
            The Basel Committee on Banking Supervision has emphasized Expected Shortfall in its Fundamental Review of the Trading Book (FRTB) framework, recognizing its advantages over traditional VaR.
            """, unsafe_allow_html=True)
            
            # Add visualization example if returns data is available
            if not filtered_returns.empty:
                # Calculate VaR and ES
                var_percentile = 1 - confidence_level
                hist_var_return = np.percentile(filtered_returns['Return'], var_percentile * 100)
                tail_returns = filtered_returns[filtered_returns['Return'] <= hist_var_return]['Return']
                hist_es_return = tail_returns.mean()
                
                fig = go.Figure()
                
                # Add histogram of returns
                fig.add_trace(go.Histogram(
                    x=filtered_returns['Return'],
                    histnorm='probability density',
                    nbinsx=30,
                    marker_color='rgba(40, 53, 147, 0.3)',
                    name='Return Distribution'
                ))
                
                # Add VaR line
                fig.add_shape(
                    type="line",
                    x0=hist_var_return, y0=0,
                    x1=hist_var_return, y1=20,
                    line=dict(color="#f57c00", width=2, dash="dash"),
                )
                
                # Add ES line
                fig.add_shape(
                    type="line",
                    x0=hist_es_return, y0=0,
                    x1=hist_es_return, y1=20,
                    line=dict(color="#d32f2f", width=2),
                )
                
                # Shade tail area
                fig.add_trace(go.Histogram(
                    x=tail_returns,
                    histnorm='probability density',
                    nbinsx=15,
                    marker_color='rgba(211, 47, 47, 0.5)',
                    name=f'Tail Region ({var_percentile*100:.0f}%)'
                ))
                
                # Add annotations
                fig.add_annotation(
                    x=hist_var_return,
                    y=15,
                    text=f"VaR: {hist_var_return:.2%}",
                    showarrow=True,
                    arrowhead=1,
                    ax=40,
                    ay=0,
                    font=dict(color="#f57c00")
                )
                
                fig.add_annotation(
                    x=hist_es_return,
                    y=10,
                    text=f"ES: {hist_es_return:.2%}",
                    showarrow=True,
                    arrowhead=1,
                    ax=-40,
                    ay=20,
                    font=dict(color="#d32f2f")
                )
                
                fig.update_layout(
                    title="Expected Shortfall Visualization",
                    xaxis_title="Return",
                    yaxis_title="Probability Density",
                    height=300,
                    margin=dict(l=20, r=20, t=50, b=20),
                    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                    xaxis=dict(tickformat='.1%'),
                    plot_bgcolor='#ffffff',
                    paper_bgcolor='#ffffff'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Add ES/VaR ratio insight
                es_var_ratio = hist_es_return / hist_var_return
                
                if es_var_ratio > 1.5:
                    ratio_insight = "significantly heavier than normal distribution tails, indicating substantial tail risk"
                elif es_var_ratio > 1.3:
                    ratio_insight = "moderately heavier than normal distribution tails, indicating elevated tail risk"
                else:
                    ratio_insight = "relatively close to normal distribution tails, indicating manageable tail risk"
                
                st.markdown(f"""
                <div style="padding: 0.75rem; background-color: #e8f0fe; border-radius: 0.5rem; border-left: 3px solid #4285f4; margin-top: 1rem;">
                    <p style="margin: 0; color: #455a64; font-size: 0.9rem;">
                        <strong>Insight:</strong> The ES/VaR ratio of {es_var_ratio:.2f}x suggests portfolio returns have {ratio_insight}.
                    </p>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.warning("No VaR data available for the selected confidence level and time horizon. Please adjust your selections or check the data source.")

# 3. Stress Testing - Enhanced with better organization and visuals
with tab3:
    st.markdown("<h1 class='main-header'>Stress Test Analysis</h1>", unsafe_allow_html=True)
    
    if not stress_test_results.empty:
        # Enhanced explanation with interactive elements
        with st.expander("Understanding Stress Testing", expanded=False):
            st.markdown("""
            ### What is Stress Testing?
            
            **Stress testing** evaluates how a portfolio would perform under extreme, but plausible, market conditions that go beyond normal VaR confidence levels.
            Unlike VaR, stress tests don't assign probabilities to scenarios but instead focus on the severity of potential losses.
            
            ### Key Types of Stress Tests:
            
            #### Historical Scenarios
            - Based on actual historical crisis periods (e.g., 2008 Financial Crisis, COVID-19 Crash)
            - Uses actual market movements from these periods
            - Advantage: Realistic scenarios that actually occurred
            - Limitation: May not capture future crises with different characteristics
            
            #### Synthetic Scenarios
            - Hypothetical scenarios designed to test specific vulnerabilities
            - Can be more severe than historical events
            - Advantage: Can test vulnerabilities without historical precedent
            - Examples: Severe market crash, liquidity crisis, rate shock scenarios
            
            #### Macroeconomic Scenarios
            - Based on economic variables like GDP growth, inflation, interest rates
            - Often used in regulatory stress tests
            - Advantage: Links market movements to underlying economic factors
            
            ### Interpreting Stress Test Results:
            
            - Absolute losses show the potential financial impact
            - Relative comparisons across scenarios show where vulnerabilities lie
            - Risk levels indicate the severity and need for mitigation strategies
            """)
        
        st.markdown(f"""
        <p class='info-text'>
            Stress testing evaluates portfolio performance under extreme market conditions that go beyond
            normal VaR confidence levels. This analysis shows how the portfolio would perform under various
            historical and hypothetical crisis scenarios.
        </p>
        """, unsafe_allow_html=True)
        
        # Improved scenario selector with categorized options
        # Group scenarios by type
        scenario_groups = {}
        if 'scenario_type' in stress_test_results.columns:
            for scenario_type in stress_test_results['scenario_type'].unique():
                scenarios_in_group = stress_test_results[stress_test_results['scenario_type'] == scenario_type]['scenario'].unique()
                scenario_groups[scenario_type] = scenarios_in_group
        
        # Create selectbox with grouped options
        scenario_options = []
        for group, scenarios in scenario_groups.items():
            for scenario in scenarios:
                scenario_options.append(f"{group}: {scenario}")
        
        if not scenario_options:
            # Fallback if grouping doesn't work
            scenario_options = list(stress_test_results['scenario'].unique())
        
        # Find index of Baseline scenario if it exists
        default_idx = 0
        for i, option in enumerate(scenario_options):
            if "Baseline" in option:
                default_idx = i
                break
        
        selected_scenario_option = st.selectbox(
            "Select Stress Scenario",
            options=scenario_options,
            index=default_idx,
            help="Choose a scenario to analyze its impact on the portfolio"
        )
        
        # Extract the actual scenario name
        if ":" in selected_scenario_option:
            selected_scenario_type, selected_scenario = selected_scenario_option.split(":", 1)
            selected_scenario = selected_scenario.strip()
        else:
            selected_scenario = selected_scenario_option
            selected_scenario_type = stress_test_results.loc[stress_test_results['scenario'] == selected_scenario, 'scenario_type'].iloc[0] if 'scenario_type' in stress_test_results.columns else "Unknown"
        
        # Get scenario data
        scenario_data = stress_test_results[stress_test_results['scenario'] == selected_scenario].iloc[0]
        
        # Get baseline data if it exists
        baseline_data = None
        baseline_scenario = "Baseline"
        if baseline_scenario in stress_test_results['scenario'].values:
            baseline_data = stress_test_results[stress_test_results['scenario'] == baseline_scenario].iloc[0]
        
        # Display scenario details with enhanced visualization
        st.markdown("<h2 class='sub-header'>Scenario Analysis</h2>", unsafe_allow_html=True)
        
        # Create 3-column layout with enhanced metric cards
        col1, col2, col3 = st.columns(3)
        
        with col1:
            var_value = scenario_data['var_value']
            var_pct = scenario_data['var_pct']
            risk_level = scenario_data['risk_level'] if 'risk_level' in scenario_data else risk_level_text(var_pct)
            risk_color = risk_level_color(var_pct)
            
            st.markdown(f"""
            <div class='metric-card'>
                <h3>Scenario VaR</h3>
                <p style='font-size: 1.8rem; font-weight: 600; margin-bottom: 0.25rem;'>{format_currency(var_value)}</p>
                <p style='margin: 0;'>
                    <span>({format_percentage(var_pct)})</span>
                    <span class='{risk_color}'>{risk_level} Risk</span>
                </p>
                <p style='margin-top: 0.5rem; font-size: 0.8rem; color: #546e7a;'>
                    Potential loss at {confidence_level:.0%} confidence
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            if baseline_data is not None:
                impact = scenario_data['var_value'] - baseline_data['var_value']
                impact_pct = (scenario_data['var_value'] / baseline_data['var_value']) - 1
                impact_color = "risk-high" if impact_pct > 0.5 else "risk-medium" if impact_pct > 0.2 else "risk-low"
                
                st.markdown(f"""
                <div class='metric-card'>
                    <h3>Impact vs. Baseline</h3>
                    <p style='font-size: 1.8rem; font-weight: 600; margin-bottom: 0.25rem;'>{format_currency(impact)}</p>
                    <p style='margin: 0;'>
                        <span class='{impact_color}'>({format_percentage(impact_pct)})</span>
                        <span>{get_arrow_emoji(impact)} vs. Baseline</span>
                    </p>
                    <p style='margin-top: 0.5rem; font-size: 0.8rem; color: #546e7a;'>
                        Additional loss beyond normal conditions
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                scenario_type = scenario_data['scenario_type'] if 'scenario_type' in scenario_data else "Unknown"
                
                # Define the scenario descriptions dictionary outside the f-string
                scenario_descriptions = {
                    'Historical': 'Based on actual historical events', 
                    'Synthetic': 'Hypothetical stress scenario', 
                    'Macroeconomic': 'Based on economic variables'
                }
                
                # Get the description from the dictionary
                scenario_description = scenario_descriptions.get(scenario_type, '')
                
                st.markdown(f"""
                <div class='metric-card'>
                    <h3>Scenario Type</h3>
                    <p style='font-size: 1.8rem; font-weight: 600; margin-bottom: 0.25rem;'>{scenario_type}</p>
                    <p style='margin: 0;'>
                        <span>{selected_scenario}</span>
                    </p>
                    <p style='margin-top: 0.5rem; font-size: 0.8rem; color: #546e7a;'>
                        {scenario_description}
                    </p>
                </div>
                """, unsafe_allow_html=True)
                    
        with col3:
            # Calculate maximum expected loss
            max_loss = investment_value * var_pct if 'investment_value' in locals() else var_value
            
            # Estimate recovery time based on average returns
            if not filtered_returns.empty:
                avg_return = filtered_returns['Return'].mean()
                if avg_return > 0:
                    recovery_days = np.log(1 / (1 - var_pct)) / avg_return if var_pct < 1 else float('inf')
                    recovery_months = recovery_days / 21  # Approximate trading days per month
                else:
                    recovery_days = float('inf')
                    recovery_months = float('inf')
                
                recovery_text = (
                    f"{recovery_months:.1f} months" if recovery_months < float('inf') and recovery_months < 24 else
                    f"{recovery_months / 12:.1f} years" if recovery_months < float('inf') else
                    "Cannot estimate"
                )
            else:
                recovery_text = "N/A (insufficient data)"
            
            st.markdown(f"""
            <div class='metric-card'>
                <h3>Recovery Estimate</h3>
                <p style='font-size: 1.8rem; font-weight: 600; margin-bottom: 0.25rem;'>{recovery_text}</p>
                <p style='margin: 0;'>
                    <span>Based on historical returns</span>
                </p>
                <p style='margin-top: 0.5rem; font-size: 0.8rem; color: #546e7a;'>
                    Estimated time to recover from losses
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        # Add scenario details section
        with st.expander("Scenario Details", expanded=True):
            # Create two columns for details
            detail_col1, detail_col2 = st.columns(2)
            
            with detail_col1:
                # Create key parameters table
                st.markdown("#### Key Parameters", unsafe_allow_html=True)
                
                parameter_html = "<table class='data-table' style='margin-top: 0;'>"
                parameter_html += "<tr><th>Parameter</th><th>Value</th></tr>"
                
                # Add scenario-specific parameters if available
                key_params = {
                    'Confidence Level': f"{confidence_level:.0%}",
                    'Time Horizon': f"{time_horizon} days",
                    'VaR Amount': format_currency(scenario_data['var_value']),
                    'VaR % of Portfolio': format_percentage(scenario_data['var_pct']),
                    'Risk Level': scenario_data['risk_level'] if 'risk_level' in scenario_data else risk_level_text(scenario_data['var_pct']),
                }
                
                # Add additional parameters if they exist in the scenario data
                if 'shock_level' in scenario_data:
                    key_params['Shock Level'] = format_percentage(scenario_data['shock_level'])
                if 'volatility_multiplier' in scenario_data:
                    key_params['Volatility Multiplier'] = f"{scenario_data['volatility_multiplier']:.2f}x"
                if 'correlation_adjustment' in scenario_data:
                    key_params['Correlation Adjustment'] = format_percentage(scenario_data['correlation_adjustment'])
                
                for param, value in key_params.items():
                    parameter_html += f"<tr><td>{param}</td><td>{value}</td></tr>"
                
                parameter_html += "</table>"
                
                st.markdown(parameter_html, unsafe_allow_html=True)
            
            with detail_col2:
                # Create scenario description
                st.markdown("#### Scenario Description", unsafe_allow_html=True)
                
                # Generate a description based on the scenario type
                if 'scenario_type' in scenario_data:
                    scenario_type = scenario_data['scenario_type']
                    
                    if scenario_type == "Historical":
                        description = f"""
                        This scenario replicates the market conditions during the **{selected_scenario}** historical period.
                        It uses actual market returns from this period to estimate potential portfolio impact
                        if similar conditions were to occur again.
                        """
                    elif scenario_type == "Synthetic":
                        description = f"""
                        This is a hypothetical **{selected_scenario}** scenario designed to test portfolio
                        resilience under specific stress conditions. It models extreme market movements
                        that may not have historical precedent but are still plausible.
                        """
                    elif scenario_type == "Macroeconomic":
                        description = f"""
                        This scenario models the impact of macroeconomic changes related to **{selected_scenario}**.
                        It estimates how portfolio returns would respond to shifts in key economic
                        variables like interest rates, inflation, growth, and market volatility.
                        """
                    elif scenario_type == "Baseline":
                        description = f"""
                        This is the **{selected_scenario}** scenario representing normal market conditions.
                        It serves as a reference point for comparing the impact of stress scenarios
                        and calculating additional losses under stress.
                        """
                    else:
                        description = f"This is a {scenario_type} scenario modeling {selected_scenario} conditions."
                else:
                    description = f"This scenario models potential portfolio performance under {selected_scenario} conditions."
                
                st.markdown(description, unsafe_allow_html=True)
        
        # Scenario comparison chart
        st.markdown("<h2 class='sub-header'>Scenario Comparison</h2>", unsafe_allow_html=True)
        
        # Sort scenarios by VaR and exclude baseline from the chart
        sorted_scenarios = stress_test_results.copy()
        if 'Baseline' in stress_test_results['scenario'].values:
            baseline_var = stress_test_results.loc[stress_test_results['scenario'] == 'Baseline', 'var_value'].values[0]
            # Add impact column
            sorted_scenarios['impact'] = sorted_scenarios['var_value'] - baseline_var
            sorted_scenarios['impact_pct'] = (sorted_scenarios['var_value'] / baseline_var) - 1
            # Filter out baseline for the chart
            chart_scenarios = sorted_scenarios[sorted_scenarios['scenario'] != 'Baseline']
        else:
            chart_scenarios = sorted_scenarios
            chart_scenarios['impact'] = chart_scenarios['var_value']
            chart_scenarios['impact_pct'] = chart_scenarios['var_pct']
        
        # Sort by VaR value
        chart_scenarios = chart_scenarios.sort_values('var_value', ascending=False)
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Create enhanced comparison chart
            fig = go.Figure()
            
            # Add bars colored by risk level with improved aesthetics
            colors = {
                'Low': '#2e7d32',    # Green
                'Medium': '#f57c00',  # Amber
                'High': '#d32f2f',    # Red
                'Extreme': '#b71c1c'  # Dark red
            }
            
            # Track scenario types for legend groups
            scenario_type_order = []
            for scenario_type in chart_scenarios['scenario_type'].unique():
                if scenario_type not in scenario_type_order:
                    scenario_type_order.append(scenario_type)
            
            # Add bars grouped by scenario type with enhanced styling
            for scenario_type in scenario_type_order:
                type_data = chart_scenarios[chart_scenarios['scenario_type'] == scenario_type]
                
                for risk_level in type_data['risk_level'].unique():
                    risk_data = type_data[type_data['risk_level'] == risk_level]
                    
                    fig.add_trace(go.Bar(
                        x=risk_data['scenario'],
                        y=risk_data['var_value'],
                        name=f"{risk_level} Risk ({scenario_type})",
                        marker_color=colors.get(risk_level, '#78909c'),
                        legendgroup=scenario_type,
                        text=[f"${v:,.0f}" for v in risk_data['var_value']],
                        textposition='outside',
                        hovertemplate='%{x}: %{y:$,.2f}<br>Risk Level: ' + risk_level + '<extra></extra>'
                    ))
            
            # Add baseline reference line if available
            if 'Baseline' in stress_test_results['scenario'].values:
                fig.add_shape(
                    type='line',
                    x0=-0.5,
                    y0=baseline_var,
                    x1=len(chart_scenarios) - 0.5,
                    y1=baseline_var,
                    line=dict(color='#455a64', width=2, dash='dash'),
                )
                
                fig.add_annotation(
                    x=len(chart_scenarios) - 1,
                    y=baseline_var,
                    text=f"Baseline: ${baseline_var:,.0f}",
                    showarrow=False,
                    yshift=10,
                    font=dict(color='#455a64')
                )
            
            fig.update_layout(
                title="Stress Scenarios Ranked by Severity",
                xaxis_title="Scenario",
                yaxis_title="Value-at-Risk ($)",
                legend_title="Risk Category",
                height=500,
                margin=dict(l=20, r=20, t=50, b=100),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                    groupclick="toggleitem"
                ),
                plot_bgcolor='#ffffff',
                paper_bgcolor='#ffffff',
                font=dict(color='#455a64')
            )
            
            # Rotate x-axis labels for better readability
            fig.update_xaxes(tickangle=45)
            
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            # Add scenario impact ranking table
            if 'impact_pct' in chart_scenarios.columns:
                # Sort by impact percentage
                impact_ranking = chart_scenarios.sort_values('impact_pct', ascending=False)[['scenario', 'impact_pct', 'scenario_type']].head(5)
                
                st.markdown("#### Top Scenarios by Impact", unsafe_allow_html=True)
                
                impact_html = "<table class='data-table' style='margin-top: 0;'>"
                impact_html += "<tr><th>Scenario</th><th>Impact vs Baseline</th></tr>"
                
                for _, row in impact_ranking.iterrows():
                    # Color code based on impact severity
                    if row['impact_pct'] > 0.5:
                        style = "color: #d32f2f; font-weight: 600;"
                    elif row['impact_pct'] > 0.2:
                        style = "color: #f57c00; font-weight: 600;"
                    else:
                        style = "color: #455a64;"
                    
                    impact_html += f"""
                    <tr>
                        <td><small>{row['scenario']}</small></td>
                        <td style="{style}">{format_percentage(row['impact_pct'])}</td>
                    </tr>
                    """
                
                impact_html += "</table>"
                
                components.html(impact_html, height=180, scrolling=False)
            
            # Add impact distribution chart
            if 'impact_pct' in chart_scenarios.columns and 'scenario_type' in chart_scenarios.columns:
                # Calculate average impact by scenario type
                impact_by_type = chart_scenarios.groupby('scenario_type')['impact_pct'].mean().reset_index()
                impact_by_type = impact_by_type.sort_values('impact_pct', ascending=False)
                
                fig = px.bar(
                    impact_by_type,
                    x='impact_pct',
                    y='scenario_type',
                    orientation='h',
                    labels={'impact_pct': 'Average Impact', 'scenario_type': 'Scenario Type'},
                    title="Average Impact by Type",
                    color='impact_pct',
                    color_continuous_scale=['#2e7d32', '#f57c00', '#d32f2f'],
                    text_auto='.1%'
                )
                
                fig.update_layout(
                    xaxis_tickformat='.0%',
                    height=180,
                    margin=dict(l=10, r=10, t=40, b=10),
                    coloraxis_showscale=False,
                    plot_bgcolor='#ffffff',
                    paper_bgcolor='#ffffff'
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Risk matrix - enhanced with better visuals and interaction
        st.markdown("<h2 class='sub-header'>Risk Matrix</h2>", unsafe_allow_html=True)
        
        # Map scenario types to probability scores (1-5) - enhance with better descriptions
        scenario_prob = {
            'Historical': 4,  # Historical events have happened before
            'Synthetic': 2,   # Synthetic scenarios are hypothetical
            'Macroeconomic': 3,  # Macro scenarios based on economic factors
            'Baseline': 5      # Baseline is current state (highest probability)
        }
        
        # Add probability score to the dataframe
        stress_test_results['probability'] = stress_test_results['scenario_type'].map(
            lambda x: scenario_prob.get(x, 2.5)  # Default to middle value if type not found
        )
        
        # Add probability labels for better understanding
        prob_labels = {
            5: "Very High",
            4: "High",
            3: "Medium",
            2: "Low",
            1: "Very Low"
        }
        
        stress_test_results['probability_label'] = stress_test_results['probability'].map(prob_labels)
        
        # Create enhanced risk matrix with quadrant labels and better visuals
        fig = go.Figure()
        
        # Add quadrant backgrounds first
        # Low Impact, Low Probability (Bottom Left)
        fig.add_shape(
            type="rect",
            x0=0, y0=0,
            x1=30000, y1=3,
            fillcolor="rgba(46, 125, 50, 0.1)",
            line=dict(color="rgba(0,0,0,0)"),
            layer="below"
        )
        
        # Low Impact, High Probability (Top Left)
        fig.add_shape(
            type="rect",
            x0=0, y0=3,
            x1=30000, y1=6,
            fillcolor="rgba(245, 124, 0, 0.1)",
            line=dict(color="rgba(0,0,0,0)"),
            layer="below"
        )
        
        # High Impact, Low Probability (Bottom Right)
        fig.add_shape(
            type="rect",
            x0=30000, y0=0,
            x1=max(stress_test_results['var_value']) * 1.1, y1=3,
            fillcolor="rgba(245, 124, 0, 0.1)",
            line=dict(color="rgba(0,0,0,0)"),
            layer="below"
        )
        
        # High Impact, High Probability (Top Right) - Danger zone
        fig.add_shape(
            type="rect",
            x0=30000, y0=3,
            x1=max(stress_test_results['var_value']) * 1.1, y1=6,
            fillcolor="rgba(211, 47, 47, 0.1)",
            line=dict(color="rgba(0,0,0,0)"),
            layer="below"
        )
        
        # Add quadrant labels
        fig.add_annotation(
            x=15000, y=1.5,
            text="Low Risk Zone",
            showarrow=False,
            font=dict(color="#2e7d32", size=10)
        )
        
        fig.add_annotation(
            x=15000, y=4.5,
            text="Medium Risk Zone",
            showarrow=False,
            font=dict(color="#f57c00", size=10)
        )
        
        fig.add_annotation(
            x=max(stress_test_results['var_value']) * 0.75, y=1.5,
            text="Medium Risk Zone",
            showarrow=False,
            font=dict(color="#f57c00", size=10)
        )
        
        fig.add_annotation(
            x=max(stress_test_results['var_value']) * 0.75, y=4.5,
            text="High Risk Zone",
            showarrow=False,
            font=dict(color="#d32f2f", size=10)
        )
        
        # Add risk level color mapping
        color_map = {
            'Low': '#2e7d32',
            'Medium': '#f57c00',
            'High': '#d32f2f',
            'Extreme': '#b71c1c'
        }
        
        # Add scatter points for each scenario with enhanced styling
        for risk_level in stress_test_results['risk_level'].unique():
            risk_data = stress_test_results[stress_test_results['risk_level'] == risk_level]
            
            fig.add_trace(go.Scatter(
                x=risk_data['var_value'],
                y=risk_data['probability'],
                mode='markers+text',
                marker=dict(
                    size=risk_data['var_pct'] * 800,  # Size proportional to VaR percentage
                    color=color_map.get(risk_level, '#78909c'),
                    line=dict(width=1, color='white'),
                    opacity=0.8
                ),
                text=risk_data['scenario'],
                textposition="top center",
                name=f"{risk_level} Risk",
                hovertemplate='<b>%{text}</b><br>VaR: $%{x:,.0f}<br>Probability: %{customdata}<extra></extra>',
                customdata=risk_data['probability_label']
            ))
        
        fig.update_layout(
            title="Risk Matrix: Impact vs. Probability",
            xaxis_title="Impact (VaR in $)",
            yaxis_title="Relative Probability",
            xaxis=dict(
                tickmode='array',
                tickvals=[10000, 30000, 50000],
                ticktext=['$10k', '$30k', '$50k']
            ),
            yaxis=dict(
                tickmode='array',
                tickvals=[1, 2, 3, 4, 5],
                ticktext=['Very Low', 'Low', 'Medium', 'High', 'Very High']
            ),
            height=500,
            margin=dict(l=20, r=20, t=50, b=50),
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ),
            plot_bgcolor='#ffffff',
            paper_bgcolor='#ffffff',
            font=dict(color='#455a64')
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Risk level distribution - enhanced with better visuals
        st.markdown("<h2 class='sub-header'>Risk Level Distribution</h2>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Create enhanced pie chart of risk levels
            risk_counts = stress_test_results['risk_level'].value_counts().reset_index()
            risk_counts.columns = ['Risk Level', 'Count']
            
            # Ensure risk levels are properly ordered
            risk_order = ['Low', 'Medium', 'High', 'Extreme']
            risk_counts['Risk Level'] = pd.Categorical(
                risk_counts['Risk Level'],
                categories=risk_order,
                ordered=True
            )
            risk_counts = risk_counts.sort_values('Risk Level')
            
            fig = px.pie(
                risk_counts,
                values='Count',
                names='Risk Level',
                color='Risk Level',
                color_discrete_map=color_map,
                title="Distribution of Risk Levels",
                hole=0.4
            )
            
            fig.update_traces(
                textposition='inside',
                textinfo='percent+label',
                marker=dict(line=dict(color='white', width=2)),
                pull=[0.05 if x == scenario_data['risk_level'] else 0 for x in risk_counts['Risk Level']]
            )
            
            fig.update_layout(
                height=400,
                margin=dict(l=20, r=20, t=50, b=20),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.1,
                    xanchor="center",
                    x=0.5
                ),
                annotations=[dict(
                    text='Risk<br>Distribution',
                    x=0.5, y=0.5,
                    font_size=12,
                    showarrow=False
                )],
                plot_bgcolor='#ffffff',
                paper_bgcolor='#ffffff'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Create enhanced bar chart of scenario types with advanced analytics
            type_counts = stress_test_results['scenario_type'].value_counts().reset_index()
            type_counts.columns = ['Scenario Type', 'Count']
            
            # Calculate average VaR by scenario type
            type_var = stress_test_results.groupby('scenario_type')['var_value'].mean().reset_index()
            type_var.columns = ['Scenario Type', 'Average VaR']
            
            # Merge counts and VaR
            type_analysis = pd.merge(type_counts, type_var, on='Scenario Type')
            
            # Add VaR percentage
            type_analysis['Average VaR %'] = type_analysis['Average VaR'] / investment_value if 'investment_value' in locals() else type_analysis['Average VaR'] / 1000000
            
            # Add risk level counts by scenario type
            if 'risk_level' in stress_test_results.columns:
                risk_by_type = stress_test_results.groupby(['scenario_type', 'risk_level']).size().unstack(fill_value=0).reset_index()
                risk_by_type = risk_by_type.rename(columns={'scenario_type': 'Scenario Type'})
                type_analysis = pd.merge(type_analysis, risk_by_type, on='Scenario Type', how='left')
            
            # Sort by average VaR
            type_analysis = type_analysis.sort_values('Average VaR', ascending=False)
            
            fig = px.bar(
                type_analysis,
                x='Scenario Type',
                y='Average VaR',
                color='Scenario Type',
                text_auto='.2s',
                title="Average VaR by Scenario Type",
                custom_data=[type_analysis['Count'], type_analysis['Average VaR %']]
            )
            
            fig.update_traces(
                hovertemplate='<b>%{x}</b><br>Average VaR: $%{y:,.2f}<br>Count: %{customdata[0]}<br>VaR Percentage: %{customdata[1]:.2%}<extra></extra>'
            )
            
            fig.update_layout(
                height=400,
                margin=dict(l=20, r=20, t=50, b=20),
                xaxis_title="",
                yaxis_title="Average VaR ($)",
                plot_bgcolor='#ffffff',
                paper_bgcolor='#ffffff',
                font=dict(color='#455a64')
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add a concise summary table of risk statistics by scenario type
            if 'risk_level' in stress_test_results.columns:
                # Create summary by scenario type
                st.markdown("#### Risk Summary by Scenario Type", unsafe_allow_html=True)
                
                summary_html = "<table class='data-table' style='margin-top: 0; font-size: 0.85rem;'>"
                summary_html += "<tr><th>Scenario Type</th><th>Avg VaR</th><th>Risk Profile</th></tr>"
                
                for _, row in type_analysis.sort_values('Average VaR', ascending=False).iterrows():
                    # Calculate risk profile
                    risk_profile = ""
                    for level in risk_order:
                        if level in row and row[level] > 0:
                            color = color_map.get(level, '#78909c')
                            risk_profile += f'<span style="color: {color}; margin-right: 3px;">■</span>{row[level]} '
                    
                    summary_html += f"""
                    <tr>
                        <td>{row['Scenario Type']}</td>
                        <td>{format_currency(row['Average VaR'])}</td>
                        <td>{risk_profile}</td>
                    </tr>
                    """
                
                summary_html += "</table>"
                
                components.html(summary_html, height=150, scrolling=False)
    else:
        st.warning("No stress test data available. Please check your data source or run the stress testing module.")
        
        # Provide sample scenarios for demonstration if no data available
        st.markdown("""
        ### Sample Stress Scenarios
        
        Here are some common stress scenarios that would typically be included in a comprehensive risk management framework:
        
        1. **Historical Scenarios**
           - Global Financial Crisis (2008)
           - COVID-19 Market Crash (2020)
           - Dot-com Bubble Burst (2000-2002)
           - Black Monday (1987)
           
        2. **Synthetic Scenarios**
           - Severe Equity Market Crash (-40%)
           - Interest Rate Shock (+200 basis points)
           - Liquidity Crisis (widening spreads, correlation breakdown)
           - Combined Scenario (market crash + rate spike)
           
        3. **Macroeconomic Scenarios**
           - Stagflation (high inflation + low growth)
           - Deflationary Recession
           - Currency Crisis
           - Credit Crunch
        
        To enable this module, run the stress testing analysis notebook and generate the required output files.
        """, unsafe_allow_html=True)

# 4. Model Validation - Enhanced with better visuals and explanations
with tab4:
    st.markdown("<h1 class='main-header'>Model Validation & Backtesting</h1>", unsafe_allow_html=True)
    
    if not var_backtest_summary.empty:
        # Enhanced explanation with interactive elements
        with st.expander("Understanding VaR Model Validation", expanded=False):
            st.markdown("""
            ### Why Backtest VaR Models?
            
            **Backtesting** is a critical component of model risk management that evaluates the accuracy of VaR models by comparing their predictions against actual outcomes.
            
            ### Key Validation Tests:
            
            #### Kupiec Test (Unconditional Coverage)
            Tests whether the frequency of VaR breaches (losses exceeding VaR) matches the expected rate.
            - **Null Hypothesis**: The model's breach rate equals the expected rate.
            - **Pass**: The model correctly estimates the frequency of tail events.
            - **Fail**: The model systematically under or overestimates risk.
            
            #### Christoffersen Test (Independence)
            Tests whether VaR breaches occur independently over time or tend to cluster.
            - **Null Hypothesis**: VaR breaches occur independently over time.
            - **Pass**: Breaches are randomly distributed over time.
            - **Fail**: Breaches tend to cluster, indicating the model doesn't capture volatility dynamics.
            
            #### Combined Test
            Evaluates both the frequency and independence of breaches simultaneously.
            
            ### Key Metrics:
            
            - **Breach Rate**: The percentage of days when actual losses exceeded the VaR estimate.
            - **Expected Rate**: The theoretical breach rate (e.g., 5% for 95% VaR).
            - **Breach Ratio**: The ratio of actual to expected breaches (ideal value is 1.0).
            
            ### Regulatory Context:
            
            Financial regulators require VaR backtesting to validate internal risk models. Under the Basel framework, models are classified into green, yellow, and red zones based on their backtesting performance.
            """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <p class='info-text'>
            Backtesting is essential to validate VaR models by comparing predicted risk levels against actual outcomes.
            This analysis shows how different VaR methodologies performed historically, including breach rates and
            statistical tests.
        </p>
        """, unsafe_allow_html=True)
        
        # Enhanced model selector with visual indicators
        models = var_backtest_summary['Model'].unique()
        
        # Add accuracy scores to model options
        model_options = []
        for model in models:
            model_data = var_backtest_summary[var_backtest_summary['Model'] == model].iloc[0]
            accuracy = None
            
            # Calculate accuracy based on breach ratio if available
            if 'Breach Ratio' in model_data:
                accuracy = 100 - min(100, abs(model_data['Breach Ratio'] - 1) * 100)
                model_options.append(f"{model} ({accuracy:.0f}% Accuracy)")
            else:
                model_options.append(model)
        
        selected_model_option = st.selectbox(
            "Select Model to Analyze",
            options=model_options,
            index=0,
            help="Choose a VaR model to examine its backtesting performance"
        )
        
        # Extract model name from selection
        selected_model = selected_model_option.split(" (")[0] if "(" in selected_model_option else selected_model_option
        
        # Get model data
        model_data = var_backtest_summary[var_backtest_summary['Model'] == selected_model].iloc[0]
        
        # Display model metrics with enhanced visual indicators
        st.markdown("<h2 class='sub-header'>Model Performance Summary</h2>", unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            breach_rate = model_data['Breach Rate (%)'] if 'Breach Rate (%)' in model_data else model_data.get('Breach Rate', 0)
            expected_rate = model_data['Expected (%)'] if 'Expected (%)' in model_data else model_data.get('Expected Rate', 5)
            
            # Convert to float if they're strings
            breach_rate = float(breach_rate) if isinstance(breach_rate, str) and '%' in breach_rate else breach_rate
            expected_rate = float(expected_rate) if isinstance(expected_rate, str) and '%' in expected_rate else expected_rate
            
            # Calculate breach ratio and determine color
            breach_ratio = breach_rate / expected_rate if expected_rate != 0 else 0
            
            # Apply conditional styling based on ratio
            if 0.8 <= breach_ratio <= 1.2:
                color = "risk-low"
                assessment = "Good"
            elif 0.5 <= breach_ratio <= 1.5:
                color = "risk-medium"
                assessment = "Fair"
            else:
                color = "risk-high"
                assessment = "Poor"
            
            # Create enhanced metric card
            st.markdown(f"""
            <div class='metric-card'>
                <h3>Breach Rate</h3>
                <p style='font-size: 1.8rem; font-weight: 600; margin-bottom: 0.25rem;'>{breach_rate:.2f}%</p>
                <p style='margin: 0;'>
                    <span>Expected: {expected_rate:.2f}%</span>
                    <span class='{color}'>(Ratio: {breach_ratio:.2f})</span>
                </p>
                <p style='margin-top: 0.5rem; font-size: 0.8rem; color: #546e7a;'>
                    Percentage of days when losses exceeded VaR
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            kupiec_result = model_data.get('Kupiec Test', 'N/A')
            kupiec_color = 'risk-low' if kupiec_result == 'Pass' else 'risk-high' if kupiec_result == 'Fail' else ''
            
            # Add kupiec p-value if available
            kupiec_pvalue = model_data.get('Kupiec p-value', None)
            kupiec_pvalue_text = f"p-value: {kupiec_pvalue:.4f}" if kupiec_pvalue is not None else ""
            
            st.markdown(f"""
            <div class='metric-card'>
                <h3>Kupiec Test</h3>
                <p style='font-size: 1.8rem; font-weight: 600; margin-bottom: 0.25rem;' class='{kupiec_color}'>{kupiec_result}</p>
                <p style='margin: 0;'>
                    <span>{kupiec_pvalue_text}</span>
                </p>
                <p style='margin-top: 0.5rem; font-size: 0.8rem; color: #546e7a;'>
                    Tests correct frequency of breaches
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            christ_result = model_data.get('Christoffersen Test', 'N/A')
            christ_color = 'risk-low' if christ_result == 'Pass' else 'risk-high' if christ_result == 'Fail' else ''
            
            # Add christoffersen p-value if available
            christ_pvalue = model_data.get('Christoffersen p-value', None)
            christ_pvalue_text = f"p-value: {christ_pvalue:.4f}" if christ_pvalue is not None else ""
            
            st.markdown(f"""
            <div class='metric-card'>
                <h3>Christoffersen Test</h3>
                <p style='font-size: 1.8rem; font-weight: 600; margin-bottom: 0.25rem;' class='{christ_color}'>{christ_result}</p>
                <p style='margin: 0;'>
                    <span>{christ_pvalue_text}</span>
                </p>
                <p style='margin-top: 0.5rem; font-size: 0.8rem; color: #546e7a;'>
                    Tests independence of breaches
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            assessment = model_data.get('Assessment', 'N/A')
            assessment_color = 'risk-low' if assessment in ['Excellent', 'Good'] else 'risk-medium' if assessment == 'Fair' else 'risk-high'
            
            # Calculate accuracy score if breach ratio is available
            accuracy_score = None
            if 'Breach Ratio' in model_data:
                accuracy_score = 100 - min(100, abs(breach_ratio - 1) * 100)
            
            accuracy_text = f"{accuracy_score:.0f}% Accuracy" if accuracy_score is not None else ""
            
            st.markdown(f"""
            <div class='metric-card'>
                <h3>Overall Assessment</h3>
                <p style='font-size: 1.8rem; font-weight: 600; margin-bottom: 0.25rem;' class='{assessment_color}'>{assessment}</p>
                <p style='margin: 0;'>
                    <span>{accuracy_text}</span>
                </p>
                <p style='margin-top: 0.5rem; font-size: 0.8rem; color: #546e7a;'>
                    Combined evaluation of model performance
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        # Model accuracy visualization - enhanced with better gauge and explanations
        st.markdown("<h2 class='sub-header'>Model Accuracy</h2>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # Create enhanced accuracy gauge chart
            accuracy_score = min(100, max(0, 100 - abs(breach_ratio - 1) * 100)) if breach_ratio else 50
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=accuracy_score,
                title={'text': "Model Accuracy Score", 'font': {'size': 20}},
                delta={'reference': 80, 'increasing': {'color': '#2e7d32'}, 'decreasing': {'color': '#d32f2f'}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': '#455a64'},
                    'bar': {'color': "#283593"},
                    'steps': [
                        {'range': [0, 50], 'color': "#d32f2f"},
                        {'range': [50, 80], 'color': "#f57c00"},
                        {'range': [80, 100], 'color': "#2e7d32"}
                    ],
                    'threshold': {
                        'line': {'color': "#455a64", 'width': 4},
                        'thickness': 0.75,
                        'value': 80
                    }
                }
            ))
            
            fig.update_layout(
                height=300,
                margin=dict(l=20, r=20, t=30, b=20),
                font=dict(color='#455a64'),
                paper_bgcolor='#ffffff'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add interpretation based on accuracy score
            if accuracy_score >= 90:
                interpret_text = "**Excellent model performance**. The model accurately predicts risk at the specified confidence level."
                interpret_color = "#2e7d32"
            elif accuracy_score >= 80:
                interpret_text = "**Good model performance**. The model provides reliable risk estimates with minor deviations."
                interpret_color = "#4caf50"
            elif accuracy_score >= 60:
                interpret_text = "**Fair model performance**. The model provides acceptable risk estimates but has notable deviations."
                interpret_color = "#f57c00"
            else:
                interpret_text = "**Poor model performance**. The model significantly under or overestimates risk and should be recalibrated."
                interpret_color = "#d32f2f"
            
            st.markdown(f"""
            <div style="padding: 0.75rem; background-color: #f5f7fa; border-radius: 0.5rem; border-left: 3px solid {interpret_color}; margin-top: 1rem;">
                <p style="margin: 0; color: #455a64; font-size: 0.9rem;">
                    <span style="color: {interpret_color}; font-weight: 600;">Interpretation:</span> {interpret_text}
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Create enhanced breach statistics
            breaches = model_data.get('Breaches', 0)
            avg_var = model_data.get('Avg VaR (%)', 0)
            max_var = model_data.get('Max VaR (%)', 0)
            
            # Create an enhanced data card
            st.markdown("""
            <h4 style="margin-top: 0;">Breach Statistics</h4>
            """, unsafe_allow_html=True)
            
            stats_html = "<table class='data-table' style='margin-top: 0;'>"
            stats_html += "<tr><th>Metric</th><th>Value</th></tr>"
            
            # Add model statistics with enhanced formatting
            stats = [
                ('Total Observations', model_data.get('Observations', 'N/A')),
                ('Total Breaches', f"{breaches:.0f}"),
                ('Breach Rate', f"{breach_rate:.2f}%"),
                ('Expected Rate', f"{expected_rate:.2f}%"),
                ('Breach Ratio', f"{breach_ratio:.2f}"),
                ('Average VaR', f"{avg_var:.2f}%"),
                ('Maximum VaR', f"{max_var:.2f}%")
            ]
            
            for metric, value in stats:
                stats_html += f"<tr><td>{metric}</td><td>{value}</td></tr>"
            
            stats_html += "</table>"
            
            st.markdown(stats_html, unsafe_allow_html=True)
            
            # Add Basel zone classification if available
            if 'Basel Zone' in model_data:
                basel_zone = model_data['Basel Zone']
                zone_color = "#2e7d32" if basel_zone == "Green" else "#f57c00" if basel_zone == "Yellow" else "#d32f2f"
                
                st.markdown(f"""
                <div style="margin-top: 1rem; padding: 0.75rem; background-color: #f5f7fa; border-radius: 0.5rem; border-left: 3px solid {zone_color};">
                    <h4 style="margin-top: 0; margin-bottom: 0.5rem;">Basel Classification</h4>
                    <p style="margin: 0; font-size: 1.2rem; font-weight: 600; color: {zone_color};">{basel_zone} Zone</p>
                    <p style="margin: 0; font-size: 0.8rem; color: #546e7a;">
                        {'Model meets regulatory standards' if basel_zone == 'Green' else
                         'Model requires additional scrutiny' if basel_zone == 'Yellow' else
                         'Model fails regulatory standards and requires revision'}
                    </p>
                </div>
                """, unsafe_allow_html=True)
        
        # Statistical test details - enhanced with better explanations
        st.markdown("<h2 class='sub-header'>Statistical Test Details</h2>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Enhanced Kupiec test explanation with interpretable results
            st.markdown("""
            #### Kupiec Test (Unconditional Coverage)
            
            The Kupiec test checks if the observed breach rate matches the expected rate based on the VaR confidence level.
            """, unsafe_allow_html=True)
            
            # Create a visual indicator for the test result
            kupiec_result = model_data.get('Kupiec Test', 'N/A')
            kupiec_pvalue = model_data.get('Kupiec p-value', None)
            
            if kupiec_result != 'N/A':
                kupiec_color = "#2e7d32" if kupiec_result == "Pass" else "#d32f2f"
                
                st.markdown(f"""
                <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                    <div style="width: 1rem; height: 1rem; border-radius: 50%; background-color: {kupiec_color}; margin-right: 0.5rem;"></div>
                    <span style="font-weight: 600; color: {kupiec_color};">{kupiec_result}</span>
                    {f'(p-value: {kupiec_pvalue:.4f})' if kupiec_pvalue is not None else ''}
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("""
            **Null Hypothesis**: The model's breach rate equals the expected rate.
            
            **Interpretation**:
            - **Pass**: The model correctly estimates the frequency of breaches
            - **Fail**: The model systematically under or overestimates risk
            
            **Formula**:
            
            $$LR_{uc} = -2 \ln\left[\frac{(1-p)^{T-N} \cdot p^N}{(1-\frac{N}{T})^{T-N} \cdot (\frac{N}{T})^N}\right]$$
            
            Where:
            - T = total number of observations
            - N = number of breaches
            - p = expected breach rate (e.g., 0.05 for 95% VaR)
            """, unsafe_allow_html=True)
            
            # Add specific interpretation based on breach ratio
            if breach_ratio < 0.8:
                interpret_text = "The model is too conservative, overestimating risk."
            elif breach_ratio > 1.2:
                interpret_text = "The model is too aggressive, underestimating risk."
            else:
                interpret_text = "The model accurately estimates the frequency of breaches."
            
            st.markdown(f"""
            **Specific Finding**: {interpret_text}
            """, unsafe_allow_html=True)
        
        with col2:
            # Enhanced Christoffersen test explanation
            st.markdown("""
            #### Christoffersen Test (Independence)
            
            The Christoffersen test checks if VaR breaches are independent or if they cluster together.
            """, unsafe_allow_html=True)
            
            # Create a visual indicator for the test result
            christ_result = model_data.get('Christoffersen Test', 'N/A')
            christ_pvalue = model_data.get('Christoffersen p-value', None)
            
            if christ_result != 'N/A':
                christ_color = "#2e7d32" if christ_result == "Pass" else "#d32f2f"
                
                st.markdown(f"""
                <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                    <div style="width: 1rem; height: 1rem; border-radius: 50%; background-color: {christ_color}; margin-right: 0.5rem;"></div>
                    <span style="font-weight: 600; color: {christ_color};">{christ_result}</span>
                    {f'(p-value: {christ_pvalue:.4f})' if christ_pvalue is not None else ''}
                </div>
                """, unsafe_allow_html=True)
            
            # Extract transition probabilities if available
            p01 = model_data.get('p01', None)
            p11 = model_data.get('p11', None)
            
            if p01 is not None and p11 is not None:
                trans_text = f"""
                <div style="margin-bottom: 0.5rem;">
                    <strong>Transition Probabilities:</strong>
                    <ul style="margin-top: 0.25rem; margin-bottom: 0.25rem;">
                        <li>p01: {p01:.2%} (prob. of breach following non-breach)</li>
                        <li>p11: {p11:.2%} (prob. of breach following breach)</li>
                    </ul>
                </div>
                """
                # Display transition probabilities if available
                st.markdown(trans_text, unsafe_allow_html=True)
            
            # Display the static content without using an f-string
            st.markdown("""
            **Null Hypothesis**: VaR breaches occur independently over time.
            
            **Interpretation**:
            - **Pass**: Breaches are randomly distributed over time
            - **Fail**: Breaches tend to cluster, indicating the model doesn't capture volatility dynamics
            
            **Formula**:
            
            $$LR_{ind} = -2 \ln\left[\frac{(1-\pi)^{n_{00}+n_{10}} \cdot \pi^{n_{01}+n_{11}}}{(1-\pi_0)^{n_{00}} \cdot \pi_0^{n_{01}} \cdot (1-\pi_1)^{n_{10}} \cdot \pi_1^{n_{11}}}\right]$$
            
            Where:
            - n₀₀ = number of transitions from no breach to no breach
            - n₀₁ = number of transitions from no breach to breach
            - n₁₀ = number of transitions from breach to no breach
            - n₁₁ = number of transitions from breach to breach
            """, unsafe_allow_html=True)
            
            # Add specific interpretation based on p01 and p11 (if available)
            if p01 is not None and p11 is not None:
                if p11 > p01 * 1.5:
                    interpret_text = "Breaches tend to cluster (one breach is likely to be followed by another)."
                elif p11 < p01 * 0.5:
                    interpret_text = "Breaches tend to be followed by non-breaches (mean-reversion pattern)."
                else:
                    interpret_text = "Breaches appear to be independent over time."
                
                st.markdown(f"""
                **Specific Finding**: {interpret_text}
                """, unsafe_allow_html=True)
        
        # Model comparison - enhanced with better visualization
        st.markdown("<h2 class='sub-header'>Model Comparison</h2>", unsafe_allow_html=True)
        
        # Create enhanced comparison chart with better visuals
        fig = go.Figure()
        
        # Extract data
        model_names = var_backtest_summary['Model']
        breach_rates = var_backtest_summary['Breach Rate (%)'] if 'Breach Rate (%)' in var_backtest_summary.columns else var_backtest_summary.get('Breach Rate', [])
        expected_rates = var_backtest_summary['Expected (%)'] if 'Expected (%)' in var_backtest_summary.columns else var_backtest_summary.get('Expected Rate', [])
        
        # Ensure numeric values
        breach_rates = pd.to_numeric(breach_rates, errors='coerce')
        expected_rates = pd.to_numeric(expected_rates, errors='coerce')
        
        # Expected rate value (should be the same for all models)
        expected_rate = expected_rates.iloc[0] if len(expected_rates) > 0 else 5
        
        # Add bars for breach rates with conditional coloring
        bar_colors = []
        for model_idx, breach_rate in enumerate(breach_rates):
            ratio = breach_rate / expected_rate if expected_rate != 0 else 0
            
            if 0.8 <= ratio <= 1.2:
                bar_colors.append('#2e7d32')  # Good - green
            elif 0.5 <= ratio <= 1.5:
                bar_colors.append('#f57c00')  # Fair - amber
            else:
                bar_colors.append('#d32f2f')  # Poor - red
        
        # Add breach rate bars
        fig.add_trace(go.Bar(
            x=model_names,
            y=breach_rates,
            name='Actual Breach Rate',
            marker_color=bar_colors,
            text=[f"{x:.2f}%" for x in breach_rates],
            textposition='outside'
        ))
        
        # Add expected rate line
        fig.add_trace(go.Scatter(
            x=model_names,
            y=[expected_rate] * len(model_names),
            name='Expected Rate',
            mode='lines+markers',
            line=dict(color='#455a64', width=2, dash='dash'),
            marker=dict(symbol='diamond', size=8)
        ))
        
        # Add acceptable range (80%-120% of expected rate)
        fig.add_traces([
            go.Scatter(
                x=model_names,
                y=[expected_rate * 0.8] * len(model_names),
                name='Lower Bound (80%)',
                mode='lines',
                line=dict(color='#4caf50', width=1, dash='dot')
            ),
            go.Scatter(
                x=model_names,
                y=[expected_rate * 1.2] * len(model_names),
                name='Upper Bound (120%)',
                mode='lines',
                line=dict(color='#4caf50', width=1, dash='dot'),
                fill='tonexty',
                fillcolor='rgba(76, 175, 80, 0.1)'
            )
        ])
        
        # Highlight the selected model
        selected_idx = model_names[model_names == selected_model].index[0] if selected_model in model_names.values else -1
        
        if selected_idx >= 0:
            fig.add_trace(go.Scatter(
                x=[model_names.iloc[selected_idx]],
                y=[breach_rates.iloc[selected_idx]],
                mode='markers',
                marker=dict(
                    color='#fff',
                    size=12,
                    line=dict(
                        color='#283593',
                        width=3
                    )
                ),
                showlegend=False
            ))
        
        fig.update_layout(
            title="VaR Breach Rates by Model",
            xaxis_title="Model",
            yaxis_title="Breach Rate (%)",
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            height=400,
            margin=dict(l=20, r=20, t=50, b=20),
            plot_bgcolor='#ffffff',
            paper_bgcolor='#ffffff',
            font=dict(color='#455a64')
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Breach clustering analysis - enhanced visualization
        st.markdown("<h2 class='sub-header'>Breach Clustering Analysis</h2>", unsafe_allow_html=True)
        
        # Create breach clustering visualization if data is available
        breach_data = None
        if 'breach_data' in locals() and breach_data is not None:
            # Use existing breach data if available
            pass
        elif 'p01' in model_data and 'p11' in model_data:
            # Simulate breach data based on transition probabilities
            p01 = model_data['p01']
            p11 = model_data['p11']
            
            # Generate synthetic breach sequence for visualization
            n_days = 100
            breaches = [0]
            for i in range(1, n_days):
                if breaches[i-1] == 0:
                    # Previous day was not a breach
                    breaches.append(1 if np.random.random() < p01 else 0)
                else:
                    # Previous day was a breach
                    breaches.append(1 if np.random.random() < p11 else 0)
            
            breach_data = pd.DataFrame({
                'Day': list(range(n_days)),
                'Breach': breaches
            })
        
        if breach_data is not None:
            # Create breach heatmap (calendar-like view)
            # Reshape breaches into weeks
            weeks = len(breach_data) // 5 + (1 if len(breach_data) % 5 > 0 else 0)
            breach_matrix = np.zeros((weeks, 5))
            
            for i, breach in enumerate(breach_data['Breach']):
                week = i // 5
                day = i % 5
                if week < weeks and day < 5:
                    breach_matrix[week, day] = breach
            
            # Plot heatmap
            fig = px.imshow(
                breach_matrix,
                labels=dict(x="Day of Week", y="Week", color="Breach"),
                x=['Mon', 'Tue', 'Wed', 'Thu', 'Fri'],
                color_continuous_scale=['#f5f7fa', '#d32f2f'],
                title="VaR Breach Pattern Visualization"
            )
            
            fig.update_layout(
                height=400,
                margin=dict(l=20, r=20, t=50, b=20),
                coloraxis_showscale=False,
                plot_bgcolor='#ffffff',
                paper_bgcolor='#ffffff'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Create run length analysis
        col1, col2 = st.columns(2)
        
        with col1:
            # Transition matrix visualization
            if 'p01' in model_data and 'p11' in model_data:
                p01 = model_data['p01']
                p11 = model_data['p11']
                p00 = 1 - p01
                p10 = 1 - p11
                
                transition_matrix = np.array([
                    [p00, p01],
                    [p10, p11]
                ])
                
                # Create heatmap with annotations
                fig = px.imshow(
                    transition_matrix,
                    labels=dict(x="To State", y="From State", color="Probability"),
                    x=['No Breach (0)', 'Breach (1)'],
                    y=['No Breach (0)', 'Breach (1)'],
                    color_continuous_scale=['#e8f5e9', '#2e7d32'],
                    text_auto='.1%',
                    title="Transition Matrix"
                )
                
                fig.update_layout(
                    height=300,
                    margin=dict(l=20, r=20, t=50, b=20),
                    plot_bgcolor='#ffffff',
                    paper_bgcolor='#ffffff'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Add interpretation
                if p11 > p01 * 1.5:
                    st.markdown("""
                    <div style="padding: 0.75rem; background-color: #ffebee; border-radius: 0.5rem; border-left: 3px solid #d32f2f;">
                        <p style="margin: 0; color: #455a64; font-size: 0.9rem;">
                            <strong>Warning:</strong> The transition matrix shows significant breach clustering. 
                            This indicates the model does not adequately capture volatility persistence.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
        
        with col2:
            # Run length distribution
            # Create theoretical or sample-based run length distribution
            if 'p11' in model_data:
                p11 = model_data['p11']
                
                # Calculate theoretical run length distribution
                # P(run length = k) = p11^(k-1) * (1-p11)
                max_run = 10
                run_lengths = list(range(1, max_run + 1))
                probabilities = [(p11 ** (k-1)) * (1-p11) for k in run_lengths]
                
                # Normalize to ensure they sum to 1
                if sum(probabilities) > 0:
                    probabilities = [p / sum(probabilities) for p in probabilities]
                
                # Create run length distribution chart
                fig = px.bar(
                    x=run_lengths,
                    y=probabilities,
                    labels={'x': 'Run Length (Days)', 'y': 'Probability'},
                    title="Breach Run Length Distribution"
                )
                
                fig.update_layout(
                    height=300,
                    margin=dict(l=20, r=20, t=50, b=20),
                    plot_bgcolor='#ffffff',
                    paper_bgcolor='#ffffff'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Calculate expected run length
                if p11 < 1:
                    expected_run = 1 / (1 - p11)
                    st.markdown(f"""
                    <div style="padding: 0.75rem; background-color: #e8f0fe; border-radius: 0.5rem; border-left: 3px solid #4285f4;">
                        <p style="margin: 0; color: #455a64; font-size: 0.9rem;">
                            <strong>Expected Run Length:</strong> {expected_run:.2f} days
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Assessment summary - enhanced with recommendations
        st.markdown("<h2 class='sub-header'>Model Assessment Summary</h2>", unsafe_allow_html=True)
        
        # Create a visual assessment summary
        assessment = model_data.get('Assessment', 'Unknown')
        assessment_color = '#2e7d32' if assessment in ['Excellent', 'Good'] else '#f57c00' if assessment == 'Fair' else '#d32f2f'
        
        # Create detailed recommendations based on test results
        recommendations = []
        
        if 'Breach Ratio' in model_data:
            breach_ratio = model_data['Breach Ratio']
            if breach_ratio < 0.8:
                recommendations.append("Consider using a lower confidence level to reduce excess conservatism.")
            elif breach_ratio > 1.2:
                recommendations.append("Increase the confidence level or add a safety buffer to VaR estimates.")
        
        if 'Kupiec Test' in model_data and model_data['Kupiec Test'] == 'Fail':
            recommendations.append("Recalibrate the model to achieve the expected breach frequency.")
        
        if 'Christoffersen Test' in model_data and model_data['Christoffersen Test'] == 'Fail':
            recommendations.append("Implement a GARCH model or other time-varying volatility approach to address breach clustering.")
        
        if len(recommendations) == 0:
            recommendations.append("The model is performing well. Continue regular monitoring and validation.")
        
        assessment_html = f"""
        <div style="padding: 1.25rem; background-color: #f5f7fa; border-radius: 0.75rem; border-left: 4px solid {assessment_color};">
            <h3 style="margin-top: 0; color: {assessment_color};">{assessment} Model Performance</h3>
            
            <h4 style="margin-top: 1rem; margin-bottom: 0.5rem;">Key Findings:</h4>
            <ul style="margin-top: 0.5rem;">
                <li>The model {'correctly estimates' if 0.8 <= breach_ratio <= 1.2 else 'overestimates' if breach_ratio < 0.8 else 'underestimates'} risk with a breach ratio of {breach_ratio:.2f}.</li>
                <li>Statistical tests show the model {'has adequate unconditional coverage' if model_data.get('Kupiec Test', '') == 'Pass' else 'fails the coverage test'}.</li>
                <li>Breaches {'appear independent over time' if model_data.get('Christoffersen Test', '') == 'Pass' else 'show clustering patterns'}.</li>
            </ul>
            
            <h4 style="margin-top: 1rem; margin-bottom: 0.5rem;">Recommendations:</h4>
            <ul style="margin-top: 0.5rem;">
                {''.join([f'<li>{rec}</li>' for rec in recommendations])}
            </ul>
        </div>
        """
        components.html(assessment_html, height=300, scrolling=False)
    else:
        st.warning("No backtesting data available. Please run the backtesting module to validate your VaR models.")
        
        # Provide explanation of what backtesting should show
        st.markdown("""
        ### Model Validation through Backtesting
        
        Backtesting is a critical component of the risk management process. When you run the backtesting module, you'll see:
        
        1. **Breach Analysis**: Comparison of expected vs. actual VaR breaches
        2. **Statistical Tests**: Kupiec and Christoffersen tests to validate model accuracy
        3. **Model Comparison**: Performance metrics across different VaR methodologies
        4. **Regime Analysis**: How models perform in different market conditions
        
        To enable this module, run the backtesting notebook and generate the required output files.
        """, unsafe_allow_html=True)

# 5. Return Analysis - Enhanced with better visualizations and insights
with tab5:
    st.markdown("<h1 class='main-header'>Portfolio Return Analysis</h1>", unsafe_allow_html=True)
    
    if not filtered_returns.empty:
        # Enhanced explanation with interactive elements
        with st.expander("Understanding Return Analysis", expanded=False):
            st.markdown("""
            ### Return Distribution Analysis
            
            Understanding the statistical properties of returns is fundamental to risk management:
            
            - **Normal Distribution**: Financial theory often assumes returns follow a normal distribution
            - **Empirical Distribution**: Actual returns typically exhibit:
              - **Fat Tails**: More extreme events than predicted by normal distribution
              - **Skewness**: Asymmetry in the distribution (negative skew common in equity markets)
              - **Excess Kurtosis**: Measure of "peakedness" and tail extremity
            
            ### Key Return Metrics:
            
            - **Mean Return**: Average daily return
            - **Volatility**: Standard deviation of returns, a measure of risk
            - **Sharpe Ratio**: Return per unit of risk (higher is better)
            - **Maximum Drawdown**: Largest peak-to-trough decline
            - **Skewness**: Measure of asymmetry (-ve values indicate longer left tail)
            - **Kurtosis**: Measure of "fatness" of tails (>3 indicates fat tails)
            
            ### Why This Matters for Risk Management:
            
            - Non-normal return distributions affect VaR accuracy
            - Fat tails mean extreme events occur more frequently than expected
            - Understanding these properties helps select appropriate risk models
            """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <p class='info-text'>
            This analysis examines the statistical properties of portfolio returns, including distribution,
            volatility patterns, and extreme events. Understanding return characteristics is fundamental to
            effective risk management.
        </p>
        """, unsafe_allow_html=True)
        
        # Return distribution - enhanced visualization and analysis
        st.markdown("<h2 class='sub-header'>Return Distribution Analysis</h2>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Return histogram with enhanced normal overlay
            fig = go.Figure()
            
            # Calculate return statistics
            returns = filtered_returns['Return'].dropna()
            mean = returns.mean()
            std = returns.std()
            
            # Create histogram bins
            hist, bins = np.histogram(returns, bins=30, density=True)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            
            # Add histogram of returns
            fig.add_trace(go.Bar(
                x=bin_centers,
                y=hist,
                name='Returns',
                marker_color='rgba(40, 53, 147, 0.6)',
                hovertemplate='Return: %{x:.2%}<br>Density: %{y:.4f}<extra></extra>'
            ))
            
            # Add normal distribution overlay
            x = np.linspace(min(returns), max(returns), 100)
            y = scipy_stats.norm.pdf(x, mean, std)
            
            fig.add_trace(go.Scatter(
                x=x,
                y=y,
                mode='lines',
                name='Normal Distribution',
                line=dict(color='#00838f', width=2, dash='dash'),
                hovertemplate='Return: %{x:.2%}<br>Density: %{y:.4f}<extra></extra>'
            ))
            
            # Add vertical line at mean
            fig.add_shape(
                type="line",
                x0=mean, y0=0,
                x1=mean, y1=max(hist) * 1.1,
                line=dict(color="#455a64", width=1.5, dash="dot"),
            )
            
            # Add VaR line if available
            if not filtered_var_results.empty:
                var_value = -filtered_var_results['historical_pct'].values[0]
                
                fig.add_shape(
                    type="line",
                    x0=var_value, y0=0,
                    x1=var_value, y1=max(hist) * 1.1,
                    line=dict(color="#d32f2f", width=2, dash="dash"),
                )
                
                fig.add_annotation(
                    x=var_value,
                    y=max(hist) * 0.9,
                    text=f"{confidence_level:.0%} VaR",
                    showarrow=True,
                    arrowhead=1,
                    ax=-40,
                    ay=0,
                    font=dict(color="#d32f2f")
                )
            
            fig.update_layout(
                title='Return Distribution vs. Normal Distribution',
                xaxis_title='Return',
                yaxis_title='Density',
                xaxis=dict(tickformat='.1%'),
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                height=400,
                margin=dict(l=20, r=20, t=50, b=20),
                plot_bgcolor='#ffffff',
                paper_bgcolor='#ffffff',
                font=dict(color='#455a64')
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add distribution statistics for context
            skewness = scipy_stats.skew(returns)
            kurtosis = scipy_stats.kurtosis(returns)
            
            # Format skewness interpretation
            if abs(skewness) < 0.1:
                skew_interp = "symmetric (close to normal)"
                skew_color = "#2e7d32"
            elif skewness < -0.5:
                skew_interp = "significant negative skew (longer left tail, more extreme losses)"
                skew_color = "#d32f2f"
            elif skewness < 0:
                skew_interp = "mild negative skew (slightly longer left tail)"
                skew_color = "#f57c00"
            elif skewness < 0.5:
                skew_interp = "mild positive skew (slightly longer right tail)"
                skew_color = "#2e7d32"
            else:
                skew_interp = "significant positive skew (longer right tail, more extreme gains)"
                skew_color = "#2e7d32"
            
            # Format kurtosis interpretation
            if abs(kurtosis) < 0.5:
                kurt_interp = "similar to normal distribution"
                kurt_color = "#2e7d32"
            elif kurtosis < 0:
                kurt_interp = "lighter tails than normal (fewer extreme values)"
                kurt_color = "#2e7d32"
            elif kurtosis < 1:
                kurt_interp = "slightly heavier tails than normal"
                kurt_color = "#f57c00"
            else:
                kurt_interp = "significantly heavier tails than normal (more frequent extreme values)"
                kurt_color = "#d32f2f"
            
            st.markdown(f"""
            <div style="padding: 0.75rem; background-color: #f5f7fa; border-radius: 0.5rem; margin-top: 1rem;">
                <h4 style="margin-top: 0; margin-bottom: 0.5rem;">Distribution Statistics</h4>
                <p style="margin: 0; font-size: 0.9rem; color: #455a64;">
                    <strong>Skewness:</strong> {skewness:.4f} - <span style="color: {skew_color};">{skew_interp}</span><br>
                    <strong>Excess Kurtosis:</strong> {kurtosis:.4f} - <span style="color: {kurt_color};">{kurt_interp}</span>
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Enhanced Q-Q plot to check normality
            fig = go.Figure()
            
            # Calculate quantiles
            returns_sorted = sorted(returns)
            n = len(returns_sorted)
            theoretical_quantiles = [scipy_stats.norm.ppf((i + 0.5) / n) for i in range(n)]
            
            # Create scatter plot with improved styling
            fig.add_trace(go.Scatter(
                x=theoretical_quantiles,
                y=returns_sorted,
                mode='markers',
                name='Return Quantiles',
                marker=dict(
                    color='rgba(40, 53, 147, 0.6)',
                    size=6,
                    line=dict(
                        color='rgba(40, 53, 147, 1.0)',
                        width=1
                    )
                ),
                hovertemplate='Theoretical: %{x:.2f}<br>Actual: %{y:.2%}<extra></extra>'
            ))
            
            # Add the diagonal line (y=x) with better positioning
            min_val = min(theoretical_quantiles)
            max_val = max(theoretical_quantiles)
            min_returns = min(returns_sorted)
            max_returns = max(returns_sorted)
            
            # Calculate extended line to cover all points
            slope = (max_returns - min_returns) / (max_val - min_val)
            intercept = min_returns - slope * min_val
            
            line_x = [min_val, max_val]
            line_y = [min_val * slope + intercept, max_val * slope + intercept]
            
            fig.add_trace(go.Scatter(
                x=line_x,
                y=line_y,
                mode='lines',
                name='Normal Line',
                line=dict(color='#00838f', width=2, dash='dash')
            ))
            
            # Add annotations for tail deviations
            # Left tail
            left_idx = int(0.01 * n)
            if left_idx < n:
                left_actual = returns_sorted[left_idx]
                left_theo = theoretical_quantiles[left_idx]
                left_expected = left_theo * slope + intercept
                deviation = left_actual - left_expected
                
                if abs(deviation) > 0.005:  # Only annotate if deviation is significant
                    fig.add_annotation(
                        x=left_theo,
                        y=left_actual,
                        text="Left Tail<br>Deviation",
                        showarrow=True,
                        arrowhead=1,
                        ax=20,
                        ay=20,
                        font=dict(size=10, color='#d32f2f')
                    )
            
            # Right tail
            right_idx = int(0.99 * n)
            if right_idx < n:
                right_actual = returns_sorted[right_idx]
                right_theo = theoretical_quantiles[right_idx]
                right_expected = right_theo * slope + intercept
                deviation = right_actual - right_expected
                
                if abs(deviation) > 0.005:  # Only annotate if deviation is significant
                    fig.add_annotation(
                        x=right_theo,
                        y=right_actual,
                        text="Right Tail<br>Deviation",
                        showarrow=True,
                        arrowhead=1,
                        ax=-20,
                        ay=-20,
                        font=dict(size=10, color='#d32f2f')
                    )
            
            fig.update_layout(
                title='Q-Q Plot (Testing for Normality)',
                xaxis_title='Theoretical Quantiles',
                yaxis_title='Empirical Quantiles',
                yaxis=dict(tickformat='.1%'),
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                height=400,
                margin=dict(l=20, r=20, t=50, b=20),
                plot_bgcolor='#ffffff',
                paper_bgcolor='#ffffff',
                font=dict(color='#455a64')
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add normality test results
            jarque_bera = scipy_stats.jarque_bera(returns)
            shapiro = scipy_stats.shapiro(returns.sample(min(5000, len(returns))) if len(returns) > 5000 else returns)
            
            st.markdown(f"""
            <div style="padding: 0.75rem; background-color: #f5f7fa; border-radius: 0.5rem; margin-top: 1rem;">
                <h4 style="margin-top: 0; margin-bottom: 0.5rem;">Normality Tests</h4>
                <p style="margin: 0; font-size: 0.9rem; color: #455a64;">
                    <strong>Jarque-Bera Test:</strong> statistic={jarque_bera[0]:.4f}, p-value={jarque_bera[1]:.6f}<br>
                    <strong>Shapiro-Wilk Test:</strong> statistic={shapiro[0]:.4f}, p-value={shapiro[1]:.6f}<br>
                    <strong>Conclusion:</strong> <span style="color: #d32f2f; font-weight: 600;">{'Returns are not normally distributed' if jarque_bera[1] < 0.05 else 'Returns appear normally distributed'}</span>
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        # Add normality analysis conclusion with risk management implications
        normal_dist = jarque_bera[1] >= 0.05
        
        implications_html = f"""
        <div style="padding: 1rem; background-color: {'#e8f5e9' if normal_dist else '#ffebee'}; border-radius: 0.5rem; margin-top: 1rem;">
            <h4 style="margin-top: 0; margin-bottom: 0.5rem; color: {'#2e7d32' if normal_dist else '#d32f2f'};">
                Return Distribution Assessment
            </h4>
            <p style="margin: 0; color: #455a64; font-size: 0.9rem;">
                {'The portfolio returns appear to follow a normal distribution, which supports the use of parametric VaR models.' if normal_dist else 
                'The portfolio returns <strong>do not</strong> follow a normal distribution, suggesting parametric VaR models may underestimate tail risk.'}
            </p>
            
            <h4 style="margin-top: 1rem; margin-bottom: 0.5rem;">Risk Management Implications:</h4>
            <ul style="margin-top: 0; margin-bottom: 0; color: #455a64; font-size: 0.9rem;">
                {'<li>Parametric VaR is likely to provide accurate risk estimates</li>' +
                '<li>Simple VaR models based on normal distribution are appropriate</li>' +
                '<li>Expected Shortfall (ES) will be proportional to VaR</li>' if normal_dist else
                '<li>Historical or Monte Carlo VaR methods are more appropriate than parametric approaches</li>' +
                '<li>Consider using Expected Shortfall (ES) to better capture tail risk</li>' +
                '<li>Stress testing becomes particularly important to understand tail events</li>'}
            </ul>
        </div>
        """
        
        components.html(implications_html, height=250, scrolling=False)
        
        # Return time series and volatility - enhanced visualization
        st.markdown("<h2 class='sub-header'>Return Time Series & Volatility</h2>", unsafe_allow_html=True)
        
        # Calculate rolling metrics
        window = min(21, len(filtered_returns) // 4)  # Use smaller window if data is limited
        if window > 5:  # Only calculate if we have enough data
            rolling_vol = filtered_returns['Return'].rolling(window=window).std() * np.sqrt(252)  # Annualized
            rolling_vol_series = pd.Series(rolling_vol, index=filtered_returns.index)
            
            # Create enhanced figure with annotations and styling
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Add return trace with better styling
            fig.add_trace(
                go.Scatter(
                    x=filtered_returns.index,
                    y=filtered_returns['Return'],
                    mode='lines',
                    name='Daily Return',
                    line=dict(color='rgba(40, 53, 147, 0.7)', width=1),
                    hovertemplate='%{x|%Y-%m-%d}: %{y:.2%}<extra></extra>'
                ),
                secondary_y=False
            )
            
            # Add volatility trace with enhanced styling
            fig.add_trace(
                go.Scatter(
                    x=rolling_vol_series.index,
                    y=rolling_vol_series,
                    mode='lines',
                    name=f'{window}-Day Rolling Volatility (Ann.)',
                    line=dict(color='#d32f2f', width=2),
                    hovertemplate='%{x|%Y-%m-%d}: %{y:.2%}<extra></extra>'
                ),
                secondary_y=True
            )
            
            # Add VaR thresholds if available
            if not filtered_var_results.empty:
                var_value = -filtered_var_results['historical_pct'].values[0]
                
                fig.add_trace(
                    go.Scatter(
                        x=[filtered_returns.index.min(), filtered_returns.index.max()],
                        y=[var_value, var_value],
                        mode='lines',
                        name=f'{confidence_level:.0%} VaR',
                        line=dict(color='#4caf50', width=2, dash='dash'),
                        hovertemplate='%{y:.2%}<extra>VaR</extra>'
                    ),
                    secondary_y=False
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=[filtered_returns.index.min(), filtered_returns.index.max()],
                        y=[-var_value, -var_value],
                        mode='lines',
                        name=f'Positive Threshold',
                        line=dict(color='#4caf50', width=2, dash='dash'),
                        hovertemplate='%{y:.2%}<extra>Positive Threshold</extra>'
                    ),
                    secondary_y=False
                )
            
            # Identify high volatility periods
            if len(rolling_vol_series.dropna()) > 0:
                high_vol_threshold = rolling_vol_series.quantile(0.75)
                high_vol_periods = rolling_vol_series[rolling_vol_series > high_vol_threshold]
                
                # Add high volatility period highlighting
                for i, (date, _) in enumerate(high_vol_periods.items()):
                    # Skip highlighting isolated points
                    if i > 0 and i < len(high_vol_periods) - 1:
                        prev_date = high_vol_periods.index[i-1]
                        next_date = high_vol_periods.index[i+1]
                        
                        # Only highlight if consecutive dates
                        if (date - prev_date).days <= window and (next_date - date).days <= window:
                            fig.add_vrect(
                                x0=date, x1=date,
                                fillcolor="#d32f2f",
                                opacity=0.1,
                                layer="below", line_width=0,
                            )
            
            # Update axes labels with better styling
            fig.update_xaxes(title_text="Date", gridcolor='#e0e0e0')
            fig.update_yaxes(
                title_text="Daily Return",
                tickformat='.1%',
                secondary_y=False,
                gridcolor='#e0e0e0',
                zerolinecolor='#9e9e9e'
            )
            fig.update_yaxes(
                title_text="Annualized Volatility",
                tickformat='.1%',
                secondary_y=True,
                gridcolor='#e0e0e0'
            )
            
            fig.update_layout(
                title="Daily Returns and Rolling Volatility",
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                height=500,
                margin=dict(l=20, r=20, t=50, b=20),
                plot_bgcolor='#ffffff',
                paper_bgcolor='#ffffff',
                font=dict(color='#455a64'),
                hovermode='x unified'
            )
            
            # Add range selector for better interactivity
            fig.update_xaxes(
                rangeslider_visible=False,
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(count=1, label="YTD", step="year", stepmode="todate"),
                        dict(count=1, label="1y", step="year", stepmode="backward"),
                        dict(step="all")
                    ])
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            # Simple plot for limited data
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=filtered_returns.index,
                y=filtered_returns['Return'],
                mode='lines',
                name='Daily Return',
                line=dict(color='#283593')
            ))
            
            fig.update_layout(
                title="Daily Returns (Limited Data)",
                xaxis_title="Date",
                yaxis_title="Return",
                yaxis=dict(tickformat='.1%'),
                height=400,
                plot_bgcolor='#ffffff',
                paper_bgcolor='#ffffff'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.info("Limited data available. More observations needed for rolling volatility analysis.")
        
        # Extreme value analysis - enhanced with better visuals and insights
        st.markdown("<h2 class='sub-header'>Extreme Value Analysis</h2>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Identify extreme events (beyond 3 standard deviations)
            mean = filtered_returns['Return'].mean()
            std = filtered_returns['Return'].std()
            
            extreme_threshold = 3 * std
            extremes = filtered_returns[abs(filtered_returns['Return'] - mean) > extreme_threshold].copy()
            
            # Categorize as positive or negative
            extremes['Type'] = np.where(extremes['Return'] > mean, 'Positive', 'Negative')
            
            # Display extreme events with enhanced styling
            if not extremes.empty:
                extremes['Normalized'] = (extremes['Return'] - mean) / std
                extremes_display = extremes.reset_index()
                
                # Create a more visually appealing table
                extremes_html = "<table class='data-table'>"
                extremes_html += "<tr><th>Date</th><th>Return</th><th>Std. Dev.</th><th>Type</th></tr>"
                
                for _, row in extremes_display.iterrows():
                    # Color by extreme type
                    type_color = "#2e7d32" if row['Type'] == 'Positive' else "#d32f2f"
                    
                    extremes_html += f"""
                    <tr>
                        <td>{row['Date'].strftime('%Y-%m-%d')}</td>
                        <td style="color: {type_color}; font-weight: 600;">{row['Return']:.2%}</td>
                        <td>{row['Normalized']:.2f}σ</td>
                        <td style="color: {type_color};">{row['Type']}</td>
                    </tr>
                    """
                
                extremes_html += "</table>"
                
                st.markdown(f"**Extreme Events (Beyond ±3σ): {len(extremes)} events**", unsafe_allow_html=True)
                
                components.html(extremes_html, height=150, scrolling=False)
                
                # Add contextual interpretation
                if len(extremes) > 10:
                    st.markdown(f"""
                    <div style="padding: 0.75rem; background-color: #ffebee; border-radius: 0.5rem; margin-top: 1rem; border-left: 3px solid #d32f2f;">
                        <p style="margin: 0; color: #455a64; font-size: 0.9rem;">
                            <strong>Finding:</strong> The portfolio shows {len(extremes)} extreme events, more than expected for a normal distribution.
                            This confirms the fat-tailed nature of returns and suggests using tail risk-sensitive measures.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No extreme events (beyond ±3σ) found in the selected date range.")
        
        with col2:
            # Create enhanced visualization of extreme events
            if not extremes.empty:
                # Distribution by type
                type_counts = extremes['Type'].value_counts().reset_index()
                type_counts.columns = ['Type', 'Count']
                
                # Add colors
                type_counts['Color'] = type_counts['Type'].map({
                    'Positive': '#2e7d32',  # Green
                    'Negative': '#d32f2f'   # Red
                })
                
                # Create a more informative pie chart
                fig = px.pie(
                    type_counts,
                    values='Count',
                    names='Type',
                    color='Type',
                    color_discrete_map={
                        'Positive': '#2e7d32',
                        'Negative': '#d32f2f'
                    },
                    title="Distribution of Extreme Events"
                )
                
                fig.update_traces(
                    textposition='inside',
                    textinfo='percent+value',
                    marker=dict(line=dict(color='#ffffff', width=2))
                )
                
                fig.update_layout(
                    height=300,
                    margin=dict(l=20, r=20, t=50, b=20),
                    plot_bgcolor='#ffffff',
                    paper_bgcolor='#ffffff'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Extreme events statistics with enhanced visualization
                negative = extremes[extremes['Type'] == 'Negative']['Return']
                positive = extremes[extremes['Type'] == 'Positive']['Return']
                
                # Create comparison bar chart
                comparison_data = []
                
                if len(negative) > 0:
                    comparison_data.append({
                        'Category': 'Negative Extremes',
                        'Average': negative.mean(),
                        'Count': len(negative),
                        'Min': negative.min(),
                        'Max': negative.max()
                    })
                
                if len(positive) > 0:
                    comparison_data.append({
                        'Category': 'Positive Extremes',
                        'Average': positive.mean(),
                        'Count': len(positive),
                        'Min': positive.min(),
                        'Max': positive.max()
                    })
                
                if comparison_data:
                    comparison_df = pd.DataFrame(comparison_data)
                    
                    fig = go.Figure()
                    
                    # Add average bars
                    fig.add_trace(go.Bar(
                        x=comparison_df['Category'],
                        y=comparison_df['Average'],
                        text=[f"{x:.2%}" for x in comparison_df['Average']],
                        textposition='outside',
                        marker_color=['#d32f2f', '#2e7d32'],
                        name='Average'
                    ))
                    
                    # Add min/max error bars
                    fig.add_trace(go.Scatter(
                        x=comparison_df['Category'],
                        y=comparison_df['Average'],
                        error_y=dict(
                            type='data',
                            symmetric=False,
                            array=comparison_df['Max'] - comparison_df['Average'],
                            arrayminus=comparison_df['Average'] - comparison_df['Min']
                        ),
                        mode='markers',
                        marker=dict(
                            color=['#d32f2f', '#2e7d32'],
                            size=10,
                            symbol='diamond'
                        ),
                        name='Min/Max Range'
                    ))
                    
                    fig.update_layout(
                        title="Extreme Event Statistics",
                        yaxis=dict(tickformat='.1%'),
                        height=250,
                        margin=dict(l=20, r=20, t=50, b=20),
                        plot_bgcolor='#ffffff',
                        paper_bgcolor='#ffffff',
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No extreme events to analyze in the selected period.")
                
                # Add educational content if no extremes
                st.markdown("""
                ### About Extreme Events
                
                In financial markets, extreme events (events beyond 3 standard deviations) should be rare in a normal distribution:
                
                - Expected frequency: ~0.27% of observations
                - For daily data: ~0.7 days per year
                
                When extremes are more frequent, this indicates **fat tails** in the return distribution, which has important risk management implications:
                
                - Models assuming normality can underestimate risk
                - Extreme Value Theory (EVT) can help model tail behavior
                - Stress testing becomes critical for preparedness
                """, unsafe_allow_html=True)
        
        # Return metrics table - enhanced with better visuals and comparisons
        st.markdown("<h2 class='sub-header'>Return Statistics Summary</h2>", unsafe_allow_html=True)
        
        # Calculate enhanced return metrics
        cum_return = (1 + filtered_returns['Return']).prod() - 1
        ann_return = (1 + filtered_returns['Return'].mean()) ** 252 - 1
        ann_vol = filtered_returns['Return'].std() * np.sqrt(252)
        sharpe = ann_return / ann_vol if ann_vol != 0 else 0
        
        # Calculate maximum drawdown
        cum_returns = (1 + filtered_returns['Return']).cumprod()
        running_max = cum_returns.cummax()
        drawdown = (cum_returns / running_max) - 1
        if isinstance(drawdown, pd.DataFrame):
            max_drawdown = drawdown.min().min()  # Get the global minimum across all columns
        else:
            max_drawdown = drawdown.min() if not drawdown.empty else 0     
              
        # Calculate win rate
        positive_days = (filtered_returns['Return'] > 0).sum()
        win_rate = positive_days / len(filtered_returns)
        
        # Calculate other risk metrics
        sortino_ratio = None
        if ann_return > 0:
            # Calculate downside deviation (only negative returns)
            neg_returns = filtered_returns[filtered_returns['Return'] < 0]['Return']
            if len(neg_returns) > 0:
                downside_dev = neg_returns.std() * np.sqrt(252)
                sortino_ratio = ann_return / downside_dev if downside_dev != 0 else np.nan
        
        calmar_ratio = abs(ann_return / max_drawdown) if max_drawdown != 0 else np.nan
        
        # Create metrics table with enhanced formatting and organization
        metrics = [
            {'category': 'Return Metrics', 'metric': 'Total Return', 'value': cum_return, 'format': 'percentage', 
             'tooltip': 'Cumulative return over the entire period'},
            {'category': 'Return Metrics', 'metric': 'Annualized Return', 'value': ann_return, 'format': 'percentage',
             'tooltip': 'Return expressed as an annual rate'},
            {'category': 'Return Metrics', 'metric': 'Best Day', 'value': filtered_returns['Return'].max(), 'format': 'percentage',
             'tooltip': 'Highest daily return in the period'},
            {'category': 'Return Metrics', 'metric': 'Worst Day', 'value': filtered_returns['Return'].min(), 'format': 'percentage',
             'tooltip': 'Lowest daily return in the period'},
            
            {'category': 'Risk Metrics', 'metric': 'Annualized Volatility', 'value': ann_vol, 'format': 'percentage',
             'tooltip': 'Standard deviation of returns expressed as an annual rate'},
            {'category': 'Risk Metrics', 'metric': 'Maximum Drawdown', 'value': max_drawdown, 'format': 'percentage',
             'tooltip': 'Largest peak-to-trough decline in portfolio value'},
            {'category': 'Risk Metrics', 'metric': 'VaR (95%)', 'value': filtered_var_results['historical_pct'].values[0] if not filtered_var_results.empty else np.nan, 'format': 'percentage',
             'tooltip': 'Maximum expected loss with 95% confidence'},
            {'category': 'Risk Metrics', 'metric': 'Conditional VaR', 'value': filtered_var_results['historical_es'].values[0]/investment_value if not filtered_var_results.empty else np.nan, 'format': 'percentage',
             'tooltip': 'Average loss beyond VaR threshold'},
            
            {'category': 'Risk-Adjusted Metrics', 'metric': 'Sharpe Ratio', 'value': sharpe, 'format': 'float',
             'tooltip': 'Return per unit of risk (assuming 0% risk-free rate)'},
            {'category': 'Risk-Adjusted Metrics', 'metric': 'Sortino Ratio', 'value': sortino_ratio, 'format': 'float',
             'tooltip': 'Return per unit of downside risk (negative volatility)'},
            {'category': 'Risk-Adjusted Metrics', 'metric': 'Calmar Ratio', 'value': calmar_ratio, 'format': 'float',
             'tooltip': 'Return relative to maximum drawdown'},
            {'category': 'Risk-Adjusted Metrics', 'metric': 'Win Rate', 'value': win_rate, 'format': 'percentage',
             'tooltip': 'Percentage of days with positive returns'},

            {'category': 'Distribution Metrics', 'metric': 'Skewness', 'value': skewness, 'format': 'float',
             'tooltip': 'Measure of asymmetry in the return distribution'},
            {'category': 'Distribution Metrics', 'metric': 'Excess Kurtosis', 'value': kurtosis, 'format': 'float',
             'tooltip': 'Measure of the "fatness" of distribution tails'},
            {'category': 'Distribution Metrics', 'metric': 'Jarque-Bera p-value', 'value': jarque_bera[1], 'format': 'float',
             'tooltip': 'Tests if returns follow a normal distribution (p<0.05 rejects normality)'},
            {'category': 'Distribution Metrics', 'metric': 'Autocorrelation', 'value': filtered_returns['Return'].autocorr(1), 'format': 'float',
             'tooltip': 'Correlation between returns and lagged returns'},
        ]
        
        # Group metrics by category
        metrics_by_category = {}
        for item in metrics:
            category = item['category']
            if category not in metrics_by_category:
                metrics_by_category[category] = []
            metrics_by_category[category].append(item)
        
        # Display metrics in a balanced multi-column layout
        col1, col2 = st.columns(2)
        
        with col1:
            # First two categories
            categories = list(metrics_by_category.keys())
            for i in range(min(2, len(categories))):
                category = categories[i]
                category_metrics = metrics_by_category[category]
                
                st.markdown(f"#### {category}", unsafe_allow_html=True)
                
                metrics_html = "<table class='data-table' style='margin-top: 0;'>"
                metrics_html += "<tr><th>Metric</th><th>Value</th></tr>"
                
                for metric in category_metrics:
                    # Format value based on type
                    value = metric['value']
                    if pd.isna(value):
                        formatted_value = "N/A"
                        style = ""
                    elif metric['format'] == 'percentage':
                        formatted_value = f"{value:.2%}"
                        style = f"color: {'#2e7d32' if value > 0 else '#d32f2f' if value < 0 else '#455a64'}; font-weight: {'600' if abs(value) > 0.05 else 'normal'};"
                    elif metric['format'] == 'float':
                        formatted_value = f"{value:.3f}"
                        if metric['metric'] == 'Sharpe Ratio':
                            style = f"color: {'#2e7d32' if value > 1 else '#f57c00' if value > 0 else '#d32f2f'}; font-weight: {'600' if abs(value) > 1 else 'normal'};"
                        elif metric['metric'] in ['Skewness', 'Excess Kurtosis']:
                            style = f"color: {'#d32f2f' if abs(value) > 1 else '#f57c00' if abs(value) > 0.5 else '#455a64'}; font-weight: {'600' if abs(value) > 1 else 'normal'};"
                        else:
                            style = ""
                    else:
                        formatted_value = str(value)
                        style = ""
                    
                    # Add tooltip
                    tooltip = metric.get('tooltip', '')
                    metric_name = f'<span class="tooltip" data-tooltip="{tooltip}">{metric["metric"]} ℹ️</span>' if tooltip else metric['metric']
                    
                    metrics_html += f"""
                    <tr>
                        <td>{metric_name}</td>
                        <td style="{style}">{formatted_value}</td>
                    </tr>
                    """
                
                metrics_html += "</table>"
                
                components.html(metrics_html, height=200, scrolling=False)
        
        with col2:
            # Remaining categories
            for i in range(2, len(categories)):
                category = categories[i]
                category_metrics = metrics_by_category[category]
                
                st.markdown(f"#### {category}", unsafe_allow_html=True)
                
                metrics_html = "<table class='data-table' style='margin-top: 0;'>"
                metrics_html += "<tr><th>Metric</th><th>Value</th></tr>"
                
                for metric in category_metrics:
                    # Format value based on type
                    value = metric['value']
                    if pd.isna(value):
                        formatted_value = "N/A"
                        style = ""
                    elif metric['format'] == 'percentage':
                        formatted_value = f"{value:.2%}"
                        style = f"color: {'#2e7d32' if value > 0 else '#d32f2f' if value < 0 else '#455a64'}; font-weight: {'600' if abs(value) > 0.05 else 'normal'};"
                    elif metric['format'] == 'float':
                        formatted_value = f"{value:.3f}"
                        if metric['metric'] == 'Jarque-Bera p-value':
                            style = f"color: {'#d32f2f' if value < 0.05 else '#2e7d32'}; font-weight: {'600' if value < 0.05 else 'normal'};"
                        else:
                            style = ""
                    else:
                        formatted_value = str(value)
                        style = ""
                    
                    # Add tooltip
                    tooltip = metric.get('tooltip', '')
                    metric_name = f'<span class="tooltip" data-tooltip="{tooltip}">{metric["metric"]} ℹ️</span>' if tooltip else metric['metric']
                    
                    metrics_html += f"""
                    <tr>
                        <td>{metric_name}</td>
                        <td style="{style}">{formatted_value}</td>
                    </tr>
                    """
                
                metrics_html += "</table>"
                
                components.html(metrics_html, height=200, scrolling=False)
            
            # Add interpretive guidance
            st.markdown("""
            <div style="padding: 0.75rem; background-color: #f5f7fa; border-radius: 0.5rem; margin-top: 1rem; border-left: 3px solid #283593;">
                <h4 style="margin-top: 0; margin-bottom: 0.5rem;">Interpreting Risk Metrics</h4>
                <ul style="margin: 0; padding-left: 1.5rem; font-size: 0.85rem; color: #455a64;">
                    <li><strong>Sharpe Ratio:</strong> >1 is good, >2 is excellent</li>
                    <li><strong>Sortino Ratio:</strong> Usually higher than Sharpe; >2 is good</li>
                    <li><strong>Calmar Ratio:</strong> >1 means annual return exceeds max drawdown</li>
                    <li><strong>Skewness:</strong> Negative values indicate longer left tail (more extreme losses)</li>
                    <li><strong>Kurtosis:</strong> >0 indicates fat tails (more extreme events than normal)</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Add drawdown visualization
        st.markdown("<h2 class='sub-header'>Drawdown Analysis</h2>", unsafe_allow_html=True)
        
        # Calculate drawdown series
        cum_returns = (1 + filtered_returns['Return']).cumprod()
        running_max = cum_returns.cummax()
        drawdown_series = (cum_returns / running_max) - 1
        
        # Create enhanced drawdown visualization
        fig = go.Figure()
        
        # Add drawdown trace
        fig.add_trace(go.Scatter(
            x=drawdown_series.index,
            y=drawdown_series,
            mode='lines',
            name='Drawdown',
            line=dict(color='#d32f2f', width=2),
            fill='tozeroy',
            fillcolor='rgba(211, 47, 47, 0.1)',
            hovertemplate='%{x|%Y-%m-%d}: %{y:.2%}<extra></extra>'
        ))
        
        # Identify major drawdown periods (below -10%)
        if drawdown_series.min() < -0.1:
            major_dd = drawdown_series[drawdown_series < -0.1]
            
            # Find start and end dates of major drawdown periods
            if not major_dd.empty:
                # Group consecutive dates
                major_dd_groups = []
                current_group = []
                
                sorted_dates = sorted(major_dd.index)
                for i, date in enumerate(sorted_dates):
                    if i == 0 or (date - sorted_dates[i-1]).days > 7:  # Start new group if gap > 7 days
                        if current_group:
                            major_dd_groups.append(current_group)
                        current_group = [date]
                    else:
                        current_group.append(date)
                
                if current_group:
                    major_dd_groups.append(current_group)
                
                # Highlight major drawdown periods
                for group in major_dd_groups:
                    start_date = min(group)
                    end_date = max(group)
                    min_dd = drawdown_series.loc[start_date:end_date].min()
                    
                    fig.add_vrect(
                        x0=start_date,
                        x1=end_date,
                        fillcolor="#d32f2f",
                        opacity=0.1,
                        layer="below",
                        line_width=0
                    )
                    
                    # Add annotation for significant drawdowns
                    if min_dd < -0.15:  # Only annotate significant drawdowns (>15%)
                        min_dd_date = drawdown_series.loc[start_date:end_date].idxmin()
                        
                        fig.add_annotation(
                            x=min_dd_date,
                            y=min_dd,
                            text=f"{min_dd:.1%}",
                            showarrow=True,
                            arrowhead=1,
                            ax=0,
                            ay=30,
                            font=dict(color="#d32f2f", size=10)
                        )
        
        # Add zero line
        fig.add_shape(
            type="line",
            x0=drawdown_series.index.min(),
            y0=0,
            x1=drawdown_series.index.max(),
            y1=0,
            line=dict(color="#455a64", width=1)
        )
        
        fig.update_layout(
            title="Portfolio Drawdown Over Time",
            xaxis_title="Date",
            yaxis_title="Drawdown",
            yaxis=dict(tickformat='.0%'),
            height=400,
            margin=dict(l=20, r=20, t=50, b=20),
            plot_bgcolor='#ffffff',
            paper_bgcolor='#ffffff',
            font=dict(color='#455a64')
        )
        
        # Add range selector
        fig.update_xaxes(
            rangeslider_visible=False,
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add drawdown recovery analysis
        if max_drawdown < -0.05:  # Only show if we have a meaningful drawdown
            st.markdown("""
            <div style="padding: 0.75rem; background-color: #f5f7fa; border-radius: 0.5rem; margin-top: 1rem;">
                <h4 style="margin-top: 0; margin-bottom: 0.5rem;">Drawdown Recovery Analysis</h4>
            """, unsafe_allow_html=True)
            
            # Calculate theoretical recovery time
            recovery_needed = 1 / (1 + max_drawdown) - 1
            if ann_return > 0:
                est_recovery_years = np.log(1 + recovery_needed) / np.log(1 + ann_return)
                est_recovery_days = est_recovery_years * 252
                
                st.markdown(f"""
                <p style="margin: 0; font-size: 0.9rem; color: #455a64;">
                    <strong>Maximum Drawdown:</strong> {max_drawdown:.2%}<br>
                    <strong>Recovery Required:</strong> {recovery_needed:.2%}<br>
                    <strong>Estimated Recovery Time:</strong> {est_recovery_days:.0f} trading days ({est_recovery_years:.1f} years) at current return rate
                </p>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <p style="margin: 0; font-size: 0.9rem; color: #455a64;">
                    <strong>Maximum Drawdown:</strong> {max_drawdown:.2%}<br>
                    <strong>Recovery Required:</strong> {recovery_needed:.2%}<br>
                    <strong>Estimated Recovery Time:</strong> Unable to estimate (negative return rate)
                </p>
                """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.warning("No return data available for the selected date range. Please adjust your date range selection.")

# Footer - Enhanced with better styling and informative elements
st.markdown("""
<footer>
    <div style="display: flex; justify-content: space-between; align-items: center; padding-top: 2rem; margin-top: 3rem; border-top: 1px solid #e0e0e0;">
        <div>
            <p style="margin: 0; color: #78909c; font-size: 0.8rem;">Risk Management Dashboard v1.5.0</p>
            <p style="margin: 0; color: #78909c; font-size: 0.8rem;">Data updated: April 26, 2025</p>
        </div>
        <div>
            <p style="margin: 0; color: #78909c; font-size: 0.8rem;">
                Created with <a href="https://streamlit.io" target="_blank" style="color: #1a237e; text-decoration: none;">Streamlit</a>
                &bull; <a href="#" style="color: #1a237e; text-decoration: none;">Documentation</a>
                &bull; <a href="#" style="color: #1a237e; text-decoration: none;">Report Issue</a>
            </p>
        </div>
    </div>
</footer>
""", unsafe_allow_html=True)