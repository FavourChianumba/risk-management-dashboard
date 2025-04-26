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

# Set page configuration
st.set_page_config(
    page_title="Risk Management Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dashboard styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1E3A8A;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        border-radius: 0.5rem;
        padding: 1rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }
    .risk-high {
        color: #DC2626;
        font-weight: 600;
    }
    .risk-medium {
        color: #F59E0B;
        font-weight: 600;
    }
    .risk-low {
        color: #10B981;
        font-weight: 600;
    }
    .info-text {
        font-size: 0.9rem;
        color: #6B7280;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #F3F4F6;
        border-radius: 4px 4px 0 0;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #E5E7EB;
        border-bottom: 2px solid #1E3A8A;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions
def format_currency(value):
    """Format value as currency."""
    return f"${value:,.2f}"

def format_percentage(value):
    """Format value as percentage."""
    return f"{value:.2%}"

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

# Data loading
@st.cache_data(ttl=3600)
def load_data():
    """Load all data files."""
    try:
        # Define base directory relative to the current file
        base_dir = Path(__file__).parent
        
        # Define paths to data files
        portfolio_returns_path = base_dir / "data" / "processed" / "portfolio_returns.csv"
        var_results_path = base_dir / "data" / "results" / "var_results.csv"
        stress_test_results_path = base_dir / "data" / "results" / "stress_test_results.csv"
        var_backtest_summary_path = base_dir / "data" / "results" / "var_backtest_summary.csv"
        
        # Load data files
        portfolio_returns = pd.read_csv(portfolio_returns_path, parse_dates=['Date'], index_col='Date')
        portfolio_returns.columns = ['Return']  # Rename column for clarity
        
        var_results = pd.read_csv(var_results_path)
        stress_test_results = pd.read_csv(stress_test_results_path)
        var_backtest_summary = pd.read_csv(var_backtest_summary_path)
        
        return portfolio_returns, var_results, stress_test_results, var_backtest_summary
    
    except Exception as e:
        st.error(f"Error loading data: {e}")
        # Provide empty dataframes with expected columns if loading fails
        portfolio_returns = pd.DataFrame(columns=['Return'])
        var_results = pd.DataFrame(columns=['confidence_level', 'time_horizon', 'historical_var', 
                                           'historical_es', 'parametric_var', 'parametric_es', 
                                           'historical_pct', 'parametric_pct'])
        stress_test_results = pd.DataFrame(columns=['scenario', 'var_value', 'var_pct', 'scenario_type', 'risk_level'])
        var_backtest_summary = pd.DataFrame(columns=['Model', 'Avg VaR (%)', 'Max VaR (%)', 'Breaches', 
                                                    'Breach Rate (%)', 'Expected (%)', 'Breach Ratio'])
        return portfolio_returns, var_results, stress_test_results, var_backtest_summary

# Load data
portfolio_returns, var_results, stress_test_results, var_backtest_summary = load_data()

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
    return dates

calendar = create_calendar(portfolio_returns)

# Sidebar filters
st.sidebar.title("Risk Management Dashboard")
st.sidebar.image("https://i.imgur.com/GLpwQUh.png", width=200)  # Generic risk management icon

# Date range filter
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
    else:
        filtered_returns = portfolio_returns
else:
    filtered_returns = portfolio_returns

# Confidence level filter
confidence_options = sorted(var_results['confidence_level'].unique()) if not var_results.empty else [0.95]
confidence_level = st.sidebar.selectbox(
    "Confidence Level",
    options=confidence_options,
    format_func=lambda x: f"{x:.0%}",
    index=1 if len(confidence_options) > 1 else 0  # Default to 95% if available
)

# Time horizon filter
horizon_options = sorted(var_results['time_horizon'].unique()) if not var_results.empty else [1]
time_horizon = st.sidebar.selectbox(
    "Time Horizon (Days)",
    options=horizon_options,
    index=0  # Default to 1 day
)

# Filter VaR results based on selections
filtered_var_results = var_results[
    (var_results['confidence_level'] == confidence_level) & 
    (var_results['time_horizon'] == time_horizon)
] if not var_results.empty else pd.DataFrame()

# Navigation tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Executive Summary", 
    "VaR Analysis", 
    "Stress Testing", 
    "Model Validation",
    "Return Analysis"
])

# 1. Executive Summary
with tab1:
    st.markdown("<h1 class='main-header'>Risk Management Executive Summary</h1>", unsafe_allow_html=True)
    
    # Key metrics
    if not filtered_var_results.empty:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            hist_var = filtered_var_results['historical_var'].values[0]
            hist_pct = filtered_var_results['historical_pct'].values[0]
            risk_class = risk_level_text(hist_pct)
            risk_color = risk_level_color(hist_pct)
            
            st.markdown(f"""
            <div class='metric-card'>
                <h3>Historical VaR</h3>
                <p style='font-size: 1.8rem;'>{format_currency(hist_var)}</p>
                <p>({format_percentage(hist_pct)}) <span class='{risk_color}'>{risk_class} Risk</span></p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            param_var = filtered_var_results['parametric_var'].values[0]
            param_pct = filtered_var_results['parametric_pct'].values[0]
            
            st.markdown(f"""
            <div class='metric-card'>
                <h3>Parametric VaR</h3>
                <p style='font-size: 1.8rem;'>{format_currency(param_var)}</p>
                <p>({format_percentage(param_pct)})</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            hist_es = filtered_var_results['historical_es'].values[0]
            es_ratio = hist_es / hist_var
            
            st.markdown(f"""
            <div class='metric-card'>
                <h3>Expected Shortfall</h3>
                <p style='font-size: 1.8rem;'>{format_currency(hist_es)}</p>
                <p>ES/VaR Ratio: {es_ratio:.2f}x</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            # Get worst stress scenario
            if not stress_test_results.empty:
                # Exclude baseline
                non_baseline = stress_test_results[stress_test_results['scenario'] != 'Baseline']
                if not non_baseline.empty:
                    worst_scenario = non_baseline.sort_values('var_value', ascending=False).iloc[0]
                    
                    st.markdown(f"""
                    <div class='metric-card'>
                        <h3>Worst Stress Scenario</h3>
                        <p style='font-size: 1.5rem;'>{worst_scenario['scenario']}</p>
                        <p>{format_currency(worst_scenario['var_value'])} ({format_percentage(worst_scenario['var_pct'])})</p>
                    </div>
                    """, unsafe_allow_html=True)
    
    # Returns summary
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
                line=dict(color='royalblue', width=2)
            ))
            
            # Add VaR threshold if available
            if not filtered_var_results.empty:
                var_threshold = -filtered_var_results['historical_pct'].values[0]
                fig.add_trace(go.Scatter(
                    x=[cum_returns.index.min(), cum_returns.index.max()],
                    y=[var_threshold, var_threshold],
                    mode='lines',
                    name=f'{confidence_level:.0%} VaR Threshold',
                    line=dict(color='red', width=2, dash='dash')
                ))
            
            fig.update_layout(
                title='Cumulative Portfolio Return',
                xaxis_title='Date',
                yaxis_title='Return',
                yaxis_tickformat='.1%',
                hovermode='x unified',
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                height=400,
                margin=dict(l=20, r=20, t=50, b=20)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Return statistics
            stats = pd.DataFrame({
                'Metric': [
                    'Total Return',
                    'Annualized Return',
                    'Volatility (Ann.)',
                    'Sharpe Ratio',
                    'Max Drawdown',
                    'VaR Ratio'
                ],
                'Value': [
                    cum_returns['Return'].iloc[-1],
                    (1 + filtered_returns['Return'].mean()) ** 252 - 1,
                    filtered_returns['Return'].std() * np.sqrt(252),
                    ((filtered_returns['Return'].mean()) * 252) / (filtered_returns['Return'].std() * np.sqrt(252)),
                    (cum_returns['Return'] + 1).div((cum_returns['Return'] + 1).cummax()).min() - 1,
                    filtered_var_results['historical_pct'].values[0] / filtered_returns['Return'].std() if not filtered_var_results.empty else np.nan
                ]
            })
            
            # Format values
            stats['Formatted'] = stats['Value'].apply(lambda x: f"{x:.2%}" if not pd.isna(x) else "N/A")
            stats.loc[stats['Metric'] == 'Sharpe Ratio', 'Formatted'] = stats.loc[stats['Metric'] == 'Sharpe Ratio', 'Value'].apply(lambda x: f"{x:.2f}" if not pd.isna(x) else "N/A")
            stats.loc[stats['Metric'] == 'VaR Ratio', 'Formatted'] = stats.loc[stats['Metric'] == 'VaR Ratio', 'Value'].apply(lambda x: f"{x:.2f}" if not pd.isna(x) else "N/A")
            
            st.dataframe(
                stats[['Metric', 'Formatted']].set_index('Metric'),
                use_container_width=True,
                height=400
            )
    
    # Model comparison
    st.markdown("<h2 class='sub-header'>Model Performance Summary</h2>", unsafe_allow_html=True)
    
    if not var_backtest_summary.empty:
        # Display model backtesting summary
        st.dataframe(
            var_backtest_summary,
            use_container_width=True,
            column_config={
                "Model": st.column_config.TextColumn("Model"),
                "Avg VaR (%)": st.column_config.NumberColumn("Avg VaR (%)", format="%.2f%%"),
                "Max VaR (%)": st.column_config.NumberColumn("Max VaR (%)", format="%.2f%%"),
                "Breach Rate (%)": st.column_config.NumberColumn("Breach Rate (%)", format="%.2f%%"),
                "Expected (%)": st.column_config.NumberColumn("Expected (%)", format="%.2f%%"),
                "Assessment": st.column_config.TextColumn("Assessment"),
            },
            hide_index=True
        )

# 2. VaR Analysis
with tab2:
    st.markdown("<h1 class='main-header'>Value-at-Risk (VaR) Analysis</h1>", unsafe_allow_html=True)
    
    if not filtered_var_results.empty:
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
            # Create comparison bar chart
            fig = go.Figure()
            
            # Add bars for VaR values
            methods = ['Historical VaR', 'Parametric VaR', 'Historical ES', 'Parametric ES']
            values = [
                filtered_var_results['historical_var'].values[0],
                filtered_var_results['parametric_var'].values[0],
                filtered_var_results['historical_es'].values[0],
                filtered_var_results['parametric_es'].values[0]
            ]
            
            # Define colors
            colors = ['#1f77b4', '#ff7f0e', '#1f77b4', '#ff7f0e']
            opacity = [1, 1, 0.6, 0.6]
            
            for i, (method, value) in enumerate(zip(methods, values)):
                fig.add_trace(go.Bar(
                    x=[method],
                    y=[value],
                    name=method,
                    marker_color=colors[i],
                    marker_opacity=opacity[i],
                    text=[f"${value:,.2f}"],
                    textposition='outside'
                ))
            
            fig.update_layout(
                title=f"Risk Metrics at {confidence_level:.0%} Confidence Level ({time_horizon}-Day Horizon)",
                xaxis_title="Methodology",
                yaxis_title="Value ($)",
                barmode='group',
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                height=400,
                margin=dict(l=20, r=20, t=50, b=20)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Create a comparison table
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
                ]
            }
            
            comparison_df = pd.DataFrame(comparison_data)
            
            # Format values
            comparison_df['Formatted'] = comparison_df['Value'].apply(lambda x: f"${x:,.2f}" if isinstance(x, (int, float)) and not pd.isna(x) else "N/A")
            comparison_df.loc[comparison_df['Metric'].str.contains('Ratio'), 'Formatted'] = comparison_df.loc[comparison_df['Metric'].str.contains('Ratio'), 'Value'].apply(lambda x: f"{x:.2f}x" if isinstance(x, (int, float)) and not pd.isna(x) else "N/A")
            
            st.dataframe(
                comparison_df[['Metric', 'Formatted']].set_index('Metric'),
                use_container_width=True,
                height=400
            )
        
        # Time horizon scaling
        st.markdown("<h2 class='sub-header'>VaR Scaling by Time Horizon</h2>", unsafe_allow_html=True)
        
        # Filter var_results for the selected confidence level but all time horizons
        horizon_data = var_results[var_results['confidence_level'] == confidence_level]
        
        if not horizon_data.empty:
            # Plot horizon scaling
            fig = go.Figure()
            
            # Add lines for each VaR method
            fig.add_trace(go.Scatter(
                x=horizon_data['time_horizon'],
                y=horizon_data['historical_var'],
                mode='lines+markers',
                name='Historical VaR',
                line=dict(color='#1f77b4', width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=horizon_data['time_horizon'],
                y=horizon_data['parametric_var'],
                mode='lines+markers',
                name='Parametric VaR',
                line=dict(color='#ff7f0e', width=2)
            ))
            
            # Add theoretical sqrt(t) scaling line for comparison
            if len(horizon_data) > 1:
                base_var = horizon_data.loc[horizon_data['time_horizon'] == 1, 'historical_var'].values[0]
                sqrt_scaling = [base_var * np.sqrt(t) for t in horizon_data['time_horizon']]
                
                fig.add_trace(go.Scatter(
                    x=horizon_data['time_horizon'],
                    y=sqrt_scaling,
                    mode='lines',
                    name='Theoretical √t Scaling',
                    line=dict(color='green', width=2, dash='dash')
                ))
            
            fig.update_layout(
                title=f"VaR Scaling with Time Horizon ({confidence_level:.0%} Confidence)",
                xaxis_title="Time Horizon (Days)",
                yaxis_title="Value ($)",
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                height=400,
                margin=dict(l=20, r=20, t=50, b=20)
            )
            
            # Format x-axis to only show integer values
            fig.update_xaxes(tickmode='array', tickvals=list(horizon_data['time_horizon']))
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show sqrt(t) rule explanation
            with st.expander("About the Square Root of Time Rule"):
                st.markdown("""
                The square root of time rule is commonly used to scale VaR from one time horizon to another. 
                If the 1-day VaR is known, the t-day VaR can be approximated as:
                
                **t-day VaR = 1-day VaR × √t**
                
                This approximation assumes returns are independently and identically distributed (i.i.d.) and 
                normally distributed. The chart above compares actual VaR calculations at different time horizons 
                with theoretical scaling based on the square root of time rule.
                """)
    else:
        st.warning("No VaR data available for the selected confidence level and time horizon.")

# 3. Stress Testing
with tab3:
    st.markdown("<h1 class='main-header'>Stress Test Analysis</h1>", unsafe_allow_html=True)
    
    if not stress_test_results.empty:
        st.markdown(f"""
        <p class='info-text'>
            Stress testing evaluates portfolio performance under extreme market conditions that go beyond
            normal VaR confidence levels. This analysis shows how the portfolio would perform under various
            historical and hypothetical crisis scenarios.
        </p>
        """, unsafe_allow_html=True)
        
        # Scenario selector
        scenarios = stress_test_results['scenario'].unique()
        # Convert scenarios to list of strings
        scenarios = [str(s) for s in scenarios]
        baseline_idx = np.where(np.array(scenarios) == 'Baseline')[0]
        # Convert NumPy int64 to native Python int
        default_idx = 0 if len(baseline_idx) == 0 else int(baseline_idx[0])

        selected_scenario = st.selectbox(
            "Select Stress Scenario",
            options=scenarios,
            index=default_idx
        )
        
        # Get scenario data
        scenario_data = stress_test_results[stress_test_results['scenario'] == selected_scenario].iloc[0]
        
        # Get baseline data if it exists
        baseline_data = None
        if 'Baseline' in scenarios:
            baseline_data = stress_test_results[stress_test_results['scenario'] == 'Baseline'].iloc[0]
        
        # Display scenario details
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class='metric-card'>
                <h3>Scenario VaR</h3>
                <p style='font-size: 1.8rem;'>{format_currency(scenario_data['var_value'])}</p>
                <p>({format_percentage(scenario_data['var_pct'])})</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            if baseline_data is not None:
                impact = scenario_data['var_value'] - baseline_data['var_value']
                impact_pct = (scenario_data['var_value'] / baseline_data['var_value']) - 1
                
                st.markdown(f"""
                <div class='metric-card'>
                    <h3>Impact vs. Baseline</h3>
                    <p style='font-size: 1.8rem;'>{format_currency(impact)}</p>
                    <p>({format_percentage(impact_pct)})</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='metric-card'>
                    <h3>Scenario Type</h3>
                    <p style='font-size: 1.8rem;'>{scenario_data['scenario_type']}</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col3:
            risk_color = risk_level_color(scenario_data['var_pct'])
            
            st.markdown(f"""
            <div class='metric-card'>
                <h3>Risk Level</h3>
                <p style='font-size: 1.8rem;' class='{risk_color}'>{scenario_data['risk_level']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Scenario comparison chart
        st.markdown("<h2 class='sub-header'>Scenario Comparison</h2>", unsafe_allow_html=True)
        
        # Sort scenarios by VaR and exclude baseline from the chart
        sorted_scenarios = stress_test_results.copy()
        if 'Baseline' in scenarios:
            baseline_var = baseline_data['var_value']
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
        
        # Create the comparison chart
        fig = go.Figure()
        
        # Add bars colored by risk level
        colors = {
            'Low': '#10B981',    # Green
            'Medium': '#F59E0B',  # Amber
            'High': '#DC2626',    # Red
            'Extreme': '#7F1D1D'  # Dark red
        }
        
        for risk in chart_scenarios['risk_level'].unique():
            risk_data = chart_scenarios[chart_scenarios['risk_level'] == risk]
            
            fig.add_trace(go.Bar(
                x=risk_data['scenario'],
                y=risk_data['var_value'],
                name=risk,
                marker_color=colors.get(risk, '#6B7280'),  # Default to gray if risk level not found
                text=[f"${v:,.0f}" for v in risk_data['var_value']],
                textposition='outside'
            ))
        
        # Add baseline reference line if available
        if 'Baseline' in scenarios:
            fig.add_shape(
                type='line',
                x0=-0.5,
                y0=baseline_var,
                x1=len(chart_scenarios) - 0.5,
                y1=baseline_var,
                line=dict(color='black', width=2, dash='dash'),
                name='Baseline'
            )
            
            fig.add_annotation(
                x=len(chart_scenarios) - 1,
                y=baseline_var,
                text=f"Baseline: ${baseline_var:,.0f}",
                showarrow=False,
                yshift=10
            )
        
        fig.update_layout(
            title="Stress Scenarios Ranked by Severity",
            xaxis_title="Scenario",
            yaxis_title="Value-at-Risk ($)",
            legend_title="Risk Level",
            height=500,
            margin=dict(l=20, r=20, t=50, b=100)
        )
        
        # Rotate x-axis labels for better readability
        fig.update_xaxes(tickangle=45)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Risk matrix
        st.markdown("<h2 class='sub-header'>Risk Matrix</h2>", unsafe_allow_html=True)
        
        # Create a risk matrix visualization
        # X-axis: Impact severity (VaR size)
        # Y-axis: Scenario type (as a proxy for probability)
        
        # Map scenario types to probability scores (1-5)
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
        
        # Create risk matrix
        fig = go.Figure()
        
        # Add scatter points for each scenario
        for risk in stress_test_results['risk_level'].unique():
            risk_data = stress_test_results[stress_test_results['risk_level'] == risk]
            
            fig.add_trace(go.Scatter(
                x=risk_data['var_value'],
                y=risk_data['probability'],
                mode='markers+text',
                marker=dict(
                    size=risk_data['var_pct'] * 1000,  # Size proportional to VaR percentage
                    color=colors.get(risk, '#6B7280'),  # Color by risk level
                    line=dict(width=1, color='black')
                ),
                text=risk_data['scenario'],
                textposition="top center",
                name=risk
            ))
        
        # Add risk zones
        fig.add_shape(
            type="rect",
            x0=0,
            y0=0,
            x1=20000,
            y1=5.5,
            line=dict(color="rgba(0,0,0,0)"),
            fillcolor="rgba(16,185,129,0.1)",
            layer="below"
        )
        
        fig.add_shape(
            type="rect",
            x0=20000,
            y0=0,
            x1=40000,
            y1=5.5,
            line=dict(color="rgba(0,0,0,0)"),
            fillcolor="rgba(245,158,11,0.1)",
            layer="below"
        )
        
        fig.add_shape(
            type="rect",
            x0=40000,
            y0=0,
            x1=max(stress_test_results['var_value']) * 1.1,
            y1=5.5,
            line=dict(color="rgba(0,0,0,0)"),
            fillcolor="rgba(220,38,38,0.1)",
            layer="below"
        )
        
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
            margin=dict(l=20, r=20, t=50, b=50)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Risk level distribution
        st.markdown("<h2 class='sub-header'>Risk Level Distribution</h2>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Create pie chart of risk levels
            risk_counts = stress_test_results['risk_level'].value_counts().reset_index()
            risk_counts.columns = ['Risk Level', 'Count']
            
            fig = px.pie(
                risk_counts,
                values='Count',
                names='Risk Level',
                color='Risk Level',
                color_discrete_map=colors,
                title="Distribution of Risk Levels"
            )
            
            fig.update_traces(textposition='inside', textinfo='percent+label')
            
            fig.update_layout(
                height=400,
                margin=dict(l=20, r=20, t=50, b=20)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Create bar chart of scenario types
            type_counts = stress_test_results['scenario_type'].value_counts().reset_index()
            type_counts.columns = ['Scenario Type', 'Count']
            
            # Calculate average VaR by scenario type
            type_var = stress_test_results.groupby('scenario_type')['var_value'].mean().reset_index()
            type_var.columns = ['Scenario Type', 'Average VaR']
            
            # Merge counts and VaR
            type_analysis = pd.merge(type_counts, type_var, on='Scenario Type')
            
            fig = px.bar(
                type_analysis,
                x='Scenario Type',
                y='Average VaR',
                color='Scenario Type',
                text_auto='.2s',
                title="Average VaR by Scenario Type"
            )
            
            fig.update_layout(
                height=400,
                margin=dict(l=20, r=20, t=50, b=20)
            )
            
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No stress test data available.")

# 4. Model Validation
with tab4:
    st.markdown("<h1 class='main-header'>Model Validation & Backtesting</h1>", unsafe_allow_html=True)
    
    if not var_backtest_summary.empty:
        st.markdown(f"""
        <p class='info-text'>
            Backtesting is essential to validate VaR models by comparing predicted risk levels against actual outcomes.
            This analysis shows how different VaR methodologies performed historically, including breach rates and
            statistical tests.
        </p>
        """, unsafe_allow_html=True)
        
        # Model selector
        models = var_backtest_summary['Model'].unique()
        selected_model = st.selectbox(
            "Select Model to Analyze",
            options=models,
            index=0
        )
        
        # Get model data
        model_data = var_backtest_summary[var_backtest_summary['Model'] == selected_model].iloc[0]
        
        # Display model metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            breach_rate = model_data['Breach Rate (%)']
            expected_rate = model_data['Expected (%)']
            breach_ratio = model_data['Breach Ratio']
            
            color = 'risk-low' if 0.8 <= breach_ratio <= 1.2 else 'risk-high'
            
            st.markdown(f"""
            <div class='metric-card'>
                <h3>Breach Rate</h3>
                <p style='font-size: 1.8rem;'>{breach_rate:.2f}%</p>
                <p>Expected: {expected_rate:.2f}% (Ratio: <span class='{color}'>{breach_ratio:.2f}</span>)</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            kupiec_result = model_data['Kupiec Test']
            kupiec_color = 'risk-low' if kupiec_result == 'Pass' else 'risk-high'
            
            st.markdown(f"""
            <div class='metric-card'>
                <h3>Kupiec Test</h3>
                <p style='font-size: 1.8rem;' class='{kupiec_color}'>{kupiec_result}</p>
                <p>Tests correct frequency of breaches</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            christ_result = model_data['Christoffersen Test']
            christ_color = 'risk-low' if christ_result == 'Pass' else 'risk-high'
            
            st.markdown(f"""
            <div class='metric-card'>
                <h3>Christoffersen Test</h3>
                <p style='font-size: 1.8rem;' class='{christ_color}'>{christ_result}</p>
                <p>Tests independence of breaches</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            assessment = model_data['Assessment']
            assessment_color = 'risk-low' if assessment in ['Excellent', 'Good'] else 'risk-medium' if assessment == 'Fair' else 'risk-high'
            
            st.markdown(f"""
            <div class='metric-card'>
                <h3>Overall Assessment</h3>
                <p style='font-size: 1.8rem;' class='{assessment_color}'>{assessment}</p>
                <p>Combined evaluation of model performance</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Model accuracy visualization
        st.markdown("<h2 class='sub-header'>Model Accuracy</h2>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # Create accuracy gauge chart
            accuracy_score = min(100, max(0, 100 - abs(breach_ratio - 1) * 100))
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=accuracy_score,
                title={'text': "Model Accuracy Score"},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "#DC2626"},
                        {'range': [50, 80], 'color': "#F59E0B"},
                        {'range': [80, 100], 'color': "#10B981"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 80
                    }
                }
            ))
            
            fig.update_layout(
                height=300,
                margin=dict(l=20, r=20, t=50, b=20)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Create breach statistics
            breaches = model_data['Breaches']
            avg_var = model_data['Avg VaR (%)']
            max_var = model_data['Max VaR (%)']
            
            stats = pd.DataFrame({
                'Metric': [
                    'Total Breaches',
                    'Breach Rate',
                    'Expected Rate',
                    'Breach Ratio',
                    'Average VaR',
                    'Maximum VaR'
                ],
                'Value': [
                    f"{breaches:.0f}",
                    f"{breach_rate:.2f}%",
                    f"{expected_rate:.2f}%",
                    f"{breach_ratio:.2f}",
                    f"{avg_var:.2f}%",
                    f"{max_var:.2f}%"
                ]
            })
            
            st.dataframe(
                stats.set_index('Metric'),
                use_container_width=True,
                height=300
            )
        
        # Statistical test details
        st.markdown("<h2 class='sub-header'>Statistical Test Details</h2>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Kupiec test explanation
            st.markdown("""
            #### Kupiec Test (Unconditional Coverage)
            
            The Kupiec test checks if the observed breach rate matches the expected rate based on the VaR confidence level.
            
            **Null Hypothesis**: The model's breach rate equals the expected rate.
            
            **Interpretation**:
            - **Pass**: The model correctly estimates the frequency of breaches
            - **Fail**: The model systematically under or overestimates risk
            """)
            
            # Add p-value if available
            if 'Kupiec p-value' in model_data:
                st.markdown(f"**P-value**: {model_data['Kupiec p-value']:.4f}")
        
        with col2:
            # Christoffersen test explanation
            st.markdown("""
            #### Christoffersen Test (Independence)
            
            The Christoffersen test checks if VaR breaches are independent or if they cluster together.
            
            **Null Hypothesis**: VaR breaches occur independently over time.
            
            **Interpretation**:
            - **Pass**: Breaches are randomly distributed over time
            - **Fail**: Breaches tend to cluster, indicating the model doesn't capture volatility dynamics
            """)
            
            # Add p-value if available
            if 'Christoffersen p-value' in model_data:
                st.markdown(f"**P-value**: {model_data['Christoffersen p-value']:.4f}")
        
        # Model comparison
        st.markdown("<h2 class='sub-header'>Model Comparison</h2>", unsafe_allow_html=True)
        
        # Create comparison chart
        fig = go.Figure()
        
        # Add bars for breach rates
        fig.add_trace(go.Bar(
            x=var_backtest_summary['Model'],
            y=var_backtest_summary['Breach Rate (%)'],
            name='Actual Breach Rate',
            marker_color='royalblue'
        ))
        
        # Add line for expected rate
        fig.add_trace(go.Scatter(
            x=var_backtest_summary['Model'],
            y=var_backtest_summary['Expected (%)'],
            name='Expected Rate',
            mode='lines+markers',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        # Add acceptable range (if using Basel standards)
        expected_rate = var_backtest_summary['Expected (%)'].iloc[0]
        fig.add_traces([
            go.Scatter(
                x=var_backtest_summary['Model'],
                y=[expected_rate * 0.8] * len(var_backtest_summary),
                name='Lower Bound (80%)',
                mode='lines',
                line=dict(color='green', width=1, dash='dot')
            ),
            go.Scatter(
                x=var_backtest_summary['Model'],
                y=[expected_rate * 1.2] * len(var_backtest_summary),
                name='Upper Bound (120%)',
                mode='lines',
                line=dict(color='green', width=1, dash='dot'),
                fill='tonexty',
                fillcolor='rgba(0, 128, 0, 0.1)'
            )
        ])
        
        fig.update_layout(
            title="VaR Breach Rates by Model",
            xaxis_title="Model",
            yaxis_title="Breach Rate (%)",
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            height=400,
            margin=dict(l=20, r=20, t=50, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Assessment summary
        assessment_counts = var_backtest_summary['Assessment'].value_counts().reset_index()
        assessment_counts.columns = ['Assessment', 'Count']
        
        # Create a color map for assessments
        assessment_colors = {
            'Excellent': '#10B981',  # Green
            'Good': '#34D399',      # Light green
            'Fair': '#F59E0B',      # Amber
            'Poor': '#DC2626'       # Red
        }
        
        fig = px.pie(
            assessment_counts,
            values='Count',
            names='Assessment',
            color='Assessment',
            color_discrete_map=assessment_colors,
            title="Model Assessment Distribution"
        )
        
        fig.update_traces(textposition='inside', textinfo='percent+label')
        
        fig.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=50, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No backtesting data available.")

# 5. Return Analysis
with tab5:
    st.markdown("<h1 class='main-header'>Portfolio Return Analysis</h1>", unsafe_allow_html=True)
    
    if not filtered_returns.empty:
        st.markdown(f"""
        <p class='info-text'>
            This analysis examines the statistical properties of portfolio returns, including distribution,
            volatility patterns, and extreme events. Understanding return characteristics is fundamental to
            effective risk management.
        </p>
        """, unsafe_allow_html=True)
        
        # Return distribution
        st.markdown("<h2 class='sub-header'>Return Distribution Analysis</h2>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Return histogram with normal overlay
            fig = go.Figure()
            
            # Add histogram of returns
            fig.add_trace(go.Histogram(
                x=filtered_returns['Return'],
                name='Returns',
                marker_color='royalblue',
                opacity=0.7,
                histnorm='probability density',
                nbinsx=30
            ))
            
            # Add normal distribution overlay
            x = np.linspace(
                filtered_returns['Return'].min(),
                filtered_returns['Return'].max(),
                100
            )
            mean = filtered_returns['Return'].mean()
            std = filtered_returns['Return'].std()
            y = scipy_stats.norm.pdf(x, mean, std)
            
            fig.add_trace(go.Scatter(
                x=x,
                y=y,
                mode='lines',
                name='Normal Distribution',
                line=dict(color='red', width=2, dash='dash')
            ))
            
            # Add VaR line if available
            if not filtered_var_results.empty:
                var_value = -filtered_var_results['historical_pct'].values[0]
                
                fig.add_trace(go.Scatter(
                    x=[var_value, var_value],
                    y=[0, scipy_stats.norm.pdf(var_value, mean, std) * 1.1],
                    mode='lines',
                    name=f'{confidence_level:.0%} VaR',
                    line=dict(color='green', width=2)
                ))
            
            fig.update_layout(
                title='Return Distribution vs. Normal Distribution',
                xaxis_title='Return',
                yaxis_title='Density',
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                height=400,
                margin=dict(l=20, r=20, t=50, b=20)
            )
            
            # Format x-axis as percentage
            fig.update_xaxes(tickformat=',.1%')
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Q-Q plot to check normality
            fig = go.Figure()
            
            # Calculate quantiles
            returns_sorted = sorted(filtered_returns['Return'].dropna())
            n = len(returns_sorted)
            theoretical_quantiles = [scipy_stats.norm.ppf((i + 0.5) / n) for i in range(n)]
            
            # Create scatter plot
            fig.add_trace(go.Scatter(
                x=theoretical_quantiles,
                y=returns_sorted,
                mode='markers',
                name='Returns',
                marker=dict(color='royalblue')
            ))
            
            # Add the diagonal line (y=x)
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
                line=dict(color='red', width=2, dash='dash')
            ))
            
            fig.update_layout(
                title='Q-Q Plot (Testing for Normality)',
                xaxis_title='Theoretical Quantiles',
                yaxis_title='Sample Quantiles',
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                height=400,
                margin=dict(l=20, r=20, t=50, b=20)
            )
            
            # Format y-axis as percentage
            fig.update_yaxes(tickformat=',.1%')
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Normality test results
        skewness = scipy_stats.skew(filtered_returns['Return'].dropna())
        kurtosis = scipy_stats.kurtosis(filtered_returns['Return'].dropna())
        jarque_bera = scipy_stats.jarque_bera(filtered_returns['Return'].dropna())
        
        st.markdown(f"""
        <p class='info-text'>
            <strong>Normality Test Results:</strong>
            Skewness: {skewness:.4f} (0 is normal) &bull;
            Excess Kurtosis: {kurtosis:.4f} (0 is normal) &bull;
            Jarque-Bera test: statistic={jarque_bera[0]:.4f}, p-value={jarque_bera[1]:.6f} 
            ({'<span class="risk-high">Returns are not normally distributed</span>' if jarque_bera[1] < 0.05 else '<span class="risk-low">Returns appear normally distributed</span>'})
        </p>
        """, unsafe_allow_html=True)
        
        # Return time series and volatility
        st.markdown("<h2 class='sub-header'>Return Time Series & Volatility</h2>", unsafe_allow_html=True)
        
        # Calculate rolling volatility
        rolling_window = 21  # 21 trading days ~ 1 month
        rolling_vol = filtered_returns['Return'].rolling(window=rolling_window).std() * np.sqrt(252)  # Annualized
        
        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add return trace
        fig.add_trace(
            go.Scatter(
                x=filtered_returns.index,
                y=filtered_returns['Return'],
                mode='lines',
                name='Daily Return',
                line=dict(color='royalblue', width=1)
            ),
            secondary_y=False
        )
        
        # Add volatility trace
        fig.add_trace(
            go.Scatter(
                x=rolling_vol.index,
                y=rolling_vol,
                mode='lines',
                name='Rolling Volatility (Ann.)',
                line=dict(color='red', width=2)
            ),
            secondary_y=True
        )
        
        # Add VaR threshold if available
        if not filtered_var_results.empty:
            var_value = -filtered_var_results['historical_pct'].values[0]
            
            fig.add_trace(
                go.Scatter(
                    x=[filtered_returns.index.min(), filtered_returns.index.max()],
                    y=[var_value, var_value],
                    mode='lines',
                    name=f'{confidence_level:.0%} VaR',
                    line=dict(color='green', width=2, dash='dash')
                ),
                secondary_y=False
            )
            
            fig.add_trace(
                go.Scatter(
                    x=[filtered_returns.index.min(), filtered_returns.index.max()],
                    y=[-var_value, -var_value],
                    mode='lines',
                    name=f'{confidence_level:.0%} VaR (positive)',
                    line=dict(color='green', width=2, dash='dash')
                ),
                secondary_y=False
            )
        
        # Update axes labels
        fig.update_xaxes(title_text="Date")
        fig.update_yaxes(title_text="Daily Return", secondary_y=False, tickformat=',.1%')
        fig.update_yaxes(title_text="Annualized Volatility", secondary_y=True, tickformat=',.1%')
        
        fig.update_layout(
            title="Daily Returns and Rolling Volatility",
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            height=500,
            margin=dict(l=20, r=20, t=50, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Extreme value analysis
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
            
            # Display extreme events
            if not extremes.empty:
                extremes['Normalized'] = (extremes['Return'] - mean) / std
                extremes_display = extremes.reset_index()
                extremes_display['Return'] = extremes_display['Return'].apply(lambda x: f"{x:.2%}")
                extremes_display['Normalized'] = extremes_display['Normalized'].apply(lambda x: f"{x:.2f}σ")
                
                st.markdown(f"**Extreme Events (Beyond ±3σ): {len(extremes)} events**")
                
                st.dataframe(
                    extremes_display[['Date', 'Return', 'Normalized', 'Type']],
                    use_container_width=True,
                    height=300
                )
            else:
                st.info("No extreme events (beyond ±3σ) found in the selected date range.")
        
        with col2:
            # Create a chart showing the distribution of extreme events
            if not extremes.empty:
                # Distribution by type
                type_counts = extremes['Type'].value_counts().reset_index()
                type_counts.columns = ['Type', 'Count']
                
                # Add colors
                type_counts['Color'] = type_counts['Type'].map({
                    'Positive': '#10B981',  # Green
                    'Negative': '#DC2626'   # Red
                })
                
                fig = px.bar(
                    type_counts,
                    x='Type',
                    y='Count',
                    color='Type',
                    color_discrete_map={
                        'Positive': '#10B981',
                        'Negative': '#DC2626'
                    },
                    text_auto='.2s',
                    title="Distribution of Extreme Events"
                )
                
                fig.update_layout(
                    height=300,
                    margin=dict(l=20, r=20, t=50, b=20)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Extreme events statistics
                negative = extremes[extremes['Type'] == 'Negative']['Return']
                positive = extremes[extremes['Type'] == 'Positive']['Return']
                
                st.markdown(f"""
                **Extreme Event Statistics:**
                - Negative extremes avg: {negative.mean():.2%} ({len(negative)} events)
                - Positive extremes avg: {positive.mean():.2%} ({len(positive)} events)
                - Ratio (neg/pos): {len(negative)/len(positive) if len(positive) > 0 else 'N/A'}
                """)
            else:
                st.info("No extreme events to analyze.")
        
        # Return metrics table
        st.markdown("<h2 class='sub-header'>Return Statistics Summary</h2>", unsafe_allow_html=True)
        
        # Calculate return metrics
        cum_return = (1 + filtered_returns['Return']).prod() - 1
        ann_return = (1 + filtered_returns['Return'].mean()) ** 252 - 1
        ann_vol = filtered_returns['Return'].std() * np.sqrt(252)
        sharpe = ann_return / ann_vol if ann_vol != 0 else 0
        
        # Calculate maximum drawdown
        cum_returns = (1 + filtered_returns['Return']).cumprod()
        running_max = cum_returns.cummax()
        drawdown = (cum_returns / running_max) - 1
        max_drawdown = drawdown.min()
        
        # Calculate win rate
        positive_days = (filtered_returns['Return'] > 0).sum()
        win_rate = positive_days / len(filtered_returns)
        
        # Create metrics table
        metrics = pd.DataFrame({
            'Metric': [
                'Total Return',
                'Annualized Return',
                'Annualized Volatility',
                'Sharpe Ratio',
                'Maximum Drawdown',
                'Win Rate',
                'Best Day',
                'Worst Day',
                'Skewness',
                'Excess Kurtosis'
            ],
            'Value': [
                cum_return,
                ann_return,
                ann_vol,
                sharpe,
                max_drawdown,
                win_rate,
                filtered_returns['Return'].max(),
                filtered_returns['Return'].min(),
                skewness,
                kurtosis
            ]
        })
        
        # Format values
        metrics['Formatted'] = metrics['Value'].apply(lambda x: f"{x:.2%}" if isinstance(x, (int, float)) and not pd.isna(x) else "N/A")
        metrics.loc[metrics['Metric'] == 'Sharpe Ratio', 'Formatted'] = metrics.loc[metrics['Metric'] == 'Sharpe Ratio', 'Value'].apply(lambda x: f"{x:.2f}" if not pd.isna(x) else "N/A")
        metrics.loc[metrics['Metric'].isin(['Skewness', 'Excess Kurtosis']), 'Formatted'] = metrics.loc[metrics['Metric'].isin(['Skewness', 'Excess Kurtosis']), 'Value'].apply(lambda x: f"{x:.4f}" if not pd.isna(x) else "N/A")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Display first half of metrics
            half = len(metrics) // 2
            st.dataframe(
                metrics.iloc[:half][['Metric', 'Formatted']].set_index('Metric'),
                use_container_width=True,
                height=200
            )
        
        with col2:
            # Display second half of metrics
            st.dataframe(
                metrics.iloc[half:][['Metric', 'Formatted']].set_index('Metric'),
                use_container_width=True,
                height=200
            )
        
        # Add explanation about key metrics
        with st.expander("Learn more about these metrics"):
            st.markdown("""
            **Total Return**: The cumulative return over the entire period.
            
            **Annualized Return**: The return expressed as an annual rate.
            
            **Annualized Volatility**: The standard deviation of returns expressed as an annual rate, a measure of risk.
            
            **Sharpe Ratio**: The excess return per unit of risk. Higher is better. A Sharpe ratio > 1 is generally considered good.
            
            **Maximum Drawdown**: The largest peak-to-trough decline in portfolio value. Smaller (less negative) is better.
            
            **Win Rate**: The percentage of days with positive returns.
            
            **Skewness**: Measures asymmetry in the return distribution. Negative skewness indicates a longer left tail (more extreme losses).
            
            **Excess Kurtosis**: Measures the "fatness" of distribution tails. Higher kurtosis indicates more frequent extreme values.
            """)
    else:
        st.warning("No return data available for the selected date range.")

# Footer
st.markdown("""
<div style="text-align: center; margin-top: 1rem; padding-top: 1rem; border-top: 1px solid #eee; color: #6B7280; font-size: 0.8rem;">
    Risk Management Dashboard v1.0.0 | Data updated: March 28, 2025 | Created with Streamlit
</div>
""", unsafe_allow_html=True)