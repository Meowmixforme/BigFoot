
# BIGFOOT ML DASHBOARD - MAIN ENTRY POINT

# Author: Meowmixforme
# Date: 2025-07-22
# Description: Interactive dashboard for Bigfoot sightings analysis using BFRO data

import streamlit as st
from config.styles import apply_custom_styles
from config.settings import PAGE_CONFIG
from components.header import render_header
from components.sidebar import render_sidebar
from components.footer import render_footer
from data.loader import load_and_prepare_data
from views import overview, geographic, temporal, machine_learning, anomaly, recommendations, explorer, about

# Page configuration
st.set_page_config(**PAGE_CONFIG)

# Apply custom styles
apply_custom_styles()

def main():
    """Main Streamlit application"""
    
    # Render header
    render_header()
    
    # Load data
    df = load_and_prepare_data()
    
    if df is None:
        st.stop()
    
    # Render sidebar and get selections
    sidebar_data = render_sidebar(df)
    
    # Apply filters
    filtered_df = df[
        (df['year'] >= sidebar_data['year_range'][0]) & 
        (df['year'] <= sidebar_data['year_range'][1]) &
        (df['classification'].isin(sidebar_data['classifications'])) &
        (df['season'].isin(sidebar_data['seasons']))
    ]
    
    # Main content router
    analysis_type = sidebar_data['analysis_type']
    
    if analysis_type == "📊 Overview Dashboard":
        overview.show_overview_dashboard(filtered_df)
    elif analysis_type == "🗺️ Geographic Analysis":
        geographic.show_geographic_analysis(filtered_df)
    elif analysis_type == "📅 Temporal Patterns":
        temporal.show_temporal_analysis(filtered_df)
    elif analysis_type == "🤖 Machine Learning":
        machine_learning.show_ml_analysis(filtered_df)
    elif analysis_type == "🔍 Anomaly Detection":
        anomaly.show_anomaly_analysis(filtered_df)
    elif analysis_type == "🎯 Location Recommendations":
        recommendations.show_recommendations(filtered_df)
    elif analysis_type == "📝 Data Explorer":
        explorer.show_data_explorer(filtered_df)
    elif analysis_type == "ℹ️ About This Project":
        about.show_about_section()
    
    # Render footer
    render_footer()

if __name__ == "__main__":
    main()
