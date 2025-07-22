"""
Sidebar component for the BigFoot dashboard
"""

import streamlit as st
from config.settings import ANALYSIS_TYPES, DATA_CONFIG

def render_sidebar(df):
    """Render sidebar controls and return user selections"""
    
    st.sidebar.markdown('<div class="sidebar-header">🔧 Dashboard Controls</div>', unsafe_allow_html=True)
    
    # BFRO Attribution in sidebar
    st.sidebar.markdown(f"""
    <div class="attribution-box">
        <strong>🦶 BFRO Data</strong><br>
        {DATA_CONFIG['attribution']}<br>
        <small>Used for educational research</small>
    </div>
    """, unsafe_allow_html=True)
    
    # Analysis selection
    analysis_type = st.sidebar.selectbox(
        "🎯 Choose Analysis Type:",
        ANALYSIS_TYPES
    )
    
    # Data filters
    st.sidebar.markdown("### 🔍 Data Filters")
    
    # Year filter
    year_range = st.sidebar.slider(
        "Select Year Range:",
        min_value=int(df['year'].min()),
        max_value=int(df['year'].max()),
        value=(int(df['year'].min()), int(df['year'].max())),
        help="Filter sightings by year range"
    )
    
    # Classification filter
    classifications = st.sidebar.multiselect(
        "Classification Types:",
        options=sorted(df['classification'].unique()),
        default=sorted(df['classification'].unique()),
        help="Class A: Clear visual, Class B: Possible visual, Class C: Second-hand"
    )
    
    # Season filter
    seasons = st.sidebar.multiselect(
        "Seasons:",
        options=['Winter', 'Spring', 'Summer', 'Fall'],
        default=['Winter', 'Spring', 'Summer', 'Fall']
    )
    
    # Filter summary
    filtered_count = len(df[
        (df['year'] >= year_range[0]) & 
        (df['year'] <= year_range[1]) &
        (df['classification'].isin(classifications)) &
        (df['season'].isin(seasons))
    ])
    
    st.sidebar.markdown(f"""
    <div class="success-box">
        <strong>📊 Filtered Data:</strong><br>
        {filtered_count:,} of {len(df):,} records<br>
        <small>({filtered_count/len(df)*100:.1f}% of total)</small>
    </div>
    """, unsafe_allow_html=True)
    
    return {
        'analysis_type': analysis_type,
        'year_range': year_range,
        'classifications': classifications,
        'seasons': seasons
    }