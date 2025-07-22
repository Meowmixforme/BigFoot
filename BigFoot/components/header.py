"""
Header component for the BigFoot dashboard
"""

import streamlit as st
from config.settings import DATA_CONFIG

def render_header():
    """Render the main header with attribution"""
    
    # Main header
    st.markdown('<div class="main-header">🦶 Bigfoot Sightings ML Dashboard</div>', unsafe_allow_html=True)
    
    # Attribution and info box
    st.markdown(f"""
    <div class="attribution-box">
        <strong>📊 Data Source:</strong> {DATA_CONFIG['attribution']}<br>
        <strong>🌐 Database:</strong> <a href="{DATA_CONFIG['bfro_url']}" target="_blank">{DATA_CONFIG['bfro_url']}</a><br>
        <strong>⚖️ Usage:</strong> Educational/Research purposes under Fair Use<br>
        <strong>👤 Created by:</strong> Meowmixforme | <strong>📅 Date:</strong> 2025-07-22
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### *Comprehensive Machine Learning Analysis of Sasquatch Sighting Reports*")
    st.markdown("---")