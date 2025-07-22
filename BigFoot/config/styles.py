"""
Custom CSS styles for the BigFoot dashboard
"""

import streamlit as st
from .settings import COLOR_PALETTE

def apply_custom_styles():
    """Apply custom CSS styles to the Streamlit app"""
    
    css = f"""
    <style>
        /* Import professional font */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        /* Global styles */
        .main {{
            font-family: 'Inter', sans-serif;
        }}
        
        /* Header styles */
        .main-header {{
            font-size: 2.5rem;
            color: {COLOR_PALETTE['primary']};
            text-align: center;
            padding: 1.5rem;
            background: linear-gradient(135deg, {COLOR_PALETTE['light']}, #FFFFFF);
            border-radius: 12px;
            margin-bottom: 2rem;
            box-shadow: 0 2px 10px rgba(44, 62, 80, 0.1);
            border: 1px solid {COLOR_PALETTE['light']};
        }}
        
        /* Metric cards */
        .metric-card {{
            background: #FFFFFF;
            padding: 1.5rem;
            border-radius: 10px;
            border-left: 4px solid {COLOR_PALETTE['accent']};
            margin-bottom: 1rem;
            box-shadow: 0 2px 8px rgba(52, 73, 94, 0.08);
            transition: transform 0.2s ease;
        }}
        
        .metric-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(52, 73, 94, 0.12);
        }}
        
        /* Sidebar styles */
        .sidebar-header {{
            color: {COLOR_PALETTE['primary']};
            font-size: 1.1rem;
            font-weight: 600;
            margin-bottom: 1rem;
            padding: 0.75rem;
            background: {COLOR_PALETTE['light']};
            border-radius: 8px;
            border: 1px solid #DEE2E6;
        }}
        
        /* Attribution box */
        .attribution-box {{
            background: #FFFFFF;
            border: 1px solid {COLOR_PALETTE['warning']};
            border-radius: 10px;
            padding: 1.25rem;
            margin: 1rem 0;
            font-size: 0.9rem;
            color: {COLOR_PALETTE['secondary']};
            box-shadow: 0 2px 6px rgba(243, 156, 18, 0.1);
        }}
        
        /* Info boxes */
        .info-box {{
            background: #FFFFFF;
            border-left: 4px solid {COLOR_PALETTE['info']};
            padding: 1.25rem;
            margin: 1rem 0;
            border-radius: 8px;
            box-shadow: 0 2px 6px rgba(93, 173, 226, 0.1);
        }}
        
        .success-box {{
            background: #FFFFFF;
            border-left: 4px solid {COLOR_PALETTE['success']};
            padding: 1.25rem;
            margin: 1rem 0;
            border-radius: 8px;
            box-shadow: 0 2px 6px rgba(39, 174, 96, 0.1);
        }}
        
        .warning-box {{
            background: #FFFFFF;
            border-left: 4px solid {COLOR_PALETTE['warning']};
            padding: 1.25rem;
            margin: 1rem 0;
            border-radius: 8px;
            box-shadow: 0 2px 6px rgba(243, 156, 18, 0.1);
        }}
        
        /* Footer styles */
        .footer {{
            margin-top: 3rem;
            padding: 2rem;
            background: {COLOR_PALETTE['light']};
            border-radius: 12px;
            text-align: center;
            color: {COLOR_PALETTE['secondary']};
            border-top: 3px solid {COLOR_PALETTE['accent']};
            box-shadow: 0 -2px 10px rgba(44, 62, 80, 0.05);
        }}
        
        /* Tabs styling */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 8px;
        }}
        
        .stTabs [data-baseweb="tab"] {{
            background-color: {COLOR_PALETTE['light']};
            border-radius: 8px;
            color: {COLOR_PALETTE['secondary']};
            font-weight: 500;
        }}
        
        .stTabs [aria-selected="true"] {{
            background-color: {COLOR_PALETTE['accent']};
            color: white;
        }}
        
        /* Sidebar styling */
        .css-1d391kg {{
            background-color: {COLOR_PALETTE['background']};
        }}
        
        /* Button styling */
        .stButton > button {{
            background-color: {COLOR_PALETTE['accent']};
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.5rem 1rem;
            font-weight: 500;
            transition: all 0.2s ease;
        }}
        
        .stButton > button:hover {{
            background-color: {COLOR_PALETTE['primary']};
            transform: translateY(-1px);
        }}
        
        /* Select box styling */
        .stSelectbox > div > div {{
            background-color: white;
            border: 1px solid {COLOR_PALETTE['light']};
            border-radius: 8px;
        }}
        
        /* Slider styling */
        .stSlider > div > div > div {{
            background-color: {COLOR_PALETTE['accent']};
        }}
        
        /* Expander styling */
        .streamlit-expanderHeader {{
            background-color: {COLOR_PALETTE['light']};
            border-radius: 8px;
            border: 1px solid #DEE2E6;
        }}
        
        /* Chart styling */
        .js-plotly-plot {{
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(44, 62, 80, 0.05);
        }}
    </style>
    """
    
    st.markdown(css, unsafe_allow_html=True)