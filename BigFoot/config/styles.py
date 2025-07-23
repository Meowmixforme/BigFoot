"""
Dark theme styling with blue accents and proper chart sizing
"""

import streamlit as st
from config.settings import COLOR_PALETTE

def apply_custom_styles():
    """Apply dark theme CSS styles with improved chart layouts"""
    
    css = f"""
    <style>
        /* Import professional font */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        /* Global dark theme */
        .stApp {{
            background-color: {COLOR_PALETTE['background']} !important;
            color: {COLOR_PALETTE['primary']} !important;
        }}
        
        .main .block-container {{
            background-color: {COLOR_PALETTE['background']} !important;
            color: {COLOR_PALETTE['primary']} !important;
            font-family: 'Inter', sans-serif;
            padding-top: 2rem;
            max-width: 1200px;
        }}
        
        /* Override Streamlit's default backgrounds */
        .main {{
            background-color: {COLOR_PALETTE['background']} !important;
        }}
        
        /* Header with dark theme */
        .main-header {{
            font-size: 2.5rem;
            color: {COLOR_PALETTE['dark']};
            text-align: center;
            padding: 1.5rem;
            background: linear-gradient(135deg, {COLOR_PALETTE['light']} 0%, #1A202C 100%);
            border-radius: 12px;
            margin-bottom: 2rem;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
            border: 1px solid #4A5568;
        }}
        
        /* Dark metric cards */
        .metric-card {{
            background: linear-gradient(135deg, {COLOR_PALETTE['light']} 0%, #2D3748 100%);
            padding: 1.5rem;
            border-radius: 12px;
            border: 1px solid #4A5568;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
            margin-bottom: 1rem;
            transition: all 0.2s ease;
            border-left: 4px solid {COLOR_PALETTE['accent']};
            color: {COLOR_PALETTE['primary']} !important;
        }}
        
        .metric-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
            border-left-color: {COLOR_PALETTE['success']};
        }}
        
        /* Dark sidebar */
        .sidebar-header {{
            color: {COLOR_PALETTE['dark']};
            font-size: 1.1rem;
            font-weight: 600;
            margin-bottom: 1rem;
            padding: 0.75rem;
            background: linear-gradient(135deg, #1A202C 0%, #2D3748 100%);
            border-radius: 8px;
            border: 1px solid #4A5568;
        }}
        
        .css-1d391kg {{
            background-color: #1A202C !important;
            border-right: 1px solid #4A5568;
        }}
        
        /* Dark info boxes */
        .info-box {{
            background: linear-gradient(135deg, {COLOR_PALETTE['light']} 0%, #2D3748 100%);
            border-left: 4px solid {COLOR_PALETTE['info']};
            padding: 1.25rem;
            margin: 1rem 0;
            border-radius: 8px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
            border: 1px solid #4A5568;
            color: {COLOR_PALETTE['primary']} !important;
        }}
        
        .success-box {{
            background: linear-gradient(135deg, {COLOR_PALETTE['light']} 0%, #2D3748 100%);
            border-left: 4px solid {COLOR_PALETTE['success']};
            padding: 1.25rem;
            margin: 1rem 0;
            border-radius: 8px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
            border: 1px solid #4A5568;
            color: {COLOR_PALETTE['primary']} !important;
        }}
        
        .warning-box {{
            background: linear-gradient(135deg, {COLOR_PALETTE['light']} 0%, #2D3748 100%);
            border-left: 4px solid {COLOR_PALETTE['warning']};
            padding: 1.25rem;
            margin: 1rem 0;
            border-radius: 8px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
            border: 1px solid #4A5568;
            color: {COLOR_PALETTE['primary']} !important;
        }}
        
        .attribution-box {{
            background: linear-gradient(135deg, {COLOR_PALETTE['light']} 0%, #2D3748 100%);
            border: 1px solid {COLOR_PALETTE['warning']};
            border-radius: 8px;
            padding: 1rem;
            margin: 1rem 0;
            font-size: 0.9rem;
            color: {COLOR_PALETTE['primary']} !important;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        }}
        
        /* Dark footer */
        .footer {{
            margin-top: 3rem;
            padding: 2rem;
            background: linear-gradient(135deg, {COLOR_PALETTE['light']} 0%, #1A202C 100%);
            border-radius: 12px;
            text-align: center;
            color: {COLOR_PALETTE['primary']} !important;
            border-top: 3px solid {COLOR_PALETTE['accent']};
            box-shadow: 0 -4px 20px rgba(0, 0, 0, 0.3);
        }}
        
        /* Dark tabs */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 8px;
            background-color: {COLOR_PALETTE['light']};
            border-radius: 8px;
            padding: 4px;
        }}
        
        .stTabs [data-baseweb="tab"] {{
            background-color: transparent;
            border-radius: 6px;
            color: {COLOR_PALETTE['primary']} !important;
            font-weight: 500;
        }}
        
        .stTabs [aria-selected="true"] {{
            background-color: {COLOR_PALETTE['accent']} !important;
            color: white !important;
        }}
        
        /* Bright buttons for dark theme */
        .stButton > button {{
            background: linear-gradient(135deg, {COLOR_PALETTE['accent']} 0%, #3182CE 100%);
            color: white !important;
            border: none;
            border-radius: 8px;
            padding: 0.6rem 1.2rem;
            font-weight: 500;
            transition: all 0.2s ease;
            box-shadow: 0 4px 15px rgba(99, 179, 237, 0.3);
        }}
        
        .stButton > button:hover {{
            background: linear-gradient(135deg, {COLOR_PALETTE['success']} 0%, #38A169 100%);
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(104, 211, 145, 0.4);
        }}
        
        /* Dark form elements */
        .stSelectbox > div > div {{
            background-color: {COLOR_PALETTE['light']} !important;
            border: 1px solid #4A5568 !important;
            border-radius: 8px;
            color: {COLOR_PALETTE['primary']} !important;
        }}
        
        .stMultiSelect > div > div {{
            background-color: {COLOR_PALETTE['light']} !important;
            border: 1px solid #4A5568 !important;
            border-radius: 8px;
            color: {COLOR_PALETTE['primary']} !important;
        }}
        
        .stSlider > div > div > div {{
            background: linear-gradient(90deg, {COLOR_PALETTE['accent']} 0%, #3182CE 100%);
        }}
        
        .streamlit-expanderHeader {{
            background-color: {COLOR_PALETTE['light']} !important;
            border-radius: 8px;
            border: 1px solid #4A5568;
            color: {COLOR_PALETTE['primary']} !important;
        }}
        
        /* Force text colors for dark theme */
        h1, h2, h3, h4, h5, h6 {{
            color: {COLOR_PALETTE['dark']} !important;
        }}
        
        p, div, span {{
            color: {COLOR_PALETTE['primary']} !important;
        }}
        
        .metric-label {{
            color: {COLOR_PALETTE['secondary']} !important;
            font-size: 0.9rem;
        }}
        
        /* FIXED: Chart backgrounds and sizing for dark theme */
        .js-plotly-plot {{
            background-color: {COLOR_PALETTE['light']} !important;
            border-radius: 8px;
            border: 1px solid #4A5568;
            margin: 1rem 0;
        }}
        
        /* Ensure charts take full width and proper height */
        .stPlotlyChart {{
            width: 100% !important;
            min-height: 400px !important;
        }}
        
        .stPlotlyChart > div {{
            width: 100% !important;
            height: auto !important;
            min-height: 400px !important;
        }}
        
        /* Column layout improvements for charts */
        .stColumn {{
            padding: 0 0.5rem;
        }}
        
        .stColumn > div {{
            width: 100%;
        }}
        
        /* Sidebar text colors */
        .css-1d391kg .stMarkdown {{
            color: {COLOR_PALETTE['primary']} !important;
        }}
        
        /* Input field styling */
        .stTextInput > div > div > input {{
            background-color: {COLOR_PALETTE['light']} !important;
            color: {COLOR_PALETTE['primary']} !important;
            border: 1px solid #4A5568 !important;
        }}
        
        /* Number input styling */
        .stNumberInput > div > div > input {{
            background-color: {COLOR_PALETTE['light']} !important;
            color: {COLOR_PALETTE['primary']} !important;
            border: 1px solid #4A5568 !important;
        }}
        
        /* Fix column spacing */
        .row-widget.stHorizontal {{
            gap: 1rem;
        }}
        
        /* Ensure proper container width */
        .block-container {{
            max-width: none !important;
            padding-left: 2rem;
            padding-right: 2rem;
        }}
    </style>
    """
    
    st.markdown(css, unsafe_allow_html=True)

def get_custom_css():
    """Alternative function that returns CSS string (for compatibility)"""
    return apply_custom_styles()