"""
Custom styles for the BigFoot dashboard
"""

import streamlit as st
from config.settings import COLOR_PALETTE

def apply_custom_styles():
    """Apply custom CSS styles to the Streamlit app"""
    
    st.markdown(f"""
    <style>
        /* Main app background - Professional brown */
        .stApp {{
            background-color: #2D2B2A;
        }}
        
        /* Main content area */
        .main .block-container {{
            background-color: #2D2B2A;
            padding-top: 2rem;
            padding-bottom: 2rem;
        }}
        
        /* Header styling with brown theme */
        .main-header {{
            background: linear-gradient(135deg, #3E3B39 0%, #2D2B2A 100%);
            padding: 2rem;
            border-radius: 15px;
            margin-bottom: 2rem;
            text-align: center;
            color: white;
            box-shadow: 0 4px 15px rgba(45, 43, 42, 0.3);
            border: 1px solid #4A4642;
        }}
        
        .main-header h1 {{
            color: white;
            margin: 0;
            font-size: 2.5rem;
            font-weight: 700;
        }}
        
        .main-header p {{
            color: #E8E6E3;
            margin: 0.5rem 0 0 0;
            font-size: 1.1rem;
        }}
        
        /* Sidebar header - darker brown */
        .sidebar-header {{
            background: linear-gradient(135deg, #1F1D1C 0%, #161514 100%);
            color: white;
            padding: 1rem;
            border-radius: 10px;
            text-align: center;
            font-weight: 600;
            margin-bottom: 1rem;
            border: 1px solid #2D2B2A;
        }}
        
        /* Metric cards with brown theme */
        .metric-card {{
            background: #3E3B39;
            padding: 1.5rem;
            border-radius: 12px;
            text-align: center;
            margin: 0.5rem 0;
            box-shadow: 0 3px 10px rgba(45, 43, 42, 0.2);
            border: 1px solid #4A4642;
        }}
        
        .metric-card h1, .metric-card h2, .metric-card h3 {{
            color: white;
        }}
        
        .metric-card p {{
            color: #C7C4C1;
        }}
        
        /* Attribution box - darker for sidebar */
        .attribution-box {{
            background: #1F1D1C;
            border-left: 4px solid {COLOR_PALETTE['accent']};
            padding: 1.25rem;
            margin: 1rem 0;
            border-radius: 8px;
            box-shadow: 0 2px 6px rgba(31, 29, 28, 0.3);
            color: #E8E6E3;
            border: 1px solid #2D2B2A;
        }}
        
        /* Warning boxes with brown theme */
        .warning-box {{
            background: #3E3B39;
            border-left: 4px solid {COLOR_PALETTE['warning']};
            padding: 1.25rem;
            margin: 1rem 0;
            border-radius: 8px;
            box-shadow: 0 2px 6px rgba(243, 156, 18, 0.1);
            color: #E8E6E3;
            border: 1px solid #4A4642;
        }}
        
        /* Info boxes with brown theme */
        .info-box {{
            background: #3E3B39;
            border-left: 4px solid {COLOR_PALETTE['info']};
            padding: 1.25rem;
            margin: 1rem 0;
            border-radius: 8px;
            box-shadow: 0 2px 6px rgba(93, 173, 226, 0.1);
            color: #E8E6E3;
            border: 1px solid #4A4642;
        }}
        
        .success-box {{
            background: #1F1D1C;
            border-left: 4px solid {COLOR_PALETTE['success']};
            padding: 1.25rem;
            margin: 1rem 0;
            border-radius: 8px;
            box-shadow: 0 2px 6px rgba(39, 174, 96, 0.1);
            color: #E8E6E3;
            border: 1px solid #2D2B2A;
        }}
        
        /* Footer styles with brown theme */
        .footer {{
            margin-top: 3rem;
            padding: 2rem;
            background: #3E3B39;
            border-radius: 12px;
            text-align: center;
            color: #C7C4C1;
            border-top: 3px solid {COLOR_PALETTE['accent']};
            box-shadow: 0 -2px 10px rgba(45, 43, 42, 0.2);
            border: 1px solid #4A4642;
        }}
        
        /* Tabs styling with brown theme */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 8px;
        }}
        
        .stTabs [data-baseweb="tab"] {{
            background-color: #3E3B39;
            border-radius: 8px;
            color: #C7C4C1;
            font-weight: 500;
            border: 1px solid #4A4642;
        }}
        
        .stTabs [aria-selected="true"] {{
            background-color: {COLOR_PALETTE['accent']};
            color: white;
        }}
        
        /* Sidebar styling - much darker brown */
        .css-1d391kg {{
            background-color: #161514 !important;
        }}
        
        /* Additional sidebar selectors */
        section[data-testid="stSidebar"] {{
            background-color: #161514 !important;
        }}
        
        .css-1d391kg > div {{
            background-color: #161514 !important;
        }}
        
        /* Sidebar content styling */
        .css-1d391kg .stSelectbox > div > div {{
            background-color: #1F1D1C;
            border: 1px solid #2D2B2A;
            color: #E8E6E3;
        }}
        
        .css-1d391kg .stMultiSelect > div > div {{
            background-color: #1F1D1C;
            border: 1px solid #2D2B2A;
            color: #E8E6E3;
        }}
        
        .css-1d391kg .stSlider {{
            color: #E8E6E3;
        }}
        
        /* Sidebar text colors */
        .css-1d391kg .stMarkdown {{
            color: #E8E6E3;
        }}
        
        .css-1d391kg label {{
            color: #E8E6E3 !important;
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
        
        /* Select box styling with brown theme */
        .stSelectbox > div > div {{
            background-color: #3E3B39;
            border: 1px solid #4A4642;
            border-radius: 8px;
            color: #E8E6E3;
        }}
        
        /* Slider styling */
        .stSlider > div > div > div {{
            background-color: {COLOR_PALETTE['accent']};
        }}
        
        /* Expander styling with brown theme */
        .streamlit-expanderHeader {{
            background-color: #3E3B39;
            border-radius: 8px;
            border: 1px solid #4A4642;
            color: #E8E6E3;
        }}
        
        /* Text color adjustments */
        .stMarkdown, .stText {{
            color: #E8E6E3;
        }}
        
        /* Charts background */
        .js-plotly-plot .plotly {{
            background-color: #3E3B39 !important;
        }}
        
        /* Streamlit widgets text color */
        .stSelectbox label, .stSlider label, .stMultiSelect label {{
            color: #E8E6E3 !important;
        }}
        
        /* Input fields */
        .stTextInput > div > div > input {{
            background-color: #3E3B39;
            color: #E8E6E3;
            border: 1px solid #4A4642;
        }}
        
        /* Number input */
        .stNumberInput > div > div > input {{
            background-color: #3E3B39;
            color: #E8E6E3;
            border: 1px solid #4A4642;
        }}
    </style>
    """, unsafe_allow_html=True)