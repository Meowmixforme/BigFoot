"""
Configuration settings for the BigFoot dashboard
"""

# Page configuration
PAGE_CONFIG = {
    "page_title": "🦶 Bigfoot Sightings ML Dashboard",
    "page_icon": "🦶",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# Data source configuration
DATA_CONFIG = {
    "source_url": "https://raw.githubusercontent.com/Meowmixforme/BigFoot/refs/heads/main/timothyrenner-bfro-sightings-data/bfro_locations.csv",
    "bfro_url": "http://bfro.net/GDB/",
    "attribution": "Bigfoot Field Researchers Organization (BFRO)"
}

# Color palette - Professional theme
COLOR_PALETTE = {
    "primary": "#2C3E50",        # Dark blue-gray
    "secondary": "#34495E",      # Medium blue-gray
    "accent": "#3498DB",         # Professional blue
    "success": "#27AE60",        # Muted green
    "warning": "#F39C12",        # Professional orange
    "danger": "#E74C3C",         # Muted red
    "light": "#ECF0F1",          # Light gray
    "dark": "#2C3E50",           # Dark
    "info": "#5DADE2",           # Light blue
    "background": "#FAFBFC"      # Very light background
}

# Analysis types
ANALYSIS_TYPES = [
    "📊 Overview Dashboard", 
    "🗺️ Geographic Analysis", 
    "📅 Temporal Patterns", 
    "🤖 Machine Learning", 
    "🔍 Anomaly Detection", 
    "🎯 Location Recommendations", 
    "📝 Data Explorer", 
    "ℹ️ About This Project"
]

# Classification mappings
CLASSIFICATION_MAPPING = {
    'Class A': 'Class A', 'Class B': 'Class B', 'Class C': 'Class C',
    'class a': 'Class A', 'class b': 'Class B', 'class c': 'Class C',
    'A': 'Class A', 'B': 'Class B', 'C': 'Class C'
}

# Map configuration
MAP_CONFIG = {
    "default_zoom": 4,
    "cluster_colors": ['#E74C3C', '#3498DB', '#27AE60', '#9B59B6', '#F39C12'],
    "classification_colors": {
        'Class A': '#E74C3C',    # Red - Clear visual sighting (highest quality)
        'Class B': '#3498DB',    # Blue - Sounds/tracks/blurry (medium quality) 
        'Class C': '#27AE60'     # Green - Second-hand reports (lowest quality)
    }
}
