"""
Data loading and preprocessing utilities
"""

import streamlit as st
import pandas as pd
import numpy as np
from config.settings import DATA_CONFIG, CLASSIFICATION_MAPPING

@st.cache_data
def load_and_prepare_data():
    """Load and prepare the BFRO dataset with comprehensive cleaning"""
    try:
        # Display loading status
        with st.spinner("Loading BFRO Bigfoot data... 🦶"):
            # Load the dataset
            df = pd.read_csv(DATA_CONFIG["source_url"])
            
            # Initial data info
            st.sidebar.success(f"📊 Raw data loaded: {len(df):,} records")
            
            # Clean and prepare the data
            original_count = len(df)
            
            # Remove rows with missing critical data
            df = df.dropna(subset=['latitude', 'longitude', 'timestamp'])
            
            # Validate coordinate ranges
            df = df[
                (df['latitude'] >= -90) & (df['latitude'] <= 90) &
                (df['longitude'] >= -180) & (df['longitude'] <= 180)
            ]
            
            # Parse timestamps safely
            def safe_parse_date(timestamp_str):
                try:
                    for fmt in ['%Y-%m-%d', '%m/%d/%Y', '%Y-%m-%d %H:%M:%S', '%m/%d/%y']:
                        try:
                            return pd.to_datetime(timestamp_str, format=fmt)
                        except:
                            continue
                    return pd.to_datetime(timestamp_str, errors='coerce')
                except:
                    return pd.NaT
            
            df['parsed_date'] = df['timestamp'].apply(safe_parse_date)
            df = df.dropna(subset=['parsed_date'])
            
            # Extract date components
            df['year'] = df['parsed_date'].dt.year
            df['month'] = df['parsed_date'].dt.month
            df['day_of_week'] = df['parsed_date'].dt.dayofweek
            df['day_of_year'] = df['parsed_date'].dt.dayofyear
            
            # Add season
            df['season'] = df['month'].apply(lambda x: 
                'Winter' if x in [12,1,2] else
                'Spring' if x in [3,4,5] else
                'Summer' if x in [6,7,8] else 'Fall')
            
            # Filter reasonable years (1850-2024)
            df = df[(df['year'] >= 1850) & (df['year'] <= 2024)]
            
            # Clean classification column
            df['classification'] = df['classification'].map(CLASSIFICATION_MAPPING).fillna('Class C')
            
            # Add derived features
            df['distance_from_center'] = np.sqrt(
                (df['latitude'] - df['latitude'].mean())**2 + 
                (df['longitude'] - df['longitude'].mean())**2
            )
            
            # Calculate data retention
            retention_rate = len(df) / original_count * 100
            st.sidebar.info(f"✅ Data cleaned: {len(df):,} valid records ({retention_rate:.1f}% retained)")
            
            return df
            
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        st.info("Please check your internet connection or try again later.")
        return None