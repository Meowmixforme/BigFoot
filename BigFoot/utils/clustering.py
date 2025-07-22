"""
Clustering algorithms for geographic analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

@st.cache_data
def perform_clustering(df):
    """Perform geographic clustering analysis"""
    try:
        # Prepare features for clustering
        features = df[['latitude', 'longitude', 'distance_from_center']].copy()
        
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # K-means clustering
        kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
        df_clustered = df.copy()
        df_clustered['cluster'] = kmeans.fit_predict(features_scaled)
        
        # Add cluster statistics
        cluster_stats = df_clustered.groupby('cluster').agg({
            'latitude': ['mean', 'std', 'count'],
            'longitude': ['mean', 'std'],
            'year': ['min', 'max'],
            'classification': lambda x: (x == 'Class A').mean()
        }).round(3)
        
        return df_clustered, kmeans, cluster_stats
        
    except Exception as e:
        st.error(f"❌ Error in clustering: {e}")
        return df, None, None