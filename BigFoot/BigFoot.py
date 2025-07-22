# ========================================
# BIGFOOT ML DASHBOARD - COMPLETE STREAMLIT APP
# ========================================
# Author: Meowmixforme
# Date: 2025-07-22
# Description: Interactive dashboard for Bigfoot sightings analysis using BFRO data

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="🦶 Bigfoot Sightings ML Dashboard",
    page_icon="🦶",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E7D32;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #E8F5E8, #C8E6C9);
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: #F1F8E9;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #4CAF50;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .sidebar-header {
        color: #1B5E20;
        font-size: 1.2rem;
        font-weight: bold;
        margin-bottom: 1rem;
        padding: 0.5rem;
        background: #E8F5E8;
        border-radius: 5px;
    }
    .attribution-box {
        background: #FFF3E0;
        border: 1px solid #FF9800;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        font-size: 0.9rem;
        color: #E65100;
    }
    .info-box {
        background: #E3F2FD;
        border-left: 4px solid #2196F3;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .success-box {
        background: #E8F5E8;
        border-left: 4px solid #4CAF50;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .footer {
        margin-top: 3rem;
        padding: 2rem;
        background: #F5F5F5;
        border-radius: 10px;
        text-align: center;
        color: #666;
        border-top: 3px solid #4CAF50;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_and_prepare_data():
    """Load and prepare the BFRO dataset with comprehensive cleaning"""
    try:
        # Display loading status
        with st.spinner("Loading BFRO Bigfoot data... 🦶"):
            # Load the dataset
            url = "https://raw.githubusercontent.com/kittychew/bigfoot-sightings-analysis/main/bfro_locations.csv"
            df = pd.read_csv(url)
            
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
            classification_mapping = {
                'Class A': 'Class A', 'Class B': 'Class B', 'Class C': 'Class C',
                'class a': 'Class A', 'class b': 'Class B', 'class c': 'Class C',
                'A': 'Class A', 'B': 'Class B', 'C': 'Class C'
            }
            df['classification'] = df['classification'].map(classification_mapping).fillna('Class C')
            
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

@st.cache_data
def detect_anomalies(df):
    """Detect anomalous sightings using Isolation Forest"""
    try:
        # Prepare features for anomaly detection
        features = ['latitude', 'longitude', 'year', 'month', 'distance_from_center']
        X = df[features].copy()
        
        # Handle any remaining missing values
        X = X.fillna(X.mean())
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Isolation Forest
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        anomaly_labels = iso_forest.fit_predict(X_scaled)
        anomaly_scores = iso_forest.score_samples(X_scaled)
        
        df_anomaly = df.copy()
        df_anomaly['is_anomaly'] = anomaly_labels == -1
        df_anomaly['anomaly_score'] = anomaly_scores
        
        return df_anomaly, iso_forest
        
    except Exception as e:
        st.error(f"❌ Error in anomaly detection: {e}")
        return df, None

@st.cache_data
def train_ml_model(df):
    """Train machine learning model for classification prediction"""
    try:
        # Feature engineering
        ml_df = df.copy()
        
        # Create cyclical features
        ml_df['month_sin'] = np.sin(2 * np.pi * ml_df['month'] / 12)
        ml_df['month_cos'] = np.cos(2 * np.pi * ml_df['month'] / 12)
        ml_df['day_sin'] = np.sin(2 * np.pi * ml_df['day_of_year'] / 365)
        ml_df['day_cos'] = np.cos(2 * np.pi * ml_df['day_of_year'] / 365)
        
        # Normalize year
        ml_df['year_normalized'] = (ml_df['year'] - ml_df['year'].min()) / (ml_df['year'].max() - ml_df['year'].min())
        
        # Encode target variable
        le = LabelEncoder()
        ml_df['classification_encoded'] = le.fit_transform(ml_df['classification'])
        
        # Select features
        feature_columns = [
            'latitude', 'longitude', 'year_normalized', 'month', 
            'month_sin', 'month_cos', 'day_sin', 'day_cos',
            'distance_from_center', 'day_of_year'
        ]
        
        X = ml_df[feature_columns]
        y = ml_df['classification_encoded']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train Random Forest model
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        rf_model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = rf_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Feature importance
        importance_df = pd.DataFrame({
            'feature': feature_columns,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return {
            'model': rf_model,
            'accuracy': accuracy,
            'feature_importance': importance_df,
            'label_encoder': le,
            'test_predictions': y_pred,
            'test_actual': y_test,
            'feature_columns': feature_columns
        }
        
    except Exception as e:
        st.error(f"❌ Error in ML model training: {e}")
        return None

def main():
    """Main Streamlit application"""
    
    # Header with attribution
    st.markdown('<div class="main-header">🦶 Bigfoot Sightings ML Dashboard</div>', unsafe_allow_html=True)
    
    # Attribution and info box
    st.markdown("""
    <div class="attribution-box">
        <strong>📊 Data Source:</strong> Bigfoot Field Researchers Organization (BFRO)<br>
        <strong>🌐 Database:</strong> <a href="http://bfro.net/GDB/" target="_blank">bfro.net/GDB/</a><br>
        <strong>⚖️ Usage:</strong> Educational/Research purposes under Fair Use<br>
        <strong>👤 Created by:</strong> Meowmixforme | <strong>📅 Date:</strong> 2025-07-22
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### *Comprehensive Machine Learning Analysis of Sasquatch Sighting Reports*")
    st.markdown("---")
    
    # Load data
    df = load_and_prepare_data()
    
    if df is None:
        st.stop()
    
    # Sidebar
    st.sidebar.markdown('<div class="sidebar-header">🔧 Dashboard Controls</div>', unsafe_allow_html=True)
    
    # BFRO Attribution in sidebar
    st.sidebar.markdown("""
    <div class="attribution-box">
        <strong>🦶 BFRO Data</strong><br>
        Bigfoot Field Researchers<br>
        Organization Database<br>
        <small>Used for educational research</small>
    </div>
    """, unsafe_allow_html=True)
    
    # Analysis selection
    analysis_type = st.sidebar.selectbox(
        "🎯 Choose Analysis Type:",
        ["📊 Overview Dashboard", "🗺️ Geographic Analysis", "📅 Temporal Patterns", 
         "🤖 Machine Learning", "🔍 Anomaly Detection", "🎯 Location Recommendations", 
         "📝 Data Explorer", "ℹ️ About This Project"]
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
    
    # Apply filters
    filtered_df = df[
        (df['year'] >= year_range[0]) & 
        (df['year'] <= year_range[1]) &
        (df['classification'].isin(classifications)) &
        (df['season'].isin(seasons))
    ]
    
    # Filter summary
    st.sidebar.markdown(f"""
    <div class="success-box">
        <strong>📊 Filtered Data:</strong><br>
        {len(filtered_df):,} of {len(df):,} records<br>
        <small>({len(filtered_df)/len(df)*100:.1f}% of total)</small>
    </div>
    """, unsafe_allow_html=True)
    
    # Main content router
    if analysis_type == "📊 Overview Dashboard":
        show_overview_dashboard(filtered_df)
    elif analysis_type == "🗺️ Geographic Analysis":
        show_geographic_analysis(filtered_df)
    elif analysis_type == "📅 Temporal Patterns":
        show_temporal_analysis(filtered_df)
    elif analysis_type == "🤖 Machine Learning":
        show_ml_analysis(filtered_df)
    elif analysis_type == "🔍 Anomaly Detection":
        show_anomaly_analysis(filtered_df)
    elif analysis_type == "🎯 Location Recommendations":
        show_recommendations(filtered_df)
    elif analysis_type == "📝 Data Explorer":
        show_data_explorer(filtered_df)
    elif analysis_type == "ℹ️ About This Project":
        show_about_section()

def show_overview_dashboard(df):
    """Display comprehensive overview dashboard"""
    st.header("📊 Bigfoot Sightings Overview")
    
    # Key metrics row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Sightings", f"{len(df):,}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        years_span = df['year'].max() - df['year'].min()
        st.metric("Years Covered", f"{years_span}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        peak_year = df['year'].value_counts().idxmax()
        peak_count = df['year'].value_counts().max()
        st.metric("Peak Year", f"{peak_year}")
        st.caption(f"{peak_count} sightings")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        class_a_pct = (df['classification'] == 'Class A').mean() * 100
        st.metric("Class A Reports", f"{class_a_pct:.1f}%")
        st.caption("High-quality sightings")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col5:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        lat_span = df['latitude'].max() - df['latitude'].min()
        lon_span = df['longitude'].max() - df['longitude'].min()
        st.metric("Geographic Span", f"{lat_span:.1f}° × {lon_span:.1f}°")
        st.caption("Latitude × Longitude")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Main charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Yearly trend with trend line
        yearly_counts = df.groupby('year').size().reset_index(name='sightings')
        
        fig = px.scatter(yearly_counts, x='year', y='sightings', 
                        title="📈 Sightings Over Time",
                        trendline="lowess",
                        color_discrete_sequence=['#2E7D32'])
        
        fig.update_traces(marker=dict(size=8, opacity=0.7))
        fig.update_layout(
            showlegend=False,
            hovermode='x unified',
            xaxis_title="Year",
            yaxis_title="Number of Sightings"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Enhanced classification distribution
        class_counts = df['classification'].value_counts()
        colors = ['#D32F2F', '#FF9800', '#FFC107']  # Red, Orange, Yellow
        
        fig = px.pie(values=class_counts.values, names=class_counts.index,
                    title="🏷️ Classification Distribution",
                    color_discrete_sequence=colors)
        
        fig.update_traces(
            textposition='inside', 
            textinfo='percent+label',
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Secondary charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Enhanced monthly patterns
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        monthly_counts = df.groupby('month').size().reindex(range(1, 13), fill_value=0)
        
        fig = px.bar(x=month_names, y=monthly_counts.values,
                    title="📅 Seasonal Patterns",
                    color=monthly_counts.values,
                    color_continuous_scale='Greens')
        
        fig.update_layout(
            showlegend=False, 
            xaxis_title="Month", 
            yaxis_title="Number of Sightings",
            hovermode='x'
        )
        
        # Add annotations for peaks
        max_month_idx = monthly_counts.argmax()
        fig.add_annotation(
            x=month_names[max_month_idx],
            y=monthly_counts.iloc[max_month_idx],
            text=f"Peak: {monthly_counts.iloc[max_month_idx]} sightings",
            arrowhead=2,
            arrowcolor='red'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Enhanced geographic distribution
        # Sample for performance if needed
        plot_df = df.sample(min(2000, len(df)), random_state=42) if len(df) > 2000 else df
        
        fig = px.scatter(plot_df, x='longitude', y='latitude', 
                        color='classification',
                        title="🗺️ Geographic Distribution",
                        color_discrete_sequence=['#D32F2F', '#FF9800', '#FFC107'],
                        opacity=0.6,
                        hover_data=['year', 'season'])
        
        fig.update_layout(
            showlegend=True,
            xaxis_title="Longitude",
            yaxis_title="Latitude",
            legend_title="Classification Type"
        )
        
        # Add US state borders outline
        fig.update_geos(projection_type="natural earth")
        
        st.plotly_chart(fig, use_container_width=True)
        
        if len(df) > 2000:
            st.caption(f"📝 Showing sample of {len(plot_df):,} points for performance")
    
    # Additional insights
    st.subheader("🔍 Key Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="info-box">
            <strong>📈 Temporal Trends</strong><br>
            • Most active decade: {}<br>
            • Peak season: {}<br>
            • Average per year: {:.1f}
        </div>
        """.format(
            f"{(df['year'].mode().iloc[0] // 10) * 10}s",
            df['season'].mode().iloc[0],
            len(df) / (df['year'].max() - df['year'].min())
        ), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-box">
            <strong>🗺️ Geographic Patterns</strong><br>
            • Northernmost: {:.1f}°N<br>
            • Southernmost: {:.1f}°N<br>
            • Most western: {:.1f}°W
        </div>
        """.format(
            df['latitude'].max(),
            df['latitude'].min(),
            abs(df['longitude'].min())
        ), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="info-box">
            <strong>📊 Data Quality</strong><br>
            • Class A (Visual): {:.1f}%<br>
            • Class B (Sounds): {:.1f}%<br>
            • Class C (Stories): {:.1f}%
        </div>
        """.format(
            (df['classification'] == 'Class A').mean() * 100,
            (df['classification'] == 'Class B').mean() * 100,
            (df['classification'] == 'Class C').mean() * 100
        ), unsafe_allow_html=True)

def show_geographic_analysis(df):
    """Display detailed geographic analysis with clustering"""
    st.header("🗺️ Geographic Analysis & Clustering")
    
    # Perform clustering
    with st.spinner("🔄 Performing geographic clustering analysis..."):
        df_clustered, kmeans_model, cluster_stats = perform_clustering(df)
    
    if kmeans_model is None:
        st.error("❌ Clustering analysis failed")
        return
    
    # Interactive map
    st.subheader("🗺️ Interactive Sightings Map with Clusters")
    
    # Map controls
    col1, col2, col3 = st.columns(3)
    with col1:
        map_style = st.selectbox("Map Style:", ["OpenStreetMap", "Satellite", "Terrain"])
    with col2:
        show_clusters = st.checkbox("Show Clusters", value=True)
    with col3:
        max_points = st.slider("Max Points to Display:", 500, 3000, 1500)
    
    # Sample data for performance
    if len(df_clustered) > max_points:
        df_map = df_clustered.sample(max_points, random_state=42)
        st.info(f"📊 Displaying {len(df_map):,} of {len(df_clustered):,} sightings for optimal performance")
    else:
        df_map = df_clustered
    
    # Create folium map
    center_lat = df_map['latitude'].mean()
    center_lon = df_map['longitude'].mean()
    
    # Map style mapping
    tile_mapping = {
        "OpenStreetMap": "OpenStreetMap",
        "Satellite": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        "Terrain": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Terrain_Base/MapServer/tile/{z}/{y}/{x}"
    }
    
    if map_style == "OpenStreetMap":
        m = folium.Map(location=[center_lat, center_lon], zoom_start=4, tiles="OpenStreetMap")
    else:
        m = folium.Map(location=[center_lat, center_lon], zoom_start=4)
        folium.TileLayer(
            tiles=tile_mapping[map_style],
            attr="Esri",
            name=map_style,
            overlay=False,
            control=True
        ).add_to(m)
    
    # Color schemes
    cluster_colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightred', 'beige', 'darkblue', 'darkgreen']
    classification_colors = {'Class A': 'red', 'Class B': 'orange', 'Class C': 'yellow'}
    
    # Add markers
    for idx, row in df_map.iterrows():
        if show_clusters and 'cluster' in row:
            color = cluster_colors[row['cluster'] % len(cluster_colors)]
            popup_text = f"""
            <div style="width: 220px; font-family: Arial;">
                <b>🦶 Report #{row.get('number', 'Unknown')}</b><br>
                <b>📍 Cluster:</b> {row['cluster']}<br>
                <b>🏷️ Class:</b> {row['classification']}<br>
                <b>📅 Date:</b> {str(row['timestamp'])[:10]}<br>
                <b>🌍 Location:</b> {row['latitude']:.3f}, {row['longitude']:.3f}<br>
                <b>🍂 Season:</b> {row['season']}
            </div>
            """
        else:
            color = classification_colors.get(row['classification'], 'gray')
            popup_text = f"""
            <div style="width: 200px; font-family: Arial;">
                <b>🦶 Report #{row.get('number', 'Unknown')}</b><br>
                <b>🏷️ Class:</b> {row['classification']}<br>
                <b>📅 Date:</b> {str(row['timestamp'])[:10]}<br>
                <b>🌍 Location:</b> {row['latitude']:.3f}, {row['longitude']:.3f}
            </div>
            """
        
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=6,
            popup=folium.Popup(popup_text, max_width=250),
            color='black',
            fillColor=color,
            fillOpacity=0.8,
            weight=1,
            opacity=0.8
        ).add_to(m)
    
    # Add legend
    if show_clusters:
        legend_html = '''
        <div style="position: fixed; bottom: 50px; left: 50px; width: 200px; height: 140px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:12px; padding: 10px; border-radius: 5px;">
        <p style="margin: 0 0 10px 0; font-weight: bold;">🗺️ Cluster Map</p>
        <p style="margin: 2px 0;"><span style="color:red; font-size:16px;">●</span> Cluster 0</p>
        <p style="margin: 2px 0;"><span style="color:blue; font-size:16px;">●</span> Cluster 1</p>
        <p style="margin: 2px 0;"><span style="color:green; font-size:16px;">●</span> Cluster 2</p>
        <p style="margin: 2px 0;"><span style="color:purple; font-size:16px;">●</span> Cluster 3</p>
        <p style="margin: 2px 0;"><span style="color:orange; font-size:16px;">●</span> Cluster 4</p>
        </div>
        '''
    else:
        legend_html = '''
        <div style="position: fixed; bottom: 50px; left: 50px; width: 200px; height: 120px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:12px; padding: 10px; border-radius: 5px;">
        <p style="margin: 0 0 10px 0; font-weight: bold;">🏷️ Classifications</p>
        <p style="margin: 3px 0;"><span style="color:red; font-size:16px;">●</span> Class A - Visual</p>
        <p style="margin: 3px 0;"><span style="color:orange; font-size:16px;">●</span> Class B - Sounds</p>
        <p style="margin: 3px 0;"><span style="color:yellow; font-size:16px;">●</span> Class C - Stories</p>
        </div>
        '''
    
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Display map
    map_data = st_folium(m, width=700, height=500)
    
    # Analysis results
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Cluster Analysis Results")
        
        if 'cluster' in df_clustered.columns:
            cluster_summary = df_clustered.groupby('cluster').agg({
                'latitude': ['mean', 'count'],
                'longitude': 'mean',
                'year': ['min', 'max'],
                'classification': lambda x: (x == 'Class A').mean()
            }).round(3)
            
            cluster_summary.columns = ['Avg_Lat', 'Count', 'Avg_Lon', 'First_Year', 'Last_Year', 'Quality_Score']
            
            # Display cluster information
            for cluster_id in sorted(df_clustered['cluster'].unique()):
                cluster_data = cluster_summary.loc[cluster_id]
                
                with st.expander(f"🎯 Cluster {cluster_id} - {int(cluster_data['Count'])} sightings"):
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.write(f"📍 **Center:** {cluster_data['Avg_Lat']:.2f}°N, {cluster_data['Avg_Lon']:.2f}°W")
                        st.write(f"📊 **Count:** {int(cluster_data['Count'])} sightings")
                    with col_b:
                        st.write(f"📅 **Period:** {int(cluster_data['First_Year'])}-{int(cluster_data['Last_Year'])}")
                        st.write(f"⭐ **Quality:** {cluster_data['Quality_Score']:.1%} Class A")
            
            # Cluster visualization
            cluster_counts = df_clustered['cluster'].value_counts().sort_index()
            fig = px.bar(x=cluster_counts.index, y=cluster_counts.values,
                        title="Sightings per Cluster",
                        color=cluster_counts.values,
                        color_continuous_scale='Viridis',
                        labels={'x': 'Cluster ID', 'y': 'Number of Sightings'})
            
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Geographic Hotspots")
        
        # Density analysis
        st.write("**🔥 High-Density Regions:**")
        
        # Calculate state-level density (approximate)
        df_density = df_clustered.copy()
        df_density['lat_bin'] = (df_density['latitude'] // 2) * 2  # 2-degree bins
        df_density['lon_bin'] = (df_density['longitude'] // 2) * 2
        df_density['region'] = df_density['lat_bin'].astype(str) + "," + df_density['lon_bin'].astype(str)
        
        region_counts = df_density.groupby('region').agg({
            'latitude': ['mean', 'count'],
            'longitude': 'mean'
        }).round(1)
        
        region_counts.columns = ['Avg_Lat', 'Count', 'Avg_Lon']
        top_regions = region_counts.nlargest(5, 'Count')
        
        for i, (region, data) in enumerate(top_regions.iterrows(), 1):
            st.write(f"**{i}.** {data['Avg_Lat']}°N, {data['Avg_Lon']}°W - **{int(data['Count'])}** sightings")
        
        # Quality distribution by cluster
        if 'cluster' in df_clustered.columns:
            quality_by_cluster = df_clustered.groupby(['cluster', 'classification']).size().unstack(fill_value=0)
            
            fig = px.bar(quality_by_cluster, 
                        title="Classification Quality by Cluster",
                        color_discrete_sequence=['#D32F2F', '#FF9800', '#FFC107'])
            
            fig.update_layout(
                xaxis_title="Cluster ID",
                yaxis_title="Number of Reports",
                legend_title="Classification"
            )
            st.plotly_chart(fig, use_container_width=True)

def show_temporal_analysis(df):
    """Display comprehensive temporal pattern analysis"""
    st.header("📅 Temporal Pattern Analysis")
    
    # Time series overview
    st.subheader("⏱️ Time Series Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Enhanced yearly trend
        yearly_data = df.groupby('year').size().reset_index(name='sightings')
        yearly_data['cumulative'] = yearly_data['sightings'].cumsum()
        yearly_data['rolling_avg'] = yearly_data['sightings'].rolling(window=5, center=True).mean()
        
        fig = go.Figure()
        
        # Actual data
        fig.add_trace(go.Scatter(
            x=yearly_data['year'], 
            y=yearly_data['sightings'],
            mode='lines+markers', 
            name='Annual Sightings',
            line=dict(color='lightblue', width=2),
            marker=dict(size=6, opacity=0.7),
            hovertemplate='<b>%{x}</b><br>Sightings: %{y}<extra></extra>'
        ))
        
        # Rolling average
        fig.add_trace(go.Scatter(
            x=yearly_data['year'], 
            y=yearly_data['rolling_avg'],
            mode='lines', 
            name='5-Year Average',
            line=dict(color='red', width=3),
            hovertemplate='<b>%{x}</b><br>5-Year Avg: %{y:.1f}<extra></extra>'
        ))
        
        fig.update_layout(
            title="📈 Yearly Trends with Moving Average",
            xaxis_title="Year",
            yaxis_title="Number of Sightings",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Cumulative sightings
        fig = px.line(yearly_data, x='year', y='cumulative',
                     title="📊 Cumulative Sightings Over Time",
                     color_discrete_sequence=['#2E7D32'])
        
        fig.update_layout(
            xaxis_title="Year",
            yaxis_title="Cumulative Sightings",
            hovermode='x'
        )
        
        # Add milestone annotations
        milestones = [1000, 2000, 3000, 4000]
        for milestone in milestones:
            if yearly_data['cumulative'].max() >= milestone:
                milestone_year = yearly_data[yearly_data['cumulative'] >= milestone]['year'].iloc[0]
                fig.add_annotation(
                    x=milestone_year,
                    y=milestone,
                    text=f"{milestone:,}th report",
                    arrowhead=2,
                    arrowcolor='red',
                    bgcolor='yellow',
                    bordercolor='red'
                )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed temporal patterns
    st.subheader("🔍 Detailed Temporal Patterns")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Enhanced monthly heatmap
        monthly_class = df.groupby(['month', 'classification']).size().unstack(fill_value=0)
        
        fig = px.imshow(monthly_class.T, 
                       title="🔥 Monthly Activity by Classification",
                       color_continuous_scale='Greens',
                       aspect='auto',
                       labels={'x': 'Month', 'y': 'Classification', 'color': 'Sightings'})
        
        # Custom month labels
        month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        fig.update_xaxes(tickvals=list(range(12)), ticktext=month_labels)
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Day of week analysis
        if 'day_of_week' in df.columns:
            dow_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            dow_counts = df.groupby('day_of_week').size().reindex(range(7), fill_value=0)
            
            # Create radar chart for day of week
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=dow_counts.values,
                theta=dow_names,
                fill='toself',
                name='Sightings by Day',
                line_color='blue'
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, dow_counts.max()])
                ),
                title="📅 Day of Week Pattern",
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Day of week data not available")
    
    with col3:
        # Seasonal analysis with classification
        seasonal_data = df.groupby(['season', 'classification']).size().unstack(fill_value=0)
        
        fig = px.bar(seasonal_data, 
                    title="🍂 Seasonal Distribution by Class",
                    color_discrete_sequence=['#D32F2F', '#FF9800', '#FFC107'],
                    barmode='stack')
        
        fig.update_layout(
            xaxis_title="Season",
            yaxis_title="Number of Sightings",
            legend_title="Classification"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Advanced temporal analysis
    st.subheader("📊 Advanced Temporal Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Decade analysis
        df_decade = df.copy()
        df_decade['decade'] = (df_decade['year'] // 10) * 10
        decade_counts = df_decade.groupby('decade').size()
        decade_quality = df_decade.groupby(['decade', 'classification']).size().unstack(fill_value=0)
        
        fig = px.bar(decade_quality, 
                    title="📊 Sightings by Decade with Quality",
                    color_discrete_sequence=['#D32F2F', '#FF9800', '#FFC107'])
        
        fig.update_layout(
            xaxis_title="Decade",
            yaxis_title="Number of Sightings",
            legend_title="Classification"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Temporal clustering
        # Group sightings by time periods to find patterns
        df_temporal = df.copy()
        df_temporal['month_year'] = df_temporal['year'].astype(str) + '-' + df_temporal['month'].astype(str).str.zfill(2)
        
        # Calculate monthly statistics
        monthly_stats = df_temporal.groupby('month_year').agg({
            'latitude': 'count',
            'classification': lambda x: (x == 'Class A').mean()
        }).rename(columns={'latitude': 'count', 'classification': 'quality_ratio'})
        
        # Filter to recent years for clarity
        recent_years = df['year'].max() - 10
        recent_monthly = df[df['year'] >= recent_years].groupby(['year', 'month']).size().reset_index(name='sightings')
        
        if len(recent_monthly) > 0:
            fig = px.line(recent_monthly, x='month', y='sightings', color='year',
                         title=f"📈 Monthly Patterns (Last 10 Years)",
                         color_discrete_sequence=px.colors.qualitative.Set3)
            
            fig.update_layout(
                xaxis_title="Month",
                yaxis_title="Number of Sightings",
                legend_title="Year"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Insufficient recent data for monthly trend analysis")
    
    # Key temporal insights
    st.subheader("🔍 Key Temporal Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        peak_month = df['month'].mode().iloc[0]
        month_names = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        st.markdown(f"""
        <div class="info-box">
            <strong>📅 Peak Activity</strong><br>
            • Peak month: {month_names[peak_month]}<br>
            • Peak season: {df['season'].mode().iloc[0]}<br>
            • Most active year: {df['year'].mode().iloc[0]}
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        growth_rate = (yearly_data['sightings'].iloc[-5:].mean() - yearly_data['sightings'].iloc[:5].mean()) / len(yearly_data)
        
        st.markdown(f"""
        <div class="info-box">
            <strong>📈 Growth Trends</strong><br>
            • Reports per year: {len(df) / (df['year'].max() - df['year'].min()):.1f}<br>
            • Growth rate: {growth_rate:.1f}/year<br>
            • Peak decade: {decade_counts.idxmax()}s
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        weekend_count = len(df[df['day_of_week'].isin([5, 6])]) if 'day_of_week' in df.columns else 0
        weekend_pct = weekend_count / len(df) * 100 if len(df) > 0 else 0
        
        st.markdown(f"""
        <div class="info-box">
            <strong>📊 Activity Patterns</strong><br>
            • Weekend reports: {weekend_pct:.1f}%<br>
            • Summer reports: {(df['season'] == 'Summer').mean() * 100:.1f}%<br>
            • Recent uptick: {yearly_data['sightings'].iloc[-3:].mean():.0f}/year
        </div>
        """, unsafe_allow_html=True)

def show_ml_analysis(df):
    """Display comprehensive machine learning analysis"""
    st.header("🤖 Machine Learning Analysis")
    
    # Train ML model
    with st.spinner("🔄 Training machine learning models..."):
        ml_results = train_ml_model(df)
    
    if ml_results is None:
        st.error("❌ Machine learning analysis failed")
        return
    
    # Model performance overview
    st.subheader("🎯 Model Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Model Accuracy", f"{ml_results['accuracy']:.3f}")
        st.caption("Random Forest Classifier")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Features Used", len(ml_results['feature_columns']))
        st.caption("Engineered features")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        precision = precision_score(ml_results['test_actual'], ml_results['test_predictions'], average='weighted', zero_division=0)
        st.metric("Precision", f"{precision:.3f}")
        st.caption("Weighted average")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        from sklearn.metrics import f1_score
        f1 = f1_score(ml_results['test_actual'], ml_results['test_predictions'], average='weighted', zero_division=0)
        st.metric("F1 Score", f"{f1:.3f}")
        st.caption("Weighted average")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Detailed analysis
    col1, col2 = st.columns(2)
    
    with col1:
        # Feature importance
        st.subheader("🎯 Feature Importance Analysis")
        
        importance_df = ml_results['feature_importance'].head(10)
        
        fig = px.bar(importance_df, x='importance', y='feature',
                    title="Top 10 Most Important Features",
                    orientation='h',
                    color='importance',
                    color_continuous_scale='Greens')
        
        fig.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            xaxis_title="Importance Score",
            yaxis_title="Features"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance insights
        st.write("**🔍 Key Insights:**")
        top_feature = importance_df.iloc[0]
        st.write(f"• Most important: **{top_feature['feature']}** ({top_feature['importance']:.3f})")
        
        geographic_features = ['latitude', 'longitude', 'distance_from_center']
        temporal_features = ['year_normalized', 'month', 'day_of_year', 'month_sin', 'month_cos', 'day_sin', 'day_cos']
        
        geo_importance = importance_df[importance_df['feature'].isin(geographic_features)]['importance'].sum()
        temp_importance = importance_df[importance_df['feature'].isin(temporal_features)]['importance'].sum()
        
        st.write(f"• Geographic factors: **{geo_importance:.3f}** total importance")
        st.write(f"• Temporal factors: **{temp_importance:.3f}** total importance")
    
    with col2:
        # Confusion matrix
        st.subheader("📊 Confusion Matrix")
        
        cm = confusion_matrix(ml_results['test_actual'], ml_results['test_predictions'])
        class_names = ml_results['label_encoder'].classes_
        
        # Create confusion matrix heatmap
        fig = px.imshow(cm, 
                       text_auto=True,
                       aspect="auto",
                       color_continuous_scale='Blues',
                       title="Classification Confusion Matrix")
        
        fig.update_xaxes(tickvals=list(range(len(class_names))), ticktext=class_names, title="Predicted")
        fig.update_yaxes(tickvals=list(range(len(class_names))), ticktext=class_names, title="Actual")
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Classification report
        st.write("**📋 Classification Report:**")
        report = classification_report(ml_results['test_actual'], ml_results['test_predictions'], 
                                     target_names=class_names, output_dict=True, zero_division=0)
        
        for class_name in class_names:
            if class_name in report:
                precision = report[class_name]['precision']
                recall = report[class_name]['recall']
                f1 = report[class_name]['f1-score']
                st.write(f"• **{class_name}**: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}")
    
    # Advanced ML analysis
    st.subheader("🧠 Advanced Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Learning curves / Model complexity
        st.write("**🔬 Model Insights**")
        
        # Feature correlation with target
        ml_df_temp = df.copy()
        le_temp = LabelEncoder()
        ml_df_temp['classification_encoded'] = le_temp.fit_transform(ml_df_temp['classification'])
        
        correlations = {}
        for feature in ['latitude', 'longitude', 'year', 'month', 'distance_from_center']:
            if feature in ml_df_temp.columns:
                corr = ml_df_temp[feature].corr(ml_df_temp['classification_encoded'])
                correlations[feature] = abs(corr)
        
        corr_df = pd.DataFrame(list(correlations.items()), columns=['Feature', 'Correlation'])
        corr_df = corr_df.sort_values('Correlation', ascending=False)
        
        fig = px.bar(corr_df, x='Feature', y='Correlation',
                    title="Feature Correlation with Classification",
                    color='Correlation',
                    color_continuous_scale='Reds')
        
        fig.update_layout(xaxis_tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Prediction confidence analysis
        st.write("**🎯 Prediction Analysis**")
        
        if hasattr(ml_results['model'], 'predict_proba'):
            # Get prediction probabilities for a sample
            sample_indices = np.random.choice(len(df), min(100, len(df)), replace=False)
            sample_df = df.iloc[sample_indices].copy()
            
            # Prepare sample features
            sample_df['month_sin'] = np.sin(2 * np.pi * sample_df['month'] / 12)
            sample_df['month_cos'] = np.cos(2 * np.pi * sample_df['month'] / 12)
            sample_df['day_sin'] = np.sin(2 * np.pi * sample_df['day_of_year'] / 365)
            sample_df['day_cos'] = np.cos(2 * np.pi * sample_df['day_of_year'] / 365)
            sample_df['year_normalized'] = (sample_df['year'] - sample_df['year'].min()) / (sample_df['year'].max() - sample_df['year'].min())
            
            X_sample = sample_df[ml_results['feature_columns']]
            probabilities = ml_results['model'].predict_proba(X_sample)
            
            # Get max probability (confidence) for each prediction
            confidences = np.max(probabilities, axis=1)
            
            fig = px.histogram(x=confidences, nbins=20,
                             title="Model Prediction Confidence Distribution",
                             labels={'x': 'Confidence Score', 'y': 'Number of Predictions'},
                             color_discrete_sequence=['#2E7D32'])
            
            fig.add_vline(x=confidences.mean(), line_dash="dash", line_color="red",
                         annotation_text=f"Mean: {confidences.mean():.3f}")
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.write(f"• Average confidence: **{confidences.mean():.3f}**")
            st.write(f"• High confidence (>0.8): **{(confidences > 0.8).mean()*100:.1f}%**")
        else:
            st.info("Probability predictions not available for this model type")
    
    # Interactive prediction tool
    st.subheader("🔮 Interactive Prediction Tool")
    
    st.write("**Try making predictions with custom inputs:**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        pred_lat = st.slider("Latitude:", float(df['latitude'].min()), float(df['latitude'].max()), float(df['latitude'].mean()))
        pred_lon = st.slider("Longitude:", float(df['longitude'].min()), float(df['longitude'].max()), float(df['longitude'].mean()))
    
    with col2:
        pred_year = st.slider("Year:", int(df['year'].min()), int(df['year'].max()), int(df['year'].max()))
        pred_month = st.slider("Month:", 1, 12, 6)
    
    with col3:
        pred_day = st.slider("Day of Year:", 1, 365, 180)
    
    if st.button("🔍 Make Prediction"):
        # Prepare prediction data
        pred_data = pd.DataFrame({
            'latitude': [pred_lat],
            'longitude': [pred_lon],
            'year_normalized': [(pred_year - df['year'].min()) / (df['year'].max() - df['year'].min())],
            'month': [pred_month],
            'month_sin': [np.sin(2 * np.pi * pred_month / 12)],
            'month_cos': [np.cos(2 * np.pi * pred_month / 12)],
            'day_sin': [np.sin(2 * np.pi * pred_day / 365)],
            'day_cos': [np.cos(2 * np.pi * pred_day / 365)],
            'distance_from_center': [np.sqrt((pred_lat - df['latitude'].mean())**2 + (pred_lon - df['longitude'].mean())**2)],
            'day_of_year': [pred_day]
        })
        
        prediction = ml_results['model'].predict(pred_data[ml_results['feature_columns']])[0]
        predicted_class = ml_results['label_encoder'].inverse_transform([prediction])[0]
        
        if hasattr(ml_results['model'], 'predict_proba'):
            proba = ml_results['model'].predict_proba(pred_data[ml_results['feature_columns']])[0]
            confidence = np.max(proba)
            
            st.success(f"🎯 **Predicted Classification:** {predicted_class}")
            st.info(f"🎲 **Confidence:** {confidence:.3f}")
            
            # Show probability distribution
            prob_df = pd.DataFrame({
                'Class': ml_results['label_encoder'].classes_,
                'Probability': proba
            })
            
            fig = px.bar(prob_df, x='Class', y='Probability',
                        title="Prediction Probabilities",
                        color='Probability',
                        color_continuous_scale='Greens')
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success(f"🎯 **Predicted Classification:** {predicted_class}")

def show_anomaly_analysis(df):
    """Display comprehensive anomaly detection analysis"""
    st.header("🔍 Anomaly Detection Analysis")
    
    # Perform anomaly detection
    with st.spinner("🔄 Detecting anomalous sightings..."):
        df_anomaly, iso_forest = detect_anomalies(df)
    
    if iso_forest is None:
        st.error("❌ Anomaly detection failed")
        return
    
    # Anomaly overview
    st.subheader("📊 Anomaly Detection Overview")
    
    n_anomalies = df_anomaly['is_anomaly'].sum()
    anomaly_pct = n_anomalies / len(df_anomaly) * 100
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Anomalies", f"{n_anomalies:,}")
        st.caption(f"{anomaly_pct:.1f}% of all sightings")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        most_anomalous_score = df_anomaly['anomaly_score'].min()
        st.metric("Most Suspicious Score", f"{most_anomalous_score:.3f}")
        st.caption("Lower = more anomalous")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        avg_anomaly_distance = df_anomaly[df_anomaly['is_anomaly']]['distance_from_center'].mean()
        avg_normal_distance = df_anomaly[~df_anomaly['is_anomaly']]['distance_from_center'].mean()
        st.metric("Anomaly Dispersion", f"{avg_anomaly_distance:.2f}°")
        st.caption(f"vs {avg_normal_distance:.2f}° normal")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        anomaly_class_a_pct = (df_anomaly[df_anomaly['is_anomaly']]['classification'] == 'Class A').mean() * 100
        st.metric("Anomaly Quality", f"{anomaly_class_a_pct:.1f}%")
        st.caption("Class A anomalies")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Visualizations
    st.subheader("🗺️ Anomaly Visualization")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Geographic distribution of anomalies
        plot_df = df_anomaly.sample(min(2000, len(df_anomaly)), random_state=42) if len(df_anomaly) > 2000 else df_anomaly
        
        fig = px.scatter(plot_df, x='longitude', y='latitude',
                        color='is_anomaly',
                        title="🌍 Geographic Distribution of Anomalies",
                        color_discrete_map={True: '#FF5722', False: '#2196F3'},
                        hover_data=['year', 'classification', 'anomaly_score'],
                        opacity=0.7)
        
        fig.update_layout(
            legend_title="Is Anomaly",
            xaxis_title="Longitude",
            yaxis_title="Latitude"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        if len(df_anomaly) > 2000:
            st.caption(f"📝 Showing sample of {len(plot_df):,} points for performance")
    
    with col2:
        # Anomaly score distribution
        fig = px.histogram(df_anomaly, x='anomaly_score',
                          title="📊 Anomaly Score Distribution",
                          color_discrete_sequence=['#FF9800'],
                          nbins=30)
        
        # Add threshold line
        threshold = df_anomaly['anomaly_score'].quantile(0.1)  # 10% most anomalous
        fig.add_vline(x=threshold, line_dash="dash", line_color="red",
                     annotation_text="Anomaly Threshold")
        
        fig.update_layout(
            xaxis_title="Anomaly Score",
            yaxis_title="Number of Sightings",
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed anomaly analysis
    st.subheader("🔍 Detailed Anomaly Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Anomaly characteristics
        st.write("**🎯 Anomaly vs Normal Characteristics:**")
        
        anomalies = df_anomaly[df_anomaly['is_anomaly']]
        normal = df_anomaly[~df_anomaly['is_anomaly']]
        
        comparison_metrics = {
            'Average Latitude': [anomalies['latitude'].mean(), normal['latitude'].mean()],
            'Average Longitude': [anomalies['longitude'].mean(), normal['longitude'].mean()],
            'Average Year': [anomalies['year'].mean(), normal['year'].mean()],
            'Class A %': [(anomalies['classification'] == 'Class A').mean() * 100,
                         (normal['classification'] == 'Class A').mean() * 100]
        }
        
        comparison_df = pd.DataFrame(comparison_metrics, index=['Anomalies', 'Normal'])
        
        for metric in comparison_metrics.keys():
            anom_val = comparison_df.loc['Anomalies', metric]
            norm_val = comparison_df.loc['Normal', metric]
            diff = anom_val - norm_val
            
            st.write(f"• **{metric}:**")
            st.write(f"  - Anomalies: {anom_val:.2f}")
            st.write(f"  - Normal: {norm_val:.2f}")
            st.write(f"  - Difference: {diff:.2f}")
    
    with col2:
        # Temporal distribution of anomalies
        anomaly_by_year = df_anomaly.groupby('year')['is_anomaly'].agg(['sum', 'count']).reset_index()
        anomaly_by_year['anomaly_rate'] = anomaly_by_year['sum'] / anomaly_by_year['count'] * 100
        
        fig = px.line(anomaly_by_year, x='year', y='anomaly_rate',
                     title="📈 Anomaly Rate Over Time",
                     color_discrete_sequence=['#E91E63'])
        
        fig.update_layout(
            xaxis_title="Year",
            yaxis_title="Anomaly Rate (%)",
            hovermode='x'
        )
        
        # Add trend line
        if len(anomaly_by_year) > 5:
            fig.add_scatter(x=anomaly_by_year['year'], 
                           y=anomaly_by_year['anomaly_rate'].rolling(5).mean(),
                           mode='lines', name='5-Year Trend', 
                           line=dict(dash='dash', color='red'))
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Most anomalous sightings
    st.subheader("🚨 Most Anomalous Sightings")
    
    most_anomalous = df_anomaly.nsmallest(10, 'anomaly_score')[
        ['number', 'title', 'latitude', 'longitude', 'year', 'month', 'classification', 'anomaly_score']
    ].copy()
    
    # Format the dataframe for better display
    most_anomalous['anomaly_score'] = most_anomalous['anomaly_score'].round(4)
    most_anomalous['latitude'] = most_anomalous['latitude'].round(3)
    most_anomalous['longitude'] = most_anomalous['longitude'].round(3)
    
    # Add ranking
    most_anomalous['rank'] = range(1, len(most_anomalous) + 1)
    most_anomalous = most_anomalous[['rank', 'number', 'title', 'latitude', 'longitude', 
                                   'year', 'classification', 'anomaly_score']]
    
    st.dataframe(most_anomalous, use_container_width=True, hide_index=True)
    
    # Anomaly investigation tools
    st.subheader("🔬 Anomaly Investigation Tools")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Filter anomalies by score
        score_threshold = st.slider(
            "Anomaly Score Threshold:",
            float(df_anomaly['anomaly_score'].min()),
            float(df_anomaly['anomaly_score'].max()),
            float(df_anomaly['anomaly_score'].quantile(0.1)),
            step=0.01
        )
        
        filtered_anomalies = df_anomaly[df_anomaly['anomaly_score'] <= score_threshold]
        st.write(f"**Found {len(filtered_anomalies)} anomalies with score ≤ {score_threshold}**")
        
        if len(filtered_anomalies) > 0:
            # Classification distribution of filtered anomalies
            class_dist = filtered_anomalies['classification'].value_counts()
            
            fig = px.pie(values=class_dist.values, names=class_dist.index,
                        title="Classification of Filtered Anomalies",
                        color_discrete_sequence=['#D32F2F', '#FF9800', '#FFC107'])
            
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Seasonal anomaly patterns
        seasonal_anomalies = df_anomaly.groupby('season')['is_anomaly'].agg(['sum', 'count']).reset_index()
        seasonal_anomalies['anomaly_rate'] = seasonal_anomalies['sum'] / seasonal_anomalies['count'] * 100
        
        fig = px.bar(seasonal_anomalies, x='season', y='anomaly_rate',
                    title="🍂 Anomaly Rate by Season",
                    color='anomaly_rate',
                    color_continuous_scale='Reds')
        
        fig.update_layout(
            xaxis_title="Season",
            yaxis_title="Anomaly Rate (%)",
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Anomaly insights
    st.subheader("💡 Anomaly Detection Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="info-box">
            <strong>🎯 Detection Summary</strong><br>
            • Total analyzed: {len(df_anomaly):,}<br>
            • Anomalies found: {n_anomalies:,}<br>
            • Detection rate: {anomaly_pct:.1f}%<br>
            • Most suspicious: Score {most_anomalous_score:.3f}
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        peak_anomaly_season = seasonal_anomalies.loc[seasonal_anomalies['anomaly_rate'].idxmax(), 'season']
        peak_anomaly_rate = seasonal_anomalies['anomaly_rate'].max()
        
        st.markdown(f"""
        <div class="info-box">
            <strong>📊 Pattern Analysis</strong><br>
            • Peak anomaly season: {peak_anomaly_season}<br>
            • Peak rate: {peak_anomaly_rate:.1f}%<br>
            • Geographic spread: {'High' if avg_anomaly_distance > avg_normal_distance else 'Low'}<br>
            • Quality correlation: {'Positive' if anomaly_class_a_pct > 30 else 'Negative'}
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        recent_anomalies = len(anomalies[anomalies['year'] >= df_anomaly['year'].max() - 5])
        total_recent = len(df_anomaly[df_anomaly['year'] >= df_anomaly['year'].max() - 5])
        recent_rate = recent_anomalies / total_recent * 100 if total_recent > 0 else 0
        
        st.markdown(f"""
        <div class="info-box">
            <strong>📈 Recent Trends</strong><br>
            • Last 5 years: {recent_anomalies} anomalies<br>
            • Recent rate: {recent_rate:.1f}%<br>
            • Trend: {'Increasing' if recent_rate > anomaly_pct else 'Stable'}<br>
            • Investigation priority: {'High' if recent_rate > 15 else 'Medium'}
        </div>
        """, unsafe_allow_html=True)

def show_recommendations(df):
    """Display location recommendations for Bigfoot research"""
    st.header("🎯 Research Location Recommendations")
    
    # Calculate grid-based recommendations
    with st.spinner("🔄 Calculating optimal research locations..."):
        # Create geographic grid
        df_grid = df.copy()
        df_grid['lat_grid'] = (df_grid['latitude'] // 1).astype(int)
        df_grid['lon_grid'] = (df_grid['longitude'] // 1).astype(int)
        df_grid['grid_id'] = df_grid['lat_grid'].astype(str) + '_' + df_grid['lon_grid'].astype(str)
        
        # Calculate grid metrics
        grid_metrics = df_grid.groupby('grid_id').agg({
            'latitude': 'mean',
            'longitude': 'mean',
            'number': 'count',
            'year': ['min', 'max'],
            'classification': lambda x: (x == 'Class A').mean(),
            'season': lambda x: x.mode().iloc[0] if len(x) > 0 else 'Unknown'
        }).round(3)
        
        # Flatten column names
        grid_metrics.columns = ['avg_lat', 'avg_lon', 'sighting_count', 'first_year', 'last_year', 'quality_score', 'dominant_season']
        
        # Calculate recommendation scores
        grid_metrics['recency_score'] = (grid_metrics['last_year'] - grid_metrics['first_year'].min()) / (grid_metrics['last_year'].max() - grid_metrics['first_year'].min())
        grid_metrics['density_score'] = (grid_metrics['sighting_count'] - grid_metrics['sighting_count'].min()) / (grid_metrics['sighting_count'].max() - grid_metrics['sighting_count'].min())
        grid_metrics['consistency_score'] = (grid_metrics['last_year'] - grid_metrics['first_year']) / max(1, grid_metrics['last_year'].max() - grid_metrics['first_year'].min())
    
    # User preferences
    st.subheader("🎛️ Customize Your Research Preferences")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        priority = st.selectbox(
            "Research Priority:",
            ["Balanced Approach", "High Activity Zones", "High Quality Reports", "Recent Activity", "Consistent Activity"],
            help="Choose what type of locations you want to prioritize"
        )
        
        min_sightings = st.slider("Minimum Sightings per Location:", 1, 50, 5,
                                 help="Filter out locations with too few sightings")
    
    with col2:
        season_pref = st.selectbox(
            "Preferred Research Season:",
            ["Any Season", "Spring", "Summer", "Fall", "Winter"],
            help="Choose optimal season for field research"
        )
        
        max_results = st.slider("Number of Recommendations:", 5, 25, 10,
                               help="How many location recommendations to show")
    
    with col3:
        quality_weight = st.slider("Quality Importance:", 0.0, 1.0, 0.3, 0.1,
                                  help="How much to prioritize high-quality sightings")
        
        recency_weight = st.slider("Recency Importance:", 0.0, 1.0, 0.2, 0.1,
                                  help="How much to prioritize recent sightings")
    
    # Calculate final scores based on preferences
    density_weight = 1.0 - quality_weight - recency_weight
    if density_weight < 0:
        density_weight = 0.1
        st.warning("⚠️ Weights adjusted to ensure they sum to ≤ 1.0")
    
    # Filter and score locations
    filtered_grid = grid_metrics[grid_metrics['sighting_count'] >= min_sightings].copy()
    
    # Adjust scoring based on priority
    if priority == "High Activity Zones":
        filtered_grid['final_score'] = filtered_grid['density_score']
    elif priority == "High Quality Reports":
        filtered_grid['final_score'] = filtered_grid['quality_score']
    elif priority == "Recent Activity":
        filtered_grid['final_score'] = filtered_grid['recency_score']
    elif priority == "Consistent Activity":
        filtered_grid['final_score'] = filtered_grid['consistency_score']
    else:  # Balanced Approach
        filtered_grid['final_score'] = (
            filtered_grid['density_score'] * density_weight +
            filtered_grid['quality_score'] * quality_weight +
            filtered_grid['recency_score'] * recency_weight
        )
    
    # Get top recommendations
    top_recommendations = filtered_grid.nlargest(max_results, 'final_score')
    
    # Display results
    st.subheader("🏆 Top Research Location Recommendations")
    
    if len(top_recommendations) == 0:
        st.warning("⚠️ No locations match your criteria. Try adjusting the filters.")
        return
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Interactive map of recommendations
        fig = px.scatter_mapbox(
            top_recommendations, 
            lat='avg_lat', lon='avg_lon',
            size='sighting_count',
            color='final_score',
            hover_name=top_recommendations.index,
            hover_data={
                'sighting_count': True,
                'quality_score': ':.3f',
                'final_score': ':.3f',
                'first_year': True,
                'last_year': True
            },
            color_continuous_scale='Reds',
            size_max=20,
            zoom=3,
            title="🗺️ Recommended Research Locations"
        )
        
        fig.update_layout(
            mapbox_style="open-street-map",
            height=500,
            margin={"r":0,"t":30,"l":0,"b":0}
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Top recommendations list
        st.write("**🥇 Top 5 Locations:**")
        
        for i, (grid_id, row) in enumerate(top_recommendations.head(5).iterrows(), 1):
            with st.expander(f"#{i} - Grid {grid_id} ({row['final_score']:.3f})"):
                st.write(f"📍 **Coordinates:** {row['avg_lat']:.2f}°N, {row['avg_lon']:.2f}°W")
                st.write(f"🦶 **Total Sightings:** {int(row['sighting_count'])}")
                st.write(f"⭐ **Quality Score:** {row['quality_score']:.3f}")
                st.write(f"📅 **Active Period:** {int(row['first_year'])}-{int(row['last_year'])}")
                st.write(f"🍂 **Best Season:** {row['dominant_season']}")
                st.write(f"🎯 **Overall Score:** {row['final_score']:.3f}")
                
                # Add quick facts
                years_active = int(row['last_year']) - int(row['first_year']) + 1
                avg_per_year = row['sighting_count'] / years_active
                st.write(f"📊 **Activity:** {avg_per_year:.1f} sightings/year")
    
    # Detailed recommendations table
    st.subheader("📋 Detailed Recommendations Table")
    
    display_df = top_recommendations.copy()
    display_df['rank'] = range(1, len(display_df) + 1)
    display_df = display_df.round(3)
    
    # Reorder and rename columns for better display
    display_columns = {
        'rank': 'Rank',
        'avg_lat': 'Latitude',
        'avg_lon': 'Longitude', 
        'sighting_count': 'Sightings',
        'quality_score': 'Quality',
        'final_score': 'Score',
        'first_year': 'First Year',
        'last_year': 'Last Year',
        'dominant_season': 'Best Season'
    }
    
    display_df = display_df[list(display_columns.keys())].rename(columns=display_columns)
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    # Research planning insights
    st.subheader("📊 Research Planning Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Seasonal distribution
        season_dist = top_recommendations['dominant_season'].value_counts()
        
        fig = px.pie(values=season_dist.values, names=season_dist.index,
                    title="🍂 Optimal Seasons for Top Locations",
                    color_discrete_sequence=['lightblue', 'lightgreen', 'orange', 'lightcoral'])
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Quality vs quantity analysis
        fig = px.scatter(top_recommendations, x='sighting_count', y='quality_score',
                        size='final_score', color='final_score',
                        title="🎯 Quality vs Quantity Analysis",
                        color_continuous_scale='Viridis',
                        labels={'sighting_count': 'Number of Sightings', 'quality_score': 'Quality Score'})
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        # Activity timeline
        timeline_data = []
        for _, row in top_recommendations.head(5).iterrows():
            for year in range(int(row['first_year']), int(row['last_year']) + 1):
                timeline_data.append({
                    'Location': f"Grid {_}",
                    'Year': year,
                    'Active': 1
                })
        
        if timeline_data:
            timeline_df = pd.DataFrame(timeline_data)
            activity_by_year = timeline_df.groupby('Year')['Active'].sum().reset_index()
            
            fig = px.line(activity_by_year, x='Year', y='Active',
                         title="📈 Location Activity Timeline",
                         color_discrete_sequence=['#2E7D32'])
            
            fig.update_layout(
                yaxis_title="Number of Active Locations",
                xaxis_title="Year"
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Export recommendations
    st.subheader("📤 Export Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Generate research plan
        if st.button("📋 Generate Research Plan"):
            research_plan = f"""
# 🦶 BIGFOOT RESEARCH EXPEDITION PLAN
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 🎯 OBJECTIVE
{priority} - Field research at high-potential Bigfoot sighting locations

## 📍 TOP RECOMMENDED LOCATIONS

"""
            for i, (grid_id, row) in enumerate(top_recommendations.head(5).iterrows(), 1):
                research_plan += f"""
### Location #{i} - Grid {grid_id}
- **Coordinates:** {row['avg_lat']:.3f}°N, {row['avg_lon']:.3f}°W
- **Sightings:** {int(row['sighting_count'])} reports
- **Quality Score:** {row['quality_score']:.3f}
- **Activity Period:** {int(row['first_year'])}-{int(row['last_year'])}
- **Optimal Season:** {row['dominant_season']}
- **Overall Score:** {row['final_score']:.3f}
"""
            
            research_plan += f"""

## 🛠️ RESEARCH PARAMETERS
- Minimum sightings per location: {min_sightings}
- Quality importance: {quality_weight}
- Recency importance: {recency_weight}
- Total locations analyzed: {len(filtered_grid)}
- Locations recommended: {len(top_recommendations)}

## 📊 SEASONAL INSIGHTS
"""
            for season, count in season_dist.items():
                pct = count / len(top_recommendations) * 100
                research_plan += f"- {season}: {count} locations ({pct:.1f}%)\n"
            
            research_plan += """

## 🎒 RECOMMENDED EQUIPMENT
- GPS device with waypoint capability
- Trail cameras for long-term monitoring
- Audio recording equipment
- Plaster casting materials for tracks
- Weather-resistant notebook for field notes
- Emergency communication device

## ⚠️ SAFETY CONSIDERATIONS
- Always inform others of your research plans
- Carry emergency supplies and first aid kit
- Research local wildlife and terrain hazards
- Consider hiring local guides familiar with the area
- Follow Leave No Trace principles

---
Generated by Bigfoot ML Dashboard | Data: BFRO Database
"""
            
            st.download_button(
                label="📄 Download Research Plan",
                data=research_plan,
                file_name=f"bigfoot_research_plan_{datetime.now().strftime('%Y%m%d')}.md",
                mime="text/markdown"
            )
    
    with col2:
        # Generate CSV export
        if st.button("📊 Export Data as CSV"):
            csv_data = top_recommendations.to_csv(index=True)
            st.download_button(
                label="💾 Download CSV Data",
                data=csv_data,
                file_name=f"bigfoot_recommendations_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    # Success metrics
    st.subheader("✅ Research Success Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_sightings = top_recommendations['sighting_count'].sum()
        avg_quality = top_recommendations['quality_score'].mean()
        
        st.markdown(f"""
        <div class="success-box">
            <strong>📈 Coverage Metrics</strong><br>
            • Total sightings covered: {int(total_sightings):,}<br>
            • Average quality score: {avg_quality:.3f}<br>
            • Coverage: {total_sightings/len(df)*100:.1f}% of all sightings
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        years_span = top_recommendations['last_year'].max() - top_recommendations['first_year'].min()
        avg_activity = top_recommendations['sighting_count'].mean()
        
        st.markdown(f"""
        <div class="success-box">
            <strong>⏰ Temporal Metrics</strong><br>
            • Time span covered: {int(years_span)} years<br>
            • Average activity: {avg_activity:.1f} sightings/location<br>
            • Most recent: {int(top_recommendations['last_year'].max())}
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        lat_span = top_recommendations['avg_lat'].max() - top_recommendations['avg_lat'].min()
        lon_span = top_recommendations['avg_lon'].max() - top_recommendations['avg_lon'].min()
        
        st.markdown(f"""
        <div class="success-box">
            <strong>🗺️ Geographic Metrics</strong><br>
            • Latitude span: {lat_span:.1f}°<br>
            • Longitude span: {lon_span:.1f}°<br>
            • Research area: {lat_span * lon_span:.1f} degree²
        </div>
        """, unsafe_allow_html=True)

def show_data_explorer(df):
    """Display comprehensive data explorer"""
    st.header("📝 Data Explorer & Quality Analysis")
    
    # Dataset overview
    st.subheader("📊 Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Records", f"{len(df):,}")
        st.caption("Cleaned and validated")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Data Features", len(df.columns))
        st.caption("Available columns")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        date_span = df['year'].max() - df['year'].min()
        st.metric("Date Range", f"{date_span} years")
        st.caption(f"{df['year'].min()}-{df['year'].max()}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        completeness = (1 - df.isnull().any(axis=1).mean()) * 100
        st.metric("Data Completeness", f"{completeness:.1f}%")
        st.caption("Records with all fields")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Advanced search and filtering
    st.subheader("🔍 Advanced Search & Filtering")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Text search
        search_term = st.text_input("🔍 Search in titles/descriptions:", 
                                   placeholder="Enter keywords to search...")
        
        search_column = st.selectbox("Search in column:", 
                                   ['title', 'description'] if 'description' in df.columns else ['title'])
    
    with col2:
        # Advanced filters
        st.write("**Advanced Filters:**")
        
        # Coordinate range filter
        coord_filter = st.checkbox("Enable coordinate range filter")
        if coord_filter:
            lat_range = st.slider("Latitude range:", 
                                 float(df['latitude'].min()), 
                                 float(df['latitude'].max()),
                                 (float(df['latitude'].min()), float(df['latitude'].max())))
            lon_range = st.slider("Longitude range:",
                                 float(df['longitude'].min()),
                                 float(df['longitude'].max()),
                                 (float(df['longitude'].min()), float(df['longitude'].max())))
    
    # Apply filters
    filtered_data = df.copy()
    
    # Apply text search
    if search_term:
        if search_column in filtered_data.columns:
            mask = filtered_data[search_column].str.contains(search_term, case=False, na=False)
            filtered_data = filtered_data[mask]
            st.info(f"🔍 Found {len(filtered_data)} records containing '{search_term}'")
    
    # Apply coordinate filter
    if coord_filter:
        filtered_data = filtered_data[
            (filtered_data['latitude'] >= lat_range[0]) & 
            (filtered_data['latitude'] <= lat_range[1]) &
            (filtered_data['longitude'] >= lon_range[0]) & 
            (filtered_data['longitude'] <= lon_range[1])
        ]
        st.info(f"📍 Coordinate filter applied: {len(filtered_data)} records in range")
    
    # Display filtered data
    st.subheader("📋 Data Table")
    
    if len(filtered_data) > 0:
        # Column selection for display
        available_columns = list(filtered_data.columns)
        default_columns = ['number', 'title', 'classification', 'timestamp', 'latitude', 'longitude']
        default_columns = [col for col in default_columns if col in available_columns]
        
        selected_columns = st.multiselect(
            "Select columns to display:",
            available_columns,
            default=default_columns
        )
        
        if selected_columns:
            # Pagination
            page_size = st.selectbox("Records per page:", [25, 50, 100, 200], index=1)
            total_pages = (len(filtered_data) - 1) // page_size + 1
            
            if total_pages > 1:
                page = st.number_input("Page:", 1, total_pages, 1) - 1
                start_idx = page * page_size
                end_idx = min(start_idx + page_size, len(filtered_data))
                
                st.write(f"Showing records {start_idx + 1}-{end_idx} of {len(filtered_data)}")
                display_data = filtered_data.iloc[start_idx:end_idx][selected_columns]
            else:
                display_data = filtered_data[selected_columns]
            
            # Format display data
            if 'latitude' in display_data.columns:
                display_data['latitude'] = display_data['latitude'].round(4)
            if 'longitude' in display_data.columns:
                display_data['longitude'] = display_data['longitude'].round(4)
            
            st.dataframe(display_data, use_container_width=True, hide_index=True)
            
            # Export filtered data
            if st.button("📤 Export Filtered Data"):
                csv_data = filtered_data[selected_columns].to_csv(index=False)
                st.download_button(
                    label="💾 Download CSV",
                    data=csv_data,
                    file_name=f"bigfoot_filtered_data_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
        else:
            st.warning("⚠️ Please select at least one column to display")
    else:
        st.warning("⚠️ No records match your search criteria")
    
    # Data quality analysis
    st.subheader("📊 Data Quality Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Missing data analysis
        st.write("**Missing Data Analysis:**")
        
        missing_data = df.isnull().sum()
        missing_pct = (missing_data / len(df)) * 100
        
        quality_df = pd.DataFrame({
            'Column': missing_data.index,
            'Missing_Count': missing_data.values,
            'Missing_Percentage': missing_pct.values
        }).sort_values('Missing_Percentage', ascending=False)
        
        # Only show columns with missing data
        quality_df = quality_df[quality_df['Missing_Count'] > 0]
        
        if len(quality_df) > 0:
            fig = px.bar(quality_df, x='Column', y='Missing_Percentage',
                        title="Missing Data by Column",
                        color='Missing_Percentage',
                        color_continuous_scale='Reds',
                        text='Missing_Count')
            
            fig.update_layout(xaxis_tickangle=45)
            fig.update_traces(texttemplate='%{text}', textposition='outside')
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("✅ No missing data found in any columns!")
    
    with col2:
        # Data distribution analysis
        st.write("**Data Distribution Analysis:**")
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if numeric_columns:
            selected_numeric = st.selectbox("Select numeric column to analyze:", numeric_columns)
            
            if selected_numeric:
                fig = px.histogram(df, x=selected_numeric,
                                  title=f"Distribution of {selected_numeric}",
                                  color_discrete_sequence=['#2E7D32'])
                
                # Add statistics
                mean_val = df[selected_numeric].mean()
                median_val = df[selected_numeric].median()
                
                fig.add_vline(x=mean_val, line_dash="dash", line_color="red",
                             annotation_text=f"Mean: {mean_val:.2f}")
                fig.add_vline(x=median_val, line_dash="dash", line_color="blue",
                             annotation_text=f"Median: {median_val:.2f}")
                
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("ℹ️ No numeric columns available for distribution analysis")
    
    # Statistical summary
    st.subheader("📈 Statistical Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Geographic Statistics:**")
        
        geo_stats = {
            'Latitude': {
                'Min': df['latitude'].min(),
                'Max': df['latitude'].max(),
                'Mean': df['latitude'].mean(),
                'Std': df['latitude'].std()
            },
            'Longitude': {
                'Min': df['longitude'].min(),
                'Max': df['longitude'].max(), 
                'Mean': df['longitude'].mean(),
                'Std': df['longitude'].std()
            }
        }
        
        geo_df = pd.DataFrame(geo_stats).round(4)
        st.dataframe(geo_df, use_container_width=True)
        
        # Geographic insights
        lat_span = df['latitude'].max() - df['latitude'].min()
        lon_span = df['longitude'].max() - df['longitude'].min()
        
        st.write(f"📍 **Coverage Area:** {lat_span:.1f}° × {lon_span:.1f}°")
        st.write(f"🗺️ **Geographic Center:** {df['latitude'].mean():.2f}°N, {df['longitude'].mean():.2f}°W")
    
    with col2:
        st.write("**Temporal Statistics:**")
        
        temporal_stats = {
            'Years': {
                'Min': int(df['year'].min()),
                'Max': int(df['year'].max()),
                'Span': int(df['year'].max() - df['year'].min()),
                'Mode': int(df['year'].mode().iloc[0])
            },
            'Records': {
                'Total': len(df),
                'Per Year': len(df) / (df['year'].max() - df['year'].min()),
                'Peak Year Count': df['year'].value_counts().max(),
                'Unique Years': df['year'].nunique()
            }
        }
        
        for category, stats in temporal_stats.items():
            st.write(f"**{category}:**")
            for stat, value in stats.items():
                if isinstance(value, float):
                    st.write(f"• {stat}: {value:.1f}")
                else:
                    st.write(f"• {stat}: {value:,}")
    
    # Classification analysis
    st.subheader("🏷️ Classification Quality Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Classification distribution
        class_counts = df['classification'].value_counts()
        class_percentages = (class_counts / len(df) * 100).round(1)
        
        st.write("**Classification Breakdown:**")
        for class_type, count in class_counts.items():
            pct = class_percentages[class_type]
            st.write(f"• **{class_type}:** {count:,} ({pct}%)")
        
        # Quality score
        quality_score = (df['classification'] == 'Class A').mean() * 100
        st.metric("Overall Quality Score", f"{quality_score:.1f}%", 
                 help="Percentage of Class A (highest quality) reports")
    
    with col2:
        # Temporal quality trends
        quality_by_year = df.groupby('year')['classification'].apply(lambda x: (x == 'Class A').mean() * 100).reset_index()
        quality_by_year.columns = ['Year', 'Quality_Percentage']
        
        fig = px.line(quality_by_year, x='Year', y='Quality_Percentage',
                     title="Quality Trends Over Time",
                     color_discrete_sequence=['#2E7D32'])
        
        fig.update_layout(
            yaxis_title="Class A Percentage (%)",
            xaxis_title="Year"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        # Geographic quality distribution
        if len(df) > 100:  # Only if we have enough data
            # Create quality heatmap by region
            df_quality = df.copy()
            df_quality['lat_bin'] = (df_quality['latitude'] // 2) * 2
            df_quality['lon_bin'] = (df_quality['longitude'] // 2) * 2
            
            quality_by_region = df_quality.groupby(['lat_bin', 'lon_bin'])['classification'].apply(
                lambda x: (x == 'Class A').mean() * 100
            ).reset_index()
            quality_by_region.columns = ['Latitude', 'Longitude', 'Quality']
            
            if len(quality_by_region) > 1:
                fig = px.scatter(quality_by_region, x='Longitude', y='Latitude', 
                               color='Quality', size='Quality',
                               title="Quality Distribution by Region",
                               color_continuous_scale='RdYlGn')
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Insufficient data for geographic quality analysis")
        else:
            st.info("More data needed for regional quality analysis")

def show_about_section():
    """Display information about the project"""
    st.header("ℹ️ About This Project")
    
    # Project overview
    st.markdown("""
    <div class="info-box">
        <h3>🦶 Bigfoot Sightings ML Dashboard</h3>
        <p>A comprehensive machine learning analysis platform for exploring patterns in Bigfoot/Sasquatch sighting reports 
        from the Bigfoot Field Researchers Organization (BFRO) database.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Features overview
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Dashboard Features")
        
        features = [
            "📊 **Overview Dashboard** - Key metrics and trends",
            "🗺️ **Geographic Analysis** - Clustering and hotspot detection",
            "📅 **Temporal Patterns** - Seasonal and yearly analysis",
            "🤖 **Machine Learning** - Classification prediction models",
            "🔍 **Anomaly Detection** - Identify suspicious reports",
            "🎯 **Location Recommendations** - Optimal research locations",
            "📝 **Data Explorer** - Advanced search and filtering"
        ]
        
        for feature in features:
            st.markdown(feature)
    
    with col2:
        st.subheader("🛠️ Technical Stack")
        
        technologies = [
            "**Frontend:** Streamlit",
            "**Visualization:** Plotly, Folium",
            "**Machine Learning:** Scikit-learn",
            "**Data Processing:** Pandas, NumPy",
            "**Maps:** OpenStreetMap, Folium",
            "**Deployment:** Streamlit Cloud",
            "**Language:** Python 3.8+"
        ]
        
        for tech in technologies:
            st.markdown(f"• {tech}")
    
    # Data information
    st.subheader("📊 Data Source Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **🦶 BFRO Database Details:**
        - **Source:** Bigfoot Field Researchers Organization
        - **Website:** [bfro.net](http://bfro.net/GDB/)
        - **Coverage:** North America (primarily USA/Canada)
        - **Time Span:** 1960s to present
        - **Record Types:** Eyewitness reports, track findings, audio encounters
        """)
    
    with col2:
        st.markdown("""
        **📋 Classification System:**
        - **Class A:** Clear visual sighting (highest quality)
        - **Class B:** Possible visual or clear audio encounter
        - **Class C:** Second-hand reports or stories
        - **Validation:** Reports investigated by BFRO researchers
        - **Geocoding:** GPS coordinates when available
        """)
    
    # Methodology
    st.subheader("🔬 Analysis Methodology")
    
    methodology_tabs = st.tabs(["Data Processing", "Machine Learning", "Visualization", "Quality Assurance"])
    
    with methodology_tabs[0]:
        st.markdown("""
        **📋 Data Processing Pipeline:**
        1. **Data Ingestion:** Load BFRO CSV data from GitHub repository
        2. **Data Cleaning:** Remove invalid coordinates, parse timestamps
        3. **Feature Engineering:** Extract temporal features, calculate distances
        4. **Validation:** Filter unrealistic years, standardize classifications
        5. **Enrichment:** Add seasonal indicators, geographic clusters
        
        **🧹 Quality Controls:**
        - Coordinate validation (-90° to 90° lat, -180° to 180° lon)
        - Date parsing with multiple format support
        - Classification standardization and mapping
        - Outlier detection and removal
        """)
    
    with methodology_tabs[1]:
        st.markdown("""
        **🤖 Machine Learning Approaches:**
        
        **Classification Models:**
        - Random Forest for sighting quality prediction
        - Feature importance analysis
        - Cross-validation for model validation
        
        **Clustering Analysis:**
        - K-Means clustering for geographic hotspots
        - DBSCAN for density-based clustering
        - Silhouette analysis for optimal cluster count
        
        **Anomaly Detection:**
        - Isolation Forest for outlier identification
        - Multi-dimensional anomaly scoring
        - Temporal and geographic anomaly patterns
        
        **Feature Engineering:**
        - Cyclical encoding for temporal features
        - Geographic distance calculations
        - Derived categorical variables
        """)
    
    with methodology_tabs[2]:
        st.markdown("""
        **📊 Visualization Strategy:**
        
        **Interactive Maps:**
        - Folium for geographic visualization
        - Multiple map styles and layers
        - Custom markers and popups
        
        **Statistical Charts:**
        - Plotly for interactive charts
        - Time series analysis
        - Distribution plots and heatmaps
        
        **Dashboard Design:**
        - Responsive layout for all devices
        - Real-time filtering and updates
        - Export capabilities for research use
        
        **User Experience:**
        - Intuitive navigation and controls
        - Performance optimization for large datasets
        - Comprehensive help and documentation
        """)
    
    with methodology_tabs[3]:
        st.markdown("""
        **✅ Quality Assurance Measures:**
        
        **Data Validation:**
        - Multi-stage data cleaning pipeline
        - Coordinate and date validation
        - Statistical outlier detection
        
        **Model Validation:**
        - Cross-validation for all ML models
        - Performance metrics tracking
        - Feature importance analysis
        
        **Testing:**
        - Error handling for edge cases
        - Performance optimization
        - User interface testing
        
        **Documentation:**
        - Comprehensive code documentation
        - User guide and tutorials
        - Technical methodology disclosure
        """)
    
    # Usage guidelines
    st.subheader("📋 Usage Guidelines & Limitations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **✅ Appropriate Uses:**
        - Academic research and analysis
        - Educational exploration of data science
        - Pattern recognition in geographic data
        - Statistical analysis of temporal trends
        - Hypothesis generation for field research
        """)
    
    with col2:
        st.markdown("""
        **⚠️ Limitations & Disclaimers:**
        - Data represents reported sightings, not verified encounters
        - Geographic accuracy varies by report
        - Reporting bias may affect patterns
        - Model predictions are for research purposes only
        - Results should not be considered scientific proof
        """)
    
    # Contact and credits
    st.subheader("👥 Credits & Attribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **👤 Project Creator:**
        - **Developer:** Meowmixforme
        - **Date:** July 22, 2025
        - **Purpose:** Educational Data Science Project
        """)
    
    with col2:
        st.markdown("""
        **🙏 Data & Tools:**
        - **Data Source:** BFRO (Bigfoot Field Researchers Organization)
        - **Hosting:** Streamlit Community Cloud
        - **Maps:** OpenStreetMap contributors
        - **Icons:** Streamlit emoji support
        """)
    
    # Legal information
    st.subheader("⚖️ Legal Information")
    
    st.markdown("""
    <div class="attribution-box">
        <strong>📜 Terms of Use & Attribution</strong><br><br>
        
        <strong>Data Usage:</strong> This application uses data from the Bigfoot Field Researchers Organization (BFRO) 
        under Fair Use provisions for educational and research purposes. The BFRO database is publicly accessible 
        and widely used in academic research.<br><br>
        
        <strong>Educational Purpose:</strong> This dashboard is created for educational purposes to demonstrate 
        data science and machine learning techniques. It is not intended for commercial use.<br><br>
        
        <strong>Disclaimer:</strong> This application provides analysis of reported sightings and should not be 
        considered as scientific evidence for the existence of Bigfoot/Sasquatch. All analyses are based on 
        user-reported data and are subject to reporting biases and inaccuracies.<br><br>
        
        <strong>Privacy:</strong> No personal information is collected or stored by this application. All data 
        displayed is already publicly available through the BFRO database.<br><br>
        
        <strong>Attribution:</strong> When using results from this dashboard, please cite both the original BFRO 
        database and this analysis tool appropriately.
    </div>
    """, unsafe_allow_html=True)
    
    # Version information
    st.subheader("🔢 Version Information")
    
    version_info = {
        "Dashboard Version": "1.0.0",
        "Release Date": "July 22, 2025",
        "Python Version": "3.8+",
        "Streamlit Version": "1.28+",
        "Last Updated": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "Dataset Version": "BFRO Current (as of July 2025)"
    }
    
    for key, value in version_info.items():
        st.write(f"• **{key}:** {value}")

def show_footer():
    """Display application footer"""
    st.markdown("""
    <div class="footer">
        <strong>🦶 Bigfoot ML Dashboard</strong><br>
        Created by <strong>Meowmixforme</strong> | July 22, 2025<br><br>
        
        <strong>Data Source:</strong> Bigfoot Field Researchers Organization (BFRO)<br>
        <strong>Built with:</strong> Streamlit • Plotly • Folium • Scikit-learn<br><br>
        
        <em>🔬 Educational project demonstrating machine learning and data visualization techniques</em><br>
        <em>⚖️ Used under Fair Use for educational and research purposes</em><br><br>
        
        <small>💡 For questions or suggestions, please refer to the About section</small>
    </div>
    """, unsafe_allow_html=True)

# Import required metrics functions
from sklearn.metrics import precision_score

# Run the application
if __name__ == "__main__":
    main()
    show_footer()
