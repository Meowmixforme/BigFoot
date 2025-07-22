"""
Geographic analysis page with clustering
"""

import streamlit as st
import plotly.express as px
import folium
from streamlit_folium import st_folium
from utils.clustering import perform_clustering
from config.settings import COLOR_PALETTE, MAP_CONFIG

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
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=MAP_CONFIG['default_zoom'])
    
    # Add markers with professional colors
    for idx, row in df_map.iterrows():
        if show_clusters and 'cluster' in row:
            color = MAP_CONFIG['cluster_colors'][row['cluster'] % len(MAP_CONFIG['cluster_colors'])]
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
            color = MAP_CONFIG['classification_colors'].get(row['classification'], '#666666')
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
            color='white',
            fillColor=color,
            fillOpacity=0.8,
            weight=2,
            opacity=1.0
        ).add_to(m)
    
    # Display map
    map_data = st_folium(m, width=700, height=500)
    
    # Analysis results
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Cluster Analysis Results")
        
        if 'cluster' in df_clustered.columns:
            # Display cluster information
            for cluster_id in sorted(df_clustered['cluster'].unique()):
                cluster_data = df_clustered[df_clustered['cluster'] == cluster_id]
                
                with st.expander(f"🎯 Cluster {cluster_id} - {len(cluster_data)} sightings"):
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.write(f"📍 **Center:** {cluster_data['latitude'].mean():.2f}°N, {cluster_data['longitude'].mean():.2f}°W")
                        st.write(f"📊 **Count:** {len(cluster_data)} sightings")
                    with col_b:
                        st.write(f"📅 **Period:** {cluster_data['year'].min()}-{cluster_data['year'].max()}")
                        st.write(f"⭐ **Quality:** {(cluster_data['classification'] == 'Class A').mean():.1%} Class A")
    
    with col2:
        st.subheader("🎯 Geographic Hotspots")
        
        # Calculate density
        df_density = df_clustered.copy()
        df_density['lat_bin'] = (df_density['latitude'] // 2) * 2
        df_density['lon_bin'] = (df_density['longitude'] // 2) * 2
        
        region_counts = df_density.groupby(['lat_bin', 'lon_bin']).size().reset_index(name='count')
        top_regions = region_counts.nlargest(5, 'count')
        
        st.write("**🔥 High-Density Regions:**")
        for i, row in enumerate(top_regions.itertuples(), 1):
            st.write(f"**{i}.** {row.lat_bin}°N, {row.lon_bin}°W - **{row.count}** sightings")