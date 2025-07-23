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
    
    # Create folium map with proper tile handling
    center_lat = df_map['latitude'].mean()
    center_lon = df_map['longitude'].mean()
    
    # Create map based on selected style with working tile services
    if map_style == "OpenStreetMap":
        m = folium.Map(location=[center_lat, center_lon], zoom_start=MAP_CONFIG['default_zoom'], tiles="OpenStreetMap")
    elif map_style == "Satellite":
        m = folium.Map(location=[center_lat, center_lon], zoom_start=MAP_CONFIG['default_zoom'])
        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Esri WorldImagery",
            name="Satellite",
            overlay=False,
            control=True
        ).add_to(m)
    else:  # Terrain
        m = folium.Map(location=[center_lat, center_lon], zoom_start=MAP_CONFIG['default_zoom'])
        folium.TileLayer(
            tiles="https://tile.opentopomap.org/{z}/{x}/{y}.png",
            attr="OpenTopoMap",
            name="Terrain",
            overlay=False,
            control=True,
            max_zoom=17
        ).add_to(m)
    
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
    
    # Display map with full width
    map_data = st_folium(m, use_container_width=True, height=500)
    
    # Visual legend below the map
    st.markdown("### 🎨 Map Legend")
    
    if show_clusters and 'cluster' in df_clustered.columns:
        st.markdown("**Cluster Colors:**")
        
        # Create visual legend for clusters
        unique_clusters = sorted(df_clustered['cluster'].unique())
        legend_cols = st.columns(min(len(unique_clusters), 5))  # Max 5 columns
        
        for i, cluster_id in enumerate(unique_clusters[:5]):  # Show first 5 clusters
            color = MAP_CONFIG['cluster_colors'][i % len(MAP_CONFIG['cluster_colors'])]
            with legend_cols[i]:
                st.markdown(f"""
                <div style="display: flex; align-items: center; margin: 5px 0;">
                    <div style="width: 20px; height: 20px; background-color: {color}; 
                                border: 2px solid white; border-radius: 50%; margin-right: 8px;
                                box-shadow: 0 2px 4px rgba(0,0,0,0.3);"></div>
                    <span style="font-weight: bold;">Cluster {cluster_id}</span>
                </div>
                """, unsafe_allow_html=True)
        
        if len(unique_clusters) > 5:
            st.info(f"ℹ️ Showing first 5 clusters. Total clusters: {len(unique_clusters)}")
    
    else:
        st.markdown("**Classification Colors:**")
        
        # Create visual legend for classifications with NEW COLORS
        classifications = [
            ('Class A', '#E74C3C', 'Clear Visual Sighting'),
            ('Class B', '#3498DB', 'Sounds/Tracks/Blurry'),
            ('Class C', '#27AE60', 'Second-hand Reports')
        ]
        
        legend_cols = st.columns(3)
        
        for i, (class_name, color, description) in enumerate(classifications):
            with legend_cols[i]:
                st.markdown(f"""
                <div style="display: flex; align-items: center; margin: 5px 0;">
                    <div style="width: 20px; height: 20px; background-color: {color}; 
                                border: 2px solid white; border-radius: 50%; margin-right: 8px;
                                box-shadow: 0 2px 4px rgba(0,0,0,0.3);"></div>
                    <div>
                        <div style="font-weight: bold;">{class_name}</div>
                        <div style="font-size: 12px; color: #666;">{description}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown("---")
    
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