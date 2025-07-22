"""
Location recommendations page
"""

import streamlit as st
import plotly.express as px
from config.settings import COLOR_PALETTE

def show_recommendations(df):
    """Display location recommendations for research"""
    st.header("🎯 Location Recommendations")
    
    st.markdown("""
    <div class="info-box">
        <strong>🔍 Research Location Analysis</strong><br>
        Based on sighting density, quality, and recent activity patterns.
    </div>
    """, unsafe_allow_html=True)
    
    # Calculate recommendation scores
    df_rec = df.copy()
    df_rec['lat_bin'] = (df_rec['latitude'] // 1) * 1  # 1-degree bins
    df_rec['lon_bin'] = (df_rec['longitude'] // 1) * 1
    
    recommendations = df_rec.groupby(['lat_bin', 'lon_bin']).agg({
        'classification': ['count', lambda x: (x == 'Class A').mean()],
        'year': 'max'
    }).round(3)
    
    recommendations.columns = ['total_sightings', 'quality_score', 'last_sighting']
    recommendations['recency_score'] = (recommendations['last_sighting'] - 1950) / (2025 - 1950)
    recommendations['recommendation_score'] = (
        recommendations['total_sightings'] * 0.4 +
        recommendations['quality_score'] * 100 * 0.4 +
        recommendations['recency_score'] * 100 * 0.2
    )
    
    top_locations = recommendations.nlargest(10, 'recommendation_score').reset_index()
    
    # Display top recommendations
    st.subheader("🏆 Top Research Locations")
    
    for i, row in top_locations.iterrows():
        with st.expander(f"#{i+1} - {row['lat_bin']}°N, {row['lon_bin']}°W (Score: {row['recommendation_score']:.1f})"):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"📊 **Total Sightings:** {int(row['total_sightings'])}")
                st.write(f"⭐ **Quality Score:** {row['quality_score']:.1%}")
            with col2:
                st.write(f"📅 **Last Sighting:** {int(row['last_sighting'])}")
                st.write(f"🎯 **Recommendation Score:** {row['recommendation_score']:.1f}")
    
    # Visualization
    fig = px.scatter(top_locations, x='lon_bin', y='lat_bin',
                    size='total_sightings',
                    color='recommendation_score',
                    title="🗺️ Recommended Research Locations",
                    color_continuous_scale='Reds')
    
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    st.plotly_chart(fig, use_container_width=True)