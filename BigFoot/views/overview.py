"""
Overview dashboard page with dark theme charts
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from config.settings import COLOR_PALETTE
from utils.chart_theme import apply_dark_theme_to_fig

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
    
    # Main charts with dark theme
    col1, col2 = st.columns(2)
    
    with col1:
        # Yearly trend with dark theme
        yearly_counts = df.groupby('year').size().reset_index(name='sightings')
        
        fig = px.scatter(yearly_counts, x='year', y='sightings', 
                        title="📈 Sightings Over Time",
                        trendline="lowess",
                        color_discrete_sequence=[COLOR_PALETTE['accent']])
        
        fig.update_traces(marker=dict(size=8, opacity=0.8))
        fig = apply_dark_theme_to_fig(fig)
        fig.update_layout(
            showlegend=False,
            hovermode='x unified',
            xaxis_title="Year",
            yaxis_title="Number of Sightings",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Enhanced classification distribution with dark theme
        class_counts = df['classification'].value_counts()
        colors = [COLOR_PALETTE['danger'], COLOR_PALETTE['warning'], COLOR_PALETTE['info']]
        
        fig = px.pie(values=class_counts.values, names=class_counts.index,
                    title="🏷️ Classification Distribution",
                    color_discrete_sequence=colors)
        
        fig.update_traces(
            textposition='inside', 
            textinfo='percent+label',
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>',
            textfont={'color': 'white', 'size': 12}
        )
        fig = apply_dark_theme_to_fig(fig)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Add more charts following the same pattern...