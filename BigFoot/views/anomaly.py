"""
Anomaly detection analysis page
"""

import streamlit as st
import plotly.express as px
from utils.ml_models import detect_anomalies
from config.settings import COLOR_PALETTE

def show_anomaly_analysis(df):
    """Display anomaly detection analysis"""
    st.header("🔍 Anomaly Detection Analysis")
    
    # Perform anomaly detection
    with st.spinner("🔄 Detecting anomalous sightings..."):
        df_anomaly, iso_forest = detect_anomalies(df)
    
    if iso_forest is None:
        st.error("❌ Anomaly detection failed")
        return
    
    # Anomaly overview
    anomaly_count = df_anomaly['is_anomaly'].sum()
    anomaly_pct = anomaly_count / len(df_anomaly) * 100
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Anomalies", f"{anomaly_count:,}")
        st.caption(f"{anomaly_pct:.1f}% of all sightings")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        normal_count = len(df_anomaly) - anomaly_count
        st.metric("Normal Sightings", f"{normal_count:,}")
        st.caption(f"{100-anomaly_pct:.1f}% of all sightings")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        avg_score = df_anomaly['anomaly_score'].mean()
        st.metric("Avg Anomaly Score", f"{avg_score:.3f}")
        st.caption("Lower = more anomalous")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Anomaly visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Geographic distribution of anomalies
        fig = px.scatter(df_anomaly, x='longitude', y='latitude',
                        color='is_anomaly',
                        title="🗺️ Geographic Distribution of Anomalies",
                        color_discrete_map={True: COLOR_PALETTE['danger'], False: COLOR_PALETTE['accent']},
                        opacity=0.6)
        
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Anomaly scores distribution
        fig = px.histogram(df_anomaly, x='anomaly_score',
                          title="📊 Anomaly Score Distribution",
                          color_discrete_sequence=[COLOR_PALETTE['accent']])
        
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Top anomalies
    st.subheader("🚨 Most Anomalous Sightings")
    
    top_anomalies = df_anomaly[df_anomaly['is_anomaly']].nsmallest(10, 'anomaly_score')
    
    for idx, row in top_anomalies.iterrows():
        with st.expander(f"🚨 Anomaly #{idx} - Score: {row['anomaly_score']:.3f}"):
            col_a, col_b = st.columns(2)
            with col_a:
                st.write(f"📍 **Location:** {row['latitude']:.3f}°N, {row['longitude']:.3f}°W")
                st.write(f"📅 **Date:** {str(row['timestamp'])[:10]}")
            with col_b:
                st.write(f"🏷️ **Classification:** {row['classification']}")
                st.write(f"🍂 **Season:** {row['season']}")