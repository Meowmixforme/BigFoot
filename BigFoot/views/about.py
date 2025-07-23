"""
About page for the BigFoot dashboard
"""

import streamlit as st

def show_about_section():
    """Display about section with project information"""
    st.header("ℹ️ About This Project")
    
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
    
    # Legal information - FIXED VERSION
    st.subheader("⚖️ Legal Information")
    
    # Replace the HTML version with proper Streamlit markdown
    st.markdown("""
    **📜 Terms of Use & Attribution**
    
    **Data Usage:** This application uses data from the Bigfoot Field Researchers Organization (BFRO) 
    under Fair Use provisions for educational and research purposes.
    
    **Educational Purpose:** This dashboard is created for educational purposes to demonstrate 
    data science and machine learning techniques.
    
    **Disclaimer:** This application provides analysis of reported sightings and should not be 
    considered as scientific evidence for the existence of Bigfoot/Sasquatch.
    """)