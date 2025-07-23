"""
Footer component for the BigFoot dashboard
"""

import streamlit as st
from datetime import datetime

def render_footer():
    """Display application footer"""
    current_date = datetime.now().strftime("%B %d, %Y")
    
    st.markdown("---")
    
    # Create centered columns for footer content
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown(f"""
        **🦶 Bigfoot**
        
        Created by **Meowmixforme** | {current_date}
        
        **Data Source:** Bigfoot Field Researchers Organization (BFRO)  
        **Built with:** Streamlit • Plotly • Folium • Scikit-learn
        
        🔬 *Educational project demonstrating machine learning and data visualization techniques*  
        ⚖️ *Used under Fair Use for educational and research purposes*
        
        💡 *For questions or suggestions, please refer to the About section*
        """)