"""
Data explorer page
"""

import streamlit as st

def show_data_explorer(df):
    """Display advanced data exploration tools"""
    st.header("📝 Data Explorer")
    
    # Search functionality
    search_term = st.text_input("🔍 Search in data:", placeholder="Enter search term...")
    
    if search_term:
        # Simple search implementation
        mask = df.astype(str).apply(lambda x: x.str.contains(search_term, case=False, na=False)).any(axis=1)
        filtered_df = df[mask]
        st.write(f"Found {len(filtered_df)} records matching '{search_term}'")
    else:
        filtered_df = df
    
    # Data table
    st.subheader("📊 Data Table")
    st.dataframe(filtered_df.head(100), use_container_width=True)
    
    if len(filtered_df) > 100:
        st.info(f"Showing first 100 of {len(filtered_df)} records. Use search to filter.")
    
    # Download option
    if st.button("📥 Download Filtered Data"):
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="bigfoot_data_filtered.csv",
            mime="text/csv"
        )