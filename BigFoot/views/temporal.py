"""
Temporal pattern analysis page
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from config.settings import COLOR_PALETTE

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
            line=dict(color=COLOR_PALETTE['accent'], width=2),
            marker=dict(size=6, opacity=0.7),
            hovertemplate='<b>%{x}</b><br>Sightings: %{y}<extra></extra>'
        ))
        
        # Rolling average
        fig.add_trace(go.Scatter(
            x=yearly_data['year'], 
            y=yearly_data['rolling_avg'],
            mode='lines', 
            name='5-Year Average',
            line=dict(color=COLOR_PALETTE['danger'], width=3),
            hovertemplate='<b>%{x}</b><br>5-Year Avg: %{y:.1f}<extra></extra>'
        ))
        
        fig.update_layout(
            title="📈 Yearly Trends with Moving Average",
            xaxis_title="Year",
            yaxis_title="Number of Sightings",
            hovermode='x unified',
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Cumulative sightings
        fig = px.line(yearly_data, x='year', y='cumulative',
                     title="📊 Cumulative Sightings Over Time",
                     color_discrete_sequence=[COLOR_PALETTE['success']])
        
        fig.update_layout(
            xaxis_title="Year",
            yaxis_title="Cumulative Sightings",
            hovermode='x',
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Monthly and seasonal patterns
    col1, col2 = st.columns(2)
    
    with col1:
        # Monthly distribution
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        monthly_counts = df.groupby('month').size().reindex(range(1, 13), fill_value=0)
        
        fig = px.bar(x=month_names, y=monthly_counts.values,
                    title="📅 Monthly Distribution",
                    color=monthly_counts.values,
                    color_continuous_scale='Blues')
        
        fig.update_layout(
            showlegend=False,
            xaxis_title="Month",
            yaxis_title="Number of Sightings",
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Seasonal analysis
        seasonal_data = df.groupby('season').size().reset_index(name='count')
        
        fig = px.bar(seasonal_data, x='season', y='count',
                    title="🍂 Seasonal Distribution",
                    color='count',
                    color_continuous_scale='Greens')
        
        fig.update_layout(
            showlegend=False,
            xaxis_title="Season",
            yaxis_title="Number of Sightings",
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Key insights
    st.subheader("🔍 Key Temporal Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        peak_month = df['month'].mode().iloc[0]
        month_names_full = ['', 'January', 'February', 'March', 'April', 'May', 'June',
                           'July', 'August', 'September', 'October', 'November', 'December']
        
        st.markdown(f"""
        <div class="info-box">
            <strong>📅 Peak Activity</strong><br>
            • Peak month: {month_names_full[peak_month]}<br>
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
            • Recent trend: {growth_rate:.1f}/year<br>
            • Data span: {df['year'].max() - df['year'].min()} years
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        summer_pct = (df['season'] == 'Summer').mean() * 100
        
        st.markdown(f"""
        <div class="info-box">
            <strong>📊 Activity Patterns</strong><br>
            • Summer reports: {summer_pct:.1f}%<br>
            • Recent activity: {yearly_data['sightings'].iloc[-3:].mean():.0f}/year<br>
            • Peak decade: {(df['year'].mode().iloc[0] // 10) * 10}s
        </div>
        """, unsafe_allow_html=True)