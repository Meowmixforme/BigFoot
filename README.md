# 🦶 BigFoot

## Project Overview

BigFoot is a comprehensive analytics tool designed for the analysis of Bigfoot sighting data, utilising machine learning and data visualisation. The application imports verified sighting reports from the Bigfoot Field Researchers Organisation (BFRO) database and processes them to reveal insights about when, where, and what types of encounters occur.

## Data Source

The project employs data from the BFRO (Bigfoot Field Researchers Organisation) database, which includes:

- Over 4,000 North American sightings from the 1960s to the present
- Reports classified into Class A, B, and C categories
- Geographic coordinates and timestamps
- Eyewitness reports, track findings, and audio encounters

**BFRO Classification System:**
- **Class A:** Clear visual sighting in good conditions by a reliable witness (highest quality)
- **Class B:** Possible visual or clear audio encounter, but lacking Class A clarity
- **Class C:** Second-hand reports, stories, or anecdotal evidence (least reliable)

## Repository Structure

```
BigFoot/
├── BigFoot.py              # Main Streamlit application entry point
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
├── Dockerfile              # Container configuration
├── BigFoot/                # Main application package
│   ├── config/             # Configuration files and styles
│   ├── components/         # UI components (header, sidebar, footer)
│   ├── data/               # Data loading and processing
│   └── views/              # Individual page views
```

## Main Application Features

The BigFoot dashboard comprises several interactive Streamlit pages:

1. **Overview Dashboard**
   - Key metrics and statistics about the dataset
   - Time trends and data insights
   - Summary of sighting counts by classification
   - Basic data quality information

2. **Geographic Analysis**
   - Interactive maps showing sighting locations
   - Geographic clustering using KMeans and DBSCAN algorithms
   - Hotspot detection and visualisation
   - State and regional analysis
   - Coordinate validation and mapping

3. **Temporal Analysis**
   - Seasonal patterns in sighting frequency
   - Year-over-year trends
   - Month and day-of-week analysis
   - Time-based anomalies
   - Historical reporting patterns

4. **Machine Learning Analysis**
   - Random Forest classification models for predicting report credibility
   - Feature importance analysis
   - Cross-validation and model performance metrics
   - Prediction tools for hypothetical sightings
   - Model training on spatial and temporal features

5. **Anomaly Detection**
   - Isolation Forest algorithm for identifying suspicious reports
   - Statistical outlier detection
   - Quality control flagging
   - Unusual pattern identification
   - Data integrity checking

6. **Location Recommendations**
   - Data-driven expedition planning
   - Ranking geographical regions for future research
   - Analysis based on sighting density, quality, and recency
   - Research hotspot identification
   - Field investigation planning tools

7. **Advanced Data Explorer**
   - Powerful search and filtering capabilities
   - Statistical summaries and analysis
   - Data export functionality (CSV, JSON formats)
   - Interactive data tables
   - Custom filtering options
   - Report detail viewing

8. **About Section**
   - Project methodology and data processing pipeline
   - Machine learning approaches and algorithms
   - Visualisation techniques
   - Quality assurance measures
   - Legal information and attribution
   - Version information and contact details

## Technical Implementation

### Data Processing Pipeline

- **Data Ingestion:** Load BFRO CSV data from GitHub repository
- **Data Cleaning:** Remove invalid coordinates, parse timestamps
- **Feature Engineering:** Extract temporal features, calculate distances
- **Validation:** Filter unrealistic years, standardise classifications
- **Enrichment:** Add seasonal indicators, geographic clusters

### Quality Controls

- Coordinate validation (-90° to 90° latitude, -180° to 180° longitude)
- Date parsing with multiple format support
- Classification standardisation and mapping
- Outlier detection and removal

### Machine Learning Approaches

- Random Forest for sighting quality prediction
- Isolation Forest for anomaly detection
- Feature importance analysis
- Cross-validation for model validation
- Clustering algorithms for geographic analysis

## Key Capabilities

- **Data Cleaning and Preparation:** Validates coordinates, timestamps, and standardises classifications
- **Geospatial Analysis:** Interactive mapping with Folium and Plotly
- **Temporal Analysis:** Examines trends by various time periods
- **Report Classification:** ML models to predict report credibility
- **Anomaly Detection:** Identifies unusual or suspicious reports
- **Interactive Prediction Tool:** Users can input hypothetical scenarios
- **Exportable Research Plans:** Generate and export filtered data for research

## Installation and Usage

**Requirements:**
- Python 3.8 or higher
- Streamlit
- Various data science libraries (pandas, scikit-learn, folium, plotly)

**Setup:**
```bash
git clone https://github.com/Meowmixforme/BigFoot.git
cd BigFoot
pip install -r requirements.txt
streamlit run BigFoot.py
```

The application will be available at [http://localhost:8501](http://localhost:8501)

## Research Applications

The dashboard helps address questions such as:

- Where are Bigfoot sightings most common?
- Are there seasonal or annual trends in reports?
- Which locations have the highest quality sightings?
- Are there geographical or temporal anomalies?
- Where should researchers focus future expeditions?
- Can machine learning predict report credibility?

## Legal and Attribution

- Data used under Fair Use for educational and research purposes
- Original data from BFRO publicly accessible database
- No personal information collected or stored
- Educational tool for demonstrating data science techniques
- Includes appropriate disclaimers about scientific evidence

## Version Information

- **Dashboard Version:** 1.0.0
- **Release Date:** 22 July 2025
- **Python Version:** 3.8+
- **Streamlit Community Cloud hosted**

---

This comprehensive dashboard provides researchers, data scientists, and enthusiasts with powerful tools to explore patterns in cryptid encounters, assess report quality, and generate data-driven recommendations for field investigations.
