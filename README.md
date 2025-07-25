# 🦶 BigFoot

## Project Overview

BigFoot is a comprehensive analytics tool designed for the analysis of Bigfoot sighting data, utilising machine learning and data visualisation. The application imports verified sighting reports from the [BFRO](https://www.bfro.net/) (Bigfoot Field Researchers Organisation) and provides interactive dashboards, maps, and predictive tools for exploring cryptid reports.

> _I'm a longtime fan of the BFRO and the TV series [Finding Bigfoot](https://www.imdb.com/title/tt1948830/?ref_=nv_sr_srsg_0_tt_8_nm_0_in_0_q_finding%2520bigf), which inspired me to create this project to explore and visualise Bigfoot sighting data._



<img width="1024" height="1024" alt="freepik__the-style-is-candid-image-photography-with-natural__3245" src="https://github.com/user-attachments/assets/08ad5fed-8333-457c-a9bb-2d3632600b10" />


## Data Source

The primary dataset used in this project is [bfro_locations.csv](https://raw.githubusercontent.com/kittychew/bigfoot-sightings-analysis/main/bfro_locations.csv), originally sourced and cleaned by [@kittychew/bigfoot-sightings-analysis](https://github.com/kittychew/bigfoot-sightings-analysis) from the [@timothyrenner/bfro_sightings_data](https://data.world/timothyrenner/bfro-sightings-data) dataset, and ultimately derived from the [Bigfoot Field Researchers Organization (BFRO)](https://www.bfro.net/) database. Additional data files (such as bfro_reports.json and bfro_reports_geocoded.csv) from @kittychew’s repository are also referenced.

**Attribution:**  
- Special thanks to the Bigfoot Field Researchers Organization (BFRO) for providing the original dataset.  
- Thanks to [@kittychew/bigfoot-sightings-analysis](https://github.com/kittychew/bigfoot-sightings-analysis) for data cleaning, processing, and  to [@timothyrenner/bfro_sightings_data](https://data.world/timothyrenner/bfro-sightings-data) for making the dataset publicly available under the MIT License.

The project employs data from the BFRO database, which includes:
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
├── BigFoot/                # Main application package
│   ├── config/             # Configuration files and styles
│   ├── components/         # UI components (header, sidebar, footer)
│   ├── data/               # Data loading and processing
│   └── views/              # Individual page views
├── notebooks/              # Jupyter notebooks for data analysis
│   └── Bigfoot.ipynb       # Exploratory analysis notebook
```

## Main Application Features

The BigFoot dashboard comprises several interactive Streamlit pages:

1. **Overview Dashboard**
   - Key metrics and statistics about the dataset
   - Time trends and data insights
   - Summary of sighting counts by classification
   - Basic data quality information
  

<img width="3777" height="1622" alt="Screenshot 2025-07-23 045821" src="https://github.com/user-attachments/assets/9fb6dd4f-f241-46e7-aecd-e1e378eea3fa" />



2. **Geographic Analysis**
   - Interactive maps showing sighting locations
   - Geographic clustering using [KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html) and [DBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html) algorithms
   - Hotspot detection and visualisation
   - State and regional analysis
   - Coordinate validation and mapping
  

<img width="3139" height="1863" alt="Screenshot 2025-07-23 045924" src="https://github.com/user-attachments/assets/75d8b7bc-6546-4fea-b0e3-b43376c33b68" />


3. **Temporal Analysis**
   - Seasonal patterns in sighting frequency
   - Year-over-year trends
   - Month and day-of-week analysis
   - Time-based anomalies
   - Historical reporting patterns
  

  <img width="3158" height="1844" alt="Screenshot 2025-07-23 045945" src="https://github.com/user-attachments/assets/d73b0ce7-0ed2-4733-b9f8-1b888c712954" />


4. **Machine Learning Analysis**
   - [Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) classification models for predicting report credibility
   - Feature importance analysis
   - Cross-validation and model performance metrics
   - Prediction tools for hypothetical sightings
   - Model training on spatial and temporal features
  
  
<img width="3172" height="1425" alt="Screenshot 2025-07-23 050013" src="https://github.com/user-attachments/assets/8b538928-752c-4412-b043-1a2a05218891" />


5. **Anomaly Detection**
   - [Isolation Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html) algorithm for identifying suspicious reports
   - Statistical outlier detection
   - Quality control flagging
   - Unusual pattern identification
   - Data integrity checking
  
  
<img width="3140" height="1704" alt="Screenshot 2025-07-23 050027" src="https://github.com/user-attachments/assets/3652741f-afd8-4b69-a8b3-2e61763d3af5" />



6. **Location Recommendations**
   - Data-driven expedition planning
   - Ranking geographical regions for future research
   - Analysis based on sighting density, quality, and recency
   - Research hotspot identification
   - Field investigation planning tools
  
<img width="3147" height="1723" alt="Screenshot 2025-07-23 050046" src="https://github.com/user-attachments/assets/a8ea85db-9c7e-42d4-8036-d68f2b172b45" />


7. **Advanced Data Explorer**
   - Powerful search and filtering capabilities
   - Statistical summaries and analysis
   - Data export functionality ([CSV](https://en.wikipedia.org/wiki/Comma-separated_values), [JSON](https://www.json.org/json-en.html) formats)
   - Interactive data tables
   - Custom filtering options
   - Report detail viewing
  

  <img width="3127" height="1177" alt="Screenshot 2025-07-23 050058" src="https://github.com/user-attachments/assets/7daef287-2b9a-4c6e-91af-153a8b74c374" />

8. **About Section**
   - Project methodology and data processing pipeline
   - Machine learning approaches and algorithms
   - Visualisation techniques
   - Quality assurance measures
   - Legal information and attribution
   - Version information and contact details

## Technical Implementation

### Data Processing Pipeline

- **Data Ingestion:** Load BFRO CSV data from [bfro_locations.csv](https://raw.githubusercontent.com/kittychew/bigfoot-sightings-analysis/main/bfro_locations.csv)
- **Data Cleaning:** Remove invalid coordinates, parse timestamps
- **Feature Engineering:** Extract temporal features, calculate distances
- **Validation:** Filter unrealistic years, standardise classifications
- **Enrichment:** Add seasonal indicators, geographic clusters

### Quality Controls

- Coordinate validation (`-90°` to `90°` latitude, `-180°` to `180°` longitude)
- Date parsing with multiple format support
- Classification standardisation and mapping
- Outlier detection and removal

### Machine Learning Approaches

- [Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) for sighting quality prediction
- [Isolation Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html) for anomaly detection
- Feature importance analysis
- Cross-validation for model validation
- Clustering algorithms for geographic analysis

## Key Capabilities

- **Data Cleaning and Preparation:** Validates coordinates, timestamps, and standardises classifications
- **Geospatial Analysis:** Interactive mapping with [Folium](https://python-visualization.github.io/folium/) and [Plotly](https://plotly.com/python/)
- **Temporal Analysis:** Examines trends by various time periods
- **Report Classification:** ML models to predict report credibility
- **Anomaly Detection:** Identifies unusual or suspicious reports
- **Interactive Prediction Tool:** Users can input hypothetical scenarios
- **Exportable Research Plans:** Generate and export filtered data for research

## Installation and Usage

**Requirements:**
- [Python 3.8+](https://www.python.org/)
- [Streamlit](https://streamlit.io/)
- Data science libraries ([pandas](https://pandas.pydata.org/), [scikit-learn](https://scikit-learn.org/), [folium](https://python-visualization.github.io/folium/), [plotly](https://plotly.com/python/))

**Setup:**
```bash
git clone https://github.com/Meowmixforme/BigFoot.git
cd BigFoot
pip install -r requirements.txt
streamlit run BigFoot.py
```

## Research Applications

The dashboard helps address questions such as:

- Where are Bigfoot sightings most common?
- Are there seasonal or annual trends in reports?
- Which locations have the highest quality sightings?
- Are there geographical or temporal anomalies?
- Where should researchers focus future expeditions?
- Can machine learning predict report credibility?

## Legal and Attribution

- Data used under [Fair Use](https://www.copyright.gov/fair-use/more-info.html) for educational and research purposes
- Original data from [BFRO publicly accessible database](https://www.bfro.net/GDB/)
- Cleaned and processed dataset from [@kittychew/bigfoot-sightings-analysis](https://github.com/kittychew/bigfoot-sightings-analysis), released under the [MIT License](https://github.com/kittychew/bigfoot-sightings-analysis/blob/main/LICENSE)
- No personal information collected or stored
- Educational tool for demonstrating data science techniques
- Includes appropriate disclaimers about scientific evidence

**MIT License Notice:**  
This project incorporates datasets and code from [@kittychew/bigfoot-sightings-analysis](https://github.com/kittychew/bigfoot-sightings-analysis), originally licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Version Information

- **Dashboard Version:** 1.0.0
- **Release Date:** 22 July 2025
- **Python Version:** 3.8+

---

This comprehensive dashboard provides researchers, data scientists, and enthusiasts with powerful tools to explore patterns in cryptid encounters, assess report quality, and generate data-driven recommendations for future field research and analysis.
