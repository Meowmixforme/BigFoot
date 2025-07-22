# 🦶 BigFoot ML Dashboard

An advanced, interactive dashboard for analysing Bigfoot (Sasquatch) sighting reports with machine learning and data visualisation. This project empowers researchers, data scientists, and enthusiasts to explore patterns in cryptid encounters, identify geographical and temporal hotspots, assess report credibility, and generate recommendations for field investigations.

---

## 📚 What Does This Project Do?

**BigFoot ML Dashboard** is a comprehensive analytics tool for Bigfoot sighting data, built in Python with Streamlit.  
It imports authentic sighting reports from the [Bigfoot Field Researchers Organization (BFRO)](http://bfro.net/GDB/), then processes, explores, and models the data to reveal actionable insights about when, where, and what types of encounters occur.

### Main Capabilities

- **Data Cleaning and Preparation:** Parses raw BFRO data, validates coordinates and timestamps, and standardises classifications (Class A/B/C).
- **Geospatial Analysis:** Maps sightings, clusters geographical hotspots, and presents interactive locations using Folium and Plotly.
- **Temporal Analysis:** Examines trends by year, month, season, and day of week to highlight patterns in sighting frequency.
- **Report Classification:** Employs Random Forest models to predict report credibility (Class A/B/C) based on spatial and temporal features.
- **Anomaly Detection:** Identifies unusual or suspicious reports using Isolation Forest, flagging outliers for further scrutiny.
- **Location Recommendations:** Ranks geographical regions for future research or expedition planning, based on sighting density, quality, and recency.
- **Interactive Prediction Tool:** Allows users to input hypothetical locations and times, and see machine learning predictions for sighting quality.
- **Advanced Data Explorer:** Powerful search, filtering, statistical summaries, and export functionality for in-depth data analysis.
- **Exportable Research Plans:** Automatically generates recommended field research plans and allows users to export filtered data.

### About BFRO Sighting Classes

BFRO sighting reports are categorised into three main classes:

- **Class A:**  
  Sighting involving clear visual observation in good conditions, typically by a reliable witness. These reports often include physical evidence (such as tracks) and are considered the highest quality and most credible.

- **Class B:**  
  Sighting involving possible visual observation or clear audio encounter (such as vocalisations or knocks), but lacking the clarity or reliability of Class A. These may be brief glimpses, indistinct shapes, or strong sounds, but without enough detail to be classified as Class A.

- **Class C:**  
  Reports based on second-hand information, stories, or anecdotal evidence. These are less reliable, often submitted by someone who heard about an encounter from another party, or where details are vague/incomplete.

The dashboard uses these classes extensively in its data analysis, machine learning models, and visualisations.

### Example Questions You Can Explore

- Where are Bigfoot sightings most common?
- Are there seasonal or annual trends in reports?
- Which locations have the highest quality (Class A) sightings?
- Are there geographical or temporal anomalies in the data?
- Where should field researchers focus future expeditions?
- Can machine learning predict the credibility of a new report?

---

## 🚀 Features

- **Interactive Dashboard:** Key metrics, time trends, and data insights
- **Geographical Clustering:** KMeans/DBSCAN for hotspot discovery
- **Temporal Insights:** Seasonality, trends, and anomalies
- **Machine Learning:** Classification, anomaly detection, feature importance
- **Field Recommendations:** Data-driven expedition planning
- **Advanced Search and Export:** Filter, explore, and save results

---

## 📦 Installation

### Requirements

- Python 3.8 or higher
- pip

### Local Setup

```bash
git clone https://github.com/Meowmixforme/BigFoot.git
cd BigFoot
pip install -r requirements.txt
streamlit run bigfoot_dashboard.py
```

Then visit [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🗂️ Repository Structure

```
BigFoot/
├── bigfoot_dashboard.py      # Main Streamlit app
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
├── Dockerfile                # Containerisation (optional)
└── (assets/, docs/, tests/, etc.)
```

---

## 📊 Data Source

- **Bigfoot Field Researchers Organization (BFRO)**
- [BFRO Geographical Database](http://bfro.net/GDB/)
- Over 4,000 North American sightings, 1960s–present
- Class A/B/C reports, coordinates, timestamps

**Legal:** Used under Fair Use for educational and research purposes. See [BFRO Terms](http://bfro.net/GDB/) for more.

---

## 🔬 Methodology

- **Data Cleaning:** Coordinate validation, timestamp parsing, classification standardisation
- **Feature Engineering:** Temporal (season, day of week), spatial (distance from centre), cyclical encoding
- **Machine Learning Models:** Random Forest (classification), KMeans (clustering), Isolation Forest (anomaly detection)
- **Visualisation:** Interactive maps (Folium), charts (Plotly)
- **Performance:** Caching, data sampling for large datasets

---

## 📝 Usage

- Designed for researchers, educators, and enthusiasts
- **Not** scientific proof — for pattern analysis and exploration only
- **Do not** use location data for harassment or trespass

---

## 👫 Contributing

Pull requests, suggestions, and bug reports are welcome!

---

## 📖 Licence

MIT Licence (see [LICENSE](LICENSE))

---

## 🙏 Acknowledgements

- Data: BFRO
- Libraries: Streamlit, Plotly, Folium, Scikit-learn, Pandas, NumPy
- Icon: Streamlit emoji set

---

## 📞 Contact

Created by [Meowmixforme](https://github.com/Meowmixforme)

---
