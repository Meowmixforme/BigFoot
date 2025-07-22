"""
Machine learning models for classification and anomaly detection
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

@st.cache_data
def detect_anomalies(df):
    """Detect anomalous sightings using Isolation Forest"""
    try:
        # Prepare features for anomaly detection
        features = ['latitude', 'longitude', 'year', 'month', 'distance_from_center']
        X = df[features].copy()
        
        # Handle any remaining missing values
        X = X.fillna(X.mean())
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Isolation Forest
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        anomaly_labels = iso_forest.fit_predict(X_scaled)
        anomaly_scores = iso_forest.score_samples(X_scaled)
        
        df_anomaly = df.copy()
        df_anomaly['is_anomaly'] = anomaly_labels == -1
        df_anomaly['anomaly_score'] = anomaly_scores
        
        return df_anomaly, iso_forest
        
    except Exception as e:
        st.error(f"❌ Error in anomaly detection: {e}")
        return df, None

@st.cache_data
def train_ml_model(df):
    """Train machine learning model for classification prediction"""
    try:
        # Feature engineering
        ml_df = df.copy()
        
        # Create cyclical features
        ml_df['month_sin'] = np.sin(2 * np.pi * ml_df['month'] / 12)
        ml_df['month_cos'] = np.cos(2 * np.pi * ml_df['month'] / 12)
        ml_df['day_sin'] = np.sin(2 * np.pi * ml_df['day_of_year'] / 365)
        ml_df['day_cos'] = np.cos(2 * np.pi * ml_df['day_of_year'] / 365)
        
        # Normalize year
        ml_df['year_normalized'] = (ml_df['year'] - ml_df['year'].min()) / (ml_df['year'].max() - ml_df['year'].min())
        
        # Encode target variable
        le = LabelEncoder()
        ml_df['classification_encoded'] = le.fit_transform(ml_df['classification'])
        
        # Select features
        feature_columns = [
            'latitude', 'longitude', 'year_normalized', 'month', 
            'month_sin', 'month_cos', 'day_sin', 'day_cos',
            'distance_from_center', 'day_of_year'
        ]
        
        X = ml_df[feature_columns]
        y = ml_df['classification_encoded']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train Random Forest model
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        rf_model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = rf_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Feature importance
        importance_df = pd.DataFrame({
            'feature': feature_columns,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return {
            'model': rf_model,
            'accuracy': accuracy,
            'feature_importance': importance_df,
            'label_encoder': le,
            'test_predictions': y_pred,
            'test_actual': y_test,
            'feature_columns': feature_columns
        }
        
    except Exception as e:
        st.error(f"❌ Error in ML model training: {e}")
        return None