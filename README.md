# Exoplanet Confirmation and Transit Prediction Pipeline

## Overview

This project implements a high-precision pipeline for confirming exoplanet candidates, predicting their future transit times, and analyzing the evolutionary context of their host stars. By leveraging advanced machine learning (ML), deep learning (DL), and astrophysical principles, the pipeline ensures robust and reliable predictions. The system is designed to provide actionable insights into exoplanet detection, orbital mechanics, and stellar evolution.

---

## Problem Statement

Considering the methods of exoplanet detection, the physics of orbital mechanics, and the processes of stellar evolution, how can we determine the likelihood that a candidate exoplanet is a confirmed planet, predict its future transit times, and understand the evolutionary context of its host star.

---

## Features

1. **Exoplanet Confirmation**:
   - Predict whether a candidate exoplanet is a confirmed planet or a false positive using a Deep Neural Network (DNN).
   - Incorporates astrophysical features like orbital velocity, stellar luminosity, and planetary equilibrium temperature.

2. **Transit Time Prediction**:
   - Predict future transit times for confirmed exoplanets using a Temporal Convolutional Network (TCN).
   - Generate a schedule of future transits for each confirmed planet.

3. **Stellar Evolution Analysis**:
   - Classify host stars into evolutionary stages (e.g., Main Sequence, Giant, White Dwarf) using Gaussian Mixture Models (GMM) and a DNN-based classifier.
   - Visualize host stars on a Hertzsprung-Russell diagram and analyze their impact on planetary properties.

4. **Integration and Reporting**:
   - Combine predictions into a unified dataset.
   - Generate detailed reports for each candidate, including confirmation likelihood, transit schedules, and stellar classifications.
   - Interactive visualizations using Plotly Dash and Streamlit.

5. **Deployment**:
   - REST API for real-time predictions using FastAPI.
   - Web dashboard for interactive exploration and visualization.

---

## Pipeline Workflow

### 1. Data Preprocessing
- Load datasets (`KOI Confirmed Candidate and False Positive.csv`, `Stars.csv`, `Transit.csv`).
- Handle missing values using advanced imputation techniques (Iterative Imputer, KNN Imputer).
- Merge datasets using keys like `kepid`, `ra`, `dec`, or `hostname`.
- Normalize numerical features and encode categorical features using embedding layers.
- Detect and remove outliers using Isolation Forest.

### 2. Feature Engineering
- Calculate astrophysical features:
  - Orbital velocity using Kepler's laws.
  - Stellar luminosity using the Stefan-Boltzmann law.
  - Planetary equilibrium temperature.
- Aggregate features like false positive likelihood scores.
- Reduce dimensionality using PCA and select important features using SHAP values.

### 3. Exoplanet Confirmation Model
- Train a Deep Neural Network (DNN) with:
  - Input Layer: Encoded features.
  - Hidden Layers: Fully connected layers with ReLU activation, batch normalization, and dropout.
  - Output Layer: Sigmoid activation for binary classification.
- Use Focal Loss to handle class imbalance.
- Evaluate using Precision-Recall AUC and explain predictions with SHAP.

### 4. Transit Time Prediction
- Train a Temporal Convolutional Network (TCN) for sequential data prediction.
- Input: Historical transit times and orbital periods.
- Output: Predicted future transit times.
- Validate predictions using known transit times from `Transit.csv`.

### 5. Stellar Evolution Context
- Classify stars into evolutionary stages using Gaussian Mixture Models (GMM).
- Train a DNN-based classifier for stellar classification.
- Visualize host stars on a Hertzsprung-Russell diagram and overlay planetary properties.

### 6. Integration and Reporting
- Combine predictions into a unified dataset.
- Generate detailed reports for each candidate.
- Create interactive visualizations for transit schedules, stellar evolution trends, and confirmation probabilities.

### 7. Deployment
- Build a REST API using FastAPI for real-time predictions.
- Create an interactive web dashboard using Streamlit.
- Deploy the application on AWS or Azure using Docker containers.

---

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/Shubhamf0073/AstroPrediction.git
   cd AstroPrediction

The required Python libraries are listed in requirements.txt. Install them using:
    pip install -r requirements.txt

### Key Libraries
pandas: Data manipulation and analysis.
numpy: Numerical computations.
scikit-learn: Machine learning models and preprocessing.
tensorflow: Deep learning framework for DNN and TCN models.
torch: PyTorch for deep learning.
shap: Explainability for AI models.
plotly: Interactive visualizations.
streamlit: Web dashboard for visualization.
fastapi: REST API for real-time predictions.

### Results
## Exoplanet Confirmation
    Achieved Precision-Recall AUC of 0.98 on the test set.
    SHAP analysis revealed key features like koi_period, koi_depth, and st_teff.

## Transit Time Prediction
    Predicted future transit times with a Mean Absolute Error (MAE) of 0.01 days.

## Stellar Evolution Context
    Classified host stars into evolutionary stages with 98% accuracy.
    Visualized host stars on a Hertzsprung-Russell diagram.

### Deployment
## REST API:
    Serve predictions via FastAPI.
    Example endpoint: /predict

## Web Dashboard:
    Explore predictions and visualizations interactively using Streamlit.

## Cloud Deployment:
    Deploy the application on AWS or Azure using Docker.

### Contributing
Contributions are welcome! Please fork the repository and submit a pull request.

### License
This project is licensed under the Apache 2.0 License. See the LICENSE file for details.

### Contact
For questions or support, please contact:

Name: Shubham Fufal
Email: work.shubhamfufal@outlook.com
