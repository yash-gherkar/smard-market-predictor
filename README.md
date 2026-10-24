# SMARD Market Inference Desk

An end-to-end Machine Learning pipeline designed to predict day-ahead wholesale electricity clearing prices and price spike volatility in the German/Luxembourg (DE/LU) bidding zone. This project utilizes official market generation data from SMARD (Bundesnetzagentur).

## Architecture & Tech Stack

* **Machine Learning:** Scikit-Learn (Random Forest Regressor for clearing price prediction; Random Forest Classifier for >100 €/MWh volatility alerts).
* **Backend REST API:** FastAPI, containerized with Docker for isolated execution.
* **Frontend Analytics:** Streamlit, featuring interactive Plotly visualizations (Probability Gauges, Treemaps, and Historical Area Charts).
* **Data Engineering:** Pandas, optimized with Parquet file formats.

## Local Setup & Execution Guide

Due to GitHub's file size limits, the trained machine learning models and processed Parquet datasets are hosted via GitHub Releases.

### 1. Clone the Repository

    git clone https://github.com/yash-gherkar/smard-market-predictor.git
    cd smard-market-predictor

### 2. Download Model Artifacts

1. Navigate to the **Releases** tab on this GitHub repository.
2. Download `smard_artifacts.zip`.
3. Extract the contents directly into the root directory of this project. Ensure the `.parquet` files are in the root folder and the `.pkl` files are inside the `models/` directory.

### 3. Launch the API Backend (Docker)

Open a terminal in the project root and spin up the FastAPI backend:

    docker build -t smard-price-api .
    docker run -p 8000:8000 smard-price-api

*The API will be available at http://127.0.0.1:8000.*

### 4. Launch the Frontend Dashboard

Open a second terminal window, install the local dependencies, and start the Streamlit application:

    pip install -r requirements.txt
    streamlit run src/dashboard.py

*The dashboard will automatically open in your default web browser.*
