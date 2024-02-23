# AnomaData (Automated Anomaly Detection for Predictive Maintenance)

## Description

AnomaData is a Streamlit application that automates anomaly detection for predictive maintenance tasks. It allows users to view data, train machine learning models, and make predictions based on input features.

## Features

- **Data Viewer:** View the dataset and its first 100 rows.
- **Model Training:** Train machine learning models using various algorithms such as Random Forest, Logistic Regression, XGBoost, LightGBM, and CatBoost.
- **Final Prediction:** Enter input features and predict whether the data is normal or an anomaly.

## Installation

1. Clone the repository:

   ```bash
   git clone <repository_url>
   cd AnomaData
   ```
2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit application:

   ```bash
   streamlit run app.py
   ```

## Usage

1. Select the desired option from the navigation sidebar:

   - **Home:** Welcome page.
   - **Data Viewer:** View the dataset.
   - **Model Training:** Train machine learning models.
   - **Final Prediction:** Make predictions based on input features.
2. Follow the instructions on each page to interact with the application.

## Dataset

- The dataset used in this project is stored in "AnomaData (1).xlsx".

## Models

- Trained machine learning models are saved as .sav files.

## Contributing

Pull requests and feedback are welcome. For major changes, please open an issue first to discuss what you would like to change.
