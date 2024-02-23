import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import xgboost as xgb
import lightgbm as lgb
import catboost as catb
from imblearn.over_sampling import SMOTE
import pickle

# Load dataset
@st.cache
def load_data():
    # Load your dataset here
    df = pd.read_csv("final_data.csv")
    
    return df  # Make sure df is defined

# Sidebar options
option = st.sidebar.selectbox(
    'Menu',
    ('Home', 'Data Viewer', 'Model Training', 'Final Prediction')
)

# Main content
if option == 'Home':
    st.title('AnomaData (Automated Anomaly Detection for Predictive Maintenance)')
    st.write('Welcome to the Home page.')
    st.image('img.gif', use_column_width=True) 

elif option == 'Data Viewer':
    st.title('Data Viewer')
    data_load_state = st.text('Loading data...')
    df = load_data()
    data_load_state.text('Loading data...done!')
    
    st.write('Displaying first 100 rows of the dataset:')
    st.write(df.head(100))

elif option == 'Model Training':
    st.title('Model Training')
    df = load_data()

    
    
    # Assuming your dataset has features and target
    X = df.drop(columns=['y'])
    y = df['y']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Apply SMOTE to the training data
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    
    models = {
        'Random Forest': RandomForestClassifier(),
        'Logistic Regression': LogisticRegression(),
        'XGBoost': xgb.XGBClassifier(),
        'LightGBM': lgb.LGBMClassifier(),
        'CatBoost': catb.CatBoostClassifier(verbose=False)
    }
    
    for name, model in models.items():
        st.subheader(name)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f'Accuracy: {accuracy:.2f}')
    
    # Save the models
    for name, model in models.items():
        filename = f'{name.lower().replace(" ", "_")}_model.sav'
        pickle.dump(model, open(filename, 'wb'))

elif option == 'Final Prediction':
    st.title('Final Prediction')
    df = load_data()
    
    # Input fields for each feature
    input_features = [
        'x3', 'x2', 'x19', 'x17', 'x18', 'x10', 'x39', 'x48', 'x38', 'x22',
        'x9', 'x25', 'x11', 'x60', 'x24', 'x55', 'x35', 'x15', 'x14', 'x27'
    ]
    
    input_data = {}
    for feature in input_features:
        input_data[feature] = st.text_input(f'Enter the Value for {feature}')
    
    input_df = pd.DataFrame(input_data, index=[0])  # Create DataFrame from input data
    # Convert input data to float
    input_df = input_df.astype(float, errors='ignore')

    # Load models and make predictions
    predictions = {}
    for name in ['random_forest', 'logistic_regression', 'xgboost', 'lightgbm', 'catboost']:
        filename = f'{name}_model.sav'
        loaded_model = pickle.load(open(filename, 'rb'))
        predictions[name] = loaded_model.predict(input_df)
        
    
    st.write('Predictions:')
    for name, prediction in predictions.items():
        st.write(f'{name.capitalize()}: {prediction[0]}')
