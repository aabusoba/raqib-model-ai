# Import pandas for data frame manipulation
import pandas as pd
# Import numpy for numerical and array operations
import numpy as np
# Import train_test_split to divide data into training and validation sets
from sklearn.model_selection import train_test_split
# Import LabelEncoder to convert categorical labels into numerical values
from sklearn.preprocessing import LabelEncoder
# Import metrics to evaluate the performance of classification models
from sklearn.metrics import classification_report, accuracy_score
# Import XGBClassifier for high-performance gradient boosting models
from xgboost import XGBClassifier
# Import SMOTE for oversampling minority classes to balance the dataset
from imblearn.over_sampling import SMOTE
# Import pickle to serialize and save the trained models to files
import pickle
# Import the operating system module for path manipulations
import os
# Import custom preprocessing function from data_processor module
from data_processor import load_and_preprocess

# Configuration settings: Define rows to load (None uses the entire dataset)
DATA_ROWS = None
# Configuration settings: Define the ratio of the test set size
TEST_SIZE = 0.2
# Configuration settings: Set a fixed random state for reproducibility
RANDOM_STATE = 42
# Configuration settings: Set the number of boosting iterations (trees)
N_ESTIMATORS = 200
# Configuration settings: Set the step size shrinkage for updates
LEARNING_RATE = 0.1

# Define the main function to orchestrate the training process
def train_models():
    # Print an initial message to track the start of training
    print(f"🚀 Starting dual-model training for model_desgin_2 (N_ESTIMATORS={N_ESTIMATORS})...")
    
    # Load and clean the data directly from Kaggle using kagglehub
    df = load_and_preprocess(row_limit=DATA_ROWS)
    # Output the total number of processed rows to the console
    print(f"📊 Dataset loaded: {len(df)} rows")
    
    # Define the list of feature columns to be used for prediction
    features = ['Hour', 'Day_of_Week', 'Light_Conditions', 'Weather_Conditions', 'Road_Surface_Conditions', 'Road_Type', 'Urban_or_Rural_Area']
    
    # Extract the feature data into a separate dataframe
    X = df[features].copy()
    
    # Initialize a dictionary to store label encoders for each feature
    encoders = {}
    # Iterate through each feature column for encoding
    for col in features:
        # Create a new LabelEncoder instance
        le = LabelEncoder()
        # Transform the categorical strings into numerical codes
        X[col] = le.fit_transform(X[col].astype(str))
        # Store the encoder for future inverse transformation
        encoders[col] = le
    
    # Initialize a results dictionary to collect model performance data
    results = {}
    
    # 1. Severity Model section
    print("🧠 Training Severity Model...")
    # Calculate frequencies of each accident severity level
    counts_s = df['Accident_Severity'].value_counts()
    # Identify classes with enough representation (at least 10 samples)
    valid_classes_s = counts_s[counts_s >= 10].index
    # Filter the dataframe to include only valid severity classes
    df_s = df[df['Accident_Severity'].isin(valid_classes_s)].copy()
    
    # Initialize an encoder for the severity target labels
    le_sev = LabelEncoder()
    # Encode the accident severity strings into numerical targets
    y_sev = le_sev.fit_transform(df_s['Accident_Severity'])
    # Align the feature matrix with the filtered labels
    X_s = X.loc[df_s.index]
    
    # Split the data into training and testing sets for evaluation
    X_train, X_test, y_train, y_test = train_test_split(X_s, y_sev, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    # Initialize SMOTE for balancing the training data classes
    sm = SMOTE(random_state=RANDOM_STATE)
    # Upsample the training set to produce a balanced class distribution
    X_res, y_res = sm.fit_resample(X_train, y_train)
    # Instantiate the XGBoost classifier with specified parameters
    m_sev = XGBClassifier(n_estimators=N_ESTIMATORS, learning_rate=LEARNING_RATE, max_depth=6, random_state=RANDOM_STATE)
    # Train the severity model on the balanced data
    m_sev.fit(X_res, y_res)
    # Store the trained model, encoder, and its evaluation metrics
    results['severity'] = (m_sev, le_sev, accuracy_score(y_test, m_sev.predict(X_test)), classification_report(y_test, m_sev.predict(X_test), target_names=le_sev.classes_, output_dict=True))
    
    # 2. Vehicle Model section
    print("🧠 Training Vehicle Model...")
    # Calculate frequencies of each vehicle type in the data
    counts_v = df['Vehicle_Type'].value_counts()
    # Identify vehicle types with enough samples to be statistically significant
    valid_classes_v = counts_v[counts_v >= 10].index
    # Filter the dataframe to isolate supported vehicle classes
    df_v = df[df['Vehicle_Type'].isin(valid_classes_v)].copy()
    
    # Initialize an encoder for vehicle type targets
    le_veh = LabelEncoder()
    # Encode the vehicle type strings into numerical targets
    y_veh = le_veh.fit_transform(df_v['Vehicle_Type'])
    # Align the feature set with the vehicle-filtered index
    X_v = X.loc[df_v.index]
    
    # Split the vehicle data into training and testing subsets
    X_train_v, X_test_v, y_train_v, y_test_v = train_test_split(X_v, y_veh, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    # Apply SMOTE to the vehicle training data to balance types
    X_res_v, y_res_v = sm.fit_resample(X_train_v, y_train_v)
    # Instantiate another XGBoost model for vehicle classification
    m_veh = XGBClassifier(n_estimators=N_ESTIMATORS, learning_rate=LEARNING_RATE, max_depth=6, random_state=RANDOM_STATE)
    # Train the vehicle model on the resampled training set
    m_veh.fit(X_res_v, y_res_v)
    # Calculate and store model accuracy and classification report
    results['vehicle'] = (m_veh, le_veh, accuracy_score(y_test_v, m_veh.predict(X_test_v)), classification_report(y_test_v, m_veh.predict(X_test_v), target_names=le_veh.classes_, output_dict=True))
    
    # --- New: Save Dashboard Stats to JSON for lightweight deployment ---
    print("📊 Generating dashboard statistics summary...")
    from analyzer import get_dangerous_locations, get_peak_times, get_severity_report
    
    stats = {
        "severity_distribution": get_severity_report(df).to_dict(),
        "peak_hours": [{"hour": int(h), "count": int(c)} for h, c in get_peak_times(df).items()],
        "top_cities": [{"city": city, "accidents": int(count)} for city, count in get_dangerous_locations(df).items()],
        "total_rows": len(df)
    }
    
    import json
    with open('model_desgin_2/v2_stats.json', 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=4)
    # --- End Stats ---

    # Save both models to binary files
    with open('model_desgin_2/model_severity_v2.pkl', 'wb') as f:
        # Save the severity model, its encoders, features, and metrics
        pickle.dump((m_sev, encoders, features, {'test_acc': results['severity'][2], 'report': results['severity'][3]}), f)
    # Open the second target file for vehicle model output
    with open('model_desgin_2/model_vehicle_v2.pkl', 'wb') as f:
        # Save the vehicle classification model and its corresponding metadata
        pickle.dump((m_veh, encoders, features, {'test_acc': results['vehicle'][2], 'report': results['vehicle'][3]}), f)
    
    print("✅ Models and Stats saved successfully! You can now run the API without the dataset.")


# Ensure the script is run directly and not imported
if __name__ == "__main__":
    # Execute the training function
    train_models()
