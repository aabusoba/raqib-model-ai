# Import pandas for data frame management and dummy variable generation
import pandas as pd
# Import numpy for array manipulation and numerical operations
import numpy as np
# Import train_test_split to separate data into training and validation sets
from sklearn.model_selection import train_test_split
# Import XGBClassifier for the core gradient boosting classification logic
from xgboost import XGBClassifier
# Import SMOTE for oversampling to address class imbalances in training data
from imblearn.over_sampling import SMOTE
# Import LabelEncoder to map categorical text labels to integer codes
from sklearn.preprocessing import LabelEncoder
# Import metrics to generate performance reports and measure accuracy
from sklearn.metrics import classification_report, accuracy_score
# Import pickle for saving and loading serialized model objects
import pickle
# Import the operating system module for handling file paths
import os

# Define the model training function for severity and vehicle classification
def train_models(df_in, n_estimators=200, learning_rate=0.1):
    # Create a local copy of the input dataframe to avoid side effects
    df = df_in.copy()
    # Initialize a label encoder for the accident severity target
    le_severity = LabelEncoder()
    # Encode the severity strings into numerical target values
    df['Severity_Encoded'] = le_severity.fit_transform(df['Accident_Severity'])
    # List the categorical features to be used as predictive inputs
    features = ['Hour', 'Day_of_Week', 'Light_Conditions', 'Weather_Conditions', 'Road_Surface_Conditions', 'Junction_Control', 'Road_Type', 'Urban_or_Rural_Area']
    # Perform one-hot encoding on features and drop the first dummy to avoid multicollinearity
    X = pd.get_dummies(df[features], drop_first=True)
    # Define the target variable for the severity model
    y = df['Severity_Encoded']
    # Split the encoded data into 80% training and 20% testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize SMOTE with a fixed seed for reproducibility
    sm = SMOTE(random_state=42)
    # Resample the training dataset to balance the class distribution
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
    
    # Initialize the XGBoost classifier with specified hyperparameters
    model_severity = XGBClassifier(n_estimators=n_estimators, learning_rate=learning_rate, random_state=42)
    # Fit the severity model on the resampled training data
    model_severity.fit(X_train_res, y_train_res)
    # Generate predictions on the test set for evaluation
    y_pred_s = model_severity.predict(X_test)
    # Generate predictions on the training set to check for overfitting
    y_train_pred_s = model_severity.predict(X_train)
    
    # Identify unique labels present in both true and predicted sets for the report
    unique_labels_s = np.unique(np.concatenate((y_test, y_pred_s)))
    # Map the numerical labels back to their original severity names
    target_names_s = [le_severity.classes_[i] for i in unique_labels_s]
    # Compile a metrics dictionary containing accuracies and a full classification report
    metrics_s = {'train_acc': accuracy_score(y_train, y_train_pred_s), 'test_acc': accuracy_score(y_test, y_pred_s), 'report': classification_report(y_test, y_pred_s, output_dict=True, labels=unique_labels_s, target_names=target_names_s, zero_division=0)}
    
    # Initialize a new label encoder for the vehicle group classification
    le_vehicle = LabelEncoder()
    # Encode the vehicle group strings into numerical target codes
    df['Vehicle_Encoded'] = le_vehicle.fit_transform(df['Vehicle_Group'])
    # Set the vehicle classification target variable
    y_v = df['Vehicle_Encoded']
    # Split the data into separate training and testing sets for the vehicle model
    X_train_v, X_test_v, y_train_v, y_test_v = train_test_split(X, y_v, test_size=0.2, random_state=42)
    
    # Initialize SMOTE for vehicles with a low neighbor count to handle sparse classes
    sm_v = SMOTE(random_state=42, k_neighbors=2) 
    # Resample the vehicle training data to achieve class balance
    X_train_res_v, y_train_res_v = sm_v.fit_resample(X_train_v, y_train_v)
    
    # Initialize the XGBoost classifier for the vehicle model
    model_vehicle = XGBClassifier(n_estimators=n_estimators, learning_rate=learning_rate, random_state=42)
    # Fit the vehicle model on the balanced training samples
    model_vehicle.fit(X_train_res_v, y_train_res_v)
    # Predict vehicle types for the test set
    y_pred_v = model_vehicle.predict(X_test_v)
    # Predict vehicle types for the training set evaluation
    y_train_pred_v = model_vehicle.predict(X_train_v)
    
    # Collect unique vehicle labels appearing in current results
    unique_labels_v = np.unique(np.concatenate((y_test_v, y_pred_v)))
    # Resolve the target names for the vehicle classification report
    target_names_v = [le_vehicle.classes_[i] for i in unique_labels_v]
    # Store all relevant performance metrics for the vehicle classification model
    metrics_v = {'train_acc': accuracy_score(y_train_v, y_train_pred_v), 'test_acc': accuracy_score(y_test_v, y_pred_v), 'report': classification_report(y_test_v, y_pred_v, output_dict=True, labels=unique_labels_v, target_names=target_names_v, zero_division=0)}
    
    # Return both models, encoders, final feature columns, and metrics
    return model_severity, model_vehicle, le_severity, le_vehicle, X.columns.tolist(), metrics_s, metrics_v

# Define a function to generate a risk prediction from new input data
def predict_risk(model, le, input_data, feature_cols):
    # Convert input dictionary into a single-row dataframe
    input_df = pd.DataFrame([input_data])
    # Apply one-hot encoding and reindex to match the training feature schema
    input_df = pd.get_dummies(input_df).reindex(columns=feature_cols, fill_value=0)
    # Use the model to predict the class index
    prediction = model.predict(input_df)
    # Inverse transform the numeric prediction back to the original label text
    return le.inverse_transform(prediction)[0]
