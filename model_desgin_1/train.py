# Import the operating system module
import os
# Import utility for high-level file operations
import shutil
# Import pandas for data manipulation
import pandas as pd
# Import numpy for numerical operations
import numpy as np
# Import train_test_split to divide data into training and testing sets
from sklearn.model_selection import train_test_split
# Import custom data preprocessing function
from data_processor import load_and_preprocess
# Import custom model training function
from predictor import train_models
# Import pickle for object serialization
import pickle

# Define the number of rows to load (None means all)
DATA_ROWS = None
# Define the percentage of data to use for testing
TEST_SIZE = 0.2
# Define a random seed for reproducibility
RANDOM_STATE = 42
# Define the number of trees in the boosting model
N_ESTIMATORS = 200
# Define the learning rate for the optimizer
LEARNING_RATE = 0.1

# Define a function to reset the dataset storage folder
def prepare_dataset_folder():
    # Check if the dataset folder already exists
    if os.path.exists('dataset'):
        # Delete the existing dataset folder and its contents
        shutil.rmtree('dataset')
    # Create a fresh dataset folder
    os.makedirs('dataset')

# Define the main training execution function
def run_training():
    # Print training configuration parameters
    print(f"--- High Accuracy Training: Rows={DATA_ROWS}, Estimators={N_ESTIMATORS} ---")
    # Log the start of data loading
    print("Loading data...")
    
    # Get the directory path of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the full path to the CSV data file
    data_path = os.path.join(current_dir, 'Road Accident Data.csv')
    
    # Load and preprocess the raw data
    df = load_and_preprocess(data_path)
    # Select all data or a specific subset based on DATA_ROWS
    df_subset = df if DATA_ROWS is None else df.head(DATA_ROWS)
    
    # Log the start of data splitting and balancing
    print("Splitting & Balancing data (SMOTE)...")
    # Split the dataset into training and testing portions
    train_df, test_df = train_test_split(df_subset, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    
    # Prepare local folders for CSV export
    prepare_dataset_folder()
    # Save the training set to a CSV file
    train_df.to_csv('dataset/train.csv', index=False)
    # Save the testing set to a CSV file
    test_df.to_csv('dataset/test.csv', index=False)
    
    # Log the start of the XGBoost training process
    print("Starting XGBoost training...")
    # Train models using the processed training dataframe
    results = train_models(train_df, n_estimators=N_ESTIMATORS, learning_rate=LEARNING_RATE)
    # Extract models, encoders, and metrics from results
    m_s, m_v, l_s, l_v, cols, met_s, met_v = results
    
    # Print a decorative separator for severity metrics
    print("\n" + "█"*40)
    # Print header for severity model performance
    print("  أداء نموذج خطورة الحوادث")
    # Print another decorative separator
    print("█"*40)
    # Display the final training accuracy for severity
    print(f"دقة التدريب: {met_s['train_acc']*100:.2f}%")
    # Display the final testing accuracy for severity
    print(f"دقة الاختبار:  {met_s['test_acc']*100:.2f}%")
    
    # Iterate through potential severity target classes
    for cls in ['قاتل', 'خطير', 'بسيط']:
        # Check if the class exists in the metrics report
        if cls in met_s['report']:
            # Access metrics for the specific class
            m = met_s['report'][cls]
            # Print precision, recall, and F1-score for the class
            print(f"  [{cls:7}] دقة: {m['precision']:.2f} | استدعاء: {m['recall']:.2f} | F1: {m['f1-score']:.2f}")

    # Print a decorative separator for vehicle metrics
    print("\n" + "█"*40)
    # Print header for vehicle model performance
    print("  أداء نموذج نوع المركبة")
    # Print another decorative separator
    print("█"*40)
    # Display the training accuracy for vehicle classification
    print(f"دقة التدريب: {met_v['train_acc']*100:.2f}%")
    # Display the testing accuracy for vehicle classification
    print(f"دقة الاختبار:  {met_v['test_acc']*100:.2f}%")
    # Get the top 5 vehicle types based on sample support
    top_v = sorted(met_v['report'].items(), key=lambda x: x[1]['support'] if isinstance(x[1], dict) else 0, reverse=True)[:5]
    # Iterate through the top vehicle types
    for cls, m in top_v:
        # Validate that metrics are in dictionary form
        if isinstance(m, dict):
            # Print precision and recall for each vehicle type
            print(f"  [{cls[:15]:15}] Prec: {m['precision']:.2f} | Rec: {m['recall']:.2f}")

    # Open target file for severity model persistence
    with open('model_severity.pkl', 'wb') as f:
        # Serialize severity model and its metadata
        pickle.dump((m_s, l_s, cols, met_s), f)
    # Open target file for vehicle model persistence
    with open('model_vehicle.pkl', 'wb') as f:
        # Serialize vehicle model and its metadata
        pickle.dump((m_v, l_v, cols, met_v), f)
    # Log final completion message
    print("\nTraining Complete. Models saved.")

# Check if script is being run directly
if __name__ == "__main__":
    # Execute the primary training routine
    run_training()
