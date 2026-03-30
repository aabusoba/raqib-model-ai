# Import pandas for data frame manipulation and processing
import pandas as pd
# Import numpy for structured data handling and array operations
import numpy as np

# Define a function to perform risk prediction using the second model design
def predict_risk(model, encoders, input_dict, feature_cols):
    # Create a local dataframe from the input dictionary to maintain data isolation
    data = pd.DataFrame([input_dict])
    
    # Iterate through the required list of feature columns
    for col in feature_cols:
        # Check if the required column is missing from the input data
        if col not in data.columns:
            # Initialize the missing column with a default zero value
            data[col] = 0
            
    # Iterate through the pre-fitted encoders for each feature
    for col, le in encoders.items():
        # Validate that the column exists within the input dataframe
        if col in data.columns:
            # Extract the raw string value from the current column
            current_val = str(data[col].iloc[0])
            # Check if the value was present during the model training phase
            if current_val in le.classes_:
                # Transform the recognized string into its corresponding numerical code
                data[col] = le.transform([current_val])[0]
            # Handle unrecognized values to prevent prediction errors
            else:
                # Assign a default fallback value for unseen categories
                data[col] = 0 
                
    # Reorder the dataframe columns to exactly match the training feature schema
    data = data[feature_cols]
    
    # Use the trained model to generate a prediction (returns an index)
    pred_idx = model.predict(data)[0]
    
    # Note: Target label decoding should be handled by the calling function using metadata
    # The current implementation returns the raw numeric prediction index
    return pred_idx
