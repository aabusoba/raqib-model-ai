# Import pandas for data manipulation and analysis
import pandas as pd
# Import numpy for efficient numerical and vectorized operations
import numpy as np
# Import the operating system module to interact with files and directories
import os
# Import json for parsing and handling city metadata archives
import json

import kagglehub

# Define the primary loading and preprocessing function for the second design
def load_and_preprocess(row_limit=None):
    print("📥 Syncing dataset from Kaggle (tsiaras/uk-road-safety-accidents-and-vehicles)...")
    dataset_path = kagglehub.dataset_download("tsiaras/uk-road-safety-accidents-and-vehicles")
    
    # Locate files in the downloaded directory
    acc_path = os.path.join(dataset_path, "Accident_Information.csv")
    veh_path = os.path.join(dataset_path, "Vehicle_Information.csv")
    
    # Check if files exist, if not, search for them (some datasets have subdirectories)
    if not os.path.exists(acc_path):
        for root, dirs, files in os.walk(dataset_path):
            if "Accident_Information.csv" in files:
                acc_path = os.path.join(root, "Accident_Information.csv")
            if "Vehicle_Information.csv" in files:
                veh_path = os.path.join(root, "Vehicle_Information.csv")

    print(f"📖 Loading Accident_Information.csv (limit={row_limit})...")
    acc_df = pd.read_csv(acc_path, nrows=row_limit, encoding='latin-1', low_memory=False)
    
    print(f"📖 Loading Vehicle_Information.csv (limit={row_limit})...")
    veh_df = pd.read_csv(veh_path, nrows=row_limit, encoding='latin-1', low_memory=False)
    
    # Merge both datasets on the common 'Accident_Index' column using an inner join
    df = pd.merge(acc_df, veh_df, on='Accident_Index', how='inner')

    
    # Define a list of high-priority columns to retain for modeling
    cols = [
        'Accident_Severity', 'Date', 'Day_of_Week', 'Latitude', 'Longitude',
        'Light_Conditions', 'Road_Surface_Conditions', 'Road_Type', 
        'Speed_limit', 'Time', 'Urban_or_Rural_Area', 'Weather_Conditions',
        'Vehicle_Type', 'Age_of_Vehicle', 'Engine_Capacity_.CC.', 'Sex_of_Driver'
    ]
    # Filter the dataframe for the selected columns and remove rows without coordinates or severity
    df = df[cols].dropna(subset=['Accident_Severity', 'Latitude', 'Longitude'])
    
    # Apply geo-localization and Arabic translations for the Libyan context
    df = localize_to_libya(df)
    
    # Extract temporal features by parsing the 'Time' column
    df['Time'] = pd.to_datetime(df['Time'], format='%H:%M', errors='coerce')
    # Derive 'Hour' as an integer feature, defaulting to 12 if parsing fails
    df['Hour'] = df['Time'].dt.hour.fillna(12).astype(int)
    
    # Return the fully merged and preprocessed dataset
    return df

# Define a function to translate data and simulate Libyan geographic deployment
def localize_to_libya(df):
    # Create a mapping for weather conditions from English to Arabic
    weather_map = {
        'Fine no high winds': 'صافي - لا رياح',
        'Raining no high winds': 'مطر - لا رياح',
        'Snowing no high winds': 'ثلج - لا رياح',
        'Fine + high winds': 'صافي + رياح قوية',
        'Raining + high winds': 'مطر + رياح قوية',
        'Snowing + high winds': 'ثلج + رياح قوية',
        'Fog or mist': 'ضباب',
        'Other': 'أخرى',
        'Unknown': 'غير معروف'
    }
    # Create a mapping for accident severity levels
    sev_map = {'Slight': 'بسيط', 'Serious': 'خطير', 'Fatal': 'قاتل'}
    # Create a mapping for days of the week to Arabic
    day_map = {
        'Sunday': 'الأحد', 'Monday': 'الاثنين', 'Tuesday': 'الثلاثاء', 
        'Wednesday': 'الأربعاء', 'Thursday': 'الخميس', 'Friday': 'الجمعة', 'Saturday': 'السبت'
    }
    
    # Apply weather mapping and provide a fallback for missing values
    df['Weather_Conditions'] = df['Weather_Conditions'].map(weather_map).fillna('أخرى')
    # Apply severity mapping and default to 'Slight' if not found
    df['Accident_Severity'] = df['Accident_Severity'].map(sev_map).fillna('بسيط')
    # Apply day of week mapping with a default value
    df['Day_of_Week'] = df['Day_of_Week'].map(day_map).fillna('الأحد')
    
    # Define the file path for Libyan city coordinate data
    json_path = os.path.join(os.path.dirname(__file__), 'libyan_cities.json')
    # Check if the city metadata file exists at the specified location
    if os.path.exists(json_path):
        # Open and read the JSON file with UTF-8 encoding
        with open(json_path, 'r', encoding='utf-8') as f:
            # Load the data list from the JSON object
            cities_data = json.load(f)['data']
            # Collect Arabic city names from the loaded JSON data
            lib_cities = [c['name'][0]['ar'] for c in cities_data if 'ar' in c['name'][0]]
            # Extract corresponding latitude coordinates
            lib_lats = [c['latitude'] for c in cities_data]
            # Extract corresponding longitude coordinates
            lib_lons = [c['longitude'] for c in cities_data]
            
            # Determine the total number of rows in the current dataframe
            num_rows = len(df)
            # Generate random assignments for city indices for simulation variety
            indices = np.random.randint(0, len(lib_cities), size=num_rows)
            # Assign simulated Libyan city names to a new column
            df['Libyan_City'] = np.array(lib_cities)[indices]
            # Shift latitude coordinates to align with Libyan geography plus random jitter
            df['Latitude'] = np.array(lib_lats)[indices] + np.random.uniform(-0.1, 0.1, size=num_rows)
            # Shift longitude coordinates to align with Libyan geography plus random jitter
            df['Longitude'] = np.array(lib_lons)[indices] + np.random.uniform(-0.1, 0.1, size=num_rows)
            
    # Return the localized dataframe
    return df
