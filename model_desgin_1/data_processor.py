# Import pandas for flexible data manipulation and analysis
import pandas as pd
# Import numpy for array computations and numerical operations
import numpy as np
# Import the operating system module to interact with the file system
import os

# Define a helper function to categorize various vehicle types into simplified groups
def group_vehicle_types(v_type):
    # Convert input to lowercase string for consistent string matching
    v_type = str(v_type).lower()
    # Categorize cars and taxis as Passenger Car
    if 'car' in v_type or 'taxi' in v_type: return 'Passenger Car'
    # Categorize any motorcycle-related strings as Motorcycle
    if 'motorcycle' in v_type: return 'Motorcycle'
    # Categorize buses and minibuses as Public Transport
    if 'bus' in v_type or 'minibus' in v_type: return 'Public Transport'
    # Categorize goods vehicles and agricultural machinery as Heavy/Special
    if 'goods' in v_type or 'agricultural' in v_type: return 'Heavy/Special'
    # Categorize bicycles as Pedal Cycle
    if 'cycle' in v_type: return 'Pedal Cycle'
    # Default to 'Other' for any unrecognized vehicle types
    return 'Other'

# Define a function to translate data values and simulate Libyan geographic context
def localize_to_libya(df):
    # Import json inside the function for loading city data from file
    import json
    # Define a translation dictionary for mapping English values to Arabic for Libyan use
    translations = {
        'Accident_Severity': {'Slight': 'بسيط', 'Serious': 'خطير', 'Fatal': 'قاتل'},
        'Day_of_Week': {'Sunday': 'الأحد', 'Monday': 'الاثنين', 'Tuesday': 'الثلاثاء', 'Wednesday': 'الأربعاء', 'Thursday': 'الخميس', 'Friday': 'الجمعة', 'Saturday': 'السبت'},
        'Light_Conditions': {'Daylight': 'ضوء النهار', 'Darkness - lights lit': 'ظلام - مصابيح مشتعلة', 'Darkness - lights unlit': 'ظلام - مصابيح غير مشتعلة', 'Darkness - no lighting': 'ظلام - لا توجد إضاءة'},
        'Weather_Conditions': {'Fine no high winds': 'صافي - لا رياح', 'Raining no high winds': 'ممطر - لا رياح', 'Snowing no high winds': 'ثلج - لا رياح', 'Fine + high winds': 'صافي + رياح عالية', 'Raining + high winds': 'ممطر + رياح عالية', 'Snowing + high winds': 'ثلج + رياح عالية', 'Fog or mist': 'ضباب', 'Other': 'أخرى'},
        'Road_Surface_Conditions': {'Dry': 'جاف', 'Wet or damp': 'رطب أو مبلل', 'Snow': 'ثلج', 'Frost or ice': 'صقيع أو جليد', 'Flood over 3cm. deep': 'فيضان أكثر من 3 سم'},
        'Road_Type': {'Single carriageway': 'طريق فردي', 'Dual carriageway': 'طريق مزدوج', 'Roundabout': 'دوار', 'One way street': 'شارع اتجاه واحد', 'Slip road': 'طريق منحدر'},
        'Urban_or_Rural_Area': {'Urban': 'حضري', 'Rural': 'ريفي'},
        'Vehicle_Group': {'Passenger Car': 'سيارة ركاب', 'Motorcycle': 'دراجة نارية', 'Public Transport': 'نقل عام', 'Heavy/Special': 'ثقيلة/خاصة', 'Pedal Cycle': 'دراجة هوائية', 'Other': 'أخرى'}
    }
    # Loop through each column defined in the translations dictionary
    for col, mapping in translations.items():
        # Check if the column exists within the dataframe
        if col in df.columns:
            # Apply the translation mapping and keep original values if no match found
            df[col] = df[col].map(mapping).fillna(df[col])

    # Check if the Libyan cities data file exists on disk
    if os.path.exists('libyan_cities.json'):
        # Open and read the Libyan cities JSON file with UTF-8 encoding
        with open('libyan_cities.json', 'r', encoding='utf-8') as f:
            # Parse the JSON content into a dictionary
            cities_data = json.load(f)
            # Initialize a list to hold extracted city metadata
            libyan_cities_info = []
            # Iterate through the raw city data list
            for city in cities_data['data']:
                # Check for the existence of Arabic names in the primary name index
                if 'ar' in city['name'][0]:
                    # Append formatted city object with Arabic name and coordinates
                    libyan_cities_info.append({
                        'name': city['name'][0]['ar'],
                        'lat': city['latitude'],
                        'lon': city['longitude']
                    })
    # Provide fallback coordinate data if the JSON file is missing
    else:
        libyan_cities_info = [{'name': 'طرابلس', 'lat': 32.8872, 'lon': 13.1913}]
        
    # Check for the district authority column to perform geographic simulation
    if 'Local_Authority_(District)' in df.columns:
        # Get unique district names from the dataset
        unique_districts = df['Local_Authority_(District)'].unique()
        # Create a mapping from raw district names to Libyan city names (cycling through cities)
        city_map = {dist: libyan_cities_info[i % len(libyan_cities_info)]['name'] for i, dist in enumerate(unique_districts)}
        # Create a mapping for base latitude coordinates
        lat_map = {dist: libyan_cities_info[i % len(libyan_cities_info)]['lat'] for i, dist in enumerate(unique_districts)}
        # Create a mapping for base longitude coordinates
        lon_map = {dist: libyan_cities_info[i % len(libyan_cities_info)]['lon'] for i, dist in enumerate(unique_districts)}
        
        # Apply latent mapping and add random noise to simulate geographic spread
        df['Latitude'] = df['Local_Authority_(District)'].map(lat_map) + np.random.uniform(-0.05, 0.05, size=len(df))
        # Apply longitude mapping and add random noise for realistic density simulation
        df['Longitude'] = df['Local_Authority_(District)'].map(lon_map) + np.random.uniform(-0.05, 0.05, size=len(df))
        # Replace original district names with the simulated Libyan city names
        df['Local_Authority_(District)'] = df['Local_Authority_(District)'].map(city_map)
        
    # Return the translated and geographically localized dataframe
    return df

# Main preprocessing function to load, clean, and transform the dataset
def load_and_preprocess(file_path, row_limit=None):
    # Read CSV file with specific encoding and memory optimization flags
    df = pd.read_csv(file_path, nrows=row_limit, encoding='latin-1', low_memory=False)
    # Filter out numeric placeholder strings if they exist in the target column
    df = df[~df['Accident_Severity'].isin(['1', '2', '3'])]
    # Remove rows containing missing values in critical predictive columns
    df = df.dropna(subset=['Accident_Severity', 'Light_Conditions', 'Weather_Conditions', 'Latitude', 'Longitude', 'Vehicle_Type'])
    # Apply the grouping logic to consolidate various vehicle types
    df['Vehicle_Group'] = df['Vehicle_Type'].apply(group_vehicle_types)
    
    # Process the 'Time' column to extract numerical hours for modeling
    if 'Time' in df.columns:
        # Attempt to parse time using the H:M format
        df['Hour'] = pd.to_datetime(df['Time'], format='%H:%M', errors='coerce').dt.hour
        # Detect rows where initial parsing failed
        mask = df['Hour'].isna()
        # Check if any rows failed the first time formatting
        if mask.any():
            # Attempt fallback parsing with H:M:S format for failed rows
            df.loc[mask, 'Hour'] = pd.to_datetime(df.loc[mask, 'Time'], format='%H:%M:%S', errors='coerce').dt.hour
        # Fill any remaining NaNs with 0 and convert the column to integer type
        df['Hour'] = df['Hour'].fillna(0).astype(int)
    
    # Apply localization and translations to the cleaned dataframe
    df = localize_to_libya(df)
    # Standardize the accident date format for time-series analysis
    df['Accident Date'] = pd.to_datetime(df['Accident Date'], format='%d/%m/%Y', errors='coerce')
    
    # Define a list of columns that should be treated as categorical types
    categorical_cols = ['Day_of_Week', 'Junction_Control', 'Road_Surface_Conditions', 'Road_Type', 'Urban_or_Rural_Area', 'Vehicle_Group']
    # Loop through categorical columns and set appropriate memory-efficient types
    for col in categorical_cols:
        # Cast column to pandas 'category' type
        df[col] = df[col].astype('category')
    
    # Return the fully preprocessed and Libyan-localized dataframe
    return df

# Entry point for testing the processor independently
if __name__ == "__main__":
    # Test execution with the provided dataset path
    data = load_and_preprocess('Road Accident Data.csv')
    # Print summary information about the resulting dataframe
    print(data.info())
