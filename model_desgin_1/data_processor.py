# Import pandas for flexible data manipulation and analysis
import pandas as pd
# Import numpy for array computations and numerical operations
import numpy as np
import os
import kagglehub
import shutil

# Define a helper function to categorize various vehicle types into simplified groups
def group_vehicle_types(v_type):
    v_type = str(v_type).lower()
    if 'car' in v_type or 'taxi' in v_type: return 'Passenger Car'
    if 'motorcycle' in v_type: return 'Motorcycle'
    if 'bus' in v_type or 'minibus' in v_type: return 'Public Transport'
    if 'goods' in v_type or 'agricultural' in v_type: return 'Heavy/Special'
    if 'cycle' in v_type: return 'Pedal Cycle'
    return 'Other'

# Define a function to translate data values and simulate Libyan geographic context
def localize_to_libya(df):
    import json
    translations = {
        'Accident_Severity': {'Slight': 'بسيط', 'Serious': 'خطير', 'Fatal': 'قاتل'},
        'Day_of_Week': {'Sunday': 'الأحد', 'Monday': 'الاثنين', 'Tuesday': 'الثلاثاء', 'Wednesday': 'الأربعاء', 'Thursday': 'الخميس', 'Friday': 'الجمعة', 'Saturday': 'السبت'},
        'Light_Conditions': {'Daylight': 'ضوء النهار', 'Darkness - lights lit': 'ضوء النهار', 'Darkness - lights unlit': 'ظلام - مصابيح غير مشتعلة', 'Darkness - no lighting': 'ظلام - لا توجد إضاءة'},
        'Weather_Conditions': {'Fine no high winds': 'صافي - لا رياح', 'Raining no high winds': 'ممطر - لا رياح', 'Snowing no high winds': 'ثلج - لا رياح', 'Fine + high winds': 'صافي + رياح عالية', 'Raining + high winds': 'ممطر + رياح عالية', 'Snowing + high winds': 'ثلج + رياح عالية', 'Fog or mist': 'ضباب', 'Other': 'أخرى'},
        'Road_Surface_Conditions': {'Dry': 'جاف', 'Wet or damp': 'رطب أو مبلل', 'Snow': 'ثلج', 'Frost or ice': 'صقيع أو جليد', 'Flood over 3cm. deep': 'فيضان أكثر من 3 سم'},
        'Road_Type': {'Single carriageway': 'طريق فردي', 'Dual carriageway': 'طريق مزدوج', 'Roundabout': 'دوار', 'One way street': 'شارع اتجاه واحد', 'Slip road': 'طريق منحدر'},
        'Urban_or_Rural_Area': {'Urban': 'حضري', 'Rural': 'ريفي'},
        'Vehicle_Group': {'Passenger Car': 'سيارة ركاب', 'Motorcycle': 'دراجة نارية', 'Public Transport': 'نقل عام', 'Heavy/Special': 'ثقيلة/خاصة', 'Pedal Cycle': 'دراجة هوائية', 'Other': 'أخرى'}
    }
    for col, mapping in translations.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna(df[col])

    json_path = os.path.join(os.path.dirname(__file__), 'libyan_cities.json')
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            cities_data = json.load(f)
            libyan_cities_info = []
            for city in cities_data['data']:
                if 'ar' in city['name'][0]:
                    libyan_cities_info.append({'name': city['name'][0]['ar'], 'lat': city['latitude'], 'lon': city['longitude']})
    else:
        libyan_cities_info = [{'name': 'طرابلس', 'lat': 32.8872, 'lon': 13.1913}]
        
    if 'Local_Authority_(District)' in df.columns:
        unique_districts = df['Local_Authority_(District)'].unique()
        city_map = {dist: libyan_cities_info[i % len(libyan_cities_info)]['name'] for i, dist in enumerate(unique_districts)}
        lat_map = {dist: libyan_cities_info[i % len(libyan_cities_info)]['lat'] for i, dist in enumerate(unique_districts)}
        lon_map = {dist: libyan_cities_info[i % len(libyan_cities_info)]['lon'] for i, dist in enumerate(unique_districts)}
        df['Latitude'] = df['Local_Authority_(District)'].map(lat_map) + np.random.uniform(-0.05, 0.05, size=len(df))
        df['Longitude'] = df['Local_Authority_(District)'].map(lon_map) + np.random.uniform(-0.05, 0.05, size=len(df))
        df['Local_Authority_(District)'] = df['Local_Authority_(District)'].map(city_map)
        
    return df

# Main preprocessing function to load, clean, and transform the dataset
def load_and_preprocess(row_limit=None):
    local_dir = os.path.join(os.path.dirname(__file__), 'source_data')
    os.makedirs(local_dir, exist_ok=True)
    
    file_name = "Road Accident Data.csv"
    local_path = os.path.join(local_dir, file_name)

    if not os.path.exists(local_path):
        print(f"🔍 {file_name} not found in model_desgin_1 folder. Syncing from Kaggle HUB...")
        dataset_path = kagglehub.dataset_download("xavierberge/road-accident-dataset")
        src_path = os.path.join(dataset_path, file_name)
        if not os.path.exists(src_path):
            for root, dirs, files in os.walk(dataset_path):
                if file_name in files:
                    src_path = os.path.join(root, file_name)
                    break
        if os.path.exists(src_path):
            print(f"📦 Copying {file_name} to project 'source_data' folder...")
            shutil.copy(src_path, local_path)

    df = pd.read_csv(local_path, nrows=row_limit, encoding='latin-1', low_memory=False)
    df = df[~df['Accident_Severity'].isin(['1', '2', '3'])]
    df = df.dropna(subset=['Accident_Severity', 'Light_Conditions', 'Weather_Conditions', 'Latitude', 'Longitude', 'Vehicle_Type'])
    df['Vehicle_Group'] = df['Vehicle_Type'].apply(group_vehicle_types)
    
    if 'Time' in df.columns:
        df['Hour'] = pd.to_datetime(df['Time'], format='%H:%M', errors='coerce').dt.hour
        mask = df['Hour'].isna()
        if mask.any():
            df.loc[mask, 'Hour'] = pd.to_datetime(df.loc[mask, 'Time'], format='%H:%M:%S', errors='coerce').dt.hour
        df['Hour'] = df['Hour'].fillna(0).astype(int)
    
    df = localize_to_libya(df)
    df['Accident Date'] = pd.to_datetime(df['Accident Date'], format='%d/%m/%Y', errors='coerce')
    categorical_cols = ['Day_of_Week', 'Junction_Control', 'Road_Surface_Conditions', 'Road_Type', 'Urban_or_Rural_Area', 'Vehicle_Group']
    for col in categorical_cols:
        df[col] = df[col].astype('category')
    
    return df
