# Import pandas for data manipulation and group-based analysis
import pandas as pd

# Define a function to identify the most dangerous districts based on accident count
def get_dangerous_locations(df, top_n=10):
    # Group by district, count occurrences, sort by frequency, and return top n results
    return df.groupby('Local_Authority_(District)').size().sort_values(ascending=False).head(top_n)

# Define a function to determine the times of day with the highest accident frequency
def get_peak_times(df):
    # Group by hour, count accidents, and sort from most to least frequent
    return df.groupby('Hour').size().sort_values(ascending=False)

# Define a function to calculate accident statistics filtered by vehicle type
def get_violation_stats(df):
    # Group by vehicle type, count instances, and sort by descending frequency
    return df.groupby('Vehicle_Type').size().sort_values(ascending=False)

# Define a function to analyze the impact of lighting conditions on accident rates
def cross_analysis_lighting(df):
    # Guard clause: return 0 if dataframe is empty or lighting column is missing
    if df.empty or 'Light_Conditions' not in df.columns: return 0
    # Calculate the average number of accidents per day across all conditions
    base_avg = len(df) / df['Day_of_Week'].nunique()
    # Filter the dataset for poor lighting conditions using keyword search in both languages
    poor_light = df[df['Light_Conditions'].str.contains('Darkness|ظلام', na=False)]
    # Return 0 if no poor lighting cases are found in the dataset
    if poor_light.empty: return 0
    # Calculate the average accident frequency specifically for poor lighting conditions
    poor_light_avg = len(poor_light) / poor_light['Day_of_Week'].nunique()
    # Return the percentage increase of poor lighting accidents compared to the baseline
    return ((poor_light_avg - base_avg) / base_avg) * 100

# Define a function to generate a summary report of accident severity distribution
def get_severity_report(df):
    # Calculate normalized value counts (percentages) for each severity level
    return df['Accident_Severity'].value_counts(normalize=True) * 100

# Main execution block for testing analysis functions locally
if __name__ == "__main__":
    # Import the local data loader within the test block
    from data_processor import load_and_preprocess
    # Load and preprocess the dataset for analysis testing
    df = load_and_preprocess('Road Accident Data.csv')
    # Print the identified dangerous locations to console
    print(get_dangerous_locations(df))
    # Print the calculated lighting impact percentage to console
    print(f"Percentage increase in accidents with poor lighting: {cross_analysis_lighting(df):.2f}%")
