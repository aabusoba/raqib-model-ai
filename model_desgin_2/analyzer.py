# Import pandas for data manipulation and frequency analysis
import pandas as pd

# Define a function to identify the most dangerous Libyan cities
def get_dangerous_locations(df):
    # Group by simulated Libyan city labels and retrieve the top 10 most frequent
    return df['Libyan_City'].value_counts().head(10)

# Define a function to extract and sort accident occurrences by hour of day
def get_peak_times(df):
    # Calculate frequency of accidents per hour and sort results by time
    return df['Hour'].value_counts().sort_index()

# Define a function to summarize the distribution of accident severity
def get_severity_report(df):
    # Count occurrences of each severity level within the dataset
    return df['Accident_Severity'].value_counts()

# Define a function to analyze the percentage difference between dark and daylight accidents
def cross_analysis_lighting(df):
    # Count accidents occurring specifically during poor lighting (darkness)
    dark_accidents = len(df[df['Light_Conditions'].str.contains('Dark', case=False, na=False)])
    # Count accidents occurring during optimal lighting (daylight)
    day_accidents = len(df[df['Light_Conditions'].str.contains('Daylight', case=False, na=False)])
    # Avoid division by zero by returning 0 if no daylight accidents are recorded
    if day_accidents == 0: return 0
    # Return the relative percentage ratio of dark-to-daylight accidents
    return (dark_accidents / day_accidents) * 100
