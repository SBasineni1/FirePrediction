# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 22:36:56 2024

@author: nja75
"""

import pandas as pd
from pytz import timezone

# Read the CSV files using pandas
df_fire = pd.read_csv('fire_perimeters.csv')
df_weather = pd.read_csv('weather_station.csv')

# Convert datetime columns to datetime objects and localize to UTC timezone

# Convert datetime columns to datetime objects
df_fire['attr_FireDiscoveryDateTime'] = pd.to_datetime(df_fire['attr_FireDiscoveryDateTime'], errors='coerce')
df_fire['attr_FireOutDateTime'] = pd.to_datetime(df_fire['attr_FireOutDateTime'], errors='coerce')

# Remove timezone information from Date_Time column
df_weather['Date_Time'] = df_weather['Date_Time'].str.replace(' UTC', '')
df_weather['Date_Time'] = df_weather['Date_Time'].str.replace(' PST', '')
df_weather['Date_Time'] = df_weather['Date_Time'].str.replace(' PDT', '')

# Convert Date_Time column to datetime objects
df_weather['Date_Time'] = pd.to_datetime(df_weather['Date_Time'], errors='coerce')

# Drop rows with missing datetime values
df_fire.dropna(subset=['attr_FireDiscoveryDateTime', 'attr_FireOutDateTime'], inplace=True)

# Create an empty list to store the merged dataframes
merged_dfs = []

# Iterate over each record in df_fire
for index, row in df_fire.iterrows():
    # Find all records in df_weather where the Date_Time falls in the date range
    mask = (df_weather['Date_Time'] >= row['attr_FireDiscoveryDateTime']) & (df_weather['Date_Time'] <= row['attr_FireOutDateTime'])
    filtered_weather = df_weather[mask].copy()  # Make a copy to avoid SettingWithCopyWarning
    
    # Add OBJECTID column to the filtered_weather dataframe
    filtered_weather['OBJECTID'] = row['OBJECTID']  # Assign OBJECTID from df_fire
    
    # Append the filtered dataframe to merged_dfs
    merged_dfs.append(filtered_weather)

# Concatenate all dataframes in merged_dfs
final_merged_df = pd.concat(merged_dfs)

# Now final_merged_df contains all weather records within the date range of each fire record

final_df = final_merged_df.merge(df_fire, on='OBJECTID', how='left')

# Remove columns with all NaN values
final_df.dropna(axis=1, how='all', inplace=True)