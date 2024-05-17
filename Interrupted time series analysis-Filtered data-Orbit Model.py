import pandas as pd
import geopandas as gpd
import numpy as np
import os
from datawrapper import Datawrapper
import matplotlib.pyplot as plt
from datetime import datetime
from orbit.models import DLT
from orbit.diagnostics.plot import plot_predicted_data

# Import data
df = pd.read_csv('https://raw.githubusercontent.com/Nastaranghorbani/Loyola-CCJ-Chicago-Incident-Tracker/main/data/inc_data_selected.csv')

# Convert the 'date' column to a pandas datetime object
df['date'] = pd.to_datetime(df['date'])

# Extract the ISO week number from the 'date' column and create a new column 'week'
df['week'] = df['date'].dt.isocalendar().week

# Extract the year from the 'date' column and create a new column 'year'
df['year'] = df['date'].dt.year

# Group the data by 'year' and 'week'
df['year'] = df['date'].dt.year
df['week'] = df['date'].dt.isocalendar().week
week_sum = df.groupby(['year', 'week'])[['Reported Incident', 'Enforcement Driven Incidents',
                                         'Simple-Cannabis', 'Gun Offense', 'Criminal Sexual Assault',
                                         'Aggravated Assault', 'Violent Offense', 'Burglary', 'Theft',
                                         'Domestic Violence', 'Robbery', 'Violent Gun Offense']].sum().reset_index()

# Create 'ISO_Week' column
week_sum['ISO_Week'] = week_sum['year'].astype(str) + '-' + week_sum['week'].apply(lambda x: f'{x:02}')

# Filter the data to include only rows up to week 38 of 2023
filtered_data = week_sum[(week_sum['ISO_Week'] <= '2023-38')]

filtered_data = filtered_data.copy()

# Convert 'ISO_Week' to datetime and assign to 'Date', using .loc for safe in-place modification
filtered_data.loc[:, 'Date'] = pd.to_datetime(filtered_data['ISO_Week'] + '0', format='%Y-%W%w')

# Set the 'intervention' column
filtered_data.loc[:, 'intervention'] = (filtered_data['Date'] >= '2023-09-18').astype(int)

def perform_its_analysis(data, crime_type):
    print(f"Analyzing {crime_type}")
    
    # Prepare data
    ts_data = data[['Date', crime_type, 'intervention']].rename(columns={'Date': 'ds', crime_type: 'y'})
    
    # The model with the intervention
    model = DLT(response_col='y', date_col='ds', seasonality=52, estimator='stan-map', 
                regression_penalty='lasso', regressor_col=['intervention'])
    
    # Fit the model
    model.fit(df=ts_data)
    
    # Predict
    predicted_df = model.predict(df=ts_data)
    
    # Plot
    plt.figure(figsize=(12, 6))
    plot_predicted_data(training_actual_df=ts_data, predicted_df=predicted_df, date_col='ds', actual_col='y', title=f"{crime_type} ITS Analysis")
    plt.show()

# Crime types columns
crime_columns = ['Reported Incident', 'Enforcement Driven Incidents', 'Simple-Cannabis', 'Gun Offense',
                 'Criminal Sexual Assault', 'Aggravated Assault', 'Violent Offense', 'Burglary', 'Theft',
                 'Domestic Violence', 'Robbery', 'Violent Gun Offense']

# Ensure the data is sorted by date
filtered_data = filtered_data.sort_values('Date')

# Check for duplicate dates and remove if necessary
if filtered_data['Date'].duplicated().any():
    print("Duplicates found. Removing duplicates.")
    filtered_data = filtered_data.drop_duplicates(subset='Date')

# ITS analysis for all crime types
for crime in crime_columns:
    perform_its_analysis(filtered_data, crime)
