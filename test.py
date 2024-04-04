import pandas as pd
import geopandas as gpd
import numpy as np
import os
from datawrapper import Datawrapper
import matplotlib.pyplot as plt
import geopandas as gpd
%matplotlib auto


# Import data
df=pd.read_csv('https://raw.githubusercontent.com/Nastaranghorbani/Loyola-CCJ-Chicago-Incident-Tracker/main/data/inc_data_selected.csv')




# Convert the 'date' column to a pandas datetime object
df['date'] = pd.to_datetime(df['date'])

# Extract the ISO week number from the 'date' column and create a new column 'week'
df['week'] = df['date'].dt.isocalendar().week

# Extract the year from the 'date' column and create a new column 'year'
df['year'] = df['date'].dt.year






# Group the data by 'year' and 'week'
df['year'] = df['date'].dt.year
df['week'] = df['date'].dt.isocalendar().week
week_sum = df.groupby(['year','week'])[[ 'Reported Incident', 'Enforcement Driven Incidents',
       'Simple-Cannabis', 'Gun Offense', 'Criminal Sexual Assault',
       'Aggravated Assault', 'Violent Offense', 'Burglary', 'Theft',
       'Domestic Violence', 'Robbery', 'Violent Gun Offense',]].sum().reset_index()
week_sum



# Create 'ISO_Week' column 
week_sum['ISO_Week'] = week_sum['year'].astype(str) + '-' + week_sum['week'].apply(lambda x: f'{x:02}')

week_sum


# Filter the data to include only dates after January 1, 2021
week_sum_filtered = week_sum[week_sum['year'] > 2020]
print(week_sum_filtered)




# Calculate the average for all weeks
avg_week_data = week_sum_filtered.groupby('week')[['Reported Incident', 'Enforcement Driven Incidents',
                                         'Simple-Cannabis', 'Gun Offense', 'Criminal Sexual Assault',
                                         'Aggravated Assault', 'Violent Offense', 'Burglary', 'Theft',
                                         'Domestic Violence', 'Robbery', 'Violent Gun Offense']].mean().reset_index()

# Merge the average data with week_sum_filtered
avg_week_data = avg_week_data.rename(lambda x: x + '_average' if x != 'week' else x, axis='columns')
week_sum_filtered = pd.merge(week_sum_filtered, avg_week_data, on='week', how='left')


week_sum_filtered


# Rename average columns
avg_week_data = avg_week_data.rename(lambda x: x + '_average' if x != 'week' else x, axis='columns')

# Merge the average data with week_sum_filtered
week_sum_filtered = pd.merge(week_sum_filtered, avg_week_data, on='week', how='left')



# Filter the data for 2024
data_2024 = week_sum_filtered[week_sum_filtered['year'] == 2024].copy()  # Use .copy() to avoid SettingWithCopyWarning

# Calculate the difference between each week's values and the corresponding average values
comparison_columns = ['Reported Incident', 'Enforcement Driven Incidents', 'Simple-Cannabis', 'Gun Offense', 'Criminal Sexual Assault', 'Aggravated Assault', 'Violent Offense', 'Burglary', 'Theft', 'Domestic Violence', 'Robbery', 'Violent Gun Offense']
for column in comparison_columns:
    data_2024[f'{column}_vs_average'] = data_2024[column] - data_2024[f'{column}_average_average']

# Display the comparison
comparison_df = data_2024[['ISO_Week'] + [f'{col}_vs_average' for col in comparison_columns]]


# Display the comparison DataFrame
comparison_df





HTML_STRING = """<b style="background-color: rgb(255, 191, 0); padding-left: 3px; padding-right: 3px ">"""


API_KEY = os.environ['DATAWRAPPER_API']
dw = Datawrapper(access_token=API_KEY)




# Get the current date and the current week number
current_date = datetime.now()
current_week = current_date.isocalendar().week
current_year = current_date.year

# Filter out the current week if the data is still being collected
if current_week == week_sum_filtered['week'].max() and current_year == week_sum_filtered['year'].max():
    latest_week = week_sum_filtered[week_sum_filtered['year'] == 2024].iloc[-2]
else:
    latest_week = week_sum_filtered[week_sum_filtered['year'] == 2024].iloc[-1]

# Get the first day of the latest week
first_day_of_week = datetime.strptime(f"{latest_week['year']}-W{int(latest_week['week'])}-1", "%Y-W%W-%w").strftime("%B %d")

# Create and update charts with dynamic descriptions
columns = ['Reported Incident', 'Enforcement Driven Incidents', 'Simple-Cannabis', 'Gun Offense', 'Criminal Sexual Assault', 'Aggravated Assault', 'Violent Offense', 'Burglary', 'Theft', 'Domestic Violence', 'Robbery', 'Violent Gun Offense']
for column in columns:
    # Calculate the percentage change, handling division by zero or NaN values
    avg_value = latest_week[f'{column}_average']
    if avg_value == 0 or np.isnan(avg_value):
        percentage_change = 0
    else:
        percentage_change = (latest_week[column] - avg_value) / avg_value * 100
    
    change_type = "increase" if percentage_change > 0 else "decrease"
    percentage_change = abs(percentage_change)
    
    # Create a new chart
    chart = dw.create_chart(title=f"Chart for {column}", chart_type="d3-lines", data=week_sum[['ISO_Week', column]])
    
    # Update the chart description
    description = f"There have been {latest_week[column]} {column.lower()} incidents in Chicago for the week of {first_day_of_week}. This is a {change_type} of {percentage_change:.2f}%. A difference of {latest_week[column] - latest_week[f'{column}_average']:.0f} incidents."
    dw.update_description(
        chart["id"],
        intro=description,
        source_name=" ",
        source_url=" ",
        byline=" "
    )
    
    # Publish the chart
    dw.publish_chart(chart["id"])

