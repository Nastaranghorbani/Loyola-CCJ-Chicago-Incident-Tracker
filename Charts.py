import pandas as pd
import geopandas as gpd
import numpy as np
import os
from datawrapper import Datawrapper
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import datetime, timedelta





# Function to check if a week has 7 days of data
def is_full_week(df, year, week):
    start_date = datetime.strptime(f'{year}-W{week}-1', "%Y-W%W-%w")
    end_date = start_date + timedelta(days=6)
    week_data = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
    return len(week_data) >= 7

# Function to filter the dataframe to include only full weeks
def filter_full_weeks(df):
    # Calculate the number of days of data available for each year and week
    df['day_of_week'] = df['date'].dt.dayofweek
    week_day_counts = df.groupby(['year', 'week'])['day_of_week'].nunique().reset_index()
    
    # Keep only the weeks with 7 days of data
    full_weeks = week_day_counts[week_day_counts['day_of_week'] == 7][['year', 'week']]
    
    # Merge to keep only full weeks in the original dataframe
    df_full_weeks = pd.merge(df, full_weeks, on=['year', 'week'], how='inner')
    return df_full_weeks.drop(columns=['day_of_week'])


HTML_STRING = """<b style="background-color: rgb(255, 191, 0); padding-left: 3px; padding-right: 3px ">"""




if __name__ == "__main__":




    
    # Import data
    df = pd.read_csv('https://raw.githubusercontent.com/Nastaranghorbani/Loyola-CCJ-Chicago-Incident-Tracker/main/data/inc_data_selected.csv')

    # Convert the 'date' column to a pandas datetime object
    df['date'] = pd.to_datetime(df['date'])

    # Extract the ISO week number from the 'date' column and create a new column 'week'
    df['week'] = df['date'].dt.isocalendar().week

    # Extract the year from the 'date' column and create a new column 'year'
    df['year'] = df['date'].dt.year

    # Filter the dataframe to include only full weeks
    df_full_weeks = filter_full_weeks(df)

    # Group the data by 'year' and 'week'
    week_sum = df_full_weeks.groupby(['year', 'week'])[['Reported Incident', 'Enforcement Driven Incidents',
                                                        'Simple-Cannabis', 'Gun Offense', 'Criminal Sexual Assault',
                                                        'Aggravated Assault', 'Violent Offense', 'Burglary', 'Theft',
                                                        'Domestic Violence', 'Robbery', 'Violent Gun Offense']].sum().reset_index()



    
    # Create 'ISO_Week' column
    week_sum['ISO_Week'] = week_sum['year'].astype(str) + '-' + week_sum['week'].apply(lambda x: f'{x:02}')

    # Filter the data to include only dates after January 1, 2021
    week_sum_filtered = week_sum[week_sum['year'] > 2020]

    # Calculate the average for all weeks
    avg_week_data = week_sum_filtered.groupby('week')[['Reported Incident', 'Enforcement Driven Incidents',
                                                       'Simple-Cannabis', 'Gun Offense', 'Criminal Sexual Assault',
                                                       'Aggravated Assault', 'Violent Offense', 'Burglary', 'Theft',
                                                       'Domestic Violence', 'Robbery', 'Violent Gun Offense']].mean().reset_index()

    # Merge the average data with week_sum_filtered
    avg_week_data = avg_week_data.rename(lambda x: x + '_average' if x != 'week' else x, axis='columns')
    week_sum_filtered = pd.merge(week_sum_filtered, avg_week_data, on='week', how='left')

    # Filter the data for 2024
    data_2024 = week_sum_filtered[week_sum_filtered['year'] == 2024].copy()  # Use .copy() to avoid SettingWithCopyWarning

    # Calculate the difference between each week's values and the corresponding average values
    comparison_columns = ['Reported Incident', 'Enforcement Driven Incidents', 'Simple-Cannabis', 'Gun Offense', 'Criminal Sexual Assault', 'Aggravated Assault', 'Violent Offense', 'Burglary', 'Theft', 'Domestic Violence', 'Robbery', 'Violent Gun Offense']
    for column in comparison_columns:
        data_2024[f'{column}_vs_average'] = data_2024[column] - data_2024[f'{column}_average']


    
    # Ensuring the latest week is a full week
    latest_week = data_2024.iloc[-1]
    if not is_full_week(df, latest_week['year'], latest_week['week']):
        data_2024 = data_2024.iloc[:-1]
        latest_week = data_2024.iloc[-1]

    # Getting the first day of the latest week
    first_day_of_week = datetime.strptime(f"{latest_week['year']}-W{int(latest_week['week'])}-1", "%Y-W%W-%w").strftime("%B %d")

    # Create and update charts with dynamic descriptions
    columns = ['Reported Incident', 'Enforcement Driven Incidents', 'Simple-Cannabis', 'Gun Offense', 'Criminal Sexual Assault', 'Aggravated Assault', 'Violent Offense', 'Burglary', 'Theft', 'Domestic Violence', 'Robbery', 'Violent Gun Offense']

    # Chart IDs dictionary to store the chart IDs for each crime type
    chart_ids = {
        'Reported Incident': 'qeS7S',
        'Enforcement Driven Incidents': 'AMeVO',
        'Simple-Cannabis': '4VXqm',
        'Gun Offense': 'VFkbY',
        'Criminal Sexual Assault': 'BkDU0',
        'Aggravated Assault': 'nis8v',
        'Violent Offense': 'NFIDi',
        'Burglary': '9jMj4',
        'Theft': 'HvDKE',
        'Domestic Violence': 'W1NrO',
        'Robbery': '2BNYv',
        'Violent Gun Offense': 'eG0Xd'
    }





    # Datawrapper API setup
    API_KEY = os.environ.get('DATAWRAPPER_API')
    dw = Datawrapper(access_token=API_KEY)

    
    for column in columns:
        # Calculate the percentage change, handling division by zero or NaN values
        avg_value = latest_week[f'{column}_average']
        if avg_value == 0 or np.isnan(avg_value):
            percentage_change = 0
        else:
            percentage_change = (latest_week[column] - avg_value) / avg_value * 100

        change_type = "increase" if percentage_change > 0 else "decrease"
        percentage_change = abs(percentage_change)

        # Check if the chart already exists
        chart_id = chart_ids[column]

        # Update the chart data
        dw.add_data(chart_id, week_sum_filtered[['ISO_Week', column]].to_csv(index=False))

        # Update the chart description
        # Update the chart description
        description = (
        f"There have been {latest_week[column]} {column.lower()} incidents in Chicago for the week of {first_day_of_week}. "
        f"This is a {change_type} of {HTML_STRING}{percentage_change:.2f}%</b>. "
        f"A difference of {HTML_STRING}{latest_week[column] - latest_week[f'{column}_average']:.0f}</b> incidents."
        )
        
        dw.update_description(
            chart_id,
            intro=description,
            source_name=" ",
            source_url=" ",
            byline=" "
        )

        

        # Publish the chart
        dw.publish_chart(chart_id)

