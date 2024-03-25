import pandas as pd
from datawrapper import Datawrapper


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



# Filter the data to include only dates after 2021 and before 2024
week_sum_filtered_for_average = week_sum[week_sum['year'] < 2024]
print(week_sum_filtered_for_average)


week_sum_filtered_for_average


# Group the data by 'week' and calculate the mean for each week
weekly_averages = week_sum_filtered_for_average.groupby('week')[[ 'Reported Incident', 'Enforcement Driven Incidents',
       'Simple-Cannabis', 'Gun Offense', 'Criminal Sexual Assault',
       'Aggravated Assault', 'Violent Offense', 'Burglary', 'Theft',
       'Domestic Violence', 'Robbery', 'Violent Gun Offense']].mean().reset_index()

weekly_averages



# Filter the 2024 data
week_sum_2024 = week_sum_filtered[week_sum_filtered['year'] == 2024]

# Merge the 2024 data with the weekly averages on the 'week' column
comparison_df = pd.merge(week_sum_2024, weekly_averages, on='week', suffixes=('_2024', '_avg'))

# Calculate the difference between 2024 values and weekly averages for each category and store in a new dataframe
diff_columns = {'week': 'week'}
for col in ['Reported Incident', 'Enforcement Driven Incidents', 'Simple-Cannabis', 'Gun Offense', 'Criminal Sexual Assault', 'Aggravated Assault', 'Violent Offense', 'Burglary', 'Theft', 'Domestic Violence', 'Robbery', 'Violent Gun Offense']:
    diff_column_name = f'{col}_diff'
    comparison_df[diff_column_name] = comparison_df[f'{col}_2024'] - comparison_df[f'{col}_avg']
    diff_columns[diff_column_name] = diff_column_name

# Select only the week number and the differences columns
comparison_df_diff_only = comparison_df[list(diff_columns.values())]

# Display the comparison dataframe with only the differences
comparison_df_diff_only


comparison_df_diff_only.to_csv('comparison.csv', index=False)



from datawrapper import Datawrapper

# Set up Datawrapper API
dw = Datawrapper(access_token="Lwu97tMzraxROmm7rNCbgJOnWGaRyXLtdXkTkRRDjx2HQD4NL6BCsvs858Q13oav")

# Loop through each column and create a chart
for column in comparison_df_diff_only.columns[1:]:  # Skip the 'week' column
    # Create a new chart
    chart = dw.create_chart(title=f"Comparison Chart for {column}", chart_type="d3-lines", data=comparison_df_diff_only[['week', column]])
    
    # Number of weeks in the data
    num_weeks = len(comparison_df_diff_only)
    
    # Update the chart description
    dw.update_description(
        chart["id"],
        source_name=" ",
        source_url=" ",
        byline=" ",
        intro=f"There were x many crimes this week in chicago that is higher/lower relative to 3 week average for {column}."
    )
    
    # Publish the chart
    dw.publish_chart(chart["id"])
