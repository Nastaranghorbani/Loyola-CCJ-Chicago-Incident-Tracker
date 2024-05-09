import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import arviz as az
import pymc as pm
from scipy.stats import norm
from orbit.models.dlt import DLT
from orbit.diagnostics.plot import plot_predicted_data
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
import datawrapper as dw 




  # Function to format the x-axis of matplotlib plots
  def format_x_axis(ax, minor=False):
      # major ticks
      ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y %b"))
      ax.xaxis.set_major_locator(mdates.YearLocator())
      ax.grid(which="major", linestyle="-", axis="x")
      # minor ticks
      if minor:
          ax.xaxis.set_minor_formatter(mdates.DateFormatter("%Y %b"))
          ax.xaxis.set_minor_locator(mdates.MonthLocator())
          ax.grid(which="minor", linestyle=":", axis="x")
      # rotate labels
      for label in ax.get_xticklabels(which="both"):
          label.set(rotation=70, horizontalalignment="right")

  # Function to plot Bayesian model predictions using ArviZ
  def plot_xY(x, Y, ax):
      quantiles = Y.quantile((0.025, 0.25, 0.5, 0.75, 0.975), dim=("chain", "draw")).transpose()

      az.plot_hdi(
          x,
          hdi_data=quantiles.sel(quantile=[0.025, 0.975]),
          fill_kwargs={"alpha": 0.25},
          smooth=False,
          ax=ax,
      )
      az.plot_hdi(
          x,
          hdi_data=quantiles.sel(quantile=[0.25, 0.75]),
          fill_kwargs={"alpha": 0.5},
          smooth=False,
          ax=ax,
      )
      ax.plot(x, quantiles.sel(quantile=0.5), color="C1", lw=3)


  # default figure sizes
  figsize = (10, 5)


  # Calculate a causal effect based on ISO week
  def causal_effect(df):
    
      return (df['ISO_Week'] > treatment_iso_week_str) * 2

  week_sum = week_sum.assign(time=np.arange(len(week_sum)))
  week_sum


 def causal_effect(df):
      # Use 'ISO_Week' for comparison
      return (df['ISO_Week'] > treatment_iso_week_str) * 2





if __name__ == "__main__":
   
# Import data
df = pd.read_csv('https://raw.githubusercontent.com/Nastaranghorbani/Loyola-CCJ-Chicago-Incident-Tracker/main/data/inc_data_selected.csv')

# Convert the 'date' column to a pandas datetime object
df['date'] = pd.to_datetime(df['date'])

# Extract the ISO week number and year from the 'date' column
df['week'] = df['date'].dt.isocalendar().week
df['year'] = df['date'].dt.year

# Group the data by 'year' and 'week'
week_sum = df.groupby(['year', 'week'])[['Reported Incident', 'Enforcement Driven Incidents',
                                         'Simple-Cannabis', 'Gun Offense', 'Criminal Sexual Assault',
                                         'Aggravated Assault', 'Violent Offense', 'Burglary', 'Theft',
                                         'Domestic Violence', 'Robbery', 'Violent Gun Offense']].sum().reset_index()

# Create 'ISO_Week' column
week_sum['ISO_Week'] = week_sum['year'].astype(str) + '-' + week_sum['week'].apply(lambda x: f'{x:02}')

# Treatment date and convert to ISO week string
treatment_date = pd.to_datetime("2023-09-18")
treatment_iso_year = treatment_date.isocalendar().year
treatment_iso_week = treatment_date.isocalendar().week
treatment_iso_week_str = f"{treatment_iso_year}-{treatment_iso_week:02}"

# Assign time and calculate 'y'
week_sum = week_sum.assign(time=np.arange(len(week_sum)))
β0 = 0
β1 = 0.1
N = len(week_sum)
week_sum['y'] = β0 + β1 * week_sum['time'] + causal_effect(week_sum) + norm(0, 0.5).rvs(N)
week_sum.reset_index(drop=True, inplace=True)

# Split the data into pre- and post-intervention
mask = week_sum['ISO_Week'] < treatment_iso_week_str
pre = week_sum[mask]
post = week_sum[~mask]

# Plotting the intervention analysis
fig, ax = plt.subplots()
pre['y'].plot(ax=ax, label="Pre-Intervention")
post['y'].plot(ax=ax, label="Post-Intervention")
treatment_index = week_sum[week_sum['ISO_Week'] == treatment_iso_week_str].index.min()
ax.axvline(treatment_index, color="black", linestyle=":")
plt.legend()
plt.show()

# Prepare the data for more complex time series analysis
week_sum['Date'] = pd.to_datetime(week_sum['ISO_Week'] + '-1', format="%Y-%W-%w")
week_sum.set_index('Date', inplace=True)
pre = week_sum[week_sum.index < pd.to_datetime(treatment_time)]
post = week_sum[week_sum.index >= pd.to_datetime(treatment_time)]

fig, ax = plt.subplots()
pre['y'].plot(ax=ax, label="Pre-Intervention")
post['y'].plot(ax=ax, label="Post-Intervention")
ax.axvline(pd.to_datetime(treatment_time), color="black", linestyle=":")
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.xticks(rotation=45)
plt.legend()
plt.show()

# First PyMC model to predict a generic outcome 'y' using the time as predictor
with pm.Model() as model:
    time = pm.Data("time", pre["time"].to_numpy(), dims="obs_id")
    beta0 = pm.Normal("beta0", 0, 1)
    beta1 = pm.Normal("beta1", 0, 0.2)
    mu = pm.Deterministic("mu", beta0 + (beta1 * time), dims="obs_id")
    sigma = pm.HalfNormal("sigma", 2)
    pm.Normal("obs", mu=mu, sigma=sigma, observed=pre["y"].to_numpy(), dims="obs_id")

# Second PyMC model specific to the 'Theft' outcome variable
outcome_variable = 'Theft'
with pm.Model() as model:
    time = pm.Data("time", pre["time"].to_numpy(), dims="obs_id")
    observed_data = pm.Data("observed_data", pre[outcome_variable].to_numpy(), dims="obs_id")
    beta0 = pm.Normal("beta0", 0, 1)
    beta1 = pm.Normal("beta1", 0, 0.2)
    mu = pm.Deterministic("mu", beta0 + (beta1 * time), dims="obs_id")
    sigma = pm.HalfNormal("sigma", 2)
    pm.Normal("obs", mu=mu, sigma=sigma, observed=observed_data, dims="obs_id")

# Convert 'ISO_Week' to datetime to get correct indexing
week_sum['date'] = pd.to_datetime(week_sum['ISO_Week'] + '-1', format='%G-%V-%u')
week_sum.set_index('date', inplace=True)

# Define the treatment time
treatment_time = pd.to_datetime("2023-09-18")

# List of crime types
crime_types = ['Reported Incident', 'Enforcement Driven Incidents', 'Simple-Cannabis', 
               'Gun Offense', 'Criminal Sexual Assault', 'Aggravated Assault', 
               'Violent Offense', 'Burglary', 'Theft', 'Domestic Violence', 
               'Robbery', 'Violent Gun Offense']

# For each crime type, perform decomposition and interrupted time series analysis
for crime in crime_types:
    decomp = seasonal_decompose(week_sum[crime], period=52, model='additive', extrapolate_trend='freq')
    week_sum['intervention'] = (week_sum.index >= treatment_time).astype(int)
    week_sum['trend'] = decomp.trend
    week_sum['seasonal'] = decomp.seasonal
    week_sum['trend_intervention'] = week_sum['trend'] * week_sum['intervention']
    model = sm.OLS(week_sum[crime], sm.add_constant(week_sum[['trend', 'seasonal', 'trend_intervention']])).fit()
    week_sum[f'predicted_{crime}'] = model.predict(sm.add_constant(week_sum[['trend', 'seasonal', 'trend_intervention']]))
    plt.figure(figsize=(10, 6))
    plt.plot(week_sum.index, week_sum[crime], label='Actual')
    plt.plot(week_sum.index, week_sum[f'predicted_{crime}'], label='Predicted')
    plt.axvline(x=treatment_time, color='red', linestyle='--', label='Intervention')
    plt.title(crime)
    plt.legend()
    plt.show()

# Save results to CSV
week_sum.to_csv('data/complete_analysis_results.csv', index=False)

# Chart IDs dictionary
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

# Base directory for CSV files
base_dir = 'data/'

# Function to update and publish charts in Datawrapper
def update_and_publish_chart(crime):
    file_path = f'{base_dir}{crime.replace(" ", "_").lower()}_predictions.csv'
    chart_id = chart_ids[crime]
    
    with open(file_path, 'r') as file:
        csv_data = file.read()
    
    # Update data in the chart
    response = dw.add_data(chart_id=chart_id, data=csv_data)
    
    # Update chart properties
    dw.update_chart(chart_id, metadata={
        'visualize': {
            'y-grid': True,
            'y-axis-title': 'Number of Incidents',
            'x-grid': True
        }
    })
    
    # Publish the chart and get the public URL
    dw.publish_chart(chart_id)
    public_url = dw.get_chart(chart_id)['publicUrl']
    print(f'Chart for {crime} updated successfully. View at: {public_url}')

# Update and publish charts for each crime type
for crime in crime_types:
    update_and_publish_chart(crime)
