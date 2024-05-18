import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import arviz as az
import pymc as pm
from scipy.stats import norm
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
import os
import requests

# Function to format the x-axis of matplotlib plots
def format_x_axis(ax, minor=False):
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y %b"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.grid(which="major", linestyle="-", axis="x")
    if minor:
        ax.xaxis.set_minor_formatter(mdates.DateFormatter("%Y %b"))
        ax.xaxis.set_minor_locator(mdates.MonthLocator())
        ax.grid(which="minor", linestyle=":", axis="x")
    for label in ax.get_xticklabels(which="both"):
        label.set(rotation=70, horizontalalignment="right")

# Function to plot Bayesian model predictions using ArviZ
def plot_xY(x, Y, ax):
    quantiles = Y.quantile((0.025, 0.25, 0.5, 0.75, 0.975), dim=("chain", "draw")).transpose()
    az.plot_hdi(x, hdi_data=quantiles.sel(quantile=[0.025, 0.975]), fill_kwargs={"alpha": 0.25}, smooth=False, ax=ax)
    az.plot_hdi(x, hdi_data=quantiles.sel(quantile=[0.25, 0.75]), fill_kwargs={"alpha": 0.5}, smooth=False, ax=ax)
    ax.plot(x, quantiles.sel(quantile=0.5), color="C1", lw=3)

# Calculate a causal effect based on ISO week
def causal_effect(df, treatment_iso_week_str):
    return (df['ISO_Week'] > treatment_iso_week_str) * 2

# Function to update and publish charts in Datawrapper using requests
def update_and_publish_chart(crime, base_dir, chart_ids):
    file_path = f'{base_dir}{crime.replace(" ", "_").lower()}_predictions.csv'
    chart_id = chart_ids[crime]
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    with open(file_path, 'rb') as f:
        data = f.read()
    
    # Upload data to Datawrapper
    headers = {
        'Authorization': f'Bearer {os.environ["DATAWRAPPER_API"]}',
        'Content-Type': 'text/csv'
    }
    
    response = requests.put(
        f'https://api.datawrapper.de/v3/charts/{chart_id}/data',
        headers=headers,
        data=data
    )
    
    if response.status_code != 204:
        print(f"Error uploading data: {response.status_code} {response.text}")
        return

    # Update chart title and properties
    response = requests.patch(
        f'https://api.datawrapper.de/v3/charts/{chart_id}',
        headers={'Authorization': f'Bearer {os.environ["DATAWRAPPER_API"]}'},
        json={
            'title': f'{crime} - Observed vs Predicted',
            'metadata': {
                'visualize': {
                    'y-grid': True,
                    'y-axis-title': 'Number of Incidents',
                    'x-grid': True
                }
            }
        }
    )
    
    if response.status_code != 200:
        print(f"Error updating chart: {response.status_code} {response.text}")
        return
    
    # Publish chart
    response = requests.post(
        f'https://api.datawrapper.de/v3/charts/{chart_id}/publish',
        headers={'Authorization': f'Bearer {os.environ["DATAWRAPPER_API"]}'}
    )
    
    if response.status_code != 200:
        print(f"Error publishing chart: {response.status_code} {response.text}")
        return

    print(f'Chart for {crime} updated successfully. View at: {response.json()["publicUrl"]}')

if __name__ == "__main__":
    try:
        df = pd.read_csv('https://raw.githubusercontent.com/Nastaranghorbani/Loyola-CCJ-Chicago-Incident-Tracker/main/data/inc_data_selected.csv')
    except Exception as e:
        print(f"Error loading data: {e}")
        exit(1)

    df['date'] = pd.to_datetime(df['date'])
    df['week'] = df['date'].dt.isocalendar().week
    df['year'] = df['date'].dt.year

    week_sum = df.groupby(['year', 'week'])[['Reported Incident', 'Enforcement Driven Incidents',
                                             'Simple-Cannabis', 'Gun Offense', 'Criminal Sexual Assault',
                                             'Aggravated Assault', 'Violent Offense', 'Burglary', 'Theft',
                                             'Domestic Violence', 'Robbery', 'Violent Gun Offense']].sum().reset_index()

    week_sum['ISO_Week'] = week_sum['year'].astype(str) + '-' + week_sum['week'].apply(lambda x: f'{x:02}')
    treatment_date = pd.to_datetime("2023-09-18")
    treatment_iso_year = treatment_date.isocalendar().year
    treatment_iso_week = treatment_date.isocalendar().week
    treatment_iso_week_str = f"{treatment_iso_year}-{treatment_iso_week:02}"

    week_sum = week_sum.assign(time=np.arange(len(week_sum)))
    β0 = 0
    β1 = 0.1
    N = len(week_sum)
    week_sum['y'] = β0 + β1 * week_sum['time'] + causal_effect(week_sum, treatment_iso_week_str) + norm(0, 0.5).rvs(N)
    week_sum.reset_index(drop=True, inplace=True)

    treatment_time = pd.to_datetime("2023-09-18")
    week_sum['Date'] = pd.to_datetime(week_sum['ISO_Week'] + '-1', format="%Y-%W-%w")
    week_sum.set_index('Date', inplace=True)
    pre = week_sum[week_sum.index < treatment_time]
    post = week_sum[week_sum.index >= treatment_time]

    fig, ax = plt.subplots()
    pre['y'].plot(ax=ax, label="Pre-Intervention")
    post['y'].plot(ax=ax, label="Post-Intervention")
    treatment_index = week_sum[week_sum['ISO_Week'] == treatment_iso_week_str].index.min()
    ax.axvline(treatment_index, color="black", linestyle=":")
    plt.legend()
    plt.show()

    fig, ax = plt.subplots()
    pre['y'].plot(ax=ax, label="Pre-Intervention")
    post['y'].plot(ax=ax, label="Post-Intervention")
    ax.axvline(pd.to_datetime(treatment_time), color="black", linestyle=":")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    plt.legend()
    plt.show()

    with pm.Model() as model:
        time = pm.Data("time", pre["time"].to_numpy(), dims="obs_id")
        beta0 = pm.Normal("beta0", 0, 1)
        beta1 = pm.Normal("beta1", 0, 0.2)
        mu = pm.Deterministic("mu", beta0 + (beta1 * time), dims="obs_id")
        sigma = pm.HalfNormal("sigma", 2)
        pm.Normal("obs", mu=mu, sigma=sigma, observed=pre["y"].to_numpy(), dims="obs_id")

    outcome_variable = 'Theft'
    with pm.Model() as model:
        time = pm.Data("time", pre["time"].to_numpy(), dims="obs_id")
        observed_data = pm.Data("observed_data", pre[outcome_variable].to_numpy(), dims="obs_id")
        beta0 = pm.Normal("beta0", 0, 1)
        beta1 = pm.Normal("beta1", 0, 0.2)
        mu = pm.Deterministic("mu", beta0 + (beta1 * time), dims="obs_id")
        sigma = pm.HalfNormal("sigma", 2)
        pm.Normal("obs", mu=mu, sigma=sigma, observed=observed_data, dims="obs_id")

    crime_types = ['Reported Incident', 'Enforcement Driven Incidents', 'Simple-Cannabis',
                   'Gun Offense', 'Criminal Sexual Assault', 'Aggravated Assault',
                   'Violent Offense', 'Burglary', 'Theft', 'Domestic Violence',
                   'Robbery', 'Violent Gun Offense']

    base_dir = 'data/'
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    for crime in crime_types:
        decomp = seasonal_decompose(week_sum[crime], period=52, model='additive', extrapolate_trend='freq')
        week_sum[f'{crime}_seasonal'] = decomp.seasonal
        week_sum[f'{crime}_trend'] = decomp.trend
        week_sum[f'{crime}_resid'] = decomp.resid
        decomp.trend.plot(title=f"{crime} Trend", figsize=(10, 6))
        plt.savefig(f'{base_dir}{crime.replace(" ", "_").lower()}_trend.png')
        plt.clf()

    chart_ids = {
        'Reported Incident': 'chart_id_1',
        'Enforcement Driven Incidents': 'chart_id_2',
        'Simple-Cannabis': 'chart_id_3',
        'Gun Offense': 'chart_id_4',
        'Criminal Sexual Assault': 'chart_id_5',
        'Aggravated Assault': 'chart_id_6',
        'Violent Offense': 'chart_id_7',
        'Burglary': 'chart_id_8',
        'Theft': 'chart_id_9',
        'Domestic Violence': 'chart_id_10',
        'Robbery': 'chart_id_11',
        'Violent Gun Offense': 'chart_id_12'
    }

    for crime in crime_types:
        update_and_publish_chart(crime, base_dir, chart_ids)
