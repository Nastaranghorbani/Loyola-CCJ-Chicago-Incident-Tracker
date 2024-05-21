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
from datawrapper import Datawrapper

# Datawrapper API setup
API_KEY = os.environ.get('DATAWRAPPER_API')

dw = Datawrapper(access_token=API_KEY)

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

# Function to update and publish charts in Datawrapper
def update_and_publish_chart(crime, base_dir, chart_ids):
    file_path = f'{base_dir}{crime.replace(" ", "_").lower()}_predictions.csv'
    chart_id = chart_ids[crime]
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    dw.add_data(chart_id, file_path)
    
    payload = {
        'visualize': {
            'y-grid': True,
            'y-axis-title': 'Number of Incidents',
            'x-grid': True
        }
    }
    print(f"Updating chart with ID {chart_id} and payload: {payload}")
    
    dw.update_chart(chart_id, payload)
    
    dw.publish_chart(chart_id)
    public_url = dw.get_chart_metadata(chart_id)['publicUrl']
    print(f'Chart for {crime} updated successfully. View at: {public_url}')

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
    treatment_iso_year, treatment_iso_week, _ = treatment_date.isocalendar()
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
        time = pm.Data("time", pre["time"].to_numpy(), dims="obs_id", mutable=False)
        beta0 = pm.Normal("beta0", 0, 1)
        beta1 = pm.Normal("beta1", 0, 0.2)
        mu = pm.Deterministic("mu", beta0 + (beta1 * time), dims="obs_id")
        sigma = pm.HalfNormal("sigma", 2)
        pm.Normal("obs", mu=mu, sigma=sigma, observed=pre["y"].to_numpy(), dims="obs_id")

    outcome_variable = 'Theft'
    with pm.Model() as model:
        time = pm.Data("time", pre["time"].to_numpy(), dims="obs_id", mutable=False)
        observed_data = pm.Data("observed_data", pre[outcome_variable].to_numpy(), dims="obs_id", mutable=False)
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
        week_sum['intervention'] = (week_sum.index >= treatment_time).astype(int)
        week_sum['trend'] = decomp.trend
        week_sum['seasonal'] = decomp.seasonal
        week_sum['trend_intervention'] = week_sum['trend'] * week_sum['intervention']
        model = sm.OLS(week_sum[crime], sm.add_constant(week_sum[['trend', 'seasonal', 'trend_intervention']])).fit()
        week_sum[f'predicted_{crime}'] = model.predict(sm.add_constant(week_sum[['trend', 'seasonal', 'trend_intervention']]))
        plt.figure(figsize=(10, 6))
        plt.plot(week_sum[crime], label='Observed')
        plt.plot(week_sum[f'predicted_{crime}'], label='Predicted', linestyle='--')
        plt.title(f'{crime} - Observed vs Predicted')
        plt.legend()
        plt.grid()
        plt.show()

        file_path = f'{base_dir}{crime.replace(" ", "_").lower()}_predictions.csv'
        week_sum.to_csv(file_path, columns=['ISO_Week', f'predicted_{crime}'], index=False)

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

    for crime in crime_types:
        update_and_publish_chart(crime, base_dir, chart_ids)
