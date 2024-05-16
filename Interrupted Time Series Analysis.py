import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import norm
import statsmodels.api as sm
from datetime import datetime

# Function to format the x-axis of matplotlib plots
def format_x_axis(ax, minor=False):
    # Major ticks
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y %b"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.grid(which="major", linestyle="-", axis="x")
    # Minor ticks
    if minor:
        ax.xaxis.set_minor_formatter(mdates.DateFormatter("%Y %b"))
        ax.xaxis.set_minor_locator(mdates.MonthLocator())
        ax.grid(which="minor", linestyle=":", axis="x")
    # Rotate labels
    for label in ax.get_xticklabels(which="both"):
        label.set(rotation=70, horizontalalignment="right")

# Function to calculate a causal effect based on ISO week
def causal_effect(df, treatment_iso_week_str):
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
    week_sum = df.groupby(['year', 'week']).sum().reset_index()
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
    week_sum['y'] = β0 + β1 * week_sum['time'] + causal_effect(week_sum, treatment_iso_week_str) + norm(0, 0.5).rvs(N)
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
    format_x_axis(ax, minor=True)
    plt.legend()
    plt.show()
