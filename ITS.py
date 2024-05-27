import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import norm
from orbit.models.dlt import DLT
from orbit.diagnostics.plot import plot_predicted_data
from statsmodels.tsa.seasonal import seasonal_decompose
from datetime import datetime
from datawrapper import Datawrapper
import matplotlib.dates as mdates


# Function to format x-axis
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

# Function to plot predictions
def plot_predictions(pre, post, treatment_time, crime_type, week_sum):
    fig, ax = plt.subplots()
    pre['y'].plot(ax=ax, label="Pre-Intervention")
    post['y'].plot(ax=ax, label="Post-Intervention")
    ax.axvline(pd.to_datetime(treatment_time), color="black", linestyle=":")
    format_x_axis(ax)
    plt.title(f"{crime_type} - Actual vs. Predicted")
    plt.legend()
    plt.show()

# Function to fit DLT model and get predictions
def fit_dlt_model(week_sum, crime_type):
    dlt = DLT(response_col=crime_type, 
              date_col='date', 
              regressor_col=['intervention'],
              seasonality=52, 
              estimator='stan-map',  
              seed=2022)
    dlt.fit(df=week_sum)
    predicted_df = dlt.predict(df=week_sum)
    return predicted_df


# Datawrapper API setup
API_KEY = os.environ.get('DATAWRAPPER_API')
dw = Datawrapper(access_token=API_KEY)



# Chart IDs dictionary
chart_ids = {
    'Reported Incident': 'UV70R',
    'Enforcement Driven Incidents': '0ZAre',
    'Simple-Cannabis': 'qjs8a',
    'Gun Offense': 'Xn9w3',
    'Criminal Sexual Assault': 'KV02G',
    'Aggravated Assault': 't2Duz',
    'Violent Offense': 'TGCUm',
    'Burglary': '8vg1Q',
    'Theft': '3jFDQ',
    'Domestic Violence': 'hyQWa',
    'Robbery': 'feB8w',
    'Violent Gun Offense': 'NMbX2'
}

if __name__ == "__main__":
    # Import data
    df = pd.read_csv('https://raw.githubusercontent.com/Nastaranghorbani/Loyola-CCJ-Chicago-Incident-Tracker/main/data/inc_data_selected.csv')
    df['date'] = pd.to_datetime(df['date'])
    df['week'] = df['date'].dt.isocalendar().week
    df['year'] = df['date'].dt.year
    week_sum = df.groupby(['year', 'week'])[['Reported Incident', 'Enforcement Driven Incidents',
                                             'Simple-Cannabis', 'Gun Offense', 'Criminal Sexual Assault',
                                             'Aggravated Assault', 'Violent Offense', 'Burglary', 'Theft',
                                             'Domestic Violence', 'Robbery', 'Violent Gun Offense']].sum().reset_index()
    week_sum['ISO_Week'] = week_sum['year'].astype(str) + '-' + week_sum['week'].apply(lambda x: f'{x:02}')
    treatment_time = pd.to_datetime("2023-09-18")
    treatment_iso_year = treatment_time.isocalendar().year
    treatment_iso_week = treatment_time.isocalendar().week
    treatment_iso_week_str = f"{treatment_iso_year}-{treatment_iso_week:02}"
    week_sum = week_sum.assign(time=np.arange(len(week_sum)))
    week_sum['date'] = pd.to_datetime(week_sum['ISO_Week'] + '-1', format='%G-%V-%u')
    week_sum.set_index('date', inplace=True)
    week_sum['intervention'] = (week_sum.index >= treatment_time).astype(int)

    crime_types = ['Reported Incident', 'Enforcement Driven Incidents', 'Simple-Cannabis', 
                   'Gun Offense', 'Criminal Sexual Assault', 'Aggravated Assault', 
                   'Violent Offense', 'Burglary', 'Theft', 'Domestic Violence', 
                   'Robbery', 'Violent Gun Offense']

    main_title = 'Chicago Crime Incidents - Actual vs. Predicted'

    for crime_type in crime_types:
        decomp = seasonal_decompose(week_sum[crime_type], period=52, model='additive', extrapolate_trend='freq')
        week_sum['trend'] = decomp.trend
        week_sum['seasonal'] = decomp.seasonal
        week_sum['trend_intervention'] = week_sum['trend'] * week_sum['intervention']
        model = sm.OLS(week_sum[crime_type], sm.add_constant(week_sum[['trend', 'seasonal', 'trend_intervention']])).fit()
        week_sum[f'predicted_{crime_type}'] = model.predict(sm.add_constant(week_sum[['trend', 'seasonal', 'trend_intervention']]))
    
        # Updating Datawrapper chart
        chart_id = chart_ids[crime_type]
        #data_to_add = week_sum.reset_index()[['date', crime_type, f'predicted_{crime_type}']]
        #data_to_add.columns = ['Date', 'Actual', 'Predicted']
        data_to_add = week_sum.reset_index()[['date', f'predicted_{crime_type}']]
        data_to_add.columns = ['Date', 'Predicted']
        dw.add_data(chart_id, data_to_add.to_csv(index=False))
        
        dw.update_chart(chart_id, metadata={
            'title': f'{crime_type} - Actual vs. Predicted',
            'describe': main_title
        })
        dw.publish_chart(chart_id)
