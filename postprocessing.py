# Import libraries

import pandas as pd
import datetime
import geopandas as gpd
import numpy as np
from datawrapper import Datawrapper
import os
import matplotlib.pyplot as plt
import geopandas as gpd
import matplotlib.pyplot as plt


# Offense features function
def offense_features(df):
    
    # assign enforcement drive offenses
    enfor_do = ['GAMBLING', 'CONCEALED CARRY LICENSE VIOLATION', 'NARCOTICS', 'WEAPONS VIOLATION', 'OBSCENITY', 'PROSTITUTION', 'INTERFERENCE WITH PUBLIC OFFICER', 'LIQUOR LAW VIOLATION', 'OTHER NARCOTIC VIOLATION']
    df['Enforcement Driven Incidents'] = np.where(df['primary_type'].isin(enfor_do), 1, 0)
    
    #assign domestic battery
    df['Domestic Battery'] = np.where(df['description'].str.lower().str.contains('domestic|dom') == True, 1, 0)
    
    #Assign Domestic Violence
    df['Domestic Violence'] = np.where(
        (df['Domestic Battery'] == 1) |
        ((df['primary_type'] == 'BATTERY') & (df['domestic'] == True)) |
        ((df['primary_type'] == 'ASSAULT') & (df['domestic'] == True)) |
        ((df['primary_type'] == 'CRIM SEXUAL ASSAULT') & (df['domestic'] == True)),
        1, 0

    )
    # Remove simple marijuana possession (under 30g) and distribution/intent to sell (under 10g) from offense differences
    df['simple-cannabis'] =  np.where((df['primary_type'] == 'NARCOTICS') &
                                  (df['description'].isin(['POSS: CANNABIS 30GMS OR LESS', 'MANU/DEL:CANNABIS 10GM OR LESS'])), 1, 0)

    df['primary_type'] = np.where(df['simple-cannabis'] == 1, 'NARCOTICS-CANNABIS', df['primary_type'])
    
    df['is_gun'] = np.where(df['description'].str.lower().str.contains('gun|firearm'), 1, 0)
    df['crim_sex_offense'] = np.where((df['primary_type'] == 'CRIM SEXUAL ASSAULT')| 
                                (df['primary_type'].isin(['CRIMINAL SEXUAL ABUSE', 'AGG CRIMINAL SEXUAL ABUSE', 'AGG CRIMINAL SEXUAL ABUSE']) == True),
                                      1, 0)
    df['is_agg_assault'] =  np.where((df['primary_type'] == 'ASSAULT') & (df['description'].str.lower().str.contains('agg') == True), 1, 0)

    df['is_violent'] = np.where((df['primary_type'] == 'ROBBERY')|
                               (df['primary_type'] == 'HOMICIDE')|
                               (df['crim_sex_offense'] == 1)|
                                (df['is_agg_assault'] == 1), 1, 0)

    df['is_burglary'] = np.where(df['primary_type'] == 'BURGLARY', 1, 0)
    
    df['is_theft'] = np.where(df['primary_type'] == 'THEFT', 1, 0)
    
    df['is_domestic'] = np.where(df['domestic'] == 'True', 1, 0)
    
    df['is_robbery'] = np.where(df['primary_type'] == 'ROBBERY', 1, 0)
    
    df['violent_gun'] = np.where((df['is_violent'] == 1) & (df['is_gun'] == 1), 1, 0)
    
    return df



### Import Police and Ward Info

ward = gpd.read_file('https://data.cityofchicago.org/api/geospatial/sp34-6z76?method=export&format=GeoJSON')
police = gpd.read_file('https://data.cityofchicago.org/api/geospatial/fthy-xz3r?method=export&format=GeoJSON')




### Add Offense Map
offense_map = {'simple-cannabis':'Simple-Cannabis', 'is_gun':'Gun Offense',
                   'crim_sex_offense':'Criminal Sexual Assault', 'is_agg_assault':'Aggravated Assault',
                   'is_violent':'Violent Offense', 'is_burglary':'Burglary',
                   'is_theft':'Theft', 'is_domestic':'is_domestic', 'is_robbery':"Robbery", 'violent_gun':"Violent Gun Offense"}





full_feature_list = ['date','Reported Incident', 'Enforcement Driven Incidents','Simple-Cannabis',\
       'Gun Offense', 'Criminal Sexual Assault', 'Aggravated Assault',\
       'Violent Offense', 'Burglary', 'Theft', 'Domestic Violence', 'Robbery',\
       'Violent Gun Offense']






API_KEY = os.environ['DATAWRAPPER_API']
dw = Datawrapper(access_token=API_KEY)




if __name__ == "__main__":


    
    
    ### Import Data
    
    today = datetime.date.today()
    
    current_yr = today.year
    my_df = []
    
    for year in range(2018, current_yr + 1):
        inc_data = pd.read_csv(
            f'https://data.cityofchicago.org/resource/6zsd-86xi.csv?$limit=2000000&$where=date%20between%20%27{year}-01-01T00:00:00%27%20and%20%27{year}-12-31T23:59:59%27'
        )
        my_df.append(inc_data)
    
    combined_df = pd.concat(my_df, ignore_index=True)
    print(combined_df.shape)
    
    
    
    ### Join Ward Info
    
    combined_df['ward'] = combined_df['ward'].astype(str)
    
    
    ### Add Date Features
    
    combined_df['date'] = pd.to_datetime(combined_df['date'], errors='coerce')
    
    
    combined_df['year'] = combined_df['date'].dt.year
    combined_df['month'] = combined_df['date'].dt.month
    combined_df['day'] = combined_df['date'].dt.day
    
    
    
    combined_df['Time'] = combined_df['date'].dt.time
    
    
    combined_df['date'] = combined_df['date'].dt.date
    
    
    combined_df['Reported Incident'] = 1
    
    
    ### Add Offense Features
    
    inc_data_processed = offense_features(combined_df)
    
    
    inc_data_processed = inc_data_processed.rename(columns=offense_map)
    
    
    inc_data_merged = inc_data_processed.merge(ward[['ward', 'geometry']], how='left', left_on="ward", right_on='ward').rename(columns={'geometry':"Ward_Geo"})
    
    
    ### Subset the Data
    
    inc_data_selected = inc_data_merged[full_feature_list]
    inc_data_selected
    
    ### Save the selected features to a CSV file
    inc_data_selected.to_csv('data/inc_data_selected.csv', index=False)
    #inc_data_selected.to_csv('/Users/nastaranghorbani/Documents/CCJ/Codes/inc_data_selected.csv', index=False)




