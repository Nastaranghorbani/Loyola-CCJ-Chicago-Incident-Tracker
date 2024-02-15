# Loyola-CCJ-Chicago-Incident-Tracker

This is a Python script that imports the yearly incident data from the Chicago data portal from 2018.

## Execution :

# Scrape Cook County's Jail Population Report 

A Python script that collects Cook jail population and community correction data. 

# Visualization

[Cook County Jail Population](https://www.datawrapper.de/_/JoeoH/)

[Cook County Community Corrections Population](https://www.datawrapper.de/_/GlakD/)

## Execution :

- the Github Action action is scheduled daily.

- the `flat.yml` specifies the action, triggers the install of ghostscript, and installs the required dependencies. 

- the `postprocess.py` script is then trigged and parses the daily pdf table into a cleaned pandas dataframe. Then, an existing .csv of past jail records is imported and the parsed csv is appended to that file and saved at `data/cook-jail-data.csv`.

- The most recent parsed pdf table is saved at `data/...`. 


## Data Notes

- Data is taken from the [Chicago Data Portal](https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-Present/ijzp-q8t2/data)
- Data begins at 01-01-2018 and is updated daily. 
