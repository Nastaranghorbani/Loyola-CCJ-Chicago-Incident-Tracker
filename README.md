# Loyola-CCJ-Chicago-Incident-Tracker

This is a Python script that imports the yearly incident data from the Chicago data portal from 2018.


## Execution :

- the Github Action action is scheduled daily.

- the `flat.yml` specifies the action, triggers the install of ghostscript, and installs the required dependencies. 

- the `postprocess.py` script is then trigged and parses the daily pdf table into a cleaned pandas dataframe. Then, an existing .csv of past jail records is imported and the parsed csv is appended to that file and saved at `data/...`. 

- The most recent parsed pdf table is saved at `data/...`. 


## Data Notes

- Data is taken from the [Chicago Data Portal](https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-Present/ijzp-q8t2/data)
- Data begins at 01-01-2018 and is updated daily. 
