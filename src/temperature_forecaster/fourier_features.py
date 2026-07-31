import pandas as pd
import numpy as np

from temperature_forecaster.__init__ import weather_station_coords
from temperature_forecaster.paths import DATA_RAW, DATA_PROCESSED
from temperature_forecaster.find_optimize_fourier_terms import optimize_fourier_terms

def load_raw_data(city = None): # loads RAW data 
    loaded_data = []
    for location in weather_station_coords.keys():
        if ((city is None) or (location == city)):
            df = pd.read_csv(DATA_RAW/f"{location}_weather_data.csv", index_col=0, parse_dates=True)
            df["day_of_year"] = df.index.dayofyear
            loaded_data.append(df)
            print(f"Loaded from data/raw: {location}_weather_data.csv")
            if (location == city):
                return [df]
    return loaded_data

def engineer_df(dataframe, k_value): #STRICTLY A HELPER FUNCTION
    engineered_df = dataframe.copy()

    engineered_df = engineered_df[["tmax", "tmin"]]
    engineered_df["day_of_year"] = engineered_df.index.dayofyear
     
    for j in range(1, int(k_value)+1):
        engineered_df[f"Fsin{j}"] = np.sin(j*2*np.pi*engineered_df["day_of_year"]/365)
        engineered_df[f"Fcos{j}"] = np.cos(j*2*np.pi*engineered_df["day_of_year"]/365)
    return engineered_df

def create_fourier_features(variable_name = "tmax", city = None):
    data = load_raw_data(city)
    engineered_df_list = []
    k_list = optimize_fourier_terms(max_k=10, variable = variable_name, city=city) # recall that this returns a dictionary
    if (city is not None):
        # recall that load_raw_data will return a list regardless of input
        single_df = data[0]
        return [engineer_df(single_df, k_list[city])]
    for df, cityName in zip(data, weather_station_coords.keys()):
            engineered_df = engineer_df(df, int(k_list[cityName]))
            engineered_df_list.append(engineered_df)
    return engineered_df_list
    
def store_data(data, variable = "tmax", city = None):
    # city is NOT none, then data must be just one dataframe
    if (city is not None):
        sole_df = data[0]
        sole_df.to_csv(DATA_PROCESSED / f"{city}_{variable}_weather_data.csv")
        print(f"Stored to data/processed: {city}_{variable}_weather_data.csv")
        return
    for df, cityName in zip(data, weather_station_coords.keys()):
        df.to_csv(DATA_PROCESSED / f"{cityName}_{variable}_weather_data.csv")
        print(f"Stored to data/processed: {cityName}_{variable}_weather_data.csv")

def engineer_and_store_data(variable = "tmax", city = None):
    engineered_df_list = create_fourier_features(variable, city=city) # returns a list regardless if city is None or not
    store_data(engineered_df_list, variable, city=city)

