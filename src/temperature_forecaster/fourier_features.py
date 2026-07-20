import pandas as pd
import numpy as np

from temperature_forecaster.__init__ import weather_station_coords
from temperature_forecaster.paths import DATA_RAW, DATA_PROCESSED
from temperature_forecaster.find_optimize_fourier_terms import optimize_fourier_terms

optimal_k_vals =  optimize_fourier_terms(max_k=10, variable="tmax")
tmin_optimal_k_vals =  optimize_fourier_terms(max_k=10, variable="tmin")

def load_data(): # loads RAW data 
    loaded_data = []
    for location in weather_station_coords.keys():
        df = pd.read_csv(DATA_RAW/f"{location}_weather_data.csv", index_col=0, parse_dates=True)
        df["day_of_year"] = df.index.dayofyear
        loaded_data.append(df)
        print(f"Loaded from data/raw: {location}_weather_data.csv") 
    return loaded_data

def create_fourier_features(variable = "tmax"):
    data = load_data()
    engineered_df_list = []
    k_list = optimal_k_vals if (variable == "tmax") else tmin_optimal_k_vals

    for df, cityName in zip(data, weather_station_coords.keys()):
        engineered_df = df.copy()
        
        engineered_df = engineered_df[["tmax", "tmin"]]
        engineered_df["day_of_year"] = engineered_df.index.dayofyear

        optimal_k = int(k_list[cityName])
        for j in range(1, optimal_k+1):
            engineered_df[f"Fsin{j}"] = np.sin(j*2*np.pi*engineered_df["day_of_year"]/365)
            engineered_df[f"Fcos{j}"] = np.cos(j*2*np.pi*engineered_df["day_of_year"]/365)
        engineered_df_list.append(engineered_df)
    return engineered_df_list
    
def store_data(data, variable = "tmax"):
    for df, cityName in zip(data, weather_station_coords.keys()):
        df.to_csv(DATA_PROCESSED / f"{cityName}_{variable}_weather_data.csv")
        print(f"Stored to data/processed: {cityName}_{variable}_weather_data.csv")

def engineer_and_store_data(variable = "tmax"):
    engineered_df_list = create_fourier_features(variable)
    store_data(engineered_df_list, variable)
