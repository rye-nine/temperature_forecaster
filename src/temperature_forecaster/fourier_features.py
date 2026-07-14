import pandas as pd
import numpy as np

from src.temperature_forecaster.__init__ import weather_station_coords
from src.temperature_forecaster.load_weather_data import optimal_k_vals

def load_data():
    loaded_data = []
    for location in weather_station_coords.keys():
        df = pd.read_csv(f"../data/raw/{location}_weather_data.csv", index_col=0, parse_dates=True)
        df["day_of_year"] = df.index.dayofyear
        loaded_data.append(df)
    return loaded_data

def create_fourier_features():
    data = load_data()
    print(f"Number of locations: {len(data)}")
    engineered_df_list = []
    for df, cityName in zip(data, weather_station_coords.keys()):
        engineered_df = df.copy()
        
        engineered_df = engineered_df[["tmax", "tmin"]]
        engineered_df["day_of_year"] = engineered_df.index.dayofyear

        optimal_k = int(optimal_k_vals[cityName])
        for j in range(1, optimal_k+1):
            engineered_df[f"Fsin{j}"] = np.sin(j*2*np.pi*engineered_df["day_of_year"]/365)
            engineered_df[f"Fcos{j}"] = np.cos(j*2*np.pi*engineered_df["day_of_year"]/365)
        engineered_df_list.append(engineered_df)
    return engineered_df_list
    
def store_data(data):
    for df, cityName in zip(data, weather_station_coords.keys()):
        print(f"Storing engineered data for {cityName}...")
        df.to_csv(f"../data/processed/{cityName}_weather_data.csv")

def engineer_and_store_data():
    engineered_df_list = create_fourier_features()
    print(len(engineered_df_list))
    store_data(engineered_df_list)
