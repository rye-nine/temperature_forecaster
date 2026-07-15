from sklearn.linear_model import LinearRegression
import pandas as pd
from temperature_forecaster.__init__ import weather_station_coords
from temperature_forecaster.paths import DATA_PROCESSED, FOURIER_MODELS
import pickle

def load_engineered_data():
    engineered_df_list = []
    for location in weather_station_coords.keys():
        df = pd.read_csv(DATA_PROCESSED / f"{location}_weather_data.csv", index_col=0, parse_dates=True)
        engineered_df_list.append(df)
    return engineered_df_list

def train(variable="tmax"): #tmax or tmin
    engineered_df_list = load_engineered_data()
    models = []
    X_list = [df[list(df.columns[df.columns.str.contains("Fsin|Fcos")])] for df in engineered_df_list]
    y_list = [df[variable] for df in engineered_df_list]
    for features, target in zip(X_list, y_list):
        reg = LinearRegression()
        reg.fit(features, target)
        models.append(reg)
    return models

def store(models, variable="tmax"):
    for model, cityName in zip(models, weather_station_coords.keys()):
        with open(FOURIER_MODELS / f"{cityName}_{variable}_fourier_model.pkl", "wb") as f:
            pickle.dump(model, f) 
    
def train_and_store_models(variable="tmax"):
    models = train(variable)
    store(models, variable)

