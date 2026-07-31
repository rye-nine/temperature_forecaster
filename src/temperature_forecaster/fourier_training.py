from sklearn.linear_model import LinearRegression
import pandas as pd
from temperature_forecaster.__init__ import weather_station_coords
from temperature_forecaster.paths import DATA_PROCESSED, FOURIER_MODELS
import pickle

def load_engineered_data(variable = "tmax", city = None):
    engineered_df_list = []
    for location in weather_station_coords.keys():
        if ((city is None) or (city == location)):
            df = pd.read_csv(DATA_PROCESSED / f"{location}_{variable}_weather_data.csv", index_col=0, parse_dates=True)
            engineered_df_list.append(df)
            print(f"Loaded from data/processed: {location}_{variable}_weather_data.csv")
    return engineered_df_list

def train(variable="tmax", city = None): #tmax or tmin
    engineered_df_list = load_engineered_data(variable, city=city) # if city is not None, this is still a list
    models = []
    X_list = [df[list(df.columns[df.columns.str.contains("Fsin|Fcos")])] for df in engineered_df_list]
    y_list = [df[variable] for df in engineered_df_list]
    for features, target in zip(X_list, y_list):
        reg = LinearRegression()
        reg.fit(features, target)
        models.append(reg)
    return models # still also returns a list regardless of whether or not city is None

def store(models, variable="tmax", city = None):
    if (city is not None):
        singular_model = models[0]
        with open(FOURIER_MODELS / f"{city}_{variable}_fourier_model.pkl", "wb") as g:
            pickle.dump(singular_model, g)
        print(f"Stored to models/fourier_models: {city}_{variable}_fourier_model.pkl")
        return
    for model, cityName in zip(models, weather_station_coords.keys()):
        with open(FOURIER_MODELS / f"{cityName}_{variable}_fourier_model.pkl", "wb") as f:
            pickle.dump(model, f)
        print(f"Stored to models/fourier_models: {cityName}_{variable}_fourier_model.pkl")
    
def train_and_store_models(variable="tmax", city = None):
    models = train(variable, city=city)
    store(models, variable, city=city)

# below is stuff for other scripts

def load_models(variable="tmax", city = None):
    model_list = []
    for location in weather_station_coords.keys():
        if ((city is None) or (city == location)):
            with open(FOURIER_MODELS / f"{location}_{variable}_fourier_model.pkl", "rb") as f:
                model = pickle.load(f)
                model_list.append(model)
            print(f"Loaded from models/fourier_models: {location}_{variable}_fourier_model.pkl")
    return model_list # still returns a list regardless

def get_residual_list(variable="tmax", city = None):
    engineered_df_list = load_engineered_data(variable, city=city) # this should just be a list consisting of one df if city is not None
    model_list = load_models(variable, city=city) # this should just be a list consisting of one model if city is not None
    df_residuals_list = []
    for df, model in zip(engineered_df_list, model_list): # if city is not None, then this should only iterate ONCE

        df_residual = df.copy()
        cols = df_residual.columns[df_residual.columns.str.contains("Fsin|Fcos")]
        X = df_residual[cols]
        
        df_residual["predictions"] = model.predict(X)
        df_residual["residuals"] =  df_residual[variable] - df_residual["predictions"] 
        df_residuals_list.append(df_residual)
    return df_residuals_list


