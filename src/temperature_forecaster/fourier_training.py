from sklearn.linear_model import LinearRegression
import pandas as pd
from temperature_forecaster.__init__ import weather_station_coords
from temperature_forecaster.paths import DATA_PROCESSED, FOURIER_MODELS
import pickle

def load_engineered_data(variable = "tmax"):
    engineered_df_list = []
    for location in weather_station_coords.keys():
        df = pd.read_csv(DATA_PROCESSED / f"{location}_{variable}_weather_data.csv", index_col=0, parse_dates=True)
        engineered_df_list.append(df)
        print(f"Loaded from data/processed: {location}_{variable}_weather_data.csv")
    return engineered_df_list

def train(variable="tmax"): #tmax or tmin
    engineered_df_list = load_engineered_data(variable)
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
        print(f"Stored to models/fourier_models: {cityName}_{variable}_fourier_model.pkl")
    
def train_and_store_models(variable="tmax"):
    models = train(variable)
    store(models, variable)

# below is stuff for other scripts

def load_models(variable="tmax"):
    model_list = []
    for location in weather_station_coords.keys():
        with open(FOURIER_MODELS / f"{location}_{variable}_fourier_model.pkl", "rb") as f:
            model = pickle.load(f)
            model_list.append(model)
        print(f"Loaded from models/fourier_models: {location}_{variable}_fourier_model.pkl")
    return model_list

def get_residual_list(variable="tmax"):
    engineered_df_list = load_engineered_data(variable)
    model_list = load_models(variable)
    df_residuals_list = []
    for df, model in zip(engineered_df_list, model_list):

        df_residual = df.copy()
        cols = df_residual.columns[df_residual.columns.str.contains("Fsin|Fcos")]
        X = df_residual[cols]
        print(X)
        
        df_residual["predictions"] = model.predict(X)
        df_residual["residuals"] =  df_residual[variable] - df_residual["predictions"] 
        df_residuals_list.append(df_residual)
    return df_residuals_list

