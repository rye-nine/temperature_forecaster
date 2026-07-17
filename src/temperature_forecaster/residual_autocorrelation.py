from sklearn.linear_model import LinearRegression
from temperature_forecaster.__init__ import weather_station_coords
import pickle
from temperature_forecaster.fourier_training import load_engineered_data
from temperature_forecaster.paths import FOURIER_MODELS, AUTOREGRESSION_MODELS

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

def get_lagged_df(variable="tmax", lag=3):
    df_residuals_list = get_residual_list(variable)
    lagged_df_list = []
    for dataframe in df_residuals_list:
        our_df = dataframe.copy()
        for j in range(1, lag + 1):
            our_df[f"residual_lag{j}"] = our_df["residuals"].shift(j)
        our_df = our_df.dropna(axis = 0)
        lagged_df_list.append(our_df) 
    return lagged_df_list

def train_residual_models(variable="tmax", lag=3):
    lagged_df_list = get_lagged_df(variable, lag)
    residual_model_list = []
    for df in lagged_df_list:
        X = df[df.columns[df.columns.str.contains("residual_lag")]]
        y = df["residuals"]
        reg = LinearRegression()
        reg.fit(X, y)
        residual_model_list.append(reg)
    return residual_model_list

def store(models, variable, lag=3):
    for model, cityName in zip(models, weather_station_coords.keys()):
        target = AUTOREGRESSION_MODELS / f"AR({lag})_{cityName}_{variable}.pkl"
        with open(target, "wb") as f:
            pickle.dump(model, f)
        print(f"Stored to models/residual_autoregression_models: AR({lag})_{cityName}_{variable}.pkl")


def train_and_store_autocorrelations(variable="tmax", lag=3):
    residual_model_list = train_residual_models(variable, lag)
    store(residual_model_list, variable, lag)
    
