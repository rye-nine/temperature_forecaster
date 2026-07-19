from sklearn.linear_model import LinearRegression
from temperature_forecaster.__init__ import weather_station_coords
import pickle
from temperature_forecaster.paths import FOURIER_MODELS, AUTOREGRESSION_MODELS
from temperature_forecaster.optimize_autoregression import optimize_autoregressive_terms
from temperature_forecaster.fourier_training import get_residual_list

optimal_ar_terms = optimize_autoregressive_terms(10, "tmax")
tmin_optimal_ar_terms = optimize_autoregressive_terms(10, "tmin")


# HELPER FUNCTION, IGNORE IN PIPELINE
def lag_df(df, lag):
    for j in range(1, int(lag) + 1):
        df[f"residual_lag{j}"] = df["residuals"].shift(j)
    df = df.dropna(axis = 0)
    return df

def get_lagged_df(variable="tmax"):
    df_residuals_list = get_residual_list(variable)
    lagged_df_list = []
    shift_vals = optimal_ar_terms if (variable == "tmax") else tmin_optimal_ar_terms 
    for dataframe, opt_shift_val in zip(df_residuals_list, shift_vals.values()):
        our_df = dataframe.copy()
        append_df = lag_df(our_df, opt_shift_val)
        lagged_df_list.append(append_df) 
    return lagged_df_list

def train_residual_models(variable="tmax"):
    lagged_df_list = get_lagged_df(variable)
    residual_model_list = []
    for df in lagged_df_list:
        X = df[df.columns[df.columns.str.contains("residual_lag")]]
        y = df["residuals"]
        reg = LinearRegression()
        reg.fit(X, y)
        residual_model_list.append(reg)
    return residual_model_list

def store(models, variable):
    opt_vals = optimal_ar_terms if (variable == "tmax") else tmin_optimal_ar_terms
    for model, cityName, lag in zip(models, weather_station_coords.keys(), opt_vals.values()):
        target = AUTOREGRESSION_MODELS / f"AR({lag})_{cityName}_{variable}.pkl"
        with open(target, "wb") as f:
            pickle.dump(model, f)
        print(f"Stored to models/residual_autoregression_models: AR({lag})_{cityName}_{variable}.pkl")


def train_and_store_autocorrelations(variable="tmax"):
    residual_model_list = train_residual_models(variable)
    store(residual_model_list, variable)
    
