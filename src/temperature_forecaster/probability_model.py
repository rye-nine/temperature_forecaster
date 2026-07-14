from src.temperature_forecaster.residual_autocorrelation import load_models, get_lagged_df
from src.temperature_forecaster.__init__ import weather_station_coords
from src.temperature_forecaster.load_weather_data import optimal_k_vals
import pickle
import numpy as np

def load_residual_models(variable="tmax", lag=3):
    model_list = []
    for location in weather_station_coords.keys():
        with open(f"../models/residual_autoregression_models/AR({lag})_{location}_{variable}.pkl", "rb") as f:
            model_list.append(pickle.load(f))
    return model_list

def day_transform(day,k):
    max_k = int(k) + 1
    list1 = [np.sin(i*2*np.pi*day/365) for i in range(1,max_k)]
    list2 = [np.cos(j*2*np.pi*day/365) for j in range(1,max_k)]
    return [item for pair in zip(list1,list2) for item in pair]


## will need to edit this function in the future; need to make it more automated
def residual_transform(prev_temps,day, city, variable="tmax"):
    index = list(weather_station_coords.keys()).index(city)
    fourier_model = load_models(variable)[index]
    k_val = optimal_k_vals[city]
    residual_list = []
    for i in range(1, len(prev_temps) + 1):
        target_day = day - i
        transformed_day = day_transform(target_day, k_val)
        residual = prev_temps[i-1] - fourier_model.predict([transformed_day])[0]
        residual_list.append(residual)
    return residual_list

def extrema_approximation_all(prev_temps, day, variable="tmax"):
    lag = len(prev_temps)
    approximation_list = []
    fourier_model = load_models(variable)
    residual_model = load_residual_models(variable, lag)

    for city in weather_station_coords.keys():

        index = list(weather_station_coords.keys()).index(city)
        city_fourier_model = fourier_model[index]
        city_residual_model = residual_model[index]

        transformed_day = day_transform(day, optimal_k_vals[city])
        approximation = city_fourier_model.predict([transformed_day])[0] +city_residual_model.predict([residual_transform(prev_temps, day, city, variable)])[0]
        approximation_list.append(approximation)

    return approximation_list

def get_final_residuals(variable="tmax"):
    initial_residuals = get_lagged_df(variable)

    residual_models = load_residual_models(variable)
    fourier_models = load_models(variable)

    final_residuals_dataframes = []
    for i,df in enumerate(initial_residuals):
        df_temp = df.copy()

        y = df_temp[variable]
        X_fourier = df_temp[list(df_temp.columns[df_temp.columns.str.contains("Fsin|Fcos")])]
        X_residual = df_temp[list(df_temp.columns[df_temp.columns.str.contains("residual_lag")])]
        
        fourier_predict = fourier_models[i].predict(X_fourier)
        residual_predict = residual_models[i].predict(X_residual)

        df_temp["final_prediction"] = fourier_predict + residual_predict
        df_temp["final_residuals"] = y - df_temp["final_prediction"]
        final_residuals_dataframes.append(df_temp)

    return final_residuals_dataframes

def get_all_std(day, day_range=15, variable="tmax"):
    df_with_residuals_list = get_final_residuals(variable)
    city_names = list(weather_station_coords.keys())
    standard_deviation_list = []
    for df_with_residuals, city in zip(df_with_residuals_list, city_names):
        df_temp = df_with_residuals.copy()
        
        bool1 = df_temp["day_of_year"] >= day - day_range
        bool2 = df_temp["day_of_year"] <= day + day_range
        df_temp = df_temp[bool1 & bool2]
    
        standard_deviation = df_temp["final_residuals"].std()
        standard_deviation_list.append((city, standard_deviation))
    return standard_deviation_list

def normal_distribution_approximation(prev_temps, day, variable="tmax"):
    city_names = list(weather_station_coords.keys())
    mean_list = extrema_approximation_all(prev_temps, day, variable)
    std_list = get_all_std(day,15, variable)

    vals = [(mean, std) for mean, (_, std) in zip(mean_list, std_list)]
    dict = {city: val for city, val in zip(city_names, vals)}
    return dict
