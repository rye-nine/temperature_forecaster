from temperature_forecaster.fourier_training import load_models
from temperature_forecaster.residual_autocorrelation import get_lagged_df
from temperature_forecaster.__init__ import weather_station_coords
#from src.fourier_features import optimal_k_vals, tmin_optimal_k_vals
from temperature_forecaster.paths import AUTOREGRESSION_MODELS
#from src.residual_autocorrelation import optimal_ar_terms, tmin_optimal_ar_terms
from temperature_forecaster.optimize_autoregression import optimize_autoregressive_terms
from temperature_forecaster.find_optimize_fourier_terms import optimize_fourier_terms
from temperature_forecaster.__init__ import get_wrh_climate_temp
import pickle
import numpy as np
import pandas as pd

def load_residual_models(variable="tmax", city = None):
    model_list = []
    ar_list = optimize_autoregressive_terms(10, variable=variable, city=city) # should just be a dictionary containing one pair
    if (city is not None):
        with open(AUTOREGRESSION_MODELS / f"AR({int(ar_list[city])})_{city}_{variable}.pkl", "rb") as g:
            model_list.append(pickle.load(g))
        print(f"Loaded from model/residual_autoregression_models: AR({int(ar_list[city])})_{city}_{variable}.pkl")
        return model_list

    # else if city is None and ar_list is full length
    for location, opt_ar in zip(weather_station_coords.keys(), ar_list.values()):
        with open(AUTOREGRESSION_MODELS / f"AR({int(opt_ar)})_{location}_{variable}.pkl", "rb") as f:
            model_list.append(pickle.load(f))
        print(f"Loaded from model/residual_autoregression_models: AR({int(opt_ar)})_{location}_{variable}.pkl")
    return model_list

def day_transform(day,k):
    if (isinstance(day, pd.Series)):
        day_list = list(day)
        return [day_transform(day_list[i],k) for i in range(len(day_list))] 
    max_k = int(k) + 1
    list1 = [np.sin(i*2*np.pi*day/365) for i in range(1,max_k)]
    list2 = [np.cos(j*2*np.pi*day/365) for j in range(1,max_k)]
    return [item for pair in zip(list1,list2) for item in pair]

## will need to edit this function in the future; need to make it more automated
def residual_transform(prev_temps,day, city, variable="tmax"):
    fourier_model = load_models(variable, city = city)[0]
    k_val = optimize_fourier_terms(10, variable=variable, city=city)[city]
    residual_list = []
    for i in range(1, len(prev_temps) + 1):
        target_day = day - i
        transformed_day = day_transform(target_day, k_val)
        residual = prev_temps[i-1] - fourier_model.predict([transformed_day])[0]
        residual_list.append(residual)
    return residual_list

# take note, this function isnt that important to implement city = None because it's low computation time
# helper function, this is the only function that uses the get_temperatures function from __init__.py
def get_prev_temps(day: int, variable = "tmax", city = None): # override should be fixed in the future, but rn it's false cuz im just using random numbers haha
    optimal_ar = optimize_autoregressive_terms(10, variable=variable, city=city)
    # the idea is that we'll return a dictionary of each city and their prev temps in accordance to optimal ar terms
    prev_temps_dict = {}
    if (city is not None): # equivalent to if optimal_ar has only one value
        prev_temps_dict[city] = get_wrh_climate_temp(city, int(optimal_ar[city]), variable,day)
        return prev_temps_dict
    for city_name, optimal_terms in zip(weather_station_coords.keys(), optimal_ar.values()):
        prev_temps_dict[city_name] = get_wrh_climate_temp(city_name, int(optimal_terms), variable, day)
    return prev_temps_dict
    
def modified_get_prev_temps(day: int, variable = "tmax", city = None): # to be used when getting the next day's max/min temp at 12am
    # so the idea is that we're going to use the approximation for the current day as a previous temp
    approximated_prev_temp = extrema_approximation_all(day-1, variable, city_name = city, one_day_ahead = False) # a singleton set btw
    raw_prev_temps = get_prev_temps(day-1, variable,city)[city]
    return_dict = {}
    return_dict[city] = approximated_prev_temp + raw_prev_temps[:-1] # union the two sets 
    return return_dict


# OK LOOK I KNOW IT HAS THE WORD "ALL" IN IT BUT PLEASE JUST BEAR WITH ME
def extrema_approximation_all(day, variable="tmax", city_name = None, one_day_ahead = True): # after implementing get_prev_temps, will no longer need a prev_temps parameter
    approximation_list = []
    fourier_model = load_models(variable, city=city_name)
    residual_model = load_residual_models(variable, city=city_name)

    #best_ar_values = optimize_autoregressive_terms(10,variable=variable, city=city)

    dict_prev_temps = get_prev_temps(day,variable, city=city_name) if (not one_day_ahead) else modified_get_prev_temps(day, variable, city_name) # if one_day_ahead == True
    print(f"one_day_ahead: {one_day_ahead}")

    our_k_vals = optimize_fourier_terms(10, variable=variable, city=city_name)

    for city in weather_station_coords.keys():
        if ((city_name is None) or (city_name == city)):
            index = list(weather_station_coords.keys()).index(city) if (city_name is None) else 0 
            city_fourier_model = fourier_model[index]
            city_residual_model = residual_model[index]
            
            transformed_day = day_transform(day, our_k_vals[city])
            approximation = city_fourier_model.predict([transformed_day])[0] + city_residual_model.predict([residual_transform(dict_prev_temps[city], day, city, variable)])[0]
            approximation_list.append(approximation)
            if (city_name == city):
                return approximation_list # so should just be a singleton set if city_name is None
    return approximation_list


def modified_extrema_approximation(day, variable = "tmax", city_name = None):
    # edit get_wrh_climate_temp to accept a parameter that takes the day of year
    return

# investigate this function
def get_final_residuals(variable="tmax", city = None):
    initial_residuals = get_lagged_df(variable, city = city)

    residual_models = load_residual_models(variable, city = city)
    fourier_models = load_models(variable, city = city)

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

    return final_residuals_dataframes # should just return a singleton list if city is None

def find_std(df, day, day_range):
    df_temp = df.copy()
            
    bool1 = df_temp["day_of_year"] >= day - day_range
    bool2 = df_temp["day_of_year"] <= day + day_range
    df_temp = df_temp[bool1 & bool2]
        
    standard_deviation = df_temp["final_residuals"].std()
    return standard_deviation

# I KNOW IT HAS THE WORD "ALL" BUT JUST BARE WITH ME PLEASE
def get_all_std(day, day_range=15, variable="tmax", city_name = None, residual_list=None):
    if (isinstance(day, pd.Series)):
        days_list = list(day)
        temp_df_with_residuals_list = get_final_residuals(variable, city=city_name)
        return [get_all_std(days_list[i], day_range, variable, city_name, temp_df_with_residuals_list) for i in range(len(days_list))]
    df_with_residuals_list = residual_list if (residual_list is not None) else get_final_residuals(variable, city=city_name)
    standard_deviation_list = []
    if (city_name is not None):
        standard_deviation_list.append((city_name, find_std(df_with_residuals_list[0], day, day_range)))
        return standard_deviation_list
    
    city_names = list(weather_station_coords.keys())
    for df_with_residuals, city in zip(df_with_residuals_list, city_names):
        standard_dev = find_std(df_with_residuals, day, day_range)
        standard_deviation_list.append((city, standard_dev))
    return standard_deviation_list

def normal_distribution_approximation(day, variable="tmax", city = None):
    mean_list = extrema_approximation_all(day, variable, city_name=city)
    std_list = get_all_std(day,15, variable, city_name=city)

    if (city is not None):
        return_dictionary = {
            city: (mean_list[0], std_list[0])
        }
        return return_dictionary

    city_names = list(weather_station_coords.keys())
    vals = [(mean, std) for mean, (_, std) in zip(mean_list, std_list)]
    my_dictionary = {city: val for city, val in zip(city_names, vals)}
    return my_dictionary

