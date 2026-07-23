# general idea: for each day, and for each city, we will ensure that
# the probabilities of each temperature range are close enough to
# the actual observed frequencies of those temperature ranges
# spanning the historical data.

from temperature_forecaster.forecasting import run_forecasting
from temperature_forecaster.fourier_features import load_data
from temperature_forecaster.__init__ import weather_station_coords
from temperature_forecaster.residual_autocorrelation import optimal_ar_terms, tmin_optimal_ar_terms
from temperature_forecaster.fourier_training import load_models
from temperature_forecaster.probability_model import load_residual_models
# get_final_residuals is a function that returns a list of 
# dataframes that has all of the features and the final predictions
# and residuals

import pandas as pd 
import numpy as np

def form_data(variable = "tmax"):
    raw_data = load_data()
    return_list = []
    opt_shift_values = optimal_ar_terms if (variable == "tmax") else tmin_optimal_ar_terms
    for df, city in zip(raw_data, weather_station_coords.keys()):
        df_copy = df.copy()
        df_copy = df_copy[[variable, "day_of_year"]]
        lag = int(opt_shift_values[city])
        for j in range(1, lag+1):
            df_copy[f"lag_{j}"] = df_copy["tmax"].shift(j)
        df_copy = df_copy.dropna(axis = 0)
        return_list.append(df_copy)
    return return_list

def vectorized_day_transform(days):
    raise NotImplementedError

def vectorized_forecasting(mode, day_col, MIN, MAX, city, variable = "tmax"):
    # the general idea is that given a pd.Series object that contains the days of the year, our goal is to...
    # predict the temperature and generate a probability distribution in accordance to @mode
    # then we just find the probability between MIN and MAX (recall that MAX = MIN + 2)
    # this will typically use mode = 1 (use normal distribution) 
    all_city_fourier_models = load_models(variable)
    all_city_AR_models = load_residual_models(variable)
    
    city_index = list(weather_station_coords.keys()).index(city)
    city_fm = all_city_fourier_models[city_index]
    city_am = all_city_AR_models[city_index]
    raise NotImplementedError

#mode = 1 --> normal distribution
#mode = 2 --> empirical residual distribution
def calibrate_one_interval(mode, MIN, MAX, city_name, variable_name = "tmax"):
    # MAX = MIN + 2
    city_index = list(weather_station_coords.keys()).index(city_name)
    df_list = form_data()
    
    df = df_list[city_index]
    df[f"[{MIN}, {MAX}]-probability"] = run_forecasting(mode, df["day_of_year"], minimum=MIN, maximum=MAX,city=city_name, variable=variable_name)
    # df.apply(
    #     lambda row: run_forecasting(
    #         mode,
    #         row["day_of_year"], 
    #         minimum=MIN, 
    #         maximum=MAX, 
    #         city=city_name, 
    #         variable=variable_name
    #     )[0][1],
    #     axis = 1
    #)

    df["outcome"] = (
    (min <= df[variable_name]) &
    (df[variable_name] <= max)
    )
    df["outcome"] = df["outcome"].astype(int)
    return df 
    # for each day in the df (offset by lag cuz autoregression),
    # we get the mean and standard deviation for min/max temp
    # then we compute the area between min, max using normal distribution
    # we then see if the actual min/max temp is in that range 

# helper function
def form_groups(mode, minimum, maximum, city_name, variable_name = "tmax"):
    # maximum = minimum + 2
    df = calibrate_one_interval(mode, minimum, maximum, city_name, variable_name )
    return_dict = {}
    probs_col_name = f"[{minimum}, {maximum}]-probability"
    for i in np.arange(0,1,0.1):
        bucket_min = i
        bucket_max = i + 0.1

        bool1 = df[probs_col_name] >= bucket_min
        bool2 = df[probs_col_name] < bucket_max
        temp_df = df[bool1 & bool2]

        num_rows = len(temp_df)
        observed_positive = temp_df["outcome"].sum()
        return_dict[(i, i + 0.1)] = observed_positive / num_rows
    return return_dict

# def calibrate(min_temp, max_temp, city_name, lag= 3, variable_name="tmax"):
