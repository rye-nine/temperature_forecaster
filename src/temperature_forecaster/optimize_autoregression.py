from temperature_forecaster.fourier_training import get_residual_list
from temperature_forecaster.__init__ import weather_station_coords
from temperature_forecaster.find_optimize_fourier_terms import display_charts, find_optimal_k_values
from datetime import datetime
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import altair as alt
import numpy as np

def compute_heuristics(shift=10, variable="tmax"):
    current_year = datetime.now().year
    df_list = get_residual_list(variable)
    chart_list = []
    heuristics_list = []
    for df_original in df_list:

        df_iterate = df_original.copy()
        # edit the df
        df_iterate = df_iterate[[variable, "residuals"]]
        # day of year column is already included courtesy of load_data
        # no need to remove N/A values because the tmax and tmin columns
        # have no missing values (if they did, they were intrapolated)

        # feature engineering to add the fourier coefficients
        for i in range(1, shift + 1):
            df_iterate[f"residual_lag{i}"] = df_iterate["residuals"].shift(i)

        df_iterate = df_iterate.dropna(axis = 0)
        
        cols = list(df_iterate.columns[df_iterate.columns.str.contains("residual_lag")])

        # divide into training and testing df, also make the heuristics df
        df_train = df_iterate[df_iterate.index < f"{current_year}-01-01"] # train 
        df_test = df_iterate[df_iterate.index >= f"{current_year}-01-01"] # test
        df_heuristics = pd.DataFrame(columns = ["shift_amount", "train_MSE", "test_MSE"])
        
        # fill out the heuristics df 
        for potential_shift in range(1, shift + 1):

            regression = LinearRegression()

            X_train = df_train[cols]
            y_train = df_train[variable]
            X_train_only_k = X_train.iloc[:, 0:potential_shift]

            regression.fit(X_train_only_k, y_train)

            X_test = df_test[cols]
            y_test = df_test[variable]
            X_test_only_k = X_test.iloc[:, 0:potential_shift]

            train_MSE = mean_squared_error(y_train, regression.predict(X_train_only_k))
            test_MSE = mean_squared_error(y_test, regression.predict(X_test_only_k))
            
            df_heuristics.loc[len(df_heuristics)] = [potential_shift, train_MSE,test_MSE]

        # make the train_MSE and test_MSE charts
        c1 = alt.Chart(df_heuristics.copy()).mark_line(color="red").encode(
        x = "shift_amount",
        y = "train_MSE"
        )
        c2 = alt.Chart(df_heuristics.copy()).mark_line(color="blue").encode(
        x = "shift_amount",
        y = "test_MSE"
        )

        # append to the chart_list
        chart_list.append((c1,c2))
        heuristics_list.append(df_heuristics)
    return heuristics_list, chart_list

def store_charts():
    return NotImplementedError

def optimize_autoregressive_terms(max_shift = 10, variable = "tmax"):
    heuristics_list, chart_list = compute_heuristics(max_shift, variable)
    #display_charts(chart_list)
    optimal_shift_values = find_optimal_k_values(heuristics_list) # i know it's called k values but ignore it, it's shift values
    shift_dict = {}
    for i,key in enumerate(weather_station_coords):
        shift_dict[key] = optimal_shift_values[i] 
    print(f"this is our AUTOREGRESSION dictionary: {shift_dict}")
    return shift_dict
