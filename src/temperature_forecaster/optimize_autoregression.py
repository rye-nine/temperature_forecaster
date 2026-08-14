from temperature_forecaster.fourier_training import get_residual_list
from temperature_forecaster.__init__ import weather_station_coords
from temperature_forecaster.find_optimize_fourier_terms import display_charts, find_optimal_k_values
from datetime import datetime
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import altair as alt # type: ignore
import numpy as np
import pickle
from temperature_forecaster.paths import PROJECT_ROOT



def compute_heuristics(shift=10, variable="tmax", city = None):
    current_year = datetime.now().year
    df_list = get_residual_list(variable, city=city)
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
        for i in range(2, shift + 2): # changed to 2, shift + 2
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
        chart_list.append(c1+c2)
        heuristics_list.append(df_heuristics)
    return heuristics_list, chart_list # heuristics list should only have one df if city is not None


def get_html(city_name, variable, chart_bv_html, heuristics_chart_html):
    html = f"""
            <html> 
    
            <body>
    
            <h1>{city_name}_{variable} Bias-Variance Graph</h1>
    
            {chart_bv_html}
    
            <hr>
    
            <h2>Heuristics DataFrame</h2>
    
            {heuristics_chart_html}
    
            </body>
    
            </html>
            """ 
    return html

def store_charts(charts_lst, h_charts, variable = "tmax", city = None):
    if (city is not None):
        one_bv_chart = charts_lst[0]
        one_h_chart = h_charts[0]
        with open(PROJECT_ROOT / f"charts/bias_variance/{city}_{variable}.html", "w", encoding="utf-8") as g:
            g.write(get_html(city_name=city, variable=variable, chart_bv_html=one_bv_chart.to_html(), heuristics_chart_html=one_h_chart.to_html()))
        print(f"stored BV chart: {city}")
        return
    for bv_chart, city_name, heuristics_chart in zip(charts_lst, weather_station_coords.keys(), h_charts):
        html_to_write = get_html(city_name=city_name, variable=variable, chart_bv_html=bv_chart.to_html(), heuristics_chart_html=heuristics_chart.to_html())
        with open(PROJECT_ROOT / f"charts/bias_variance/{city_name}_{variable}.html", "w", encoding="utf-8") as f:
            f.write(html_to_write)
            print(f"stored BV chart: {city_name}")

def optimize_autoregressive_terms(max_shift = 10, variable = "tmax", only_charts = False, city = None):
    heuristics_list, chart_list = compute_heuristics(max_shift, variable, city = city)
    store_charts(chart_list,heuristics_list,variable, city=city)
    if only_charts:
        return 
    optimal_shift_values = find_optimal_k_values(heuristics_list) # i know it's called k values but ignore it, it's shift values
    shift_dict = {}
    if (city is not None): # if city is specified
        shift_dict[city] = optimal_shift_values[0]
    else: # if city is None
        for i,key in enumerate(weather_station_coords.keys()):
            shift_dict[key] = optimal_shift_values[i] 
    print(f"this is our AUTOREGRESSION_{variable} dictionary: {shift_dict}")
    return shift_dict
