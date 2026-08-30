#from temperature_forecaster.fourier_training import get_residual_list
from temperature_forecaster.__init__ import weather_station_coords
from temperature_forecaster.find_optimize_fourier_terms import display_charts, optimize_fourier_terms
from temperature_forecaster.fourier_features import load_raw_data, engineer_df
from datetime import datetime
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import altair as alt # type: ignore
import numpy as np
import pickle
from temperature_forecaster.paths import PROJECT_ROOT

# this is like an overriden version
def get_residual_list(variable="tmax", city=None, cutoff_year = None, var_df = None): # returns a list of 
    if (var_df is not None) and (cutoff_year > var_df.index.year.max() or cutoff_year < var_df.index.year.min()):
        raise IndexError(f"cutoff year {cutoff_year} is not within the years of given var_df")

    og_df = load_raw_data(city=city) if (var_df is None) else [var_df]
    og_df = og_df[0]

    raw_df = og_df[og_df.index.year <= cutoff_year]

    k_dict = optimize_fourier_terms(max_k = 10, variable=variable, city=city, var_df = raw_df, alternate=True) # var_df is either raw_df or the parameter var_df in function
    k_val = k_dict[city]
    
    df_to_feed = raw_df.copy()
    engineered_df = engineer_df(df_to_feed, k_val) # copied btw
    
    fourier_cols = list(engineered_df.columns[engineered_df.columns.str.contains("Fsin|Fcos")]) 
    features = engineered_df[fourier_cols]
    features = features[features.index.year < cutoff_year] # just removing cutoff_year
    target = engineered_df[variable]
    target = target[target.index.year < cutoff_year]
    
    # we now fit fourier model on the years before (not including) cutoff_year
    reg = LinearRegression()
    reg.fit(features, target)
    
    # now use the fourier model on all years in engineered_df
    engineered_df["predictions"] = reg.predict(engineered_df[fourier_cols])
    engineered_df["residuals"] = engineered_df[variable] - engineered_df["predictions"]
    
    return engineered_df

def alternate_compute_heuristics(shift=10, variable="tmax", city = None, var_df = None):
    # ok always assume that city is specified
    years_list = [2021,2022,2023,2024,2025,2026] if (var_df is None) else [m for m in range(2021, var_df.index.year.max()+1)]

    heuristics_list = []

    heuristics_df = pd.DataFrame({
        "shift": [i for i in range(1, shift + 1)],
        **{str(year): None for year in years_list}
    })
    heuristics_df = heuristics_df.set_index("shift", drop = False)
    
    
    for target_year in years_list:
        df_iterate = get_residual_list(variable=variable, city=city, cutoff_year = target_year, var_df = var_df)

        for i in range(1, shift + 1):
            df_iterate[f"residual_lag{i}"] = df_iterate["residuals"].shift(i)
    
        df_iterate = df_iterate.dropna(axis = 0)

        train_df = df_iterate[df_iterate.index.year < target_year] # get the AR terms from here
        test_df = df_iterate[df_iterate.index.year == target_year] 

        train_df_AR_terms = train_df[list(train_df.columns[train_df.columns.str.contains("residual_lag")])]
        test_df_AR_terms = test_df[list(test_df.columns[test_df.columns.str.contains("residual_lag")])]

        for i in range(1, shift+1):
            X_train = train_df_AR_terms.iloc[:, 0:i] # this is the one that changes
            y_train = train_df["residuals"]
            reg = LinearRegression()
            reg.fit(X_train, y_train)
            
            X_test = test_df_AR_terms.iloc[:, 0:i]
            y_test = test_df["residuals"]

            y_pred = reg.predict(X_test)

            mse = mean_squared_error(y_test, y_pred)

            # put into heuristics_df
            heuristics_df.loc[i,str(target_year)] = mse
    heuristics_list.append(heuristics_df)
    print("DONE!")
    return heuristics_list

def alternate_find_optimal_shift_values(heuristics_list): # gets input from alternate_compute_heuristics
    k_values = []
    for heuristics_df in heuristics_list:
        df = heuristics_df.copy()
        df = df.drop(columns = ["shift"])
        df["average_MSE"] = df.mean(axis = 1)
        k_values.append(df["average_MSE"].idxmin())
    return k_values

def compute_heuristics(shift=10, variable="tmax", city = None, var_df = None):
    current_year = datetime.now().year
    df_list = get_residual_list(variable, city=city) if (var_df is None) else [var_df]
    chart_list = []
    heuristics_list = []

    if (var_df is not None):
        latest_year_in_var_df = var_df.index.year.max()
        var_df_test = var_df[var_df.index.year == latest_year_in_var_df]

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
        df_train = df_iterate[df_iterate.index < f"{current_year}-01-01"] if (var_df is None) else df_iterate[df_iterate.index.year < latest_year_in_var_df] # train 
        df_test = df_iterate[df_iterate.index >= f"{current_year}-01-01"] if (var_df is None) else var_df_test # test

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

def optimize_autoregressive_terms(max_shift = 10, variable = "tmax", only_charts = False, city = None, var_df = None, alternate = True):
    if not alternate:
        heuristics_list, chart_list = compute_heuristics(max_shift, variable, city = city, var_df=var_df)
        store_charts(chart_list,heuristics_list,variable, city=city)
    else: # alternate is False
        print("alternate is TRUE, we now use the new method of computing heuristics")
        heuristics_list = alternate_compute_heuristics(max_shift, variable, city, var_df)

    if only_charts:
        return 
    optimal_shift_values = alternate_find_optimal_shift_values(heuristics_list)
    shift_dict = {}
    if (city is not None): # if city is specified
        shift_dict[city] = optimal_shift_values[0]
    else: # if city is None
        for i,key in enumerate(weather_station_coords.keys()):
            shift_dict[key] = optimal_shift_values[i] 
    print_statement = f"Here is our AUTOREGRESSION {variable} dictionary: {shift_dict}" if (var_df is None) else f"Here is our VAR_DF AUTOREGRESSION {variable} dictionary: {shift_dict}" 
    print(print_statement)
    return shift_dict
