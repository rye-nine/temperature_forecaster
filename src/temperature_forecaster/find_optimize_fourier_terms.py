from temperature_forecaster.__init__ import weather_station_coords
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import altair as alt # type: ignore
import pandas as pd
import numpy as np
from temperature_forecaster.paths import DATA_RAW

# after completing, make it so that load_weather_data takes
# optimal k-values from THIS script

# ok i think you just need processed_data?

def temp_load_data(city = None):
    loaded_data = []
    for location in weather_station_coords.keys():
        if ((city is None) or (city == location)):
            df = pd.read_csv(DATA_RAW / f"{location}_weather_data.csv", index_col=0, parse_dates=True)
            df["day_of_year"] = df.index.dayofyear
            loaded_data.append(df)
    return loaded_data

def alternate_compute_heuristics(max_k=10, variable="tmax", city_name = None, var_df = None):
    years_list = [2021,2022,2023,2024,2025,2026] if (var_df is None) else [m for m in range(2021, var_df.index.year.max() + 1)]
    df_list = temp_load_data(city=city_name) if (var_df is None) else [var_df] # if var_df is not None
    df = df_list[0]

    heuristics_list = []

    heuristics_df = pd.DataFrame({
        "k": [i for i in range(1, max_k + 1)],
        **{str(year): None for year in years_list}
    })
    heuristics_df = heuristics_df.set_index("k", drop = False)

    df_iterate = df.copy()
    # just populate all sine/cosine features up to max_k + 1
    for j in range(1, max_k+1):
        df_iterate[f"Fsin{j}"] = np.sin(j*2*np.pi*df_iterate["day_of_year"]/365)
        df_iterate[f"Fcos{j}"] = np.cos(j*2*np.pi*df_iterate["day_of_year"]/365)

    for target_year in years_list:
        train_df = df_iterate[df_iterate.index.year < target_year] # get the fourier terms from here
        test_df = df_iterate[df_iterate.index.year == target_year] 

        train_df_fourier_terms = train_df[list(train_df.columns[train_df.columns.str.contains("Fsin|Fcos")])]
        test_df_fourier_terms = test_df[list(test_df.columns[test_df.columns.str.contains("Fsin|Fcos")])]

        for i in range(1, max_k+1):
            X_train = train_df_fourier_terms.iloc[:, 0:(2*i)] # this is the one that changes
            y_train = train_df[variable]
            reg = LinearRegression()
            reg.fit(X_train, y_train)
            
            X_test = test_df_fourier_terms.iloc[:, 0:(2*i)]
            y_test = test_df[variable]

            y_pred = reg.predict(X_test)

            mse = mean_squared_error(y_test, y_pred)

            # put into heuristics_df
            print(f"mse: {mse}")
            heuristics_df.loc[i,str(target_year)] = mse
    heuristics_list.append(heuristics_df)
    return heuristics_list

def compute_heuristics(max_k=10, variable="tmax", city_name = None, var_df = None): # added var_df parameter to override df to be optimized
    current_year = datetime.now().year
    df_list = temp_load_data(city=city_name) if (var_df is None) else [var_df] # if var_df is not None
    chart_list = []
    heuristics_list = []

    if (var_df is not None):
        latest_year_in_var_df = var_df.index.year.max()
        var_df_test = var_df[var_df.index.year == latest_year_in_var_df]

    for df_original in df_list:

        df_iterate = df_original.copy()
        # edit the df
        df_iterate = df_iterate[[variable, "day_of_year"]]
        # day of year column is already included courtesy of load_data
        # no need to remove N/A values because the tmax and tmin columns
        # have no missing values (if they did, they were intrapolated)

        # feature engineering to add the fourier coefficients
        for i in range(1, max_k + 1):
            df_iterate[f"Fsin{i}"] = np.sin(i*2*np.pi*df_iterate["day_of_year"]/365)
            df_iterate[f"Fcos{i}"] = np.cos(i*2*np.pi*df_iterate["day_of_year"]/365)
        
        cols = list(df_iterate.columns[df_iterate.columns.str.contains("Fsin|Fcos")])

        # divide into training and testing df, also make the heuristics df
        df_train = df_iterate[df_iterate.index < f"{current_year}-01-01"] if (var_df is None) else df_iterate[df_iterate.index.year < latest_year_in_var_df] # train 
        df_test = df_iterate[df_iterate.index >= f"{current_year}-01-01"] if (var_df is None) else var_df_test # test
        df_heuristics = pd.DataFrame(columns = ["k-value", "train_MSE", "test_MSE"])
        
        # fill out the heuristics df 
        for potential_k in range(1, max_k + 1):

            regression = LinearRegression()

            X_train = df_train[cols]
            y_train = df_train[variable]
            X_train_only_k = X_train.iloc[:, 0:(2*potential_k)]

            regression.fit(X_train_only_k, y_train)

            X_test = df_test[cols]
            y_test = df_test[variable]
            X_test_only_k = X_test.iloc[:, 0:(2*potential_k)]

            train_MSE = mean_squared_error(y_train, regression.predict(X_train_only_k))
            test_MSE = mean_squared_error(y_test, regression.predict(X_test_only_k))
            
            df_heuristics.loc[len(df_heuristics)] = [potential_k, train_MSE,test_MSE]

        # make the train_MSE and test_MSE charts
        c1 = alt.Chart(df_heuristics.copy()).mark_line(color="red").encode(
        x = "k-value",
        y = "train_MSE"
        )
        c2 = alt.Chart(df_heuristics.copy()).mark_line(color="blue").encode(
        x = "k-value",
        y = "test_MSE"
        )

        # append to the chart_list
        chart_list.append((c1,c2))
        heuristics_list.append(df_heuristics)
    return heuristics_list, chart_list

def display_charts(list_of_charts):
    for left, right in list_of_charts:
        left = left.properties(title="Train MSE vs k-value", width=400, height=300)
        right = right.properties(title="Test MSE vs k-value", width=400, height=300)
        combined = left + right
        combined.display()

def alternate_find_optimal_k_values(heuristics_list): # gets input from alternate_compute_heuristics
    k_values = []
    for heuristics_df in heuristics_list:
        print(heuristics_df)
        df = heuristics_df.copy()
        df = df.drop(columns = ["k"])
        df["average_MSE"] = df.mean(axis = 1)
        k_values.append(df["average_MSE"].idxmin())
    return k_values

def find_optimal_k_values(heuristics_list):
    k_values = []
    for heuristics_df in heuristics_list:
        min_val = heuristics_df["test_MSE"].min()
        k_values.append(heuristics_df[heuristics_df["test_MSE"] == min_val].iloc[0,0])
    return k_values

def optimize_fourier_terms(max_k=10, variable="tmax", city = None, var_df = None, alternate = True):
    if alternate:
        heuristics_list = alternate_compute_heuristics(max_k=10, variable=variable, city_name=city, var_df=var_df) 
        print(f"AS EXPECTED: {heuristics_list}")
    else:
        print("DID REGULAR FOURIER OPTIMIZATION BRUH")
        heuristics_list, chart_list = compute_heuristics(max_k, variable, city, var_df=var_df)
    #display_charts(chart_list)
    print(f"heuristics_list that is fed into alternate_find_optimal_k_values: {heuristics_list}")
    optimal_k_values = alternate_find_optimal_k_values(heuristics_list)
    k_dict = {}
    for i,key in enumerate(weather_station_coords.keys()):
        if ((city is None) or (city == key)):
            k_dict[key] = optimal_k_values[i] if (city is None) else optimal_k_values[0]
    print_statement = f"Here is our FOURIER {variable} dictionary: {k_dict}" if (var_df is None) else f"Here is our VAR_DF FOURIER {variable} dictionary: {k_dict}" 
    print(print_statement)
    return k_dict

