from temperature_forecaster.fourier_features import create_fourier_features, store_data
from sklearn.linear_model import LinearRegression
from temperature_forecaster.optimize_autoregression import optimize_autoregressive_terms
from temperature_forecaster.probability_model import day_transform
from temperature_forecaster.load_weather_data import load_data
import pandas as pd
from sklearn.metrics import mean_squared_error

def get_calibration_df(start_date, city, variable = "tmax"):
    # start_date is a string "year-month-day"
    load_data(city=city)
    city_df = create_fourier_features(variable_name=variable, city=city)[0] # recall this is just a singleton set

    all_possible_days_df = city_df[city_df.index >= start_date]
    growing_training_set = city_df[city_df.index < start_date]
    
    store_data([city_df], variable=variable, city=city)
    optimal_residual_lag = optimize_autoregressive_terms(10, variable=variable, city=city)[city]
    
    j = 0
    stored_df = None
    final_df_rows = []
    for index, row in all_possible_days_df.iterrows():
        iterate_df = growing_training_set.copy()
        X_fourier_train = iterate_df[list(iterate_df.columns[iterate_df.columns.str.contains("Fsin|Fcos")])].copy()
        y_fourier_train = iterate_df[variable]


        fourier_model = LinearRegression()
        fourier_model.fit(X_fourier_train, y_fourier_train)

        iterate_df["fourier_pred"] = fourier_model.predict(X_fourier_train)
        iterate_df["fourier_residuals"] = y_fourier_train - iterate_df["fourier_pred"]
        
        # now we lag

        for i in range(2, int(optimal_residual_lag)+2): # changed to 2, int(...) + 2
            iterate_df[f"fourier_residual_lag{i}"] = iterate_df["fourier_residuals"].shift(i)
        iterate_df = iterate_df.dropna(axis = 0)
        
        X_AR_train = iterate_df[iterate_df.columns[iterate_df.columns.str.contains("fourier_residual_lag")]]
        y_AR_train = iterate_df["fourier_residuals"]

        # now train AR model

        AR_model = LinearRegression()
        AR_model.fit(X_AR_train, y_AR_train)
        
        forecast_date = growing_training_set.index.max()
        target_date = index
    
        # now transform the day
        num_fourier_terms = sum(list(iterate_df.columns.str.contains("Fsin")))
        desired_day = target_date.dayofyear
        transformed_day = day_transform(desired_day, num_fourier_terms)

        # now transform the residuals
        prev_temps = list(growing_training_set.tail(int(optimal_residual_lag))[variable])
        residual_list = []
        for i in range(2, len(prev_temps) + 2):
            date_to_use = desired_day - i
            temp_transformed_day = day_transform(date_to_use, num_fourier_terms)
            residual = prev_temps[len(prev_temps)-(i-1)] - fourier_model.predict([temp_transformed_day])[0] # this prev_temps is reverse, so we need to use prev_temps[len(prev_temps) - i]
            residual_list.append(residual)

        final_prediction = fourier_model.predict([transformed_day])[0] + AR_model.predict([residual_list])[0]
        
        final_df_rows.append({
            "forecast_date": forecast_date,
            "report_date": target_date,
            "forecast": final_prediction,
            "actual": row[variable]
            })

        growing_training_set.loc[index] = row
        
    return_df = pd.DataFrame(final_df_rows)
    return_df["residuals"] = return_df["actual"] - return_df["forecast"]

    return return_df
        
def run_calculations(df):
    mse = mean_squared_error(y_true = df["actual"], y_pred = df["forecast"])
    return mse

def backtest(start_date, city, variable = "tmax"):
    calculations = run_calculations(get_calibration_df(start_date=start_date, city=city, variable=variable))
    print(f"MSE: {calculations}")
    return calculations     


