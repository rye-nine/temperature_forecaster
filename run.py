from temperature_forecaster.exploration import populate_charts
from temperature_forecaster.forecasting import get_empirical_probability
from temperature_forecaster.forecasting import run_forecasting
from temperature_forecaster.fourier_features import engineer_and_store_data
from temperature_forecaster.fourier_training import train_and_store_models
from temperature_forecaster.residual_autocorrelation import train_and_store_autocorrelations
from temperature_forecaster.backtesting import get_calibration_df
from temperature_forecaster.evaluation import calibrate_one_interval
from temperature_forecaster.load_weather_data import load_data
from temperature_forecaster.__init__ import get_day
#from temperature_forecaster.find_optimize_fourier_terms import alternate_compute_heuristics, alternate_find_optimal_k_values
from temperature_forecaster.optimize_autoregression import alternate_compute_heuristics, alternate_find_optimal_shift_values
from temperature_forecaster.probability_model import get_prev_temps, modified_get_prev_temps

print(modified_get_prev_temps(238, "tmax", "LA"))
#df_list = alternate_compute_heuristics(city_name = "Miami")
#print(df_list)
#from temperature_forecaster.__init__ import weather_station_coords, get_temperatures
#from temperature_forecaster.evaluation import vectorized_forecasting, calibrate_one_interval
#IMPORT CALIBRATE_ONE_INTERVAL

#engineer_and_store_data(variable = "tmin")
#train_and_store_models(variable = "tmin")
#train_and_store_autocorrelations(variable = "tmin", lag = 3)


#populate_charts(variable = "tmin", open_charts=True)
#print(get_empirical_probability(196, "Miami", 84,100, variable = "tmax"))
#print(get_probability(196, "Miami", 84, 100, variable = "tmax"))

#print(run_forecasting(1, 196, 80,106, city="Miami", variable="tmax"))

#engineer_and_store_data("tmax")
#engineer_and_store_data("tmin")

#miami_coords = weather_station_coords["Miami"]
#get_temperatures(miami_coords[0], miami_coords[1], 3, "tmax")

#load_data(city="Miami")
#print(calibrate_one_interval(1, 90,92, "Miami", "tmax"))
#df = get_calibration_df(
    #start_date="2024-01-01",
    #city="LA",
    #variable="tmax",
#)
#print(df)

#print(get_day("KXHIGHLAX-26AUG04-T75"))
#df_list = alternate_compute_heuristics(city_name = "Miami")
#print(alternate_find_optimal_k_values(df_list))
