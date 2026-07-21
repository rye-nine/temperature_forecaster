from temperature_forecaster.exploration import populate_charts
from temperature_forecaster.forecasting import get_empirical_probability
from temperature_forecaster.forecasting import run_forecasting
from temperature_forecaster.fourier_features import engineer_and_store_data
from temperature_forecaster.fourier_training import train_and_store_models
from temperature_forecaster.residual_autocorrelation import train_and_store_autocorrelations
from temperature_forecaster.__init__ import weather_station_coords, get_temperatures

#engineer_and_store_data(variable = "tmin")
#train_and_store_models(variable = "tmin")
#train_and_store_autocorrelations(variable = "tmin", lag = 3)


#populate_charts(variable = "tmin", open_charts=True)
#print(get_empirical_probability(196, "Miami", 84,100, variable = "tmax"))
#print(get_probability(196, "Miami", 84, 100, variable = "tmax"))
print(run_forecasting(1, 196, 80,106, city="Miami", variable="tmax"))

#engineer_and_store_data("tmax")
#engineer_and_store_data("tmin")

#miami_coords = weather_station_coords["Miami"]
#get_temperatures(miami_coords[0], miami_coords[1], 3, "tmax")


