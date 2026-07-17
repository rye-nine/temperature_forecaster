from temperature_forecaster.load_weather_data import load_data
from temperature_forecaster.fourier_features import engineer_and_store_data

# how to remove everything from data/raw using python
#load_data()
engineer_and_store_data("tmax")
engineer_and_store_data("tmin")

