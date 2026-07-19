from temperature_forecaster.load_weather_data import load_data
from temperature_forecaster.fourier_features import engineer_and_store_data
from wipe_folder import wipe_folder
from temperature_forecaster.paths import PROJECT_ROOT

# remove everything from data/raw and data/processed

if (wipe_folder("data")):
    load_data()
    engineer_and_store_data("tmax")
    engineer_and_store_data("tmin")
else:
    print("Could not run regenerate_data.py: Unable to wipe past data, make sure directory name is correct")
