# DELETE THIS

from wipe_folder import wipe_folder
from temperature_forecaster.fourier_training import train_and_store_models
from temperature_forecaster.paths import ROOT
from temperature_forecaster.__init__ import weather_station_coords

for city in weather_station_coords.keys():
    print()
