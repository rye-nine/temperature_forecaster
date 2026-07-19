from wipe_folder import wipe_folder
from temperature_forecaster.fourier_training import train_and_store_models
from temperature_forecaster.paths import ROOT

if(wipe_folder("models/fourier_models")):
    train_and_store_models("tmax")
    train_and_store_models("tmin")
else:
    print("could not wipe")
