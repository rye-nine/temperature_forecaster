from temperature_forecaster.fourier_training import train_and_store_models
from wipe_folder import wipe_folder

if (wipe_folder("models/fourier_models")):
    train_and_store_models("tmax")
    train_and_store_models("tmin")
    print("Successfully run recompute_fourier_models.py")
else:
    print("Unable to run recompute_fourier_models.py: Failure to wipe folder")
