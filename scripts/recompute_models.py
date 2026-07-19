# store fourier models + autoregression models

# repopulate charts/diagnostics folder DONT MESS UP (also dont make it so that it opens the chart once created)

from temperature_forecaster.fourier_training import train_and_store_models
from temperature_forecaster.residual_autocorrelation import train_and_store_autocorrelations
from wipe_folder import wipe_folder
from temperature_forecaster.fourier_features import optimal_k_vals, tmin_optimal_k_vals

if (wipe_folder("models")):
    train_and_store_models("tmax")
    train_and_store_models("tmin")
    print("now we're on autocorrelations")
    train_and_store_autocorrelations("tmax")
    train_and_store_autocorrelations("tmin")
    print("Successfully run recompute_models.py")
else:
    print("Unable to run recompute_models.py: Failure to wipe folder")



