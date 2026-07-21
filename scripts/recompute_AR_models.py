# autoregression models

# repopulate charts/diagnostics folder DONT MESS UP (also dont make it so that it opens the chart once created)

from temperature_forecaster.residual_autocorrelation import train_and_store_autocorrelations
from wipe_folder import wipe_folder

if (wipe_folder("models/residual_autoregression_models")):
    train_and_store_autocorrelations("tmax")
    train_and_store_autocorrelations("tmin")
    print("Successfully run recompute_models.py")
else:
    print("Unable to run recompute_models.py: Failure to wipe folder")



