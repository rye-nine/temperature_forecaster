# store fourier models + autoregression models

# repopulate charts folder DONT MESS UP (also dont make it so that it opens the chart once created)

from temperature_forecaster\fourier_training import train_and_store_models
from temperature_forecaster\fourier_training import train_and_store_autocorrelations

train_and_store_models("tmax")
train_and_store_models("tmin")

train_and_store_autocorrelations("tmax")
train_and_store_autocorrelations("tmin")



