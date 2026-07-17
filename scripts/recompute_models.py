from temperature_forecaster.fourier_training import train_and_store_models
from temperature_forecaster.residual_autocorrelation import train_and_store_autocorrelations

train_and_store_models("tmax")
train_and_store_models("tmin")

train_and_store_autocorrelations("tmax", 3)
train_and_store_autocorrelations("tmin", 3)
