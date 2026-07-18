from temperature_forecaster.exploration import populate_charts
from temperature_forecaster.forecasting import get_empirical_probability
from temperature_forecaster.forecasting import run_forecasting
from temperature_forecaster.fourier_features import engineer_and_store_data
from temperature_forecaster.fourier_training import train_and_store_models
from temperature_forecaster.residual_autocorrelation import train_and_store_autocorrelations

#engineer_and_store_data(variable = "tmin")
#train_and_store_models(variable = "tmin")
#train_and_store_autocorrelations(variable = "tmin", lag = 3)


#populate_charts(variable = "tmin", open_charts=True)
print(get_empirical_probability(196, [94, 93, 94], "Miami", 85,100, variable = "tmin"))
#print(run_forecasting(2, 196, [94,93,94], 80,105, city="Miami", variable="tmin"))

#engineer_and_store_data("tmax")
#engineer_and_store_data("tmin")

