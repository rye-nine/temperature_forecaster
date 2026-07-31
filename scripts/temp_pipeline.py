from wipe_folder import wipe_folder
target_city = "Miami"
target_variable = "tmax"

wipe_folder("data")
from temperature_forecaster.load_weather_data import load_data
load_data(target_city)
from temperature_forecaster.fourier_features import engineer_and_store_data
engineer_and_store_data(target_variable, target_city)

wipe_folder("models/fourier_models")
from temperature_forecaster.fourier_training import train_and_store_models
train_and_store_models(target_variable, target_city)

wipe_folder("models/residual_autoregression_models")
from temperature_forecaster.residual_autocorrelation import train_and_store_autocorrelations
train_and_store_autocorrelations(target_variable, target_city)

from temperature_forecaster.forecasting import run_forecasting
print(dict(run_forecasting(1, 211, 80, 105, target_city, target_variable)))

wipe_folder("charts/diagnostics") 
wipe_folder("charts/bias_variance")

from temperature_forecaster.exploration import populate_charts
populate_charts(target_variable, open_charts=True, city=target_city)


