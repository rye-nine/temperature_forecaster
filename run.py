from temperature_forecaster.exploration import populate_charts
from temperature_forecaster.forecasting import get_empirical_probability
from temperature_forecaster.forecasting import run_forecasting

#populate_charts(variable = "tmax", open_charts=True)
#print(get_empirical_probability(196, [94, 93, 94], "Miami", 94,100, variable = "tmax"))
print(run_forecasting(2, 196, [94,93,94], 80,105, city="Miami", variable="tmax"))
