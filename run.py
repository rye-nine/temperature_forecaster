from temperature_forecaster.exploration import populate_charts
from temperature_forecaster.forecasting import get_empirical_probability

#populate_charts(variable = "tmax", open_charts=True)
print(get_empirical_probability(196, [94, 93, 94], "Miami", 94,95, variable = "tmax"))
