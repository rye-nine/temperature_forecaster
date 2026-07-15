from temperature_forecaster.probability_model import normal_distribution_approximation
from temperature_forecaster.__init__ import weather_station_coords
from scipy.stats import norm

def get_probability(day, prev_temps, city, minimum, maximum, variable="tmax"):
    all_distributions = normal_distribution_approximation(prev_temps, day, variable)

    city_distribution = all_distributions[city]

    mu = city_distribution[0]
    #print("mu: ", mu)
    sigma = city_distribution[1]
    #print("sigma: ", sigma)

    probabilities = []
    for j in range(minimum, maximum):
        prob = norm.cdf(j + 1, loc=mu, scale=sigma) - norm.cdf(j, loc=mu, scale=sigma)
        probabilities.append(((j, j+1),prob))
    return probabilities

def run_forecasting(day, prev_temps, minimum, maximum, city=None, variable="tmax"):
    city_names = list(weather_station_coords.keys())
    if (city is None): # no city is specified
        dict = {}
        for city in city_names:
            dict[city] = get_probability(day, prev_temps, city, minimum, maximum, variable)
        #print(dict)
        return dict
    probabs = get_probability(day, prev_temps, city, minimum, maximum, variable)
    #print(f"Probabilities for {city}: {probabs}")
    return probabs
