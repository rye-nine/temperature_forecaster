from temperature_forecaster.probability_model import normal_distribution_approximation
from temperature_forecaster.__init__ import weather_station_coords
from temperature_forecaster.probability_model import get_final_residuals
from scipy.stats import norm

def get_probability(day, city, minimum, maximum, variable="tmax"):
    all_distributions = normal_distribution_approximation(day, variable)

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

# testing
def get_empirical_probability(day, city, minimum, maximum, variable = "tmax"):
    df_with_residuals_list = get_final_residuals(variable)
    city_index = list(weather_station_coords.keys()).index(city)
    our_df = df_with_residuals_list[city_index]
    # now we have our desired df based on city
    # now let's get the appropriate residuals
    day_offset = 15
    bool1 = our_df["day_of_year"] >= day - day_offset
    bool2 = our_df["day_of_year"] <= day + day_offset
    good_df = our_df[bool1 & bool2]
    desired_residuals = list(good_df["final_residuals"])

    # just for getting the approximated extrema
    all_distributions = normal_distribution_approximation(day, variable)
    city_distribution = all_distributions[city]
    approximation = city_distribution[0]

    probabilities = []
    for i in range(minimum, maximum):
        observed = (desired_residuals >= i-approximation) & (desired_residuals < i+1-approximation)
        prob = observed.sum() / len(desired_residuals)
        probabilities.append(((i,i+1), prob))
    return probabilities

#mode = 1 --> normal distribution
#mode = 2 --> empirical residual distribution
def run_forecasting(mode,day, minimum, maximum, city=None, variable="tmax"):
    city_names = list(weather_station_coords.keys())
    if (city is None): # no city is specified
        my_dict = {}
        for city in city_names:
            if mode == 1:
                my_dict[city] = get_probability(day, city, minimum, maximum, variable)
            else:
                my_dict[city] = get_empirical_probability(day, city, minimum, maximum, variable)
        #print(my_dict)
        return my_dict
    probabs = get_probability(day, city, minimum, maximum, variable) if (mode == 1) else get_empirical_probability(day, city, minimum, maximum, variable)
    #print(f"Probabilities for {city}: {probabs}")
    return probabs
