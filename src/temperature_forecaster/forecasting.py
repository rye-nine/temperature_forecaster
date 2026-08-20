from temperature_forecaster.probability_model import normal_distribution_approximation, extrema_approximation_all
from temperature_forecaster.__init__ import weather_station_coords
from temperature_forecaster.probability_model import get_final_residuals
from scipy.stats import norm
import pandas as pd

def get_probability(day, city, minimum, maximum, variable="tmax"):

    city_distribution = normal_distribution_approximation(day, variable, city)[city]

    mu = city_distribution[0]
    #print("mu: ", mu)
    print(type(mu))
    sigma = city_distribution[1][1]
    #print("sigma: ", sigma)
    print(type(sigma))
    probabilities = []
    for j in range(minimum, maximum-1, 2):
        lower = j
        upper = j+2
        prob = norm.cdf(upper, loc=mu, scale=sigma) - norm.cdf(lower, loc=mu, scale=sigma)
        probabilities.append(((lower, upper),prob))
    return probabilities

# testing
def get_empirical_probability(day, city, minimum, maximum, variable = "tmax"):
    df_with_residuals_list = get_final_residuals(variable, city=city) # because city is not default to None, this variable should just be a singleton list
    our_df = df_with_residuals_list[0]
    # now we have our desired df based on city
    # now let's get the appropriate residuals
    day_offset = 15
    bool1 = our_df["day_of_year"] >= (day - day_offset) % 365
    bool2 = our_df["day_of_year"] <= (day + day_offset) % 365
    good_df = our_df[bool1 & bool2]
    desired_residuals = list(good_df["final_residuals"])

    # just for getting the approximated extrema
    approximation = extrema_approximation_all(day, variable, city_name=city)[0]

    probabilities = []
    for i in range(minimum, maximum - 1, 2):
        lower = i
        upper = i + 2
        observed = (desired_residuals >= lower-approximation) & (desired_residuals < upper-approximation) # [ )
        prob = observed.sum() / len(desired_residuals)
        probabilities.append(((lower, upper), prob))
    return probabilities

#mode = 1 --> normal distribution
#mode = 2 --> empirical residual distribution
def run_forecasting(mode,day, minimum, maximum, city=None, variable="tmax"):
    if (city is None): # no city is specified
        city_names = list(weather_station_coords.keys())
        my_dict = {}
        for city in city_names:
            my_dict[city] = get_probability(day, city, minimum, maximum, variable) if (mode == 1) else get_empirical_probability(day, city, minimum, maximum, variable)
        #print(my_dict)
        return my_dict
    probabs = get_probability(day, city, minimum, maximum, variable) if (mode == 1) else get_empirical_probability(day, city, minimum, maximum, variable)
    #print(f"Probabilities for {city}: {probabs}")
    return probabs

## BELOW IS FOR KALSHI-SPECIFIC STUFF

def get_number(market_ticker: str) -> tuple[float, str]:
    split = market_ticker.split("-")
    return float(split[2][1:]), split[2][0]

def appropriate_intervals(market_ticker: list[str]) -> tuple[int,int,dict[str,tuple[float,str]]]:
    number_and_letter = {ticker: get_number(ticker) for ticker in market_ticker}
    lst = [tup[0] for tup in list(number_and_letter.values())]
    minimum = min(lst)
    maximum = max(lst)
    return int(minimum), int(maximum + 1), number_and_letter

def sliced_intervals(market_ticker: list[str]) -> list[tuple[int,int]]:
    minimum, maximum, _ = appropriate_intervals(market_ticker)
    return_thing = [(-999, minimum)]
    return_thing += [(i, i+2) for i in range(int(minimum), int(maximum), 2)]
    return_thing += [(maximum, 999)]
    return return_thing

def kalshi_forecasting(mode, day, market_tickers, city=None, variable="tmax"): # disregard mode for now, just use mode = 1 for normal distribution approximation
    intervals = sliced_intervals(market_tickers)
    print(f"intervals: {intervals}")
    city_distribution = normal_distribution_approximation(day, variable, city)[city]
    mu = city_distribution[0]
    print(mu)
    sigma = city_distribution[1][1]
    print(sigma)
    minimum, maximum, numbers_dictionary = appropriate_intervals(market_tickers)
    
    #mapping = {interval: ticker in market_tickers for (interval, ticker) in zip(intervals, market_tickers)}
    probabilities = []
    if (mode == 1):
        upper_list = [interval[1] for interval in intervals]
        lower_list = [interval[0] for interval in intervals]
        probabilities = norm.cdf(upper_list, loc=mu, scale=sigma) - norm.cdf(lower_list, loc=mu, scale=sigma)
    else: # if mode == 2
        probabilities = [get_empirical_probability(day, city, lower, upper, variable)[0][1] for (lower, upper) in intervals]
    prob_dictionary = {interval: prob for (interval, prob) in zip(intervals, probabilities)}

    return_dict = {}
    for key in numbers_dictionary.keys(): # key is a market ticker
        tuple_val = numbers_dictionary[key]
        num = tuple_val[0]
        letter = tuple_val[1]
        print(f"maximum: {maximum}, num: {num}")
        if (letter == "T"):
            num += 1
            return_dict[key] = list(prob_dictionary.values())[-1] if maximum == num else next(iter(prob_dictionary.values()))
            print_thing1 = list(prob_dictionary.keys())[-1] if maximum == num else list(prob_dictionary.keys())[0]
            print_thing2 = return_dict[key]
            print(f"{key} corresponds to {print_thing1} with probability {print_thing2}")
        else:
            for pkey in prob_dictionary.keys(): #pkey is an interval
                interval_min = pkey[0]
                interval_max = pkey[1]
                if (num >= interval_min) and (num <= interval_max): #else if letter == "B"
                    return_dict[key] = prob_dictionary[pkey]
                    print(f"{key} corresponds to interval {pkey} with probability {prob_dictionary[pkey]}")
                    break
    return return_dict


