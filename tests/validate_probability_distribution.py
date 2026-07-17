from temperature_forecaster.forecasting import run_forecasting
import numpy as np

# to do, maybe just make it a helper function (like on a given day, just make sure all of the probabilities sum to almost 1)

def check_for_anomalies(item):
    raise NotImplementedError

verify_singular_probability(probability_sum):
    if np.isclose(probability_sum, 1.0, atol=1e-9):
        print(f"[SUCCESS] Probability distribution for {city} is valid (total: {prob_sum})")
    else:
        print(f"[FAILURE] Probability distribution for {city} is invalid (error = {total - 1:.12e})")


def check_probability(item):
    if (isinstance(item, dict)):
        for city in item.keys():
            current_tuple = item[city]
            prob_sum = current_tuple[:, 1].sum()
            verify_singular_probability(prob_sum)
    else:
        sum_prob = 0
        for tup in item:
            sum_prob += tup[1]
        verify_singular_probability(prob_sum)

            
