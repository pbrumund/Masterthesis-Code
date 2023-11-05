# Simple gaussian process using only wind data and constant noise variance

import numpy as np
import hilo_mpc as hilo
import datetime
import random

from fileloading import load_weather_data
import utils

def generate_features(weather_data, time, n_last=5, feature='last error'):
    measurements = np.zeros(n_last)
    predictions = np.zeros(n_last)

    start_time = time - datetime.timedelta(minutes=n_last*10)
    for i in range(n_last):
        measurements[i] = utils.get_wind_value(weather_data, start_time, i)
        predictions[i] = utils.get_NWP(weather_data, start_time, i)
    
    if feature == 'last error':
        return measurements - predictions
    elif feature == 'last measurement':
        return measurements
    else:
        raise ValueError("Argument 'feature' must be 'last error' or 'last measurement")



def sample_random_inputs(weather_data, n_samples, n_features):
    n_data_points = len(weather_data['times_meas'])
    sample_indices = random.sample(np.arange(n_features, n_data_points), n_samples)



if __name__ == "__main__":
    # Data loading
    start_time = datetime.datetime(2020,1,1)
    end_time = datetime.datetime(2022,12,31)

    weather_data = load_weather_data(start_time, end_time)


