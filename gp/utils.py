import numpy as np
import datetime
import random

def get_NWP(wind_table, time, steps=0, key="wind_speed_10m"):
    """Interpolates weather prediction from dict using current forecast for the given time
    
    Uses linear interpolation to get values between full hours
    Keyword Arguments:
    wind_table -- dictionary of NWP and measurements
    time -- time at current time step
    steps -- number of steps to predict ahead
    key -- string with name of value as in dict, standard: "wind_speed_10m"
    """
    r = time.hour % 6
    if key in ["wind_speed_10m", "wind_direction_10m", "air_pressure_at_sea_level", "air_temperature_2m"]:   # MET post-processed
        times_sh = wind_table["times1_sh"]
    else:
        times_sh = wind_table['times2_sh']
    dt = time - times_sh[0].astype(datetime.datetime)
    i_start = int(dt.total_seconds()//21600) * 6 #6 hours, time of last released forecast
    predicted_trajectory = np.concatenate([wind_table[key+"_sh"][i_start:i_start+6], 
                                           wind_table[key+"_mh"][i_start:i_start+6], 
                                           wind_table[key+"_lh"][i_start:]])    # Reconstruct the most recent NWP at the given time
    predicted_times = np.concatenate([wind_table["times1_sh"][i_start:i_start+6], 
                                           wind_table["times1_mh"][i_start:i_start+6], 
                                           wind_table["times1_lh"][i_start:]])    # Reconstruct the most recent NWP at the given time
    t_rounded = time.replace(minute=0)

    i = np.where(predicted_times==(time+datetime.timedelta(minutes=10*steps)).replace(minute=0))[0][0]
    # i = int((6*r + time.minute//10 + steps)//6)
    wind_interp = (predicted_trajectory[i+1]*(time+steps*datetime.timedelta(minutes=10)).minute/60 
                   + predicted_trajectory[i]*(1-(time+steps*datetime.timedelta(minutes=10)).minute/60))
    return wind_interp

def get_wind_value(wind_table, time, steps=0):
    """Returns measured wind speed, inputs as in get_NWP"""
    dt = time - wind_table['times_meas'][0]
    i = int(dt.total_seconds()/600) + steps
    return wind_table['meas'][i]

def generate_features(weather_data, time, n_last=3, feature='error', steps_ahead = 1):
    measurements = np.zeros(n_last)
    wind_predictions = np.zeros(n_last)

    start_time = time - datetime.timedelta(minutes=(n_last-1)*10)
    for i in range(n_last):
        measurements[i] = get_wind_value(weather_data, start_time, i)
        wind_predictions[i] = get_NWP(weather_data, start_time, i, 'wind_speed_10m')
    
    cape = get_NWP(weather_data, time, steps_ahead, 'specific_convective_available_potential_energy')
    temperature = get_NWP(weather_data, time, steps_ahead, 'air_temperature_2m')
    wind_prediction_at_step = get_NWP(weather_data, time, steps_ahead, 'wind_speed_10m')
    errors = measurements - wind_predictions

    # Normalize inputs
    # measurements = (measurements-weather_data['mean_meas'])/weather_data['std_meas']
    # wind_predictions = (wind_predictions-weather_data['mean_pred'])/weather_data['std_pred']
    # errors = (errors-weather_data['mean_error'])/weather_data['std_error']
    sqrt_cape = np.sqrt(cape)
    # sqrt_cape = (sqrt_cape-weather_data['mean_cape'])/weather_data['std_cape']
    # temperature = (temperature-weather_data['mean_temp'])/weather_data['std_temp']

    if feature == 'error':
        return errors
    elif feature == 'measurement':
        return measurements
    elif feature == 'measurement & error':
        return np.concatenate([measurements, errors])
    elif feature == 'prediction & error':
        return np.concatenate([wind_predictions, errors])
    elif feature == 'prediction & measurement':
        return np.concatenate([measurements, wind_prediction_at_step])
    elif feature == 'error & nwp':
        return np.concatenate([errors, [measurements[-1], wind_prediction_at_step, sqrt_cape, temperature]])
    elif feature == 'error & nwp & time':
        return np.concatenate([errors, [measurements[-1], wind_prediction_at_step, sqrt_cape, temperature, steps_ahead]])
    elif feature == 'measurement & nwp':
        return np.concatenate([measurements, [wind_prediction_at_step, sqrt_cape, temperature]])
    elif feature == 'measurement & nwp & time':
        return np.concatenate([measurements, [wind_prediction_at_step, sqrt_cape, temperature, steps_ahead]])
    else:
        raise ValueError("Unknown value for feature")

def generate_labels(weather_data, time, label='error', steps_ahead = 1):
    measurement = get_wind_value(weather_data, time, steps=steps_ahead)
    if label == 'error':
        prediction = get_NWP(weather_data, time, steps=steps_ahead)
        return measurement - prediction
    elif label == 'measurement':
        return measurement
    elif label == 'change of wind speed':
        current_measurement = get_wind_value(weather_data, time, steps=0)
        return measurement-current_measurement
    else:
        raise ValueError("Unknown value for feature")

def sample_random_inputs(weather_data, n_samples, n_features, feature='error', label='error', steps_ahead = 1, max_steps_ahead = 1):
    n_data_points = len(weather_data['times_meas'])
    sample_indices = random.sample(range(n_features, int(n_data_points*2/3)), n_samples)
    if max_steps_ahead > 1:
        steps = [random.randint(1, max_steps_ahead) for i in range(n_samples)]
    else:
        steps = steps_ahead*np.ones(n_samples)

    features = [generate_features(weather_data, weather_data['times_meas'][sample_indices[i]], n_features, feature, steps[i]) for i in range(len(sample_indices))]
    labels = [generate_labels(weather_data, weather_data['times_meas'][i], label, steps_ahead) for i in sample_indices]

    X = np.array(features)
    y = np.array(labels).reshape((-1,1))

    return X, y

def sparsify_training_data(X, y, M):
        n_sample = M
        n_sample = min(n_sample, len(y))
        rand_indices = random.sample(range(len(y)), n_sample)
        X_train_mean = X[rand_indices,:]
        y_train_mean = np.reshape(y[rand_indices], (-1,1))
        return X_train_mean, y_train_mean


def estimate_variance(X, y, M):
    # normalize x for every dimension
    n_inputs = X.shape[1]
    X_norm = np.zeros(X.shape)
    var_x = np.zeros(n_inputs)
    mean_x = np.zeros(n_inputs)
    for i in range(n_inputs):
        mean_i = np.mean(X[:,i])
        var_i = np.var(X[:,i])
        if var_i != 0:
            X_norm[:,i] = (X[:,i])/np.sqrt(var_i)
            var_x[i] = var_i
        else:
            X_norm[:,i] = X[:,i]
            var_x[i] = 1
    
    #draw random points
    n_points = M
    n_points = min(n_points, len(y))
    rand_indices = random.sample(range(len(y)), n_points)

    #find n_closest closest neighbors
    n_closest = 50
    n_closest = min(n_closest, len(y))
    X_mean = np.zeros((n_points, n_inputs))
    y_var = np.zeros(n_points)
    for i in range(n_points):
        index = rand_indices[i]
        x_i = X_norm[index,:]
        y_i = y[index]
        X_others = np.delete(X_norm,index,0)
        y_others = np.delete(y, index)
        distances = [np.linalg.norm(x_i - X_others[k,:]) for k in range(X_others.shape[0])]
        d_sort = np.argsort(distances)[:n_closest]
        X_closest = np.array([X_others[k,:] for k in d_sort]).T
        y_closest = np.array([y_others[k] for k in d_sort])
        X_mean[i,:] = np.multiply(np.mean(X_closest, axis=1), np.sqrt(var_x))
        y_var[i] = np.var(y_closest)

    X_train_var = X_mean
    mean_var = np.mean(y_var)
    y_train_var = np.reshape(np.log(y_var), (-1,1))
    return X_train_var, y_train_var
        
def add_variance(weather_data):
    prediction_errors = np.array([get_wind_value(weather_data, t, 0) - get_NWP(weather_data, t, 0) for t in weather_data['times_meas']])
    weather_data['mean_error'] = np.mean(prediction_errors)
    weather_data['std_error'] = np.sqrt(np.var(prediction_errors))
    weather_data['mean_temp'] = np.mean(weather_data['air_temperature_2m_sh'])
    weather_data['std_temp'] = np.sqrt(np.var(weather_data['air_temperature_2m_sh'])) 
    weather_data['mean_cape'] = np.mean(np.sqrt(weather_data['specific_convective_available_potential_energy_sh']))
    weather_data['std_cape'] = np.sqrt(np.var(np.sqrt(weather_data['specific_convective_available_potential_energy_sh'])))
    weather_data['mean_pred'] = np.mean(weather_data['wind_speed_10m_sh'])
    weather_data['std_pred'] = np.sqrt(np.var(weather_data['wind_speed_10m_sh']))
    weather_data['mean_meas'] = np.mean(weather_data['meas'])
    weather_data['std_meas'] = np.sqrt(np.var(weather_data['meas']))
    return weather_data