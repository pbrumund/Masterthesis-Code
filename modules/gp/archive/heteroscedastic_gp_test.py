# Simple gaussian process using only wind data and constant noise variance

import numpy as np
import hilo_mpc as hilo
import datetime
import random
import matplotlib.pyplot as plt
from scipy.stats import invgauss, norm

from fileloading import load_weather_data
import utils
from HeteroscedasticGP import MostLikelyHeteroscedasticGP

def generate_features(weather_data, time, n_last=3, feature='error', steps_ahead = 1):
    measurements = np.zeros(n_last)
    wind_predictions = np.zeros(n_last)

    start_time = time - datetime.timedelta(minutes=(n_last-1)*10)
    for i in range(n_last):
        measurements[i] = utils.get_wind_value(weather_data, start_time, i)
        wind_predictions[i] = utils.get_NWP(weather_data, start_time, i, 'wind_speed_10m')
    
    cape = utils.get_NWP(weather_data, time, steps_ahead, 'specific_convective_available_potential_energy')
    temperature = utils.get_NWP(weather_data, time, steps_ahead, 'air_temperature_2m')
    wind_prediction_at_step = utils.get_NWP(weather_data, time, steps_ahead, 'wind_speed_10m')

    if feature == 'error':
        return measurements - wind_predictions
    elif feature == 'measurement':
        return measurements
    elif feature == 'measurement & error':
        return np.concatenate([measurements, measurements-wind_predictions])
    elif feature == 'prediction & error':
        return np.concatenate([wind_predictions, measurements-wind_predictions])
    elif feature == 'prediction & measurement':
        return np.concatenate([wind_predictions, measurements])
    elif feature == 'error & nwp':
        return np.concatenate([measurements-wind_predictions, [measurements[-1], wind_prediction_at_step, np.sqrt(cape), temperature]])
    elif feature == 'error & nwp & time':
        return np.concatenate([measurements-wind_predictions, [measurements[-1], wind_prediction_at_step, np.sqrt(cape), temperature, steps_ahead]])
    elif feature == 'measurement & nwp':
        return np.concatenate([measurements, wind_predictions, [wind_prediction_at_step, np.sqrt(cape), temperature]])
    elif feature == 'measurement & nwp & time':
        return np.concatenate([measurements, wind_predictions, [wind_prediction_at_step, np.sqrt(cape), temperature, steps_ahead]])
    else:
        raise ValueError("Unknown value for feature")

def generate_labels(weather_data, time, label='error', steps_ahead = 1):
    measurement = utils.get_wind_value(weather_data, time, steps=steps_ahead)
    if label == 'error':
        prediction = utils.get_NWP(weather_data, time, steps=steps_ahead)
        return measurement - prediction
    elif label == 'measurement':
        return measurement
    elif label == 'change of wind speed':
        current_measurement = utils.get_wind_value(weather_data, time, steps=0)
        return measurement-current_measurement
    else:
        raise ValueError("Unknown value for feature")

def sample_random_inputs(weather_data, n_samples, n_features, feature='error', label='error', steps_ahead = 1, max_steps_ahead = 1):
    n_data_points = len(weather_data['times_meas'])
    sample_indices = random.sample(range(n_features, n_data_points-steps_ahead), n_samples)
    if max_steps_ahead > 1:
        steps = [random.randint(1, max_steps_ahead) for i in range(n_samples)]
    else:
        steps = steps_ahead*np.ones(n_samples)

    features = [generate_features(weather_data, weather_data['times_meas'][sample_indices[i]], n_features, feature, steps[i]) for i in range(len(sample_indices))]
    labels = [generate_labels(weather_data, weather_data['times_meas'][i], label, steps_ahead) for i in sample_indices]

    X = np.array(features).T
    y = np.array(labels).reshape((1,-1))

    return X, y
def multi_step_prediction_direct(weather_data, time, steps):
    n_inputs = 5
    mean_vec = np.zeros(steps)
    var_vec = np.zeros(steps)
    kernel1 = hilo.SquaredExponentialKernel(active_dims=np.arange(n_inputs), ard=True)
    feature_names = [f"{input_feature}_{i}" for i in range(n_inputs)]
    gp = hilo.GaussianProcess(features=feature_names,
                                labels=label,
                                kernel=kernel1)
    for i in range(steps):
        X_train, y_train = sample_random_inputs(weather_data, 100, 3, 'error & cape & temp', 'error', i)
        
        # kernel3 = hilo.LinearKernel()
        
        gp.set_training_data(X_train, y_train)
        gp.setup()
        gp.fit_model()
        features = generate_features(weather_data, time, n_inputs, 'error')
        mean, var = gp.predict(features)
        # del gp
        mean_vec[i] = mean
        var_vec[i] = var
    return mean_vec, var_vec

def multi_step_prediction_time_input(weather_data, time, steps):
    n_inputs = 6
    mean_vec = np.zeros(steps)
    var_vec = np.zeros(steps)
    kernel1 = hilo.SquaredExponentialKernel(active_dims=np.arange(n_inputs), ard=True)
    feature_names = [f"{input_feature}_{i}" for i in range(n_inputs)]
    gp = hilo.GaussianProcess(features=feature_names,
                                labels=label,
                                kernel=kernel1)
    X_train, y_train = sample_random_inputs(weather_data, 100, 3, 'measurement & nwp', 'error', max_steps_ahead=60)        
    gp.set_training_data(X_train, y_train)
    gp.setup()
    gp.fit_model()
    for i in range(steps):
        features = generate_features(weather_data, time, n_inputs, 'error')
        mean, var = gp.predict(features)
        # del gp
        mean_vec[i] = mean
        var_vec[i] = var
    return mean_vec, var_vec

if __name__ == "__main__":
    # random.seed(1)
    # Data loading
    start_time = datetime.datetime(2020,1,1)
    end_time = datetime.datetime(2020,1,31)

    n_samples = 100
    n_last = 3
    input_feature = 'error & nwp'
    label = 'error'
    steps_ahead = 1
    print(input_feature)
    print(label)
    print(f'n_last = {n_last}, steps_ahead = {steps_ahead}')

    weather_data = load_weather_data(start_time, end_time)

    X_train = np.array([])
    y_train = np.array([])

    for time in weather_data['times_meas'][n_last-1:-steps_ahead]:
        x_i = generate_features(weather_data, time, n_last=n_last, feature=input_feature, steps_ahead=steps_ahead)
        n_features = len(x_i)
        X_train = np.append(X_train, x_i)
        y_train = np.append(y_train, generate_labels(weather_data, time, label, steps_ahead))

    X_train = np.reshape(X_train, (-1,n_features)).T

    n_inputs = X_train.shape[0]
    feature_names = [f"{input_feature}_{i}" for i in range(n_inputs)]

    if 'error' in input_feature:
        n_linear = n_last
    else:
        n_linear = 2*n_last
    if 'time' in input_feature:
        n_matern = 1
    else:
        n_matern = 0
    
    kernel_mean_linear = hilo.LinearKernel(active_dims=np.arange(n_linear))
    kernel_mean_SE = hilo.SquaredExponentialKernel(active_dims=np.arange(n_linear, n_inputs-n_matern), ard=True)
    kernel_mean = hilo.kernel.Product(kernel_mean_linear, kernel_mean_SE)
    if n_matern != 0:
        kernel_mean_matern = hilo.Matern32Kernel(active_dims=(n_inputs-1))
        kernel_mean = hilo.kernel.Product(kernel_mean, kernel_mean_matern)
    # kernel_mean = hilo.SquaredExponentialKernel(active_dims=np.arange(n_inputs), ard=True)
    kernel_var = hilo.SquaredExponentialKernel(active_dims=np.arange(n_inputs), ard=True)

    gp = MostLikelyHeteroscedasticGP(features=feature_names, labels=label, kernel_mean=kernel_mean, kernel_var=kernel_var)
    gp.set_training_data(X_train, y_train)

    # kernel1 = hilo.SquaredExponentialKernel(active_dims=np.arange(n_inputs), ard=True)
    # kernel2 = hilo.SquaredExponentialKernel(active_dims=np.arange(n_inputs), ard=False)
    # kernel3 = hilo.LinearKernel()
    # gp = hilo.GaussianProcess(features=feature_names,
    #                           labels=label,
    #                           kernel=kernel1)
    # gp.set_training_data(X_train, y_train)
    gp.setup()
    gp.fit_model()

    evaluation_horizon = len(weather_data['times_meas']) - steps_ahead
    times = []
    measurements = np.zeros(evaluation_horizon-n_features)
    predictions_NWP = np.zeros(evaluation_horizon-n_features)
    predictions_GP = np.zeros(evaluation_horizon-n_features)
    predictions_GP_lower = np.zeros(evaluation_horizon-n_features)
    predictions_GP_upper = np.zeros(evaluation_horizon-n_features)
    features = np.zeros((n_inputs, evaluation_horizon-n_features))

    for i in range(n_features, evaluation_horizon):
        t = weather_data['times_meas'][i]
        prediction_NWP = utils.get_NWP(weather_data, time=t, steps=steps_ahead)
        features_i = generate_features(weather_data, time=t, n_last=n_last, feature=input_feature, steps_ahead=steps_ahead)
        gp_mean, gp_var = gp.predict(features_i)
        if label == 'error':
            prediction_GP = prediction_NWP + gp_mean
        elif label == 'measurement':
            prediction_GP = gp_mean
        elif label == 'change of wind speed':
            current_wind_speed = utils.get_wind_value(weather_data, t, steps=0)
            prediction_GP = current_wind_speed + gp_mean
        prediction_GP_lower = prediction_GP - np.sqrt(gp_var)
        prediction_GP_upper = prediction_GP + np.sqrt(gp_var)

        measurements[i-n_features] = utils.get_wind_value(weather_data, t, steps_ahead)
        predictions_NWP[i-n_features] = prediction_NWP
        predictions_GP[i-n_features] = prediction_GP
        predictions_GP_lower[i-n_features] = prediction_GP_lower
        predictions_GP_upper[i-n_features] = prediction_GP_upper
        features[:, i-n_features] = features_i
        times.append(t)

    prediction_errors_GP = measurements - predictions_GP
    prediction_errors_NWP = measurements - predictions_NWP
    print(f"RMSE GP: {np.sqrt(np.mean(np.square(prediction_errors_GP)))}")
    print(f"RMSE NWP: {np.sqrt(np.mean(np.square(prediction_errors_NWP)))}")
    print(f"Root mean standard deviation of GP: {np.sqrt(np.mean(np.square(predictions_GP_upper-predictions_GP)))}")
    alpha = 0.32
    a = norm.ppf(1-0.5*alpha)
    p_in_interval = len([i for i in range(len(predictions_GP)) if np.abs(predictions_GP[i]-measurements[i])<a*(predictions_GP_upper[i]-predictions_GP[i])])/len(predictions_GP)
    RE = len([i for i in range(len(predictions_GP)) if np.abs(predictions_GP[i]-measurements[i])<a*(predictions_GP_upper[i]-predictions_GP[i])])/len(predictions_GP)-(1-alpha)
    print(f"RE for {alpha}: {RE}")
    print(f"Percentage of points in estimated {100*(1-alpha)}%-interval: {100*p_in_interval}%")
    print(gp._gp_mean.hyperparameters)

    gp.plot(features, backend='matplotlib')
    plt.figure()
    plt.plot(times, measurements)
    plt.plot(times, predictions_NWP)
    plt.plot(times, predictions_GP)
    plt.plot(times, predictions_GP_lower, color='k')
    plt.plot(times, predictions_GP_upper, color='k')
    plt.xlabel('time')
    plt.ylabel('predicted wind speed')
    plt.legend(['measured wind speed', 'NWP', 'GP', "GP "+u"\u00B1"+" 1 std"])
    plt.figure()
    plt.plot(predictions_GP_upper-predictions_GP)
    del gp
    plt.show()
""" 
    t_start = datetime.datetime(2020,1,1,6,0)
    mean, var = multi_step_prediction_time_input(weather_data, t_start, 60)
    NWP_traj = np.zeros(len(mean))
    meas_traj = np.zeros(len(mean))
    for i in range(len(NWP_traj)):
        NWP_traj[i] = utils.get_NWP(weather_data, t_start, i)
        meas_traj[i] = utils.get_wind_value(weather_data, t_start, i)
    GP_traj = NWP_traj + mean
    GP_traj_upper = GP_traj + np.sqrt(var)
    GP_traj_lower = GP_traj - np.sqrt(var)
    t = [t_start + i*datetime.timedelta(minutes=60) for i in range(len(mean))]
    plt.figure()
    plt.plot(t,meas_traj)
    plt.plot(t, NWP_traj)
    plt.plot(t, GP_traj)
    plt.plot(t, GP_traj_lower, color='k')
    plt.plot(t, GP_traj_upper, color='k')
    plt.xlabel('time')
    plt.ylabel('wind speed')
    plt.legend(['measured wind speed', 'NWP', 'GP', "GP "+u"\u00B1"+" 1 std"])
    plt.show() """




