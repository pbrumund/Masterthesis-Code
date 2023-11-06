# Simple gaussian process using only wind data and constant noise variance

import numpy as np
import hilo_mpc as hilo
import datetime
import random
import matplotlib.pyplot as plt

from fileloading import load_weather_data
import utils

def generate_features(weather_data, time, n_last=5, feature='error'):
    measurements = np.zeros(n_last)
    predictions = np.zeros(n_last)

    start_time = time - datetime.timedelta(minutes=(n_last-1)*10)
    for i in range(n_last):
        measurements[i] = utils.get_wind_value(weather_data, start_time, i)
        predictions[i] = utils.get_NWP(weather_data, start_time, i)
    
    if feature == 'error':
        return measurements - predictions
    elif feature == 'measurement':
        return measurements
    elif feature == 'measurement & error':
        return np.concatenate([measurements, measurements-predictions])
    elif feature == 'prediction & error':
        return np.concatenate([predictions, measurements-predictions])
    elif feature == 'prediction & measurement':
        return np.concatenate([predictions, measurements])
    else:
        raise ValueError("Unknown value for feature")

def generate_labels(weather_data, time, label='error', steps_ahead = 1):
    measurement = utils.get_wind_value(weather_data, time, steps=steps_ahead)
    prediction = utils.get_NWP(weather_data, time, steps=steps_ahead)
    if label == 'error':
        return measurement - prediction
    elif label == 'measurement':
        return measurement
    else:
        raise ValueError("Argument 'label' must be 'error' or 'measurement")

def sample_random_inputs(weather_data, n_samples, n_features, feature='error', label='error', steps_ahead = 1):
    n_data_points = len(weather_data['times_meas'])
    sample_indices = random.sample(range(n_features, n_data_points-steps_ahead), n_samples)

    features = [generate_features(weather_data, weather_data['times_meas'][i], n_features, feature) for i in sample_indices]
    labels = [generate_labels(weather_data, weather_data['times_meas'][i], label, steps_ahead) for i in sample_indices]

    X = np.array(features).T
    y = np.array(labels).reshape((1,-1))

    return X, y
def multi_step_prediction_direct(weather_data, time, steps):
    n_inputs = 3
    mean_vec = np.zeros(steps)
    var_vec = np.zeros(steps)
    for i in range(steps):
        X_train, y_train = sample_random_inputs(weather_data, 100, n_inputs, 'error', 'error', i)
        kernel3 = hilo.LinearKernel()
        feature_names = [f"{input_feature}_{i}" for i in range(n_inputs)]
        gp = hilo.GaussianProcess(features=feature_names,
                                labels=label,
                                kernel=kernel3)
        gp.set_training_data(X_train, y_train)
        gp.setup()
        gp.fit_model()
        features = generate_features(weather_data, time, n_inputs, 'error')
        mean, var = gp.predict(features)
        mean_vec[i] = mean
        var_vec[i] = var
    return mean_vec, var_vec

if __name__ == "__main__":
    random.seed(1)
    # Data loading
    start_time = datetime.datetime(2020,1,1)
    end_time = datetime.datetime(2020,1,31)

    n_samples = 100
    n_features = 5
    input_feature = 'error'
    label = 'error'
    steps_ahead = 5

    weather_data = load_weather_data(start_time, end_time)

    X_train, y_train = sample_random_inputs(weather_data, n_samples, n_features, input_feature, label, steps_ahead)

    n_inputs = X_train.shape[0]
    feature_names = [f"{input_feature}_{i}" for i in range(n_inputs)]

    kernel1 = hilo.SquaredExponentialKernel(active_dims=np.arange(n_inputs), ard=True)
    kernel2 = hilo.SquaredExponentialKernel(active_dims=np.arange(n_inputs), ard=False)
    kernel3 = hilo.LinearKernel()
    gp = hilo.GaussianProcess(features=feature_names,
                              labels=label,
                              kernel=kernel3)
    gp.set_training_data(X_train, y_train)
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
        features_i = generate_features(weather_data, time=t, n_last=n_features, feature=input_feature)
        gp_mean, gp_var = gp.predict(features_i)
        if label == 'error':
            prediction_GP = prediction_NWP + gp_mean
        elif label == 'measurement':
            prediction_GP = gp_mean
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
    print((np.mean(np.abs(prediction_errors_GP))))
    print((np.mean(np.abs(prediction_errors_NWP))))

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

    t_start = datetime.datetime(2020,1,1,6,0)
    mean, var = multi_step_prediction_direct(weather_data, t_start, 60)
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
    plt.show()




