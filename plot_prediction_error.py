import numpy as np
import matplotlib.pyplot as plt
import datetime
import csv
import scipy as scp
import typing
from mpl_toolkits.mplot3d import Axes3D
import modules.gp.utils as ut
import modules.gp.fileloading as fi
from modules.gp.data_handling import DataHandler
import random

if __name__ == "__main__":
    # Data loading
    start_time = datetime.datetime(2020,1,1)
    end_time = datetime.datetime(2022,12,31)

    gp_opt = {'dt_meas': 10}
    dh = DataHandler(start_time, end_time, gp_opt)
    data_table = dh.weather_data
    plt.ion()
    # Plot predictions starting at t
    """ t = datetime.datetime(2021,7,14,6,10)
    n_predictions = 60
    wind_predicted = np.zeros(n_predictions)
    wind_measured = np.zeros(n_predictions)
    prediction_times = []
    for i in range(n_predictions):
        wind_predicted[i] = get_prediction(data_table,t,i)
        wind_measured[i] = get_wind_value(data_table,t,i)
        prediction_times.append(t + i*datetime.timedelta(minutes=10))
    plt.figure()
    plt.plot(prediction_times, wind_measured)
    plt.plot(prediction_times, wind_predicted)
    plt.show() """
    
    # Plot Historgrams for prediction error for different horizons
    """ evaluation_horizon = len(data_table['times_meas']) - 60
    predictions_10m = np.zeros(evaluation_horizon)
    errors_10m = np.zeros(evaluation_horizon)
    predictions_1h = np.zeros(evaluation_horizon)
    errors_1h = np.zeros(evaluation_horizon)
    predictions_10h = np.zeros(evaluation_horizon)
    errors_10h = np.zeros(evaluation_horizon)
    for i in range(evaluation_horizon):
        predictions_10m[i]  = get_prediction(data_table, time=data_table['times_meas'][i], steps=1)
        predictions_1h[i]   = get_prediction(data_table, time=data_table['times_meas'][i], steps=6)
        predictions_10h[i]  = get_prediction(data_table, time=data_table['times_meas'][i], steps=60)
        errors_10m[i]   = get_wind_value(data_table, time=data_table['times_meas'][i], steps=1) - predictions_10m[i]
        errors_1h[i]    = get_wind_value(data_table, time=data_table['times_meas'][i], steps=6) - predictions_1h[i]
        errors_10h[i]   = get_wind_value(data_table, time=data_table['times_meas'][i], steps=60) - predictions_10h[i]

    plt.figure()
    plt.hist(errors_10h, bins=250)
    plt.hist(errors_1h, bins=250)
    plt.hist(errors_10m, bins=250)
    plt.legend(['10 h forecast', '1h forecast', '10 min forecast'])
    plt.xlabel('prediction error')
    plt.show() """

    # Plot covariance of prediction error over number of steps ahead
    """ weather_data = fi.load_weather_data(start_time, end_time)
    max_horizon = 100
    cov = np.zeros(max_horizon)

    n_points = len(weather_data['times_meas']) - max_horizon
    for s in range(max_horizon):
        prediction_errors = np.zeros(n_points)
        for i in range(n_points):
            prediction = ut.get_NWP(weather_data, weather_data['times_meas'][i], s)
            measurement = ut.get_wind_value(weather_data, weather_data['times_meas'][i], s)
            prediction_errors[i] = measurement - prediction
        if s == 0:
            prediction_errors_now = prediction_errors
        cov[s] = np.cov(prediction_errors, prediction_errors_now)[1,0]

    np.savetxt('../Abbildungen/data_analysis/cov_over_steps.csv', cov)
    plt.figure()
    plt.plot(cov)
    plt.xlabel('Number of predicted 10-minute-steps')
    plt.ylabel('Covariance of prediction error')
    plt.show()
 """
    # Plot covariance of change of error
    """ 
    max_horizon = 10
    cov = np.zeros(max_horizon)
    for s in range(max_horizon):
        prediction_errors = np.zeros(len(data_table['times_meas']) - max_horizon)
        for i in range(len(data_table['times_meas']) - max_horizon):
            prediction = get_prediction(data_table, time=data_table['times_meas'][i], steps=s)
            prediction_errors[i] = get_wind_value(data_table, time=data_table['times_meas'][i], steps=s) - prediction
        if s == 0:
            prediction_errors_last = prediction_errors
        else:
            prediction_error_change = prediction_errors - prediction_errors_last
            prediction_errors_last = prediction_errors
        if s == 1:
            prediction_error_change_1step = prediction_error_change
        if s > 0:
            cov[s-1] = np.cov(prediction_error_change, prediction_error_change_1step)[1,0]
    plt.figure()
    plt.plot(cov)
    plt.xlabel('steps')
    plt.ylabel('covariance')
    plt.show() """

    # Plot change of error over error
     
    """ eval_horizon = len(data_table['times_meas']) - 1
    prediction_errors = np.zeros(eval_horizon)
    for i in range(eval_horizon):
        prediction = get_prediction(data_table, time=data_table['times_meas'][i], steps=1)
        prediction_errors[i] = get_wind_value(data_table, time=data_table['times_meas'][i], steps=1) - prediction
    prediction_error_change = prediction_errors[1:] - prediction_errors[:-1]

    plt.figure()
    plt.hist2d(prediction_errors[:-1], prediction_error_change, bins=1000)
    plt.xlabel('last prediction error')
    plt.ylabel('change of prediction error')
    plt.show() """
   
    # Plot current error vs error n steps ahead
    """ n = 1
    eval_horizon = len(data_table['times_meas']) - n
    prediction_error_nsteps = np.zeros(eval_horizon)
    current_error = np.zeros(eval_horizon)
    times = data_table['times_meas'][:eval_horizon]
    for i, time in enumerate(times):
        current_prediction = dh.get_NWP(time)
        current_error[i] = dh.get_measurement(time) - current_prediction
        prediction_nsteps = dh.get_NWP(time, steps=n)
        prediction_error_nsteps[i] = dh.get_measurement(time, steps=n) - prediction_nsteps
    
    plt.figure()
    plt.scatter(current_error, prediction_error_nsteps)
    plt.hist2d(current_error, prediction_error_nsteps, bins=250)
    plt.xlabel('current prediction error')
    plt.ylabel(f'prediction error for {n}-step prediction')
    plt.show() """

    # Plot prediction error over predicted wind speed
    """ n = 5
    eval_horizon = len(data_table['times_meas']) - n
    predictions = np.zeros(eval_horizon)
    measured_values = np.zeros(eval_horizon)
    for i in range(eval_horizon):
        predictions[i] = get_prediction(data_table, time=data_table['times_meas'][i], steps=0)
        measured_values[i] = get_wind_value(data_table, time=data_table['times_meas'][i], steps=0)
    prediction_errors = predictions - measured_values

    plt.figure()
    plt.hist2d(predictions, prediction_errors, bins=250)
    plt.xlabel('predicted wind speed')
    plt.ylabel(f'prediction error for {n}-step prediction')
    plt.show() """

    # Plot variance of error over predicted wind speed
    """ n = 0
    eval_horizon = len(data_table['times_meas']) - n
    predictions = np.zeros(eval_horizon)
    measured_values = np.zeros(eval_horizon)
    for i in range(eval_horizon):
        predictions[i] = get_prediction(data_table, time=data_table['times_meas'][i], steps=0)
        measured_values[i] = get_wind_value(data_table, time=data_table['times_meas'][i], steps=0)
    prediction_errors = predictions - measured_values

    prediction_var_ws = np.zeros(20)
    for ws in range(20):
        prediction_errors_at_ws = (prediction_errors[np.logical_and(predictions > ws, predictions < ws+1)])
        prediction_var_ws[ws] = np.var(prediction_errors_at_ws)
    
    plt.figure()
    plt.plot(prediction_var_ws)
    plt.xlabel('predicted wind speed')
    plt.ylabel('variance of prediction error')
    plt.show() """

    # Plot error variance over prediction error
    """ n = 0
    eval_horizon = len(data_table['times_meas']) - n
    predictions = np.zeros(eval_horizon)
    measured_values = np.zeros(eval_horizon)
    for i in range(eval_horizon):
        predictions[i] = get_prediction(data_table, time=data_table['times_meas'][i], steps=n)
        measured_values[i] = get_wind_value(data_table, time=data_table['times_meas'][i], steps=n)
    prediction_errors = predictions - measured_values

    error_ranges = np.linspace(-4,4,20)
    prediction_var_range = np.zeros(len(error_ranges)-1)
    for i in range(len(error_ranges)-1):
        prediction_errors_in_range = (prediction_errors[np.logical_and(prediction_errors > error_ranges[i], prediction_errors < error_ranges[i+1])])
        prediction_var_range[i] = np.var(prediction_errors_in_range)
        print(len(prediction_errors_in_range))
    plt.figure()
    plt.plot(error_ranges[1:], prediction_var_range)
    plt.xlabel(f'prediction error for {n} steps')
    plt.ylabel('variance of prediction error')
    plt.show() """

    # Plot future error distribution after n steps over current prediction error
    """ n = 1
    eval_horizon = len(data_table['times_meas']) - n
    predictions = np.zeros(eval_horizon)
    measured_values = np.zeros(eval_horizon)
    for i in range(eval_horizon):
        predictions[i] = get_prediction(data_table, time=data_table['times_meas'][i], steps=0)
        measured_values[i] = get_wind_value(data_table, time=data_table['times_meas'][i], steps=0)
    prediction_errors_now = predictions - measured_values
    for i in range(eval_horizon):
        predictions[i] = get_prediction(data_table, time=data_table['times_meas'][i], steps=n)
        measured_values[i] = get_wind_value(data_table, time=data_table['times_meas'][i], steps=n)
    prediction_errors = predictions - measured_values

    error_ranges = np.linspace(-4,4,20)
    prediction_var_range = np.zeros(len(error_ranges)-1)
    prediction_mean_range = np.zeros(len(error_ranges)-1)
    for i in range(len(error_ranges)-1):
        prediction_errors_in_range = (prediction_errors[np.logical_and(prediction_errors_now > error_ranges[i], prediction_errors_now < error_ranges[i+1])])
        prediction_mean_range[i] = np.mean(prediction_errors_in_range)
        prediction_var_range[i] = np.var(prediction_errors_in_range)

    plt.figure()
    plt.plot(error_ranges[1:], prediction_mean_range)
    plt.plot(error_ranges[1:], prediction_mean_range + np.sqrt(prediction_var_range), color='k')
    plt.plot(error_ranges[1:], prediction_mean_range - np.sqrt(prediction_var_range), color='k')
    plt.xlabel('prediction error')
    plt.ylabel(f'prediction error after {n} steps')
    plt.legend(['mean', '+- 1 std'])
    plt.show() """

    # Plot variance of error difference over prediction error
    """ n = 10
    eval_horizon = len(data_table['times_meas']) - n
    predictions = np.zeros(eval_horizon)
    measured_values = np.zeros(eval_horizon)
    # Current error
    for i in range(eval_horizon):
        predictions[i] = get_prediction(data_table, time=data_table['times_meas'][i], steps=0)
        measured_values[i] = get_wind_value(data_table, time=data_table['times_meas'][i], steps=0)
    prediction_errors_now = predictions - measured_values
    # Error after n steps
    for i in range(eval_horizon):
        predictions[i] = get_prediction(data_table, time=data_table['times_meas'][i], steps=n)
        measured_values[i] = get_wind_value(data_table, time=data_table['times_meas'][i], steps=n)
    prediction_errors = predictions - measured_values
    prediction_errors_diff = prediction_errors - prediction_errors_now
    error_ranges = np.linspace(-4,4,20)
    prediction_var_range = np.zeros(len(error_ranges)-1)
    prediction_mean_range = np.zeros(len(error_ranges)-1)
    cond_prediction_error_mean = np.zeros(len(error_ranges)-1)
    cond_prediction_error_var = np.zeros(len(error_ranges)-1)
    for i in range(len(error_ranges)-1):
        diff_prediction_errors_in_range = (prediction_errors_diff[np.logical_and(prediction_errors_now > error_ranges[i], prediction_errors_now < error_ranges[i+1])])
        prediction_errors_in_range = (prediction_errors[np.logical_and(prediction_errors_now > error_ranges[i], prediction_errors_now < error_ranges[i+1])])
        prediction_mean_range[i] = np.mean(diff_prediction_errors_in_range)
        prediction_var_range[i] = np.var(diff_prediction_errors_in_range)
        cond_prediction_error_mean[i] = np.mean(prediction_errors_in_range)
        cond_prediction_error_var[i] = np.var(prediction_errors_in_range)
        # print(len(prediction_errors_in_range))
    plt.figure()
    plt.plot(error_ranges[1:], prediction_var_range)
    plt.plot(error_ranges[1:], prediction_mean_range)
    plt.xlabel('current prediction error')
    plt.ylabel(f'variance/mean of prediction error change after {n} steps')
    plt.legend(['variance of prediction error difference', 'mean of prediction error change'])
    plt.figure()
    plt.hist2d(prediction_errors_now, prediction_errors_diff, bins=[50,50], range=[[-5,5], [-3,3]])
    plt.figure()
    plt.plot(error_ranges[1:], cond_prediction_error_mean)
    plt.plot(error_ranges[1:], cond_prediction_error_mean + np.sqrt(cond_prediction_error_var), color='k')
    plt.plot(error_ranges[1:], cond_prediction_error_mean - np.sqrt(cond_prediction_error_var), color='k')
    plt.xlabel('prediction error')
    plt.ylabel(f'prediction error after {n} steps')
    plt.legend(['mean', '+- 1 std'])
    plt.show() """

    # Plot variance of error change over steps
    """ max_horizon = 20
    var = np.zeros(max_horizon)
    for s in range(max_horizon):
        prediction_errors = np.zeros(len(data_table['times_meas']) - max_horizon)
        for i in range(len(data_table['times_meas']) - max_horizon):
            prediction = get_prediction(data_table, time=data_table['times_meas'][i], steps=s)
            prediction_errors[i] = get_wind_value(data_table, time=data_table['times_meas'][i], steps=s) - prediction
        if s == 0:
            prediction_errors_now = prediction_errors
        var[s] = np.var(prediction_errors - prediction_errors_now)

    plt.figure()
    plt.plot(var)
    plt.xlabel('steps')
    plt.ylabel('variance of error change')
    plt.show() """

    # Plot error change over steps
    """ max_horizon = 50
    error_changes = np.array([])
    steps = np.array([])
    indices = np.array([])
    eval_horizon = len(data_table['times_meas']) - max_horizon
    percentiles = [2.5,5,10,25,50,75,90,95,97.5]
    percentiles_vec = [np.array([]) for i in range(len(percentiles))]
    for s in range(max_horizon):
        prediction_errors = np.zeros(eval_horizon)
        for i in range(eval_horizon):
            prediction = get_prediction(data_table, time=data_table['times_meas'][i], steps=s)
            prediction_errors[i] = get_wind_value(data_table, time=data_table['times_meas'][i], steps=s) - prediction
        if s == 0:
            prediction_errors_now = prediction_errors
        error_changes = np.append(error_changes, prediction_errors - prediction_errors_now)
        errors_sorted = np.sort(prediction_errors - prediction_errors_now)
        for p in range(len(percentiles)):
            i = int(percentiles[p]/100*len(errors_sorted))
            percentiles_vec[p] = np.append(percentiles_vec[p], errors_sorted[i])
        indices = np.append(indices, s*np.ones(len(prediction_errors)))
        
    plt.figure()
    plt.hist2d(indices, error_changes, bins=[max_horizon, 250])
    plt.xlabel('steps')
    plt.ylabel('change of error')
    plt.figure()
    plt.plot(range(max_horizon), percentiles_vec[0], percentiles_vec[-1]) # 95%
    plt.plot(range(max_horizon), percentiles_vec[1], percentiles_vec[-2]) # 90%
    plt.plot(range(max_horizon), percentiles_vec[2], percentiles_vec[-3]) # 80%
    plt.plot(range(max_horizon), percentiles_vec[3], percentiles_vec[-4]) # 50%
    plt.plot(range(max_horizon), percentiles_vec[4]) # mean%
    plt.xlabel('steps')
    plt.ylabel('change of error')
    plt.figure()
    plt.hist(error_changes[(max_horizon-1)*eval_horizon:], bins=250)
    plt.hist(error_changes[eval_horizon:2*eval_horizon], bins=250)
    plt.legend(['50 step prediction', '1 step prediction'])
    plt.xlabel('Change of prediction error')
    plt.show() """

    # Plot next error change over last error change

    # 3D plot current error and time difference vs future error
    """ n_range = np.arange(50)
    n_error_ranges = 20
    prediction_mean_range_mat = []
    prediction_var_range_mat = []
    for n in n_range:
        eval_horizon = len(data_table['times_meas']) - n
        predictions = np.zeros(eval_horizon)
        measured_values = np.zeros(eval_horizon)
        for i in range(eval_horizon):
            predictions[i] = get_prediction(data_table, time=data_table['times_meas'][i], steps=0)
            measured_values[i] = get_wind_value(data_table, time=data_table['times_meas'][i], steps=0)
        prediction_errors_now = predictions - measured_values
        for i in range(eval_horizon):
            predictions[i] = get_prediction(data_table, time=data_table['times_meas'][i], steps=n)
            measured_values[i] = get_wind_value(data_table, time=data_table['times_meas'][i], steps=n)
        prediction_errors = predictions - measured_values

        error_ranges = np.linspace(-4,4,n_error_ranges)
        prediction_var_range = np.zeros(len(error_ranges)-1)
        prediction_mean_range = np.zeros(len(error_ranges)-1)
        for i in range(len(error_ranges)-1):
            prediction_errors_in_range = (prediction_errors[np.logical_and(prediction_errors_now > error_ranges[i], prediction_errors_now < error_ranges[i+1])])
            prediction_mean_range[i] = np.mean(prediction_errors_in_range)
            prediction_var_range[i] = np.var(prediction_errors_in_range)

        prediction_mean_range_mat.append(prediction_mean_range)
        prediction_var_range_mat.append(prediction_var_range)
    
    prediction_mean_range_mat = np.array(prediction_mean_range_mat)
    prediction_var_range_mat = np.array(prediction_var_range_mat)
    prediction_mat_upper = prediction_mean_range_mat + np.sqrt(prediction_var_range_mat)
    prediction_mat_lower = prediction_mean_range_mat -  np.sqrt(prediction_var_range_mat)

    X, Y = np.meshgrid(error_ranges[1:], n_range)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf_mean = ax.plot_surface(X, Y, prediction_mean_range_mat)
    surf_lower = ax.plot_surface(X, Y, prediction_mat_lower)
    surf_upper = ax.plot_surface(X, Y, prediction_mat_upper)
    ax.set_xlabel('current prediction error')
    ax.set_ylabel('number of steps')
    ax.set_zlabel('prediction error in n steps')

    #plt.plot(error_ranges[1:], prediction_mean_range)
    #plt.plot(error_ranges[1:], prediction_mean_range + np.sqrt(prediction_var_range), color='k')
    #plt.plot(error_ranges[1:], prediction_mean_range - np.sqrt(prediction_var_range), color='k')
    #plt.xlabel('prediction error')
    #plt.ylabel(f'prediction error after {n} steps')
    #plt.legend(['mean', '+- 1 std'])
    plt.show() """

    #Plot prediction error for current wind speed and predicted wind speed in certain range
    """ n = 6
    # wind_measured_range = [2,3]
    wind_predicted_range = [2,3]
    # wind_predicted_range_now = [5,8]

    eval_horizon = len(data_table['times_meas']) - n
    predictions = np.zeros(eval_horizon)
    measured_values = np.zeros(eval_horizon)
    for i in range(eval_horizon):
        predictions[i] = get_prediction(data_table, time=data_table['times_meas'][i], steps=0)
        measured_values[i] = get_wind_value(data_table, time=data_table['times_meas'][i], steps=0)
    current_measurement = measured_values.copy()
    current_prediction = predictions.copy()
    for i in range(eval_horizon):
        predictions[i] = get_prediction(data_table, time=data_table['times_meas'][i], steps=n)
        measured_values[i] = get_wind_value(data_table, time=data_table['times_meas'][i], steps=n)
    prediction_errors = predictions - measured_values

    indices_select = [i for i in range(eval_horizon) if current_measurement[i] > wind_measured_range[0] 
                      and current_measurement[i] < wind_measured_range[1]
                      and predictions[i] > wind_predicted_range[0]
                      and predictions[i] < wind_predicted_range[1]
                      and current_prediction[i] > wind_predicted_range_now[0]
                      and current_prediction[i] < wind_predicted_range_now[1]]
    
    errors_select = prediction_errors[indices_select]

    plt.figure()
    plt.hist(errors_select, bins=50, range=[-10,5])
    plt.xlabel(f'prediction error for {n} steps with predictions in range {wind_predicted_range[0]}<x<{wind_predicted_range[1]} and current wind speed in range{wind_measured_range[0]}<x<{wind_measured_range[1]}')
    plt.show() """

    # Histograms for wind speed in certain range
    # times = dh.weather_data['times_meas']

    # predictions = np.array([dh.get_NWP(t) for t in times])
    # measurements = np.array([dh.get_measurement(t) for t in times])
    # np.savetxt('../Abbildungen/data_analysis/wind_predictions.csv', predictions)
    # np.savetxt('../Abbildungen/data_analysis/wind_measurements.csv', measurements)
    """ predictions = np.loadtxt('../Abbildungen/data_analysis/wind_predictions.csv')
    measurements = np.loadtxt('../Abbildungen/data_analysis/wind_measurements.csv')

    errors = measurements - predictions

    wind_predicted_range = (2,3)
    indices_in_range = [i for i, v in enumerate(predictions) 
                        if v > wind_predicted_range[0] and v <= wind_predicted_range[1]]
    errors_in_range = errors[indices_in_range]
    measurements_in_range = measurements[indices_in_range]

    bins = 50
    fig, axs = plt.subplots(2, layout='constrained')
    labels = []
    axs[0].hist(errors_in_range, bins=bins, density=True, histtype='step')
    axs[0].set_xlabel('Prediction error')
    axs[1].hist(measurements_in_range, bins=bins, density=True, histtype='step')
    axs[1].set_xlabel('Measured wind speed')
    labels.append(fr'''${wind_predicted_range[0]}\,\mathrm{"{"}m/s{"}"}\leq 
        v_\mathrm{"{"}wind,MF{"}"}\leq {wind_predicted_range[1]}\,\mathrm{"{"}m/s{"}"}$''')

    wind_predicted_range = (4,5)
    indices_in_range = [i for i, v in enumerate(predictions) 
                        if v > wind_predicted_range[0] and v <= wind_predicted_range[1]]
    errors_in_range = errors[indices_in_range]
    measurements_in_range = measurements[indices_in_range]

    axs[0].hist(errors_in_range, bins=bins, density=True, histtype='step')
    axs[0].set_xlabel('Prediction error')
    axs[1].hist(measurements_in_range, bins=bins, density=True, histtype='step')
    axs[1].set_xlabel('Measured wind speed')
    labels.append(fr'''${wind_predicted_range[0]}\,\mathrm{"{"}m/s{"}"}\leq v_\mathrm{"{"}wind,MF{"}"}\leq {wind_predicted_range[1]}\,\mathrm{"{"}m/s{"}"}$''')


    wind_predicted_range = (10,11)
    indices_in_range = [i for i, v in enumerate(predictions) 
                        if v > wind_predicted_range[0] and v <= wind_predicted_range[1]]
    errors_in_range = errors[indices_in_range]
    measurements_in_range = measurements[indices_in_range]
    axs[0].hist(errors_in_range, bins=bins, density=True, histtype='step')
    axs[0].set_xlabel('Prediction error')
    axs[1].hist(measurements_in_range, bins=bins, density=True, histtype='step')
    axs[1].set_xlabel('Measured wind speed')
    labels.append(fr'''${wind_predicted_range[0]}\,\mathrm{"{"}m/s{"}"}\leq v_\mathrm{"{"}wind,MF{"}"}\leq {wind_predicted_range[1]}\,\mathrm{"{"}m/s{"}"}$''')
    fig.legend(labels, loc='upper center', bbox_to_anchor = (.5, 1.1), ncol=3)

    plt.figure()
    x = np.arange(-10,10)
    cov = []
    for i in x:
        if i < 0:
            predictions_i = predictions[:i]
            measurements_i = measurements[-i:]
        elif i > 0:
            predictions_i = predictions[i:]
            measurements_i = measurements[:-i]
        else:
            predictions_i = predictions
            measurements_i = measurements
        cov.append(np.cov(predictions_i, measurements_i)[0,1])
    plt.plot(x, cov) """
            # [r'$2\,\mathrm{m/s}\leq v_\mathrm{wind,MF}\leq 3\,\mathrm{m/s}$', 
    #             r'$4\,\mathrm{m/s}\leq v_\mathrm{wind,MF}\leq 5\,\mathrm{m/s}$',
    #             r'$10\,\mathrm{m}/\mathrm{s}\leq v_\mathrm{wind,MF}\leq 11\,\mathrm{m}/\mathrm{s}$']

    #Plot error variance over CAPE or other variable
    """ start_time = datetime.datetime(2020,1,1)
    end_time = datetime.datetime(2022,12,31)
    weather_data = fi.load_weather_data(start_time, end_time)

    n_points = len(weather_data['times_meas'])

    measurements = np.zeros(n_points)
    predictions = np.zeros(n_points)
    errors = np.zeros(n_points)
    cape_pred = np.zeros(n_points)

    for i in range(n_points):
        t = weather_data['times_meas'][i]
        measurements[i] = ut.get_wind_value(weather_data, t, 0)
        predictions[i] = ut.get_NWP(weather_data, t, 0, 'wind_speed_10m')
        errors[i] = measurements[i] - predictions[i]
        cape = ut.get_NWP(weather_data, t, 0, 'sqrt_specific_convective_available_potential_energy')
        if cape < 1e10:
            cape_pred[i] = cape
    
    cape_sorted = np.sort(cape_pred)
    errors_sorted = errors[np.argsort(cape_pred)]

    bins = 20
    var_errors_in_range = np.zeros(bins)
    mean_errors_in_range = np.zeros(bins)
    cape_range_mean = np.zeros(bins)
    for i in range(bins):
        index_1 = int(n_points*i/bins)
        index_2 = int(n_points*(i+1)/bins)
        cape_range_mean[i] = np.mean(cape_sorted[index_1:index_2])
        var_errors_in_range[i] = np.var(errors_sorted[index_1:index_2])
        mean_errors_in_range[i] = np.mean(errors_sorted[index_1:index_2])


    plt.figure()
    plt.scatter(cape_pred, errors)
    plt.xlabel('sqrt(CAPE)')
    plt.ylabel('prediction error')
    plt.figure()
    plt.plot(cape_range_mean, mean_errors_in_range)
    plt.plot(cape_range_mean, np.sqrt(var_errors_in_range))
    plt.legend(['mean error', 'standard deviation'])
    plt.xlabel('sqrt(CAPE)')
    plt.ylabel('Error Variance')
    plt.show() """

    #Plot error variance over CIN
    """ start_time = datetime.datetime(2020,1,1)
    end_time = datetime.datetime(2020,1,31)
    weather_data = fi.load_weather_data(start_time, end_time)

    n_points = len(weather_data['times_meas'])

    measurements = np.zeros(n_points)
    predictions = np.zeros(n_points)
    errors = np.zeros(n_points)
    cin_pred = np.zeros(n_points)

    for i in range(n_points):
        t = weather_data['times_meas'][i]
        measurements[i] = ut.get_wind_value(weather_data, t, 0)
        predictions[i] = ut.get_NWP(weather_data, t, 0, 'wind_speed_10m')
        errors[i] = measurements[i] - predictions[i]
        cin_pred[i] = ut.get_NWP(weather_data, t, 0, "atmosphere_convective_inhibition")
    
    cin_sorted = np.sort(cin_pred)
    errors_sorted = errors[np.argsort(cin_pred)]

    bins = 20
    var_errors_in_range = np.zeros(bins)
    mean_errors_in_range = np.zeros(bins)
    cin_range_mean = np.zeros(bins)
    for i in range(bins):
        index_1 = int(n_points*i/bins)
        index_2 = int(n_points*(i+1)/bins)
        cin_range_mean[i] = np.mean(cin_sorted[index_1:index_2])
        var_errors_in_range[i] = np.var(errors_sorted[index_1:index_2])
        mean_errors_in_range[i] = np.mean(errors_sorted[index_1:index_2]) 


    plt.figure()
    plt.scatter(cin_pred, errors)
    plt.xlabel('CIN')
    plt.ylabel('prediction error')
    plt.figure()
    plt.plot(cin_range_mean, var_errors_in_range)
    plt.plot(cin_range_mean, mean_errors_in_range)
    plt.xlabel('CIN')
    plt.ylabel('Error Variance')
    plt.show()"""

    # plot error over air pressure

    start_time = datetime.datetime(2020,1,1)
    end_time = datetime.datetime(2022,12,31)
    weather_data = fi.load_weather_data(start_time, end_time)

    # 2020/21
    indices = [i for i, t in enumerate(weather_data['times_meas']) if t.year<2022]
    n_points = len(indices)

    measurements = np.zeros(n_points)
    predictions = np.zeros(n_points)
    errors = np.zeros(n_points)
    p_pred = np.zeros(n_points)

    for i in indices:
        t = weather_data['times_meas'][i]
        measurements[i] = ut.get_wind_value(weather_data, t, 0)
        predictions[i] = ut.get_NWP(weather_data, t, 0, 'wind_speed_10m')
        errors[i] = measurements[i] - predictions[i]
        p_pred[i] = ut.get_NWP(weather_data, t, 0, "air_pressure_at_sea_level")
    
    p_sorted = np.sort(p_pred)
    errors_sorted = errors[np.argsort(p_pred)]

    bins = 10
    var_errors_in_range = np.zeros(bins)
    mean_errors_in_range = np.zeros(bins)
    p_range_mean = np.zeros(bins)
    for i in range(bins):
        index_1 = int(n_points*i/bins)
        index_2 = int(n_points*(i+1)/bins)
        p_range_mean[i] = np.mean(p_sorted[index_1:index_2])
        var_errors_in_range[i] = np.var(errors_sorted[index_1:index_2])
        mean_errors_in_range[i] = np.mean(errors_sorted[index_1:index_2])


    plt.figure()
    plt.scatter(p_pred, errors)
    plt.xlabel('air pressure')
    plt.ylabel('prediction error')
    plt.figure()
    plt.plot(p_range_mean, mean_errors_in_range)
    plt.plot(p_range_mean, np.sqrt(var_errors_in_range))
    plt.legend(['mean error', 'standard deviation'])
    plt.xlabel('air pressure')
    plt.ylabel('prediction error')

    # 2022
    indices = [i for i, t in enumerate(weather_data['times_meas']) if t.year==2022]
    n_points = len(indices)

    measurements = np.zeros(n_points)
    predictions = np.zeros(n_points)
    errors = np.zeros(n_points)
    p_pred = np.zeros(n_points)

    for i, k in enumerate(indices):
        t = weather_data['times_meas'][k]
        measurements[i] = ut.get_wind_value(weather_data, t, 0)
        predictions[i] = ut.get_NWP(weather_data, t, 0, 'wind_speed_10m')
        errors[i] = measurements[i] - predictions[i]
        p_pred[i] = ut.get_NWP(weather_data, t, 0, "air_pressure_at_sea_level")
    
    p_sorted = np.sort(p_pred)
    errors_sorted = errors[np.argsort(p_pred)]

    bins = 10
    var_errors_in_range = np.zeros(bins)
    mean_errors_in_range = np.zeros(bins)
    p_range_mean = np.zeros(bins)
    for i in range(bins):
        index_1 = int(n_points*i/bins)
        index_2 = int(n_points*(i+1)/bins)
        p_range_mean[i] = np.mean(p_sorted[index_1:index_2])
        var_errors_in_range[i] = np.var(errors_sorted[index_1:index_2])
        mean_errors_in_range[i] = np.mean(errors_sorted[index_1:index_2])


    plt.figure()
    plt.scatter(p_pred, errors)
    plt.xlabel('air pressure')
    plt.ylabel('prediction error')
    plt.figure()
    plt.plot(p_range_mean, mean_errors_in_range)
    plt.plot(p_range_mean, np.sqrt(var_errors_in_range))
    plt.legend(['mean error', 'standard deviation'])
    plt.xlabel('air pressure')
    plt.ylabel('prediction error')
    plt.show()

    # plot error over temperature
    """ start_time = datetime.datetime(2020,1,1)
    end_time = datetime.datetime(2022,12,31)
    weather_data = fi.load_weather_data(start_time, end_time)

    n_points = len(weather_data['times_meas'])
    n_select = n_points
    selected_points = random.sample(range(n_points), n_select)

    measurements = np.zeros(n_select)
    predictions = np.zeros(n_select)
    errors = np.zeros(n_select)
    nwp_val = np.zeros(n_select)

    for i, k in enumerate(selected_points):
        t = weather_data['times_meas'][k]
        measurements[i] = ut.get_wind_value(weather_data, t, 0)
        predictions[i] = ut.get_NWP(weather_data, t, 0, 'wind_speed_10m')
        errors[i] = measurements[i] - predictions[i]
        nwp_val[i] = (ut.get_NWP(weather_data, t, 0, "wind_speed_of_gust"))#-predictions[i])#/(predictions[i]+1e-6)
    
    nwp_val_sorted = np.sort(nwp_val)
    errors_sorted = errors[np.argsort(nwp_val)]
    predictions_sorted = predictions[np.argsort(nwp_val)]

    bins = 10
    var_errors_in_range = np.zeros(bins)
    mean_errors_in_range = np.zeros(bins)
    nwp_range_mean = np.zeros(bins)
    for i in range(bins):
        index_1 = int(n_select*i/bins)
        index_2 = int(n_select*(i+1)/bins)
        nwp_range_mean[i] = np.mean(nwp_val_sorted[index_1:index_2-1])
        var_errors_in_range[i] = np.var(errors_sorted[index_1:index_2-1])
        mean_errors_in_range[i] = np.mean(errors_sorted[index_1:index_2-1])


    plt.figure()
    plt.scatter(nwp_val, errors)
    plt.xlabel('air temperature')
    plt.ylabel('prediction error')
    plt.figure()
    plt.plot(nwp_range_mean, mean_errors_in_range)
    plt.plot(nwp_range_mean, np.sqrt(var_errors_in_range))
    plt.legend(['mean error', 'standard deviation'])
    plt.xlabel('Wind speed')
    plt.ylabel('Prediction error') """
    # histograms
    # all wind speeds
    """ plt.figure()
    legends = []
    histogram_data = []
    for i in range(bins):
        index_1 = int(n_select*i/bins)
        index_2 = int(n_select*(i+1)/bins)
        errors_i = errors_sorted[index_1:index_2-1]
        nwp_val_lb = nwp_val_sorted[index_1]
        nwp_val_ub = nwp_val_sorted[index_2-1]
        legends.append(f'Pressure between {int(nwp_val_lb/100)} and {int(nwp_val_ub/100)} hPa')
        histogram_data.append(errors_i)
        #plt.hist(errors_i, bins=20, density=True)
    plt.hist(histogram_data, bins=[
        -10,-8,-7,-6,-5,-4.5,-4,-3.5,-3,-2.75,-2.5,-2.25,-2,-1.8,-1.6,-1.4,-1.2,-1,-.8,-.6,-.4,-.2,
        0,0.2,.4,.6,.8,1,1.2,1.4,1.6,1.8,2,2.25,2.5,2.75,3,3.5,4,4.5,5,6,7,8,10], density=True)
    plt.xlabel('Prediction error')
    plt.legend(legends)
    # Wind speeds in certain range
    indices_sorted = np.argsort(nwp_val)
    wind_speed_sorted = predictions[indices_sorted]
    wind_speed_lb = 8
    wind_speed_ub = 10
    indices_in_range = [i for i in indices_sorted 
        if wind_speed_sorted[i]>wind_speed_lb and wind_speed_sorted[i] < wind_speed_ub]
    n_values = len(indices_in_range)
    measurements_in_range = measurements[indices_in_range]
    nwp_val_in_range = nwp_val_sorted[indices_in_range]
    plt.figure()
    histogram_data = []
    legends = []
    for i in range(bins):
        index_1 = int(n_values*i/bins)
        index_2 = int(n_values*(i+1)/bins)
        measurements_i = measurements_in_range[index_1:index_2-1]
        nwp_val_lb = nwp_val_in_range[index_1]
        nwp_val_ub = nwp_val_in_range[index_2-1]
        legends.append(f'Pressure between {int(nwp_val_lb/100)} and {int(nwp_val_ub/100)} hPa')
        histogram_data.append(measurements_i)
        # plt.hist(measurements_i, bins=20, density=True)
    plt.hist(histogram_data, bins=20, density=True)
    plt.xlabel('Wind speed')
    plt.legend(legends)
    """

    # Plot error over month/hour
    """ start_time = datetime.datetime(2020,1,1)
    end_time = datetime.datetime(2022,12,31)
    weather_data = fi.load_weather_data(start_time, end_time)

    n_points = len(weather_data['times_meas'])
    n_select = n_points
    selected_points = random.sample(range(n_points), n_select)

    measurements = np.zeros(n_select)
    predictions = np.zeros(n_select)
    errors = np.zeros(n_select)
    t_vec = np.zeros(n_select)

    for i, k in enumerate(selected_points):
        t = weather_data['times_meas'][k]
        measurements[i] = dh.get_measurement(t, 0)
        predictions[i] = dh.get_NWP(t, 0, 'wind_speed_10m')
        errors[i] = measurements[i] - predictions[i]
        t_vec[i] = t.hour
    
    bins = 24
    var_errors_in_range = np.zeros(bins)
    mean_errors_in_range = np.zeros(bins)
    for i in range(bins):
        var_errors_in_range[i] = np.var([error for k, error in enumerate(errors) if t_vec[k]==i])
        mean_errors_in_range[i] = np.mean([error for k, error in enumerate(errors) if t_vec[k]==i])


    plt.figure()
    plt.scatter(t_vec, errors)
    plt.xlabel('air temperature')
    plt.ylabel('prediction error')
    plt.figure()
    plt.plot(range(bins), mean_errors_in_range)
    plt.plot(range(bins), np.sqrt(var_errors_in_range))
    plt.legend(['mean error', 'standard deviation'])
    plt.xlabel('Wind speed')
    plt.ylabel('Prediction error')
    plt.show()
    np.savetxt('../Abbildungen/data_analysis/error_hour_mean.csv', mean_errors_in_range)
    np.savetxt('../Abbildungen/data_analysis/error_hour_std.csv', np.sqrt(var_errors_in_range))
    np.savetxt('../Abbildungen/data_analysis/error_hour_x.csv', np.array(range(bins))) """
    
    # Plot error over time since prediction
    """ start_time = datetime.datetime(2020,1,1)
    end_time = datetime.datetime(2022,12,31)

    n_hours = 15
    n_steps = n_hours*6
    n_values = len(dh.weather_data['times1_sh'])//6

    # dates = [start_time+datetime.timedelta(days=i) for i in range(n_values//4)]
    # times = [t + i*datetime.timedelta(hours=6) for t in dates for i in range(4)]
    # times1 = [start_time+i*datetime.timedelta(hours=6) for i in range(n_values)]

    # measurements = np.zeros((n_steps, n_values))
    # predictions = np.zeros((n_steps, n_values))
    # errors = np.zeros((n_steps, n_values))
    # dt_mat = np.zeros((n_steps, n_values))

    # for i, t in enumerate(times):
    #     for k in range(n_steps):
    #         measurements[k,i] = dh.get_measurement(t, k)
    #         predictions[k,i] = dh.get_NWP(t, k)
    #         errors[k,i] = measurements[k,i] - predictions[k,i]
    #         dt_mat[k,i] = dh.get_time_since_forecast(t, k)
    errors = np.loadtxt('../Abbildungen/data_analysis/error_steps.csv')
    means = np.zeros(n_steps)
    vars = np.zeros(n_steps)
    hours = np.zeros(n_steps)
    for k in range(n_steps):
        means[k] = np.mean(errors[k,:])
        vars[k] = np.var(errors[k,:])

    plt.figure()
    plt.plot(np.arange(15,step=1/6), means)
    plt.plot(np.arange(15,step=1/6), np.sqrt(vars))
    plt.legend(['mean error', 'standard deviation'])
    plt.xlabel('Wind speed')
    plt.ylabel('Prediction error')
    plt.show() """

    # plot covariance over time for different nwp values
    """ steps = 36
    n_points = len(dh.weather_data['times_meas'])
    times = dh.weather_data['times_meas']
    nwp_val = np.zeros(n_points)
    # prediction_errors = np.zeros(n_points)
    # nwp_val = np.array([dh.get_NWP(t, key='sqrt_specific_convective_available_potential_energy') 
    #                     for t in times])
    # for i, t in enumerate(times):
    #     nwp_val[i] = dh.get_NWP(t, key='sqrt_specific_convective_available_potential_energy')
    #     prediction = dh.get_NWP(t)
    #     measurement = dh.get_measurement(t)
    #     prediction_errors[i] = measurement - prediction

    # np.savetxt('../Abbildungen/data_analysis/prediction_errors.csv', prediction_errors)
    # np.savetxt('../Abbildungen/data_analysis/cape.csv', nwp_val) 
    n_points = len(dh.weather_data['times_meas']) - steps
    prediction_errors = np.loadtxt('../Abbildungen/data_analysis/prediction_errors.csv')
    nwp_val = np.loadtxt('../Abbildungen/data_analysis/cape.csv')
    indices_sort = np.argsort(nwp_val[:-steps])
    nwp_val_sorted = np.sort(nwp_val[:-steps])
    bins = 3

    cov = np.zeros((bins, steps))
    labels = []
    for k in range(bins):
        i_lower = int(k/bins*n_points)
        i_upper = int((k+1)/bins*n_points)-1
        indices_k = indices_sort[:-steps][i_lower:i_upper]
        labels.append(f'{int(nwp_val_sorted[i_lower])}<=cape<={int(nwp_val_sorted[i_upper])}')
        for s in range(steps):
            error_now = prediction_errors[indices_k]
            error_s = prediction_errors[indices_k+s]
            cov[k,s] = np.cov(error_now, error_s)[1,0]
        cov[k,:] = cov[k,:]/cov[k,0]

    np.savetxt('../Abbildungen/data_analysis/cov_over_steps_cape.csv', cov)
    #cov_list = [cov[k,:] for k in range(bins)]
    x = np.array([np.arange(steps)/6]*bins)
    plt.figure()
    plt.plot(x.T, cov.T, label=labels)
    plt.xlabel('Time difference (h)')
    plt.ylabel('Error correlation')
    plt.legend() """



    plt.show()

    1 == 1
    pass
    pass

    # Dated code
    """
    times_sh, wind_forecast_sh, times_mh, wind_forecast_mh = load_prediction(start_time, end_time)[:4]
    times_obs, wind_obs = load_weather_data(start_time, end_time)
    times_interp = np.array([times_obs[0]+i*np.timedelta64(10, 'm') for i in range(int((times_obs[-1].astype(float)-times_obs[0].astype(float))//600000))])
    #wind_data_interp, times_interp = interpolate_wind_prediction(wind_forecast_sh, times_sh, 6)
    interpolator_sh = scp.interpolate.CubicSpline((times_sh.astype(float)-times_sh[0].astype(float))/(times_sh[-1].astype(float)-times_sh[0].astype(float)), wind_forecast_sh)
    interpolator_mh = scp.interpolate.CubicSpline((times_mh.astype(float)-times_mh[0].astype(float))/(times_mh[-1].astype(float)-times_mh[0].astype(float)), wind_forecast_mh)
    wind_interp_sh = interpolator_sh((times_interp.astype(float)-times_interp[0].astype(float))/(times_interp[-1].astype(float)-times_interp[0].astype(float)))
    wind_interp_mh = 0.8279*np.interp((times_interp.astype(float)-times_interp[0].astype(float))/(times_interp[-1].astype(float)-times_interp[0].astype(float)), (times_mh.astype(float)-times_mh[0].astype(float))/(times_mh[-1].astype(float)-times_mh[0].astype(float)), wind_forecast_mh)
    wind_interp_sh_lin = 0.8271*np.interp((times_interp.astype(float)-times_interp[0].astype(float))/(times_interp[-1].astype(float)-times_interp[0].astype(float)), (times_sh.astype(float)-times_sh[0].astype(float))/(times_sh[-1].astype(float)-times_sh[0].astype(float)), wind_forecast_sh)
    wind_obs_interp = np.interp((times_interp.astype(float)-times_interp[0].astype(float))/(times_interp[-1].astype(float)-times_interp[0].astype(float)), (times_obs.astype(float)-times_obs[0].astype(float))/(times_obs[-1].astype(float)-times_obs[0].astype(float)), wind_obs)
    #plt.plot(times_sh, wind_forecast_sh, color='b')
    #plt.plot(times_mh, wind_forecast_mh, color='r')
    
    prediction_err_sh = wind_obs_interp-wind_interp_sh
    prediction_err_mh = wind_obs_interp-wind_interp_mh
    prediction_err_sh_lin = wind_obs_interp-wind_interp_sh_lin

    plt.figure()
    #plt.plot(times_interp, wind_interp_sh, color='b')
    plt.plot(times_interp, wind_interp_sh_lin, color='r')
    plt.plot(times_interp, wind_obs_interp, color='g')

    bins = np.arange(-20, 20, 0.1)
    plt.figure()
    plt.hist(prediction_err_sh_lin, bins)
    plt.hist(prediction_err_mh, bins)
    #plt.hist(prediction_err_sh, bins)
    plt.legend(['0-6 h horizon', '6-12 h horizon'])
    plt.figure()
    #plt.plot(times_interp, prediction_err_sh)
    #plt.plot(times_interp, prediction_err_mh)
    plt.plot(times_interp, prediction_err_sh_lin)
    # plt.show()
    indices = np.zeros(len(times_interp))
    errors_min_after_nwp = [np.array([]) for i in range(36)]
    for i in range(len(times_interp)):
        index = int((times_interp[i]-times_interp[0]).astype(np.uint64)//600000)%36
        errors_min_after_nwp[index] = np.append(errors_min_after_nwp[index], prediction_err_sh_lin[i])
        indices[i] = index
    
    error_mean = [np.mean(error) for error in errors_min_after_nwp]
    error_var = [np.var(error) for error in errors_min_after_nwp]
    plt.figure()
    plt.plot(error_mean)
    plt.figure()
    plt.plot(error_var)
    plt.figure()
    plt.imshow(np.cov([err[:4000] for err in errors_min_after_nwp]))# -np.diag([np.var(e) for e in errors_min_after_nwp]))
    plt.figure()
    plt.hist(errors_min_after_nwp[1][:4000]-errors_min_after_nwp[0][:4000],200)
    plt.hist(errors_min_after_nwp[19][:4000]-errors_min_after_nwp[18][:4000],200)
    plt.hist(errors_min_after_nwp[19][:4000]-errors_min_after_nwp[0][:4000],200)
    plt.legend(['0:10,0:0','3:10,3:00','3:10,0:00'])

    plt.figure()
    plt.plot(np.array([np.cov(prediction_err_sh_lin[:-(i+1)],prediction_err_sh_lin[i:-1])[0,1] for i in range(2000)]))

    errordiff = prediction_err_sh_lin[1:]-prediction_err_sh_lin[:-1]
    # autocorr = np.correlate(errordiff, errordiff, mode='full')
    plt.figure()
    plt.plot(times_interp[:-1],errordiff)
    plt.figure()
    # plt.plot(autocorr[len(autocorr)//2:(len(autocorr)//2+20)])

    plt.plot(np.abs(np.fft.fft(errordiff)))
    plt.figure()
    x = np.linspace(0,25,1000)
    plt.plot(wind_interp_sh_lin, wind_obs_interp, 'x')
    plt.plot(x, x)
    median_ratio = np.median(wind_obs_interp/wind_interp_sh_lin)
    std_ratio = np.sqrt(np.var(wind_obs_interp/wind_interp_sh_lin))
    fit_indices = [i for i in range(len(wind_obs_interp)) if np.abs(wind_obs_interp[i]/wind_interp_sh_lin[i] - median_ratio) < 1*std_ratio] # remove outliers
    a = np.sqrt(np.cov(np.square(wind_interp_sh_lin[fit_indices]), np.square(wind_obs_interp[fit_indices]))[1,0]/np.var(np.square(wind_interp_sh_lin[fit_indices])))
    #coeffs = np.polyfit(wind_interp_sh_lin[fit_indices], wind_obs_interp[fit_indices], 1, w=np.square(wind_obs_interp[fit_indices]))
    plt.plot(x, a*x)
    #plt.plot(x, 0.9*x)
    #plt.plot(x, coeffs[0]*x+coeffs[1])
    plt.xlabel('Predicted wind speed')
    plt.ylabel('Measured Wind speed')
    plt.show()
    """