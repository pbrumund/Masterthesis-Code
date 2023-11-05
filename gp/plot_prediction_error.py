import numpy as np
import matplotlib.pyplot as plt
import datetime
import csv
import scipy as scp
import typing
from mpl_toolkits.mplot3d import Axes3D

def load_prediction(start_time, end_time):
    filename = start_time.strftime("%Y%m%d") + "-" + end_time.strftime("%Y%m%d") + ".csv"
    with open(filename, mode="r") as f:
        reader = csv.reader(f, delimiter=';')
        table = np.array(list(reader))
        times_sh = table[:,0].astype(float)
        times_sh = np.asarray(times_sh, dtype='datetime64[s]')
        wind_forecast_sh = table[:,1].astype(float)
        times_mh = table[:,2].astype(float)
        times_mh = np.asarray(times_mh, dtype='datetime64[s]')
        wind_forecast_mh = table[:,3].astype(float)
        times_lh = table[:,4].astype(float)
        times_lh = np.asarray(times_mh, dtype='datetime64[s]')
        wind_forecast_lh = table[:,5].astype(float)
    return times_sh, wind_forecast_sh, times_mh, wind_forecast_mh, times_lh, wind_forecast_lh

def load_table(start_time, end_time):
    times_sh, wind_forecast_sh, times_mh, wind_forecast_mh, times_lh, wind_forecast_lh = load_prediction(start_time, end_time)
    times_obs, wind_speed_obs = load_weather_data(start_time, end_time)
    c_correction = np.mean(wind_speed_obs)/np.mean(wind_forecast_sh)
    times_interp = np.array([times_obs[0]+i*np.timedelta64(10, 'm') for i in range(int((times_obs[-1].astype(float)-times_obs[0].astype(float))//600000))])
    times_interp_datetime = np.array([times_obs[0].astype(datetime.datetime)+i*datetime.timedelta(minutes=10) for i in range(int((times_obs[-1].astype(float)-times_obs[0].astype(float))//600000))])
    #wind_interp_sh = c_correction*np.interp((times_interp.astype(float)-times_interp[0].astype(float))/(times_interp[-1].astype(float)-times_interp[0].astype(float)), (times_sh.astype(float)-times_sh[0].astype(float))/(times_sh[-1].astype(float)-times_sh[0].astype(float)), wind_forecast_sh)
    #wind_interp_mh = c_correction*np.interp((times_interp.astype(float)-times_interp[0].astype(float))/(times_interp[-1].astype(float)-times_interp[0].astype(float)), (times_mh.astype(float)-times_mh[0].astype(float))/(times_mh[-1].astype(float)-times_mh[0].astype(float)), wind_forecast_mh)
    #wind_interp_lh = c_correction*np.interp((times_interp.astype(float)-times_interp[0].astype(float))/(times_interp[-1].astype(float)-times_interp[0].astype(float)), (times_lh.astype(float)-times_lh[0].astype(float))/(times_lh[-1].astype(float)-times_lh[0].astype(float)), wind_forecast_lh)
    wind_obs_interp = np.interp((times_interp.astype(float)-times_interp[0].astype(float))/(times_interp[-1].astype(float)-times_interp[0].astype(float)), (times_obs.astype(float)-times_obs[0].astype(float))/(times_obs[-1].astype(float)-times_obs[0].astype(float)), wind_speed_obs)
    data = {'times_sh': times_sh,
            'pred_sh':  c_correction*wind_forecast_sh,
            'times_mh': times_mh,
            'pred_mh':  c_correction*wind_forecast_mh,
            'times_lh': times_lh,
            'pred_lh':  c_correction*wind_forecast_lh,
            'times_meas': times_interp_datetime,
            'meas':     wind_obs_interp}
    return data

def get_prediction(wind_table, time, steps):
    # r = (time.minute % 360)//10 #indices since last forecast
    r = time.hour % 6
    dt = time - wind_table['times_sh'][0].astype(datetime.datetime)
    i_start = int(dt.total_seconds()//21600) * 6 #6 hours, time of last released forecast
    predicted_trajectory = np.concatenate([wind_table['pred_sh'][i_start:i_start+6], wind_table['pred_mh'][i_start:i_start+6], wind_table['pred_lh'][i_start:]])
    prediction_times = np.concatenate([wind_table['times_sh'][i_start:i_start+6], wind_table['times_mh'][i_start:i_start+6], wind_table['times_lh'][i_start:]])
    times_obs = wind_table['times_meas']
    i = (6*r + time.minute//10 + steps)//6
    wind_interp = predicted_trajectory[i+1]*(time+steps*datetime.timedelta(minutes=10)).minute/60 + predicted_trajectory[i]*(1-(time+steps*datetime.timedelta(minutes=10)).minute/60)
    # wind_interp_np = np.interp((time+steps*datetime.timedelta(minutes=10)-times_obs[0]).total_seconds(), (prediction_times-times_obs.astype(np.datetime64)[0]).astype(float)//1000000, predicted_trajectory)
    # print(wind_interp-wind_interp_np)
    # print(wind_interp_np)
    return wind_interp

def get_wind_value(wind_table, time, steps):
    dt = time - wind_table['times_meas'][0]
    i = int(dt.total_seconds()/600) + steps
    return wind_table['meas'][i]

def load_weather_data(start_time, end_time):
    filename = start_time.strftime("%Y%m%d") + "-" + end_time.strftime("%Y%m%d") + "_observations.csv"
    with open(filename, mode="r") as f:
        reader = csv.reader(f, delimiter=';')
        table = np.array(list(reader))
        times = table[:,0].astype(np.datetime64)
        # times_sh = np.asarray(times_sh, dtype='datetime64[s]')
        wind_data = table[:,1].astype(float)
    return times, wind_data


if __name__ == "__main__":
    # Data loading
    start_time = datetime.datetime(2020,1,1)
    end_time = datetime.datetime(2022,12,31)

    data_table = load_table(start_time,end_time)

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
    """ max_horizon = 100
    cov = np.zeros(max_horizon)
    for s in range(max_horizon):
        prediction_errors = np.zeros(len(data_table['times_meas']) - max_horizon)
        for i in range(len(data_table['times_meas']) - max_horizon):
            prediction = get_prediction(data_table, time=data_table['times_meas'][i], steps=s)
            prediction_errors[i] = get_wind_value(data_table, time=data_table['times_meas'][i], steps=s) - prediction
        if s == 0:
            prediction_errors_now = prediction_errors
        cov[s] = np.cov(prediction_errors, prediction_errors_now)[1,0]

    plt.figure()
    plt.plot(cov)
    plt.xlabel('steps')
    plt.ylabel('covariance')
    plt.show() """

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
    for i in range(eval_horizon):
        current_prediction = get_prediction(data_table, time=data_table['times_meas'][i], steps=0)
        current_error[i] = get_wind_value(data_table, time=data_table['times_meas'][i], steps=0) - current_prediction
        prediction_nsteps = get_prediction(data_table, time=data_table['times_meas'][i], steps=n)
        prediction_error_nsteps[i] = get_wind_value(data_table, time=data_table['times_meas'][i], steps=n) - prediction_nsteps
    
    plt.figure()
    # plt.scatter(current_error, prediction_error_nsteps)
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
    n = 6
    wind_measured_range = [5,8]
    wind_predicted_range = [8,12]
    wind_predicted_range_now = [5,8]

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
    plt.show()




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