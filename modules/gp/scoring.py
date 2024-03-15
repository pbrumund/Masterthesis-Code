import datetime

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

from .utils import get_NWP, get_wind_value, generate_features
from .data_handling import DataHandler
from .fileloading import load_weather_data
from .gp_timeseries_model import TimeseriesModel
from .gp_timeseries_model_homoscedastic import HomoscedasticTimeseriesModel
from .gp_direct_model import DirectGPEnsemble
from .gp_simple_timeseries_model import SimpleTimeseriesModel

def get_rmse(trajectory_measured, trajectory_predicted):
    return np.sqrt(np.mean(np.square(trajectory_measured-trajectory_predicted)))

def get_mae(trajectory_measured, trajectory_predicted):
    return np.mean(np.abs(trajectory_measured-trajectory_predicted))

def get_mape(trajectory_measured, trajectory_predicted):
    return np.mean(np.abs((trajectory_measured-trajectory_predicted)/(trajectory_measured+1e-6)))

def get_RE(alpha, trajectory_measured, trajectory_predicted, var_predicted):
    # From Kou paper
    a = norm.ppf(1-0.5*alpha) # Number of ± standard deviations for 1-alpha-interval
    errors = trajectory_measured - trajectory_predicted
    i_in_interval = [i for i, error in enumerate(errors) if np.abs(error)<a*(np.sqrt(var_predicted[i]))]
    p_in_interval = len(i_in_interval)/len(trajectory_predicted)
    return p_in_interval - (1-alpha)

def get_interval_score(alpha, trajectory_measured, trajectory_predicted, var_predicted):
    # Tilmann Gneiting & Adrian E Raftery (2007) 
    # Strictly Proper Scoring Rules, Prediction, and Estimation, 
    # Journal of the American Statistical Association, 102:477, 359-378, 
    # DOI: 10.1198/016214506000001437
    std_pred = np.sqrt(var_predicted)
    a = norm.ppf(1-0.5*alpha)   # Number of ± standard deviations for 1-alpha-interval
    l = trajectory_predicted - a*std_pred   # alpha/2 quantile
    u = trajectory_predicted + a*std_pred   # 1-alpha/2 quantile
    
    interval_width = u-l # first term
    err_under = np.array([l[i] - x if x < l[i] else 0 for i, x in enumerate(trajectory_measured)])   # second term
    err_over = np.array([x - u[i] if x > u[i] else 0 for i, x in enumerate(trajectory_measured)])    # third term

    S_alpha_int = interval_width + 2/alpha*err_under + 2/alpha*err_over
    return np.mean(S_alpha_int)

def get_nlpd(trajectory_measured, trajectory_predicted, var_predicted):
    std_predicted = np.sqrt(var_predicted)
    z = (trajectory_measured - trajectory_predicted)/std_predicted
    log_density = norm.logpdf(z)
    return np.sum(-log_density)

def get_trajectory_measured(weather_data, opt):
    t_start = opt['t_start_score']
    t_end = opt['t_end_score']
    dt = t_end-t_start
    n_steps = int(dt.total_seconds()/600)
    times = [t_start+i*datetime.timedelta(minutes=10) for i in range(n_steps)]
    meas_traj = np.zeros(n_steps)
    for i, t in enumerate(times):
        meas_traj[i] = get_wind_value(weather_data, t, 0)
    return meas_traj

def get_trajectory_nwp(weather_data, opt):
    t_start = opt['t_start_score']
    t_end = opt['t_end_score']
    dt = t_end-t_start
    n_steps = int(dt.total_seconds()/600)
    times = [t_start+i*datetime.timedelta(minutes=10) for i in range(n_steps)]
    meas_traj = np.zeros(n_steps)
    for i, t in enumerate(times):
        meas_traj[i] = get_NWP(weather_data, t, 0)
    return meas_traj

def get_trajectory_gp_prior(opt):
    gp = TimeseriesModel(opt)
    dh = gp.data_handler
    t_start = opt['t_start_score']
    t_end = opt['t_end_score']
    dt = t_end-t_start
    n_steps = int(dt.total_seconds()/600)
    times = [t_start+i*datetime.timedelta(minutes=10) for i in range(n_steps)]

    NWP_traj = np.zeros(n_steps)
    mean_traj = np.zeros(n_steps)
    var_traj = np.zeros(n_steps)
    for i, t in enumerate(times):
            NWP_traj[i] = dh.get_NWP(t)
            x = dh.generate_features(t, feature='nwp & time', steps_ahead=0).reshape((1,-1))
            mean, var = gp.gp_prior.compiled_predict_y(x)
            mean_traj[i] = mean
            var_traj[i] = var
            if (i+1)%int(n_steps/20)==0:
                    print(f'{int((i+1)/int(n_steps/20)*5)}% done')
    gp_pred_traj = NWP_traj + mean_traj
    return gp_pred_traj, var_traj
    
def get_trajectory_gp_prior_homoscedastic(opt):
    gp = HomoscedasticTimeseriesModel(opt)
    dh = gp.data_handler
    t_start = opt['t_start_score']
    t_end = opt['t_end_score']
    dt = t_end-t_start
    n_steps = int(dt.total_seconds()/600)
    times = [t_start+i*datetime.timedelta(minutes=10) for i in range(n_steps)]

    NWP_traj = np.zeros(n_steps)
    mean_traj = np.zeros(n_steps)
    var_traj = np.zeros(n_steps)
    for i, t in enumerate(times):
            NWP_traj[i] = dh.get_NWP(t)
            x = dh.generate_features(t, feature='nwp & time', steps_ahead=0).reshape((1,-1))
            mean, var = gp.gp_prior.compiled_predict_y(x)
            mean_traj[i] = mean
            var_traj[i] = var
            if (i+1)%int(n_steps/20)==0:
                    print(f'{int((i+1)/int(n_steps/20)*5)}% done')
    gp_pred_traj = NWP_traj + mean_traj
    return gp_pred_traj, var_traj

def get_posterior_trajectories(opt):
    try:
        trajectories_mean = np.loadtxt(f'modules/gp/scoring/trajectories_mean_post_{opt["n_last"]}.csv')
        trajectories_var = np.loadtxt(f'modules/gp/scoring/trajectories_var_post_{opt["n_last"]}.csv')
        n_calculated = trajectories_var.shape[0]
        t_start = opt['t_start_score'] + n_calculated*datetime.timedelta(minutes=10)
        t_end = opt['t_end_score'] - opt['steps_forward']*datetime.timedelta(minutes=10)
        if t_start >= t_end: return trajectories_mean, trajectories_var

    except:
        t_start = opt['t_start_score']
        t_end = opt['t_end_score'] - opt['steps_forward']*datetime.timedelta(minutes=10)

    gp = TimeseriesModel(opt)

    dt = t_end-t_start
    n_times = int(dt.total_seconds()/600)
    times = [t_start+i*datetime.timedelta(minutes=10) for i in range(n_times)]

    trajectories_mean = np.zeros((n_times, opt['steps_forward']))
    trajectories_var = np.zeros((n_times, opt['steps_forward']))

    for i, time in enumerate(times):
        if time.minute == 0 or i==0:
            train = True
            print(f'Training timeseries GP for time {time}')
        else:
            train = False
        trajectory_mean_gp, trajectory_var_gp = gp.predict_trajectory(
            time, opt['steps_forward'], train, include_last_measurement=False)
        trajectories_mean[i,:] = trajectory_mean_gp
        trajectories_var[i,:] = trajectory_var_gp
        with open(f'modules/gp/scoring/trajectories_mean_post_{opt["n_last"]}.csv', 'a') as file:
            np.savetxt(file, trajectory_mean_gp.reshape((1,-1)))
        with open(f'modules/gp/scoring/trajectories_var_post_{opt["n_last"]}.csv', 'a') as file:
            np.savetxt(file, trajectory_var_gp.reshape((1,-1)))    
    return trajectories_mean, trajectories_var

def get_posterior_trajectories_homoscedastic(opt):
    try:
        trajectories_mean = np.loadtxt(f'modules/gp/scoring/trajectories_mean_post_{opt["n_last"]}_homoscedastic_only_nwp.csv')
        trajectories_var = np.loadtxt(f'modules/gp/scoring/trajectories_var_post_{opt["n_last"]}_homoscedastic_only_nwp.csv')
        n_calculated = trajectories_var.shape[0]
        t_start = opt['t_start_score'] + n_calculated*datetime.timedelta(minutes=10)
        t_end = opt['t_end_score'] - opt['steps_forward']*datetime.timedelta(minutes=10)
        if t_start >= t_end: return trajectories_mean, trajectories_var

    except:
        t_start = opt['t_start_score']
        t_end = opt['t_end_score'] - opt['steps_forward']*datetime.timedelta(minutes=10)

    gp = HomoscedasticTimeseriesModel(opt)

    dt = t_end-t_start
    n_times = int(dt.total_seconds()/600)
    times = [t_start+i*datetime.timedelta(minutes=10) for i in range(n_times)]

    trajectories_mean = np.zeros((n_times, opt['steps_forward']))
    trajectories_var = np.zeros((n_times, opt['steps_forward']))

    for i, time in enumerate(times):
        if time.minute == 0 or i==0:
            train = True
            print(f'Training timeseries GP for time {time}')
        else:
            train = False
        trajectory_mean_gp, trajectory_var_gp = gp.predict_trajectory(
            time, opt['steps_forward'], train, include_last_measurement=False)
        trajectories_mean[i,:] = trajectory_mean_gp
        trajectories_var[i,:] = trajectory_var_gp
        with open(f'modules/gp/scoring/trajectories_mean_post_{opt["n_last"]}_homoscedastic_only_nwp.csv', 'a') as file:
            np.savetxt(file, trajectory_mean_gp.reshape((1,-1)))
        with open(f'modules/gp/scoring/trajectories_var_post_{opt["n_last"]}_homoscedastic_only_nwp.csv', 'a') as file:
            np.savetxt(file, trajectory_var_gp.reshape((1,-1)))    
    return trajectories_mean, trajectories_var

def get_simple_timeseries_traj(opt):
    try:
        trajectories_mean = np.loadtxt('modules/gp/scoring/trajectories_mean_simple_timeseries.csv')
        trajectories_var = np.loadtxt('modules/gp/scoring/trajectories_var_simple_timeseries.csv')
        n_calculated = trajectories_var.shape[0]
        t_start = opt['t_start_score'] + n_calculated*datetime.timedelta(minutes=10)
        t_end = opt['t_end_score'] - opt['steps_forward']*datetime.timedelta(minutes=10)
        if t_start >= t_end: return trajectories_mean, trajectories_var

    except:
        t_start = opt['t_start_score']
        t_end = opt['t_end_score'] - opt['steps_forward']*datetime.timedelta(minutes=10)

    gp = SimpleTimeseriesModel(opt)

    dt = t_end-t_start
    n_times = int(dt.total_seconds()/600)
    times = [t_start+i*datetime.timedelta(minutes=10) for i in range(n_times)]

    trajectories_mean = np.zeros((n_times, opt['steps_forward']))
    trajectories_var = np.zeros((n_times, opt['steps_forward']))

    for i, time in enumerate(times):
        if time.minute == 0:
            train = True
            print(f'Training timeseries GP for time {time}')
        else:
            train = False
        trajectory_mean_gp, trajectory_var_gp = gp.predict_trajectory(
            time, opt['steps_forward'], train, include_last_measurement=False)
        trajectories_mean[i,:] = trajectory_mean_gp
        trajectories_var[i,:] = trajectory_var_gp
        with open('modules/gp/scoring/trajectories_mean_simple_timeseries.csv', 'a') as file:
            np.savetxt(file, trajectory_mean_gp.reshape((1,-1)))
        with open('modules/gp/scoring/trajectories_var_simple_timeseries.csv', 'a') as file:
            np.savetxt(file, trajectory_var_gp.reshape((1,-1)))    
    return trajectories_mean, trajectories_var

def get_direct_model_trajectories(opt):
    try:
        trajectories_mean = np.loadtxt('modules/gp/scoring/trajectories_mean_direct_only_nwp.csv')
        trajectories_var = np.loadtxt('modules/gp/scoring/trajectories_var_direct_only_nwp.csv')
        n_calculated = trajectories_var.shape[0]
        t_start = opt['t_start_score'] + n_calculated*datetime.timedelta(minutes=10)
        t_end = opt['t_end_score'] - opt['steps_forward']*datetime.timedelta(minutes=10)
        if t_start >= t_end: return trajectories_mean, trajectories_var

    except:
        t_start = opt['t_start_score']
        t_end = opt['t_end_score'] - opt['steps_forward']*datetime.timedelta(minutes=10)

    gp = DirectGPEnsemble(opt)

    dt = t_end-t_start
    n_times = int(dt.total_seconds()/600)
    times = [t_start+i*datetime.timedelta(minutes=10) for i in range(n_times)]

    trajectories_mean = np.zeros((n_times, opt['steps_forward']))
    trajectories_var = np.zeros((n_times, opt['steps_forward']))

    for i, time in enumerate(times):
        trajectory_mean_gp, trajectory_var_gp = gp.predict_trajectory(
            time, opt['steps_forward'], include_last_measurement=False)
        trajectories_mean[i,:] = trajectory_mean_gp
        trajectories_var[i,:] = trajectory_var_gp
        with open('modules/gp/scoring/trajectories_mean_direct_only_nwp.csv', 'a') as file:
            np.savetxt(file, trajectory_mean_gp.reshape((1,-1)))
        with open('modules/gp/scoring/trajectories_var_direct_only_nwp.csv', 'a') as file:
            np.savetxt(file, trajectory_var_gp.reshape((1,-1)))    
    return trajectories_mean, trajectories_var


if __name__ == '__main__':
    from get_gp_opt import get_gp_opt
    opt = get_gp_opt(n_z=200, max_epochs_second_training=10, epochs_timeseries_retrain=500, 
                     epochs_timeseries_first_train=500)
    weather_data = load_weather_data(opt['t_start'], opt['t_end'])

    try:
        trajectory_measured = np.loadtxt('modules/gp/scoring/trajectory_meas.csv')
    except:
        trajectory_measured = get_trajectory_measured(weather_data, opt)
        np.savetxt('modules/gp/scoring/trajectory_meas.csv', trajectory_measured)
    try:
        trajectory_nwp = np.loadtxt('modules/gp/scoring/trajectory_nwp.csv')
    except:
        trajectory_nwp = get_trajectory_nwp(weather_data, opt)
        np.savetxt('modules/gp/scoring/trajectory_nwp.csv', trajectory_nwp)
    try:
        trajectory_gp_prior = np.loadtxt('modules/gp/scoring/trajectory_gp_prior.csv')
        var_gp_prior = np.loadtxt('modules/gp/scoring/var_gp_prior.csv')
    except:
        trajectory_gp_prior, var_gp_prior = get_trajectory_gp_prior(weather_data, opt)
        np.savetxt('modules/gp/scoring/trajectory_gp_prior.csv', trajectory_gp_prior)
        np.savetxt('modules/gp/scoring/var_gp_prior.csv', var_gp_prior)
    rmse_nwp = get_rmse(trajectory_measured, trajectory_nwp)
    mae_nwp = get_mae(trajectory_measured, trajectory_nwp)

    rmse_gp_prior = get_rmse(trajectory_measured, trajectory_gp_prior)
    mae_gp_prior = get_mae(trajectory_measured, trajectory_gp_prior)

    alpha_vec = np.linspace(0.01,1,100)
    re_gp_prior = [get_RE(alpha, trajectory_measured, trajectory_gp_prior, var_gp_prior)
                   for alpha in alpha_vec]
    int_score_gp_prior = [get_interval_score(alpha, trajectory_measured, trajectory_gp_prior, var_gp_prior)
                   for alpha in alpha_vec]
    
    percent_in_interval_gp_prior = np.array(re_gp_prior) + (1-alpha_vec)    
    
    print(f'RMSE of NWP: {rmse_nwp}, MAE of NWP: {mae_nwp}')
    print(f'RMSE of GP: {rmse_gp_prior}, MAE of GP: {mae_gp_prior}')

    plt.figure()
    plt.plot(np.linspace(0.01,1,100), re_gp_prior)
    plt.xlabel('alpha')
    plt.ylabel('RE for NWP-based GP')
    plt.figure()
    plt.plot(np.linspace(0.01,1,100), int_score_gp_prior)
    plt.xlabel('alpha')
    plt.ylabel('Interval score for NWP-based GP')
    plt.figure()
    plt.plot(1-alpha_vec, percent_in_interval_gp_prior)
    plt.plot(1-alpha_vec, 1-alpha_vec, '--')
    plt.xlabel('1-alpha')
    plt.ylabel('actual percentage in 1-alpha-interval')
    plt.ylim((0,1))

    steps_forward = opt['steps_forward']
    rmse_post = np.zeros(steps_forward)
    mae_post = np.zeros(steps_forward)
    re_post = np.zeros(steps_forward)
    score_post = np.zeros(steps_forward)

    trajectories_mean_post, trajectories_var_post = get_posterior_trajectories(opt)
    # trajectories_mean_post = np.loadtxt('gp/scoring/trajectories_mean_post.csv')
    # trajectories_var_post = np.loadtxt('gp/scoring/trajectories_var_post.csv')
    # first dimension: time of prediction, second dimension: number of steps forward
    # n_points = len(trajectory_measured)
    alpha = 0.1
    for i in range(opt['steps_forward']):
        trajectory_post = trajectories_mean_post[:,i]
        var_post = trajectories_var_post[:,i]
        n_points = len(trajectory_post)
        rmse_post[i] = get_rmse(trajectory_measured[i:n_points+i], trajectory_post)
        mae_post[i] = get_mae(trajectory_measured[i:n_points+i], trajectory_post)
        re_post[i] = get_RE(alpha, trajectory_measured[i:n_points+i], trajectory_post, var_post)
        score_post[i] = get_interval_score(alpha, trajectory_measured[i:n_points+i], trajectory_post, var_post)
    steps = np.arange(1, steps_forward+1)
    plt.figure()
    plt.plot(steps, rmse_post)
    plt.xlabel('Number of steps predicted forward')
    plt.ylabel('RMSE of prediction')
    plt.figure()
    plt.plot(steps, mae_post)
    plt.xlabel('Number of steps predicted forward')
    plt.ylabel('MAE of prediction')
    plt.figure()
    plt.plot(steps, re_post)
    plt.xlabel('Number of steps predicted forward')
    plt.ylabel('RE of prediction interval for alpha=0.1')
    plt.figure()
    plt.plot(steps, score_post)
    plt.xlabel('Number of steps predicted forward')
    plt.ylabel('Interval score of prediction interval for alpha=0.1')

    plt.show()
    pass
