import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    """Get scores """
    plt.ion()
    from modules.gp.fileloading import load_weather_data
    from modules.gp import get_gp_opt
    from modules.gp.scoring import (get_interval_score, get_mae, get_posterior_trajectories, 
        get_rmse, get_RE, get_trajectory_gp_prior, get_trajectory_measured, get_trajectory_nwp, 
        get_direct_model_trajectories, get_simple_timeseries_traj, get_mape, get_trajectory_gp_prior_homoscedastic)

    opt = get_gp_opt(n_z=400, max_epochs_second_training=20, epochs_timeseries_retrain=500, 
                        epochs_timeseries_first_train=500, n_last=36)
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
        trajectory_gp_prior = np.loadtxt('modules/gp/scoring/trajectory_gp_prior_heteroscedastic_400.csv')
        var_gp_prior = np.loadtxt('modules/gp/scoring/var_gp_prior_heteroscedastic_400.csv')
    except:
        trajectory_gp_prior, var_gp_prior = get_trajectory_gp_prior(opt)
        np.savetxt('modules/gp/scoring/trajectory_gp_prior_heteroscedastic_400.csv', trajectory_gp_prior)
        np.savetxt('modules/gp/scoring/var_gp_prior_heteroscedastic_400.csv', var_gp_prior)
    try:
        trajectory_gp_prior_homoscedastic = np.loadtxt('modules/gp/scoring/trajectory_gp_prior_homoscedastic_400.csv')
        var_gp_prior_homoscedastic = np.loadtxt('modules/gp/scoring/var_gp_prior_homoscedastic_400.csv')
    except:
        trajectory_gp_prior_homoscedastic, var_gp_prior_homoscedastic = get_trajectory_gp_prior_homoscedastic(opt)
        np.savetxt('modules/gp/scoring/trajectory_gp_prior_homoscedastic_400.csv', trajectory_gp_prior)
        np.savetxt('modules/gp/scoring/var_gp_prior_homoscedastic_400.csv', var_gp_prior)
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
    mape_gp_prior = get_mape(trajectory_measured, trajectory_gp_prior)

    rmse_gp_prior_homoscedastic = get_rmse(trajectory_measured, trajectory_gp_prior_homoscedastic)
    mae_gp_prior_homoscedastic = get_mae(trajectory_measured, trajectory_gp_prior_homoscedastic)

    alpha_vec = np.linspace(0.01,1,100)
    re_gp_prior_homoscedastic = [get_RE(alpha, trajectory_measured, trajectory_gp_prior_homoscedastic, var_gp_prior_homoscedastic)
                    for alpha in alpha_vec]
    int_score_gp_prior_homoscedastic = [get_interval_score(alpha, trajectory_measured, trajectory_gp_prior_homoscedastic, var_gp_prior_homoscedastic)
                    for alpha in alpha_vec]

    percent_in_interval_gp_prior_homoscedastic = np.array(re_gp_prior_homoscedastic) + (1-alpha_vec)
    print(f'RMSE of NWP: {rmse_nwp}, MAE of NWP: {mae_nwp}')
    print(f'RMSE of GP: {rmse_gp_prior}, MAE of GP: {mae_gp_prior}')
    print(f'RMSE of homoscedastic GP: {rmse_gp_prior_homoscedastic}, MAE of GP: {mae_gp_prior_homoscedastic}')

    plt.figure()
    plt.plot(np.linspace(0.01,1,100), re_gp_prior)
    plt.plot(alpha_vec, re_gp_prior_homoscedastic)
    plt.xlabel('alpha')
    plt.ylabel('RE for NWP-based GP')
    plt.figure()
    plt.plot(np.linspace(0.01,1,100), int_score_gp_prior)
    plt.plot(alpha_vec, int_score_gp_prior_homoscedastic)
    plt.xlabel('alpha')
    plt.ylabel('Interval score for NWP-based GP')
    plt.figure()
    plt.plot(1-alpha_vec, percent_in_interval_gp_prior)
    plt.plot(1-alpha_vec, percent_in_interval_gp_prior_homoscedastic)
    plt.plot(1-alpha_vec, 1-alpha_vec, '--')
    plt.xlabel('1-alpha')
    plt.ylabel('actual percentage in 1-alpha-interval')
    plt.ylim((0,1))
    plt.legend(['Heteroscedastic', 'Homoscedastic'])
    plt.pause(1)
    steps_forward = opt['steps_forward']
    rmse_post = np.zeros(steps_forward)
    mae_post = np.zeros(steps_forward)
    re_post = np.zeros(steps_forward)
    score_post = np.zeros(steps_forward)

    trajectories_mean_post, trajectories_var_post = get_posterior_trajectories(opt)
    trajectories_mean_post_simple, trajectories_var_post_simple = get_simple_timeseries_traj(opt)
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

    rmse_direct = np.zeros(steps_forward)
    mae_direct = np.zeros(steps_forward)
    re_direct = np.zeros(steps_forward)
    score_direct = np.zeros(steps_forward)

    trajectories_mean_direct, trajectories_var_direct = get_direct_model_trajectories(opt)
    # trajectories_mean_post = np.loadtxt('gp/scoring/trajectories_mean_post.csv')
    # trajectories_var_post = np.loadtxt('gp/scoring/trajectories_var_post.csv')
    # first dimension: time of prediction, second dimension: number of steps forward
    # n_points = len(trajectory_measured)
    alpha = 0.1
    for i in range(opt['steps_forward']):
        trajectory_direct = trajectories_mean_direct[:,i]
        var_direct = trajectories_var_direct[:,i]
        n_points = len(trajectory_direct)
        rmse_direct[i] = get_rmse(trajectory_measured[i:n_points+i], trajectory_direct)
        mae_direct[i] = get_mae(trajectory_measured[i:n_points+i], trajectory_direct)
        re_direct[i] = get_RE(alpha, trajectory_measured[i:n_points+i], trajectory_direct, var_direct)
        score_direct[i] = get_interval_score(alpha, trajectory_measured[i:n_points+i], trajectory_direct, var_direct)
    steps = np.arange(1, steps_forward+1)
    plt.figure()
    plt.plot(steps, rmse_direct)
    plt.xlabel('Number of steps predicted forward')
    plt.ylabel('RMSE of prediction')
    plt.figure()
    plt.plot(steps, mae_direct)
    plt.xlabel('Number of steps predicted forward')
    plt.ylabel('MAE of prediction')
    plt.figure()
    plt.plot(steps, re_direct)
    plt.xlabel('Number of steps predicted forward')
    plt.ylabel('RE of prediction interval for alpha=0.1')
    plt.figure()
    plt.plot(steps, score_direct)
    plt.xlabel('Number of steps predicted forward')
    plt.ylabel('Interval score of prediction interval for alpha=0.1')

    plt.figure()
    plt.plot(steps, rmse_direct)
    plt.plot(steps, rmse_post)
    plt.xlabel('Number of steps predicted forward')
    plt.ylabel('RMSE of prediction')
    plt.legend(['Direct model', 'Timeseries model'])
    plt.figure()
    plt.plot(steps, mae_direct)
    plt.plot(steps, mae_post)
    plt.xlabel('Number of steps predicted forward')
    plt.ylabel('MAE of prediction')
    plt.legend(['Direct model', 'Timeseries model'])
    plt.figure()
    plt.plot(steps, re_direct)
    plt.plot(steps, re_post)
    plt.xlabel('Number of steps predicted forward')
    plt.ylabel('RE of prediction interval for alpha=0.1')
    plt.legend(['Direct model', 'Timeseries model'])
    plt.figure()
    plt.plot(steps, score_direct)
    plt.plot(steps, score_post)
    plt.xlabel('Number of steps predicted forward')
    plt.ylabel('Interval score of prediction interval for alpha=0.1')
    plt.legend(['Direct model', 'Timeseries model'])

    plt.show()
    pass