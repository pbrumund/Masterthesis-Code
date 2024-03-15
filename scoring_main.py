import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

if __name__ == '__main__':
    """Get scores """
    plt.ion()
    from modules.gp.fileloading import load_weather_data
    from modules.gp import get_gp_opt
    from modules.gp.scoring import (get_interval_score, get_mae, get_posterior_trajectories, get_nlpd,
        get_rmse, get_RE, get_trajectory_gp_prior, get_trajectory_measured, get_trajectory_nwp, get_posterior_trajectories_homoscedastic,
        get_direct_model_trajectories, get_simple_timeseries_traj, get_mape, get_trajectory_gp_prior_homoscedastic)
    
    rmse_list_heteroscedastic = []
    mae_list_heteroscedastic = []
    nlpd_list_heteroscedastic = []
    nlpd_list_homoscedastic = []
    re_list_heteroscedastic = []
    score_list_heteroscedastic = []
    rmse_list_homoscedastic = []
    mae_list_homoscedastic = []
    re_list_homoscedastic = []
    score_list_homoscedastic = []
    for n_z in (5,10,25,50,100,200):#,200,400):
        opt = get_gp_opt(n_z=n_z, max_epochs_second_training=50, epochs_timeseries_retrain=500, 
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
            trajectory_gp_prior = np.loadtxt(f'modules/gp/scoring/trajectory_gp_prior_heteroscedastic_{n_z}_without_time2.csv')
            var_gp_prior = np.loadtxt(f'modules/gp/scoring/var_gp_prior_heteroscedastic_{n_z}_without_time2.csv')
        except:
            trajectory_gp_prior, var_gp_prior = get_trajectory_gp_prior(opt)
            np.savetxt(f'modules/gp/scoring/trajectory_gp_prior_heteroscedastic_{n_z}_without_time2.csv', trajectory_gp_prior)
            np.savetxt(f'modules/gp/scoring/var_gp_prior_heteroscedastic_{n_z}_without_time2.csv', var_gp_prior)
        try:
            trajectory_gp_prior_homoscedastic = np.loadtxt(f'modules/gp/scoring/trajectory_gp_prior_homoscedastic_{n_z}_without_time2.csv')
            var_gp_prior_homoscedastic = np.loadtxt(f'modules/gp/scoring/var_gp_prior_homoscedastic_{n_z}_without_time2.csv')
        except:
            trajectory_gp_prior_homoscedastic, var_gp_prior_homoscedastic = get_trajectory_gp_prior_homoscedastic(opt)
            np.savetxt(f'modules/gp/scoring/trajectory_gp_prior_homoscedastic_{n_z}_without_time2.csv', trajectory_gp_prior_homoscedastic)
            np.savetxt(f'modules/gp/scoring/var_gp_prior_homoscedastic_{n_z}_without_time2.csv', var_gp_prior_homoscedastic)
        rmse_nwp = get_rmse(trajectory_measured, trajectory_nwp)
        mae_nwp = get_mae(trajectory_measured, trajectory_nwp)

        rmse_gp_prior = get_rmse(trajectory_measured, trajectory_gp_prior)
        mae_gp_prior = get_mae(trajectory_measured, trajectory_gp_prior)
        nlpd_gp_prior = get_nlpd(trajectory_measured, trajectory_gp_prior, var_gp_prior)
        rmse_list_heteroscedastic.append(rmse_gp_prior)
        mae_list_heteroscedastic.append(mae_gp_prior)
        nlpd_list_heteroscedastic.append(nlpd_gp_prior)

        alpha_vec = np.linspace(0.01,1,100)
        re_gp_prior = [get_RE(alpha, trajectory_measured, trajectory_gp_prior, var_gp_prior)
                        for alpha in alpha_vec]
        int_score_gp_prior = [get_interval_score(alpha, trajectory_measured, trajectory_gp_prior, var_gp_prior)
                        for alpha in alpha_vec]
        re_list_heteroscedastic.append(np.array(re_gp_prior))
        score_list_heteroscedastic.append(np.array(int_score_gp_prior))
        percent_in_interval_gp_prior = np.array(re_gp_prior) + (1-alpha_vec) 
        mape_gp_prior = get_mape(trajectory_measured, trajectory_gp_prior)

        rmse_gp_prior_homoscedastic = get_rmse(trajectory_measured, trajectory_gp_prior_homoscedastic)
        mae_gp_prior_homoscedastic = get_mae(trajectory_measured, trajectory_gp_prior_homoscedastic)
        nlpd_gp_prior_homoscedastic = get_nlpd(trajectory_measured, trajectory_gp_prior_homoscedastic, var_gp_prior_homoscedastic)
        alpha_vec = np.linspace(0.01,1,100)
        re_gp_prior_homoscedastic = [get_RE(alpha, trajectory_measured, trajectory_gp_prior_homoscedastic, var_gp_prior_homoscedastic)
                        for alpha in alpha_vec]
        int_score_gp_prior_homoscedastic = [get_interval_score(alpha, trajectory_measured, trajectory_gp_prior_homoscedastic, var_gp_prior_homoscedastic)
                        for alpha in alpha_vec]

        percent_in_interval_gp_prior_homoscedastic = np.array(re_gp_prior_homoscedastic) + (1-alpha_vec)
        rmse_list_homoscedastic.append(rmse_gp_prior_homoscedastic)
        mae_list_homoscedastic.append(mae_gp_prior_homoscedastic)
        nlpd_list_homoscedastic.append(nlpd_gp_prior_homoscedastic)
        re_list_homoscedastic.append(re_gp_prior_homoscedastic)
        score_list_homoscedastic.append(int_score_gp_prior_homoscedastic)
        print(f'RMSE of NWP: {rmse_nwp}, MAE of NWP: {mae_nwp}')
        print(f'RMSE of heteroscedastic GP with {n_z} inducing vars: {rmse_gp_prior}, MAE of GP: {mae_gp_prior}, NLPD of GP: {nlpd_gp_prior}')
        print(f'RMSE of homoscedastic GP with {n_z} inducing vars: {rmse_gp_prior_homoscedastic}, MAE of GP: {mae_gp_prior_homoscedastic}, NLPD of GP: {nlpd_gp_prior_homoscedastic}')

    fig, axs = plt.subplots(1,2)
    handles = []
    for i, color in enumerate(['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']):#, 'tab:gray']):
        handle, = axs[0].plot(alpha_vec, re_list_heteroscedastic[i], color=color)
        handles.append(handle)
        axs[0].plot(alpha_vec, re_list_homoscedastic[i], '--', color=color)
        axs[0].set_xlabel('alpha')
        axs[0].set_ylabel('RE')
        axs[1].plot(alpha_vec, score_list_heteroscedastic[i], color=color)
        axs[1].plot(alpha_vec, score_list_homoscedastic[i], '--', color=color)
        axs[1].set_xlabel('alpha')
        axs[1].set_ylabel('Interval score')
    fig.legend(handles=handles, labels=['5', '10', '25', '50', '100', '200'])

    fig, axs = plt.subplots(1,3)
    axs[0].plot([5,10,25,50,100,200], rmse_list_heteroscedastic,'x')
    axs[0].plot([5,10,25,50,100,200], rmse_list_homoscedastic,'o')
    axs[1].plot([5,10,25,50,100,200], mae_list_heteroscedastic,'x')
    axs[1].plot([5,10,25,50,100,200], mae_list_homoscedastic,'o')
    axs[2].plot([5,10,25,50,100,200], nlpd_list_heteroscedastic,'x')
    axs[2].plot([5,10,25,50,100,200], nlpd_list_homoscedastic,'o')
    #axs[].xaxis.set_major_locator(mticker.FixedLocator([10,25,50,100,200,400]))
    #axs[0].xaxis.set_major_locator(mticker.FixedLocator([10,25,50,100,200,400]))
    axs[0].grid()
    axs[1].grid()
    axs[0].set_xlabel('Number of inducing points')
    axs[0].set_ylabel('RMSE (m/s)')
    axs[1].set_xlabel('Number of inducing points')
    axs[1].set_ylabel('MAE (m/s)')
    fig.legend(['Heteroscedastic', 'Homoscedastic'])
    re_gp_prior = re_list_heteroscedastic[-2]
    re_gp_prior_homoscedastic = re_list_homoscedastic[-2]
    int_score_gp_prior = score_list_heteroscedastic[-2]
    int_score_gp_prior_homoscedastic = score_list_homoscedastic[-2]
    percent_in_interval_gp_prior = np.array(re_gp_prior) + (1-alpha_vec) 
    percent_in_interval_gp_prior_homoscedastic = np.array(re_gp_prior_homoscedastic) + (1-alpha_vec)
    plt.figure()
    plt.plot(np.linspace(0.01,1,100), re_gp_prior, label='Heteroscedastic GP')
    plt.plot(alpha_vec, re_gp_prior_homoscedastic, label='Homoscedastic GP')
    plt.xlabel('alpha')
    plt.ylabel('RE for NWP-based GP')
    plt.legend()
    plt.figure()
    plt.plot(np.linspace(0.01,1,100), int_score_gp_prior, label='Heteroscedastic GP')
    plt.plot(alpha_vec, int_score_gp_prior_homoscedastic, label='Homoscedastic GP')
    plt.xlabel('alpha')
    plt.ylabel('Interval score for NWP-based GP')
    plt.legend()
    plt.figure()
    plt.plot(1-alpha_vec, percent_in_interval_gp_prior, label='Heteroscedastic GP')
    plt.plot(1-alpha_vec, percent_in_interval_gp_prior_homoscedastic, label='Homoscedastic GP')
    plt.plot(1-alpha_vec, 1-alpha_vec, '--')
    plt.xlabel('1-alpha')
    plt.ylabel('actual percentage in 1-alpha-interval')
    plt.ylim((0,1))
    plt.legend()
    plt.pause(1)
    steps_forward = opt['steps_forward']
    rmse_post = np.zeros(steps_forward)
    mae_post = np.zeros(steps_forward)
    re_post = np.zeros(steps_forward)
    score_post = np.zeros(steps_forward)#
    rmse_post_simple = np.zeros(steps_forward)
    mae_post_simple = np.zeros(steps_forward)
    re_post_simple = np.zeros(steps_forward)
    score_post_simple = np.zeros(steps_forward)
    rmse_post_homoscedastic = np.zeros(steps_forward)
    mae_post_homoscedastic = np.zeros(steps_forward)
    re_post_homoscedastic = np.zeros(steps_forward)
    score_post_homoscedastic = np.zeros(steps_forward)
    opt['n_z'] = 100
    trajectories_mean_post_simple, trajectories_var_post_simple = get_simple_timeseries_traj(opt)
    trajectories_mean_post, trajectories_var_post = get_posterior_trajectories(opt)
    trajectories_mean_post_homoscedastic, trajectories_var_post_homoscedastic = get_posterior_trajectories_homoscedastic(opt)
    
    # first dimension: time of prediction, second dimension: number of steps forward
    n_points = len(trajectory_measured)
    alpha = 0.1
    for i in range(opt['steps_forward']):
        trajectory_post = trajectories_mean_post[:,i]
        var_post = trajectories_var_post[:,i]
        n_points = len(trajectory_post)
        rmse_post[i] = get_rmse(trajectory_measured[i:n_points+i], trajectory_post)
        mae_post[i] = get_mae(trajectory_measured[i:n_points+i], trajectory_post)
        re_post[i] = get_RE(alpha, trajectory_measured[i:n_points+i], trajectory_post, var_post)
        score_post[i] = get_interval_score(alpha, trajectory_measured[i:n_points+i], trajectory_post, var_post)
    for i in range(opt['steps_forward']):
        trajectory_post_homoscedastic = trajectories_mean_post_homoscedastic[:,i]
        var_post_homoscedastic = trajectories_var_post_homoscedastic[:,i]
        n_points = len(trajectory_post_homoscedastic)
        rmse_post_homoscedastic[i] = get_rmse(trajectory_measured[i:n_points+i], trajectory_post_homoscedastic)
        mae_post_homoscedastic[i] = get_mae(trajectory_measured[i:n_points+i], trajectory_post_homoscedastic)
        re_post_homoscedastic[i] = get_RE(alpha, trajectory_measured[i:n_points+i], trajectory_post_homoscedastic, var_post_homoscedastic)
        score_post_homoscedastic[i] = get_interval_score(alpha, trajectory_measured[i:n_points+i], trajectory_post_homoscedastic, var_post_homoscedastic)
    for i in range(opt['steps_forward']):
        trajectory_post = trajectories_mean_post_simple[:,i]
        var_post = trajectories_var_post_simple[:,i]
        n_points = len(trajectory_post)
        rmse_post_simple[i] = get_rmse(trajectory_measured[i:n_points+i], trajectory_post)
        mae_post_simple[i] = get_mae(trajectory_measured[i:n_points+i], trajectory_post)
        re_post_simple[i] = get_RE(alpha, trajectory_measured[i:n_points+i], trajectory_post, var_post)
        score_post_simple[i] = get_interval_score(alpha, trajectory_measured[i:n_points+i], trajectory_post, var_post)
    steps = np.arange(1, steps_forward+1)
    plt.figure()
    plt.plot(steps, rmse_post)
    plt.plot(steps, rmse_post_homoscedastic)
    plt.plot(steps, rmse_post_simple)
    plt.xlabel('Number of steps predicted forward')
    plt.ylabel('RMSE of prediction')
    plt.legend(['Heteroscedastic prior GP', 'Homoscedastic prior GP', 'Constant prior'])
    plt.figure()
    plt.plot(steps, mae_post)
    plt.plot(steps, mae_post_homoscedastic)
    plt.plot(steps, mae_post_simple)
    plt.xlabel('Number of steps predicted forward')
    plt.ylabel('MAE of prediction')
    plt.legend(['Heteroscedastic prior GP', 'Homoscedastic prior GP', 'Constant prior'])
    plt.figure()
    plt.plot(steps, re_post)
    plt.plot(steps, re_post_homoscedastic)
    plt.plot(steps, re_post_simple)
    plt.xlabel('Number of steps predicted forward')
    plt.ylabel('RE of prediction interval for alpha=0.1')
    plt.legend(['Heteroscedastic prior GP', 'Homoscedastic prior GP', 'Constant prior'])
    plt.figure()
    plt.plot(steps, score_post)
    plt.plot(steps, score_post_homoscedastic)
    plt.plot(steps, score_post_simple)
    plt.xlabel('Number of steps predicted forward')
    plt.ylabel('Interval score of prediction interval for alpha=0.1')
    plt.legend(['Heteroscedastic prior GP', 'Homoscedastic prior GP', 'Constant prior'])

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