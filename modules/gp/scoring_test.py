import numpy as np
import matplotlib.pyplot as plt
import datetime
from fileloading import load_weather_data
from scoring import get_interval_score, get_mae, get_RE, get_rmse, get_trajectory_gp_prior, get_trajectory_measured, get_trajectory_nwp

if __name__ == '__main__':
    mae = []
    rmse = []
    for n_z in (10,25,50,75,100,200):
        from get_gp_opt import get_gp_opt
        opt = get_gp_opt(max_epochs_second_training=50, n_z = n_z)
        weather_data = load_weather_data(opt['t_start'], opt['t_end'])

        trajectory_measured = get_trajectory_measured(weather_data, opt)
        trajectory_nwp = get_trajectory_nwp(weather_data, opt)
        trajectory_gp_prior, var_gp_prior = get_trajectory_gp_prior(weather_data, opt)

        rmse_nwp = get_rmse(trajectory_measured, trajectory_nwp)
        mae_nwp = get_mae(trajectory_measured, trajectory_nwp)

        rmse_gp_prior = get_rmse(trajectory_measured, trajectory_gp_prior)
        mae_gp_prior = get_mae(trajectory_measured, trajectory_gp_prior)
        rmse.append(rmse_gp_prior)
        mae.append(mae_gp_prior)

        alpha_vec = np.linspace(0.01,1,100)
        re_gp_prior = [get_RE(alpha, trajectory_measured, trajectory_gp_prior, var_gp_prior)
                        for alpha in alpha_vec]
        int_score_gp_prior = [get_interval_score(alpha, trajectory_measured, trajectory_gp_prior, var_gp_prior)
                        for alpha in alpha_vec]

        percent_in_interval_gp_prior = np.array(re_gp_prior) + (1-alpha_vec)    

        print(f'RMSE of NWP: {rmse_nwp}, MAE of NWP: {mae_nwp}')
        print(f'{n_z} inducing variables: RMSE of GP: {rmse_gp_prior}, MAE of GP: {mae_gp_prior}')

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
    plt.show()
    pass