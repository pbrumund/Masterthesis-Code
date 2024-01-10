import datetime
import random

import casadi as ca
import matplotlib.pyplot as plt
import numpy as np

from modules.mpc import NominalMPC, get_mpc_opt
from modules.models import OHPS
from modules.gp import TimeseriesModel as WindPredictionGP
from modules.gp import DataHandler
from modules.gp import get_gp_opt
from modules.plotting import TimeseriesPlot

plt.ion()
dt_pred = 5
steps = 60
std_list = (-1,0,1)
dP_min = 2000
tree_depth = 30

gp_opt = get_gp_opt(dt_pred=dt_pred)
gp = WindPredictionGP(gp_opt)

ohps = OHPS()
def get_wind_power_traj(wind_speed_traj):
    return np.array([ohps.wind_turbine.power_curve_fun(ohps.wind_turbine.scale_wind_speed(w)) 
                     for w in wind_speed_traj]).reshape(-1)

for k in range(5):
    year = 2022
    month = random.randint(1,12)
    day = random.randint(1,28)
    hour = random.randint(0,23)
    minute = 5*random.randint(0,11)
    t = datetime.datetime(year, month, day, hour, minute)
    t_vec = [t + i*datetime.timedelta(minutes=dt_pred) for i in range(steps)]
    mean, var = gp.predict_trajectory(t, steps, train=True)
    means_i = [mean]
    vars_i = [var]
    last_pseudo_inputs = [np.array([])]
    last_pseudo_indices = [np.array([])]
    means = []
    vars = []
    # last_measurements = None
    for i in range(tree_depth):  # time step
        if i%2 == 0: continue
        new_pseudo_inputs = []
        new_pseudo_indices = []
        means_i_next = []
        vars_i_next = []
        for mean_i_k, var_i_k, pseudo_inputs, pseudo_indices in zip(
                means_i, vars_i, last_pseudo_inputs, last_pseudo_indices):  # scenario index
            #pseudo_indices = np.arange(i+1)
            P_upper = ohps.wind_turbine.power_curve_fun(
                ohps.wind_turbine.scale_wind_speed(mean_i_k[i]+np.sqrt(var_i_k[i])))
            P_lower = ohps.wind_turbine.power_curve_fun(
                ohps.wind_turbine.scale_wind_speed(mean_i_k[i]-np.sqrt(var_i_k[i])))
            if np.abs(P_upper-P_lower) < dP_min:
                # do not branch, but keep trajectory
                new_pseudo_indices.append(pseudo_indices)
                new_pseudo_inputs.append(pseudo_inputs)
                means_i_next.append(mean_i_k)
                vars_i_next.append(var_i_k)
                continue

            pseudo_indices = np.append(pseudo_indices, i)
            for x in std_list:
                w = mean_i_k[i] + x*np.sqrt(var_i_k[i])
                pseudo_measurements = np.append(pseudo_inputs, w)
                pseudo_gp = gp.get_pseudo_timeseries_gp(t, pseudo_measurements, pseudo_indices)
                mean_new, var_new = gp.predict_trajectory(t, steps, pseudo_gp=pseudo_gp)
                new_pseudo_inputs.append(pseudo_measurements)
                new_pseudo_indices.append(pseudo_indices)
                # only keep values after time index i
                mean_new[:i+1] = mean_i_k[:i+1]
                var_new[:i+1] = var_i_k[:i+1]
                means.append(mean_new)
                vars.append(var_new)
                means_i_next.append(mean_new)
                vars_i_next.append(var_new)
        means_i = means_i_next
        vars_i = vars_i_next
        last_pseudo_inputs = new_pseudo_inputs
        last_pseudo_indices = new_pseudo_indices
        if len(last_pseudo_indices) > 27:
            break

            # w_1 = mean_i_k[i] + np.sqrt(var_i_k[i])
            # w_2 = mean_i_k[i]
            # w_3 = mean_i_k[i] + np.sqrt(var_i_k[i])
    if len(means) == 0:
        means.append(mean)
        vars.append(var)      
    plt.figure()
    for mean_i, var_i in zip(means, vars):
        plt.plot(t_vec, mean_i)
        plt.fill_between(t_vec, mean_i-np.sqrt(var_i), mean_i+np.sqrt(var_i), alpha=0.2)
    plt.show()
    plt.xlabel('Time')
    plt.ylabel('Predicted wind speed')
    plt.figure()
    for mean_i, var_i in zip(means, vars):
        plt.plot(t_vec, get_wind_power_traj(mean_i))
        plt.fill_between(t_vec, get_wind_power_traj(mean_i-np.sqrt(var_i)), 
                         get_wind_power_traj(mean_i+np.sqrt(var_i)), alpha=0.2)
    plt.show()
    plt.xlabel('Time')
    plt.ylabel('Predicted wind power')
    plt.pause(0.5)
# w_1 = mean[i] + np.sqrt(var[i])
# w_2 = mean[i] - np.sqrt(var[i])
# pseudo_indices = np.array(i)
# pseudo_measurements = np.array(w_1)
# pseudo_gp = gp.get_pseudo_timeseries_gp(t, pseudo_measurements, pseudo_indices)
# mean_1, var_1 = gp.predict_trajectory(t, steps, pseudo_gp = pseudo_gp)
# pseudo_indices = np.array(i)
# pseudo_measurements = np.array(w_2)
# pseudo_gp = gp.get_pseudo_timeseries_gp(t, pseudo_measurements, pseudo_indices)
# mean_2, var_2 = gp.predict_trajectory(t, steps, pseudo_gp = pseudo_gp)



# plt.figure()
# plt.plot(t_vec, mean)
# plt.fill_between(t_vec, mean - np.sqrt(var), mean + np.sqrt(var), alpha=0.2)
# plt.plot(t_vec, mean_1)
# plt.fill_between(t_vec, mean_1 - np.sqrt(var_1), mean_1 + np.sqrt(var_1), alpha=0.2)
# plt.plot(t_vec, mean_2)
# plt.fill_between(t_vec, mean_2 - np.sqrt(var_2), mean_2 + np.sqrt(var_2), alpha=0.2)
# plt.show()
pass
