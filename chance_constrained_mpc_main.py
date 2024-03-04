import datetime

import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

from modules.mpc import ChanceConstrainedMPC, get_mpc_opt
from modules.models import OHPS
from modules.gp import TimeseriesModel as WindPredictionGP
from modules.gp import DataHandler
from modules.gp import get_gp_opt
from modules.plotting import TimeseriesPlot
from modules.mpc_scoring import DataSaving
from modules.mpc import LowLevelController

plot = True
plot_predictions = False
ohps = OHPS()

mpc_opt = get_mpc_opt(N=30)#, t_start=datetime.datetime(2022,12,6), t_end=datetime.datetime(2022,12,8))
# mpc_opt['param']['r_s_x'] = 1e6
chance_constrained_mpc = ChanceConstrainedMPC(ohps, mpc_opt)
chance_constrained_mpc.get_optimization_problem()

gp_opt = get_gp_opt(dt_pred = mpc_opt['dt'], steps_forward = mpc_opt['N'], verbose = False)
gp = WindPredictionGP(gp_opt)

t_start = mpc_opt['t_start_sim']
t_end = mpc_opt['t_end_sim']
dt = datetime.timedelta(minutes=mpc_opt['dt'])
n_times = int((t_end-t_start)/dt)
times = [t_start + i*dt for i in range(n_times)]
times_plot = times

x_traj = ca.DM.zeros(n_times, chance_constrained_mpc.nx)
u_traj = ca.DM.zeros(n_times, chance_constrained_mpc.nu)
P_traj = ca.DM.zeros(n_times, 5)    # gtg, battery, wtg, total, demand
SOC_traj = ca.DM.zeros(n_times)

data_handler = DataHandler(datetime.datetime(2020,1,1), datetime.datetime(2022,12,31), gp_opt)

llc = LowLevelController(ohps, data_handler, mpc_opt)

x_k = ohps.x0
v_last = None
P_gtg_last = ohps.gtg.bounds['ubu']*0.8
P_demand_last = None

# create plots
if plot:
    plt_power = TimeseriesPlot('Time', 'Power output', 
        ['Gas turbine', 'Battery', 'Wind turbine', 'Total power generation', 'Demand'],
        title='Chance constrained MPC, Power output')
    plt_SOC = TimeseriesPlot('Time', 'Battery SOC', title='Chance constrained MPC, Battery SOC')
    plt_inputs = TimeseriesPlot('Time', 'Control input', ['Gas turbine power', 'Battery current'],
                                title='Chance constrained MPC, control inputs')
    if plot_predictions:
        fig_gp, ax_gp = plt.subplots(2, sharex=True, num='Chance constrained MPC, wind prediction')

# save trajectories to file
dims = {'Power output': 4, 'Power demand': 1, 'SOC': 1, 'Inputs': 2} # State: 1
data_saver = DataSaving('chance_constrained_mpc', mpc_opt, gp_opt, dims)

# load trajectories if possible
start = 0
values, times_load = data_saver.load_trajectories()
if values is not None:
    P_out = values['Power output']
    P_demand = values['Power demand']
    P = ca.horzcat(P_out, P_demand)
    SOC_bat = values['SOC']
    inputs = values['Inputs']
    n_vals = P.shape[0]
    x = 1-SOC_bat
    t_last = datetime.datetime.strptime(times_load[-1][0]+times_load[-1][1], '%Y-%m-%d%H:%M:%S')
    times = [t for t in times if t > t_last]
    start = n_vals
    SOC_traj[:n_vals,:] = SOC_bat
    x_traj[:n_vals,:] = x
    u_traj[:n_vals,:] = inputs
    P_traj[:n_vals,:] = P
    x_k = x[-1]

for k, t in enumerate(times, start=start):
    # get parameters: predicted wind speed, power demand, initial state
    # wind_speeds = [data_handler.get_measurement(t, i) for i in range(chance_constrained_mpc.horizon)] # perfect forecast
    wind_speeds_nwp = [data_handler.get_NWP(t,i) for i in range(chance_constrained_mpc.horizon)]
    P_wtg = [ohps.n_wind_turbines*ohps.wind_turbine.power_curve_fun(ohps.wind_turbine.scale_wind_speed(w)) 
             for w in wind_speeds_nwp]
    # wind_speeds = ca.vertcat(*wind_speeds)
    train = (k==start) or (t.minute==0)
    wind_speeds_gp, var_gp = gp.predict_trajectory(t, chance_constrained_mpc.horizon, train, include_last_measurement=False)
    std_gp = np.sqrt(var_gp)
    if P_demand_last is not None:
        P_demand = ca.vertcat(P_demand_last[1:], P_wtg[-1] + 0.8*ohps.P_gtg_max)
    else:
        P_demand = ca.vertcat(*P_wtg) + 0.8*ohps.gtg.bounds['ubu']
    p = chance_constrained_mpc.get_p_fun(x_k, P_gtg_last, wind_speeds_gp, std_gp, P_demand)

    # get initial guess
    v_init = chance_constrained_mpc.get_initial_guess(p, v_last)

    # solve optimization problem
    v_opt = chance_constrained_mpc.solver(v_init, p)
    # get cost function value for tuning
    J_opt = chance_constrained_mpc.J_fun(v_opt, p)
    print(f'Total cost: {J_opt}')
    print(f'''GTG cost terms: J_gtg: {chance_constrained_mpc.J_gtg_fun(v_opt, p)}, 
          J_gtg_P: {chance_constrained_mpc.J_gtg_P_fun(v_opt, p)}, 
          J_gtg_eta: {chance_constrained_mpc.J_gtg_eta_fun(v_opt, p)}, 
          J_gtg_dP: {chance_constrained_mpc.J_gtg_dP_fun(v_opt, p)}''')
    print(f'Battery cost term: {chance_constrained_mpc.J_bat_fun(v_opt, p)}')
    print(f'Power slack variables: {chance_constrained_mpc.J_s_P_fun(v_opt, p)}')
    print(f'State slack variables: {chance_constrained_mpc.J_s_x_fun(v_opt, p)}')
    print(f'Control variables: {chance_constrained_mpc.J_u_fun(v_opt, p)}')

    u_opt = chance_constrained_mpc.get_u_from_v_fun(v_opt)
    u_k = u_opt[0,:]
    s_P_opt = chance_constrained_mpc.get_s_from_v_fun(v_opt)
    s_P_k = s_P_opt[0]
    # Simulate with low level controller adding uncertainty to battery
    i_opt, P_gtg_opt, x_next = llc.simulate(t, x_k, u_k, s_P_k, P_demand[0], P_gtg_last)
    u_k[0] = P_gtg_opt
    u_k[1] = i_opt

    # save state, input, SOC and power trajectories
    x_traj[k,:] = x_k
    u_traj[k,:] = u_k
    w_k = data_handler.get_measurement(t)
    P_gtg = ohps.get_P_gtg(x_k, u_k, w_k)
    P_bat = ohps.get_P_bat(x_k, u_k, w_k)
    P_wtg = ohps.get_P_wtg(x_k, u_k, w_k)
    P_total = P_gtg + P_bat + P_wtg
    P_traj[k,:] = ca.vertcat(P_gtg, P_bat, P_wtg, P_total, P_demand[0])
    SOC_traj[k] = ohps.get_SOC_bat(x_k, u_k, w_k)
    if plot:
        plt_inputs.plot(times_plot[:k], u_traj[:k,:])
        plt_power.plot(times_plot[:k], P_traj[:k,:])
        plt_SOC.plot(times_plot[:k], SOC_traj[:k])

        # Plot wind power
        wind_power_NWP = np.array([ohps.n_wind_turbines*ohps.wind_turbine.power_curve_fun(
            ohps.wind_turbine.scale_wind_speed(w)) for w in wind_speeds_nwp]).reshape(-1)
        wind_speeds_meas = [data_handler.get_measurement(t,i) for i in range(chance_constrained_mpc.horizon)]
        wind_power_meas = np.array([ohps.n_wind_turbines*ohps.wind_turbine.power_curve_fun(
            ohps.wind_turbine.scale_wind_speed(w)) for w in wind_speeds_meas]).reshape(-1)
        wind_power_gp = np.array([ohps.n_wind_turbines*ohps.wind_turbine.power_curve_fun(
            ohps.wind_turbine.scale_wind_speed(w)) for w in wind_speeds_gp]).reshape(-1)
        wind_power_gp_lb = np.array([ohps.n_wind_turbines*ohps.wind_turbine.power_curve_fun(
            ohps.wind_turbine.scale_wind_speed(w)) 
            for w in (np.array(wind_speeds_gp)-chance_constrained_mpc.back_off_factor*np.sqrt(
                np.array(var_gp)))]).reshape(-1)
        wind_power_gp_ub = np.array([ohps.n_wind_turbines*ohps.wind_turbine.power_curve_fun(
            ohps.wind_turbine.scale_wind_speed(w)) 
            for w in (np.array(wind_speeds_gp)+chance_constrained_mpc.back_off_factor*np.sqrt(
                np.array(var_gp)))]).reshape(-1)
        gp_times = [t + i*datetime.timedelta(minutes=mpc_opt['dt']) for i in range(mpc_opt['N'])]
        if plot_predictions:
            ax_gp[0].clear()
            ax_gp[1].clear()
            ax_gp[0].plot(gp_times, wind_speeds_gp)
            ax_gp[0].plot(gp_times, wind_speeds_nwp)
            ax_gp[0].plot(gp_times, wind_speeds_meas)
            ax_gp[0].fill_between(gp_times, wind_speeds_gp-chance_constrained_mpc.back_off_factor*std_gp,
                                wind_speeds_gp+chance_constrained_mpc.back_off_factor*std_gp,
                                color='tab:blue', alpha=0.4)
            ax_gp[1].plot(gp_times, wind_power_gp)
            ax_gp[1].plot(gp_times, wind_power_NWP)
            ax_gp[1].plot(gp_times, wind_power_meas)
            ax_gp[1].fill_between(gp_times, wind_power_gp_lb, wind_power_gp_ub,
                                color='tab:blue', alpha=0.4)
            ax_gp[0].set_ylabel('Wind speed (m/s)')
            ax_gp[1].set_ylabel('Wind power (kW)')
            fig_gp.legend(['GP', 'NWP', 'Measurement'])
    # save data
    P_k = ca.horzcat(P_gtg, P_bat, P_wtg, P_total)
    data_save = {'Power output': P_k, 'Power demand': P_demand[0], 'SOC': SOC_traj[k],
                 'Inputs': u_traj[k,:]}
    data_saver.save_trajectories(t, data_save)

    # Print out current SOC and power outputs
    print(f'time: {t.strftime("%d.%m.%Y %H:%M")}: Battery SOC: {SOC_traj[k]}')
    print(f'Gas turbine power output: {P_gtg}, Battery power output: {P_bat}, '
          f'Wind turbine power output: {P_wtg}, Total power: {P_total}, Demand: {P_demand[0]}')
    # simulate system
    x_k = ohps.get_next_state(x_k, u_k)

    # save last solution for next iteration
    v_last = v_opt
    P_gtg_last = P_gtg
    P_demand_last = P_demand
    plt.pause(0.5)
pass