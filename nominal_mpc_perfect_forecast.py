import datetime

import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

from modules.mpc import NominalMPC, DayAheadScheduler, get_mpc_opt
from modules.models import OHPS
from modules.gp import DataHandler, get_gp_opt
from modules.plotting import TimeseriesPlot
from modules.mpc_scoring import DataSaving

ohps = OHPS()

mpc_opt = get_mpc_opt(N=30, use_soft_constraints_state=False)
t_start = mpc_opt['t_start']
t_end = mpc_opt['t_end']
# mpc_opt['param']['k_gtg_P'] = 10
# mpc_opt['param']['k_gtg_eta'] = 50
# mpc_opt['param']['k_gtg_dP'] = 1
# mpc_opt['param']['k_gtg_fuel'] = 0
# mpc_opt['param']['k_bat'] = 1
# mpc_opt['param']['R_input'] = ca.diag([0,1e-8])
nominal_mpc = NominalMPC(ohps, mpc_opt)

nominal_mpc.get_optimization_problem()

gp_opt = get_gp_opt(dt_pred = mpc_opt['dt'])
# gp = WindPredictionGP(gp_opt)



dt = datetime.timedelta(minutes=mpc_opt['dt'])
n_times = int((t_end-t_start)/dt)
times = [t_start + i*dt for i in range(n_times)]
times_plot = times

x_traj = ca.DM.zeros(n_times, nominal_mpc.nx)
u_traj = ca.DM.zeros(n_times, nominal_mpc.nu)
P_traj = ca.DM.zeros(n_times, 5)    # gtg, battery, wtg, total, demand
SOC_traj = ca.DM.zeros(n_times)

data_handler = DataHandler(datetime.datetime(2020,1,1), datetime.datetime(2022,12,31), gp_opt)
x_k = ohps.x0
P_gtg_last = ohps.gtg.bounds['ubu']
v_last = None
P_demand_last = None
dE = 0
E_tot = 0
scheduler = DayAheadScheduler(ohps, data_handler, mpc_opt)

# # create plots
plt_power = TimeseriesPlot('Time', 'Power output',
    ['Gas turbine', 'Battery', 'Wind turbine', 'Total power generation', 'Demand'],
    title = 'Nominal MPC with perfect forecast, Power output')
plt_SOC = TimeseriesPlot('Time', 'Battery SOC', title = 'Nominal MPC with perfect forecast, Battery SOC')
plt_inputs = TimeseriesPlot('Time', 'Control input', ['Gas turbine power', 'Battery current'],
                            title = 'Nominal MPC with perfect forecast, Control inputs')
fig_E_tot, ax_E_tot = plt.subplots()
# save trajectories to file
dims = {'Power output': 4, 'Power demand': 1, 'SOC': 1, 'Inputs': 2}

data_saver = DataSaving('nominal_mpc_perfect_forecast', mpc_opt, gp_opt, dims)

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
    wind_speeds = [data_handler.get_measurement(t, i) for i in range(nominal_mpc.horizon)] # perfect forecast
    wind_speeds_nwp = [data_handler.get_NWP(t, i) for i in range(nominal_mpc.horizon)]
    P_wtg = [4*ohps.wind_turbine.power_curve_fun(ohps.wind_turbine.scale_wind_speed(w)) for w in wind_speeds_nwp]
    wind_speeds = ca.vertcat(*wind_speeds)
    if P_demand_last is not None:
        P_demand = ca.vertcat(P_demand_last[1:], P_wtg[-1] + 0.8*ohps.P_gtg_max)
    else:
        P_demand = ca.vertcat(*P_wtg) + 0.8*ohps.gtg.bounds['ubu']
    # P_demand = scheduler.get_P_demand(t, x_k, dE)
    # P_demand = 8000*ca.DM.ones(nominal_mpc.horizon)
    p = nominal_mpc.get_p_fun(x_k, P_gtg_last, wind_speeds, P_demand)

    # get initial guess
    v_init = nominal_mpc.get_initial_guess(p, v_last)

    # solve optimization problem
    v_opt = nominal_mpc.solver(v_init, p)
    # get cost function value for tuning
    J_opt = nominal_mpc.J_fun(v_opt, p)
    print(f'Total cost: {J_opt}')
    print(f'''GTG cost terms: J_gtg: {nominal_mpc.J_gtg_fun(v_opt, p)}, 
          J_gtg_P: {nominal_mpc.J_gtg_P_fun(v_opt, p)}, 
          J_gtg_eta: {nominal_mpc.J_gtg_eta_fun(v_opt, p)}, 
          J_gtg_dP: {nominal_mpc.J_gtg_dP_fun(v_opt, p)}''')
    print(f'Battery cost term: {nominal_mpc.J_bat_fun(v_opt, p)}')
    print(f'Slip variables: {nominal_mpc.J_s_P_fun(v_opt, p)}')
    print(f'Control variables: {nominal_mpc.J_u_fun(v_opt, p)}')


    u_opt = nominal_mpc.get_u_from_v_fun(v_opt)
    u_k = u_opt[0,:]

    # save state, input, SOC and power trajectories
    x_traj[k,:] = x_k
    u_traj[k,:] = u_k
    w_k = data_handler.get_measurement(t)
    P_gtg = ohps.get_P_gtg(x_k, u_k, w_k)
    P_bat = ohps.get_P_bat(x_k, u_k, w_k)
    P_wtg = ohps.get_P_wtg(x_k, u_k, w_k)
    P_total = P_gtg + P_bat + P_wtg
    E_tot += P_total/6
    P_k = ca.horzcat(P_gtg, P_bat, P_wtg, P_total)
    P_traj[k,:] = ca.vertcat(P_gtg, P_bat, P_wtg, P_total, P_demand[0])
    SOC_traj[k] = ohps.get_SOC_bat(x_k, u_k, w_k)

    # dE += P_demand[0]-P_total
    plt_inputs.plot(times_plot[:k], u_traj[:k,:])
    plt_power.plot(times_plot[:k], P_traj[:k,:])
    plt_SOC.plot(times_plot[:k], SOC_traj[:k])
    # ax_E_tot.clear()
    # ax_E_tot.plot(times[:k], ca.cumsum(P_traj[:k,-1]/6000))
    # ax_E_tot.plot(times[:k], 40/6*(np.arange(k)+1), '--', color='black')
    # ax_E_tot.set_xlabel('Time')
    # ax_E_tot.set_ylabel('Generated energy (MWh)')
    # save data
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
    dE = 40000/6*(k+1)
    plt.pause(0.1)

pass