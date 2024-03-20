import datetime

import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from modules.gp import TimeseriesModel, DataHandler, get_gp_opt
from modules.models import OHPS
from modules.mpc import MultistageMPCLoadShifting, DayAheadScheduler, LowLevelController, get_mpc_opt
from modules.plotting import TimeseriesPlot
from modules.mpc_scoring import DataSaving

plot = False
plot_predictions = False
plt.ion()
ohps = OHPS(N_p=8000)

epsilon = 0.1
std_factor = norm.ppf(1-epsilon)
std_list = (-std_factor, 0, std_factor)

mpc_opt = get_mpc_opt(N=30, std_list_multistage=std_list, use_simple_scenarios=False, dE_min=5000, t_start_sim=datetime.datetime(2022,1,1), use_soft_constraints_state=False, include_last_measurement=True)#,  t_start=datetime.datetime(2022,12,6), t_end=datetime.datetime(2022,12,8))
mpc_opt['param']['k_dP'] = 10
mpc_opt['param']['r_s_E'] = 100
mpc_opt['param']['k_bat'] = 0
mpc_opt['use_path_constraints_energy'] = True
gp_opt = get_gp_opt(dt_pred = mpc_opt['dt'], steps_forward = mpc_opt['N'], verbose=False, do_not_load_predictions=True)
gp = TimeseriesModel(gp_opt)
del gp_opt['do_not_load_predictions']
ohps.setup_integrator(dt=60*mpc_opt['dt'])

multistage_mpc = MultistageMPCLoadShifting(ohps, gp, mpc_opt)

t_start = mpc_opt['t_start_sim']
t_end = mpc_opt['t_end_sim']

dt = datetime.timedelta(minutes=mpc_opt['dt'])
n_times = int((t_end-t_start)/dt)
times = [t_start + i*dt for i in range(n_times)]
times_plot = times

x_traj = ca.DM.zeros(n_times, ohps.nx)
u_traj = ca.DM.zeros(n_times, ohps.nu)
P_traj = ca.DM.zeros(n_times, 5)    # gtg, battery, wtg, total, demand
SOC_traj = ca.DM.zeros(n_times)

data_handler = DataHandler(datetime.datetime(2020,1,1), datetime.datetime(2022,12,31), gp_opt)
llc = LowLevelController(ohps, data_handler, mpc_opt)

x_k = ohps.x0
v_last = None
v_init_next = None
P_out_last = 40000
P_min = 16000
P_gtg_last = ohps.gtg.bounds['ubu']
dE = 0
E_tot = 0
E_sched = 0
dE_sched = 0
E_shifted = 0
E_target_lt = np.array([])
scheduler = DayAheadScheduler(ohps, data_handler, mpc_opt)

# create plots
if plot:
    plt_power = TimeseriesPlot('Time', 'Power output', 
        ['Gas turbine', 'Battery', 'Wind turbine', 'Total power generation', 'Demand'],
        title = 'Multi-stage MPC, Power output')
    plt_SOC = TimeseriesPlot('Time', 'Battery SOC', title='Multi-stage MPC, Battery SOC')
    plt_inputs = TimeseriesPlot('Time', 'Control input', ['Gas turbine power', 'Battery current'],
                                title = 'Multi-stage MPC, control inputs')
    plt_wind_pred = TimeseriesPlot('Time', 'Wind speed (m/s)', title='Multi-stage MPC, wind speed scenarios')
    plt_wind_P_pred = TimeseriesPlot('Time', 'Power (kW)', title='Multi-stage MPC, wind power scenarios')
    if plot_predictions:
        fig_pred, ax_pred = plt.subplots(2, sharex=True, num='Multi-stage MPC, wind predictions')
        fig_E_tot, ax_E_tot = plt.subplots(2, sharex=True, num='Multi-stage MPC, total Energy output')
# save trajectories to file
dims = {'Power output': 4, 'Power demand': 1, 'SOC': 1, 'Inputs': 2}
data_saver = DataSaving('multi-stage_mpc_shifting_without_llc_10MWh', mpc_opt, gp_opt, dims)

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
    P_out_last = P_out[-1,-1]
    P_gtg_last = P_out[-1,0]
    E_tot = ca.sum1(P_traj[:,-2]/6)
    E_sched = ca.sum1(P_traj[:,-1])
    E_target_lt = np.array([scheduler.get_E_target_lt(t)/1000 for t in times_plot if t <= t_last])
    dE = scheduler.get_E_target_lt(t_last)-E_tot # difference between generated energy and long-time average
    dE_sched = E_sched-6*E_tot   # difference between generated and scheduled energy


for k, t in enumerate(times, start=start):
    # get parameters: predicted wind speed, power demand, initial state
    wind_speeds_nwp = [data_handler.get_NWP(t, i) for i in range(multistage_mpc.horizon)]
    P_wtg = [4*ohps.wind_turbine.power_curve_fun(ohps.wind_turbine.scale_wind_speed(w)) for w in wind_speeds_nwp]
    wind_power_nwp = np.array([ohps.get_P_wtg(0,0,w) for w in wind_speeds_nwp]).reshape(-1)
    P_demand = scheduler.get_P_demand(t, x_k, dE)
    E_sched += P_demand[0]
    E_target = ca.sum1(P_demand) + dE_sched # total scheduled demand plus compensation for previously not satisfied demand

    train = (k==start) or (t.minute==0)
    # wind_speeds_gp, var_gp = gp.predict_trajectory(t, multistage_mpc.horizon, train)
    # std_gp = np.sqrt(var_gp)
    
    multistage_mpc.get_optimization_problem(t, train)

    p = multistage_mpc.get_parameters(x_k, P_gtg_last, P_out_last, P_min, E_shifted, P_demand, 10000)

    if v_init_next is None:
        v_init = multistage_mpc.get_initial_guess(v_last, P_wtg, x_k, P_demand)
        # v_init = multistage_mpc.get_initial_guess(None, None, x_k, P_demand)

    v_opt = multistage_mpc.solver(v_init, p)

    J_opt = multistage_mpc.J_fun(v_opt, p)
    print(f'Total cost: {J_opt}')
    print(f'''GTG cost terms: J_gtg: {multistage_mpc.J_gtg_fun(v_opt, p)}, 
          J_gtg_P: {multistage_mpc.J_gtg_P_fun(v_opt, p)}, 
          J_gtg_eta: {multistage_mpc.J_gtg_eta_fun(v_opt, p)}, 
          J_gtg_dP: {multistage_mpc.J_gtg_dP_fun(v_opt, p)}''')
    print(f'Battery cost term: {multistage_mpc.J_bat_fun(v_opt, p)}')
    print(f'Slip variables: {multistage_mpc.J_s_E_fun(v_opt, p)}')
    print(f'Control variables: {multistage_mpc.J_u_fun(v_opt, p)}')

    u_k = multistage_mpc.get_u_next_fun(v_opt)
    P_out = multistage_mpc.get_P_next_fun(v_opt)
    # s_P_k = multistage_mpc.get_s_P_next_fun(v_opt)
    # Simulate with low level controller adding uncertainty to battery
    # i_opt, P_gtg_opt, x_next = llc.simulate(t, x_k, u_k, 0, P_out)
    # u_k[0] = P_gtg_opt
    # u_k[1] = i_opt

    # save state, input, SOC and power trajectories
    x_traj[k,:] = x_k
    u_traj[k,:] = u_k
    w_k = data_handler.get_measurement(t)
    P_gtg = ohps.get_P_gtg(x_k, u_k, w_k)
    P_bat = ohps.get_P_bat(x_k, u_k, w_k)
    P_wtg = ohps.get_P_wtg(x_k, u_k, w_k)
    P_total = P_gtg + P_bat + P_wtg
    E_shifted += (P_total-P_demand[0])*1/6
    E_tot += P_total/6
    P_traj[k,:] = ca.vertcat(P_gtg, P_bat, P_wtg, P_total, P_demand[0])
    SOC_traj[k] = ohps.get_SOC_bat(x_k, u_k, w_k)
    if plot:
        plt_inputs.plot(times_plot[:k], u_traj[:k,:])
        plt_power.plot(times_plot[:k], P_traj[:k,:])
        plt_SOC.plot(times_plot[:k], SOC_traj[:k])

        means = [np.array(multistage_mpc.means[-1])]
        means_P = [np.array([ohps.get_P_wtg(0,0,w) for w in multistage_mpc.means[-1]]).reshape(-1)]
        parents_i = multistage_mpc.parent_nodes[-1]
        # plot scenarios
        if plot_predictions:
            for i in range(2, multistage_mpc.horizon+1):
                means_parents = np.array(multistage_mpc.means[-(i)])[parents_i]
                mean_wind_power = np.array([ohps.get_P_wtg(0,0,w) for w in means_parents]).reshape(-1)
                parents_i = np.array(multistage_mpc.parent_nodes[-(i)])[parents_i]
                means.append(means_parents)
                means_P.append(mean_wind_power)
            means.reverse()
            means_P.reverse()
            means = np.array(means)
            means_P = np.array(means_P)
            times_plot_power = [t+i*datetime.timedelta(minutes=mpc_opt['dt']) for i in range(multistage_mpc.horizon)]
            ax_pred[0].clear()
            ax_pred[1].clear()
            plt_sc = ax_pred[0].plot(times_plot_power, means, color='tab:blue', label='Scenarios')[0]
            ax_pred[1].plot(times_plot_power, means_P, color='tab:blue', label='Scenarios')
            meas = np.array([data_handler.get_measurement(t, i) for i in range(multistage_mpc.horizon)])
            P_meas = np.array([ohps.get_P_wtg(0,0,w) for w in meas]).reshape(-1)
            gp_mean, gp_var = gp.predict_trajectory(t, multistage_mpc.horizon)
            P_mean = np.array([ohps.get_P_wtg(0,0,w) for w in gp_mean]).reshape(-1)
            P_lower = np.array([ohps.get_P_wtg(0,0,w-std_factor*np.sqrt(v)) for w, v in zip(gp_mean, gp_var)]).reshape(-1)
            P_upper = np.array([ohps.get_P_wtg(0,0,w+std_factor*np.sqrt(v)) for w, v in zip(gp_mean, gp_var)]).reshape(-1)
            plt_meas, = ax_pred[0].plot(times_plot_power, meas, color='tab:green', label='Measurement')
            ax_pred[1].plot(times_plot_power, P_meas, color='tab:green', label='Measurement')
            plt_gp, = ax_pred[0].plot(times_plot_power, gp_mean, color='tab:orange', label='GP mean and confidence interval')
            ax_pred[1].plot(times_plot_power, P_mean, color='tab:orange', label='GP mean and confidence interval')
            ax_pred[0].fill_between(times_plot_power, gp_mean-std_factor*np.sqrt(gp_var), 
                gp_mean+std_factor*np.sqrt(gp_var), color='tab:orange', alpha=0.4)
            ax_pred[1].fill_between(times_plot_power, P_lower, P_upper, color='tab:orange', alpha=0.4)
            plt_nwp, = ax_pred[0].plot(times_plot_power, np.array(wind_speeds_nwp).reshape(-1), color='tab:red', label='NWP')
            ax_pred[1].plot(times_plot_power, np.array(wind_power_nwp), color='tab:red', label='NWP')
            ax_pred[0].set_ylabel('Wind speed (m/s)')
            ax_pred[1].set_ylabel('Wind power (kW)')
            ax_E_tot[0].clear()
            ax_E_tot[1].clear()
            E_target_lt = np.append(E_target_lt, scheduler.get_E_target_lt(t)/1000)
            ax_E_tot[0].plot(times_plot[:k+1], ca.cumsum(P_traj[:k+1,-2]/6000))
            ax_E_tot[0].plot(times_plot[:k+1], E_target_lt.reshape(-1), '--', color='black')
            ax_E_tot[0].set_xlabel('Time')
            ax_E_tot[0].set_ylabel('Generated energy (MWh)')
            ax_E_tot[1].plot(times_plot[:k+1], ca.cumsum(P_traj[:k+1,-2]/6000)-ca.cumsum(P_traj[:k+1,-1]/6000))
            ax_E_tot[0].set_xlabel('Time')
            ax_E_tot[1].set_ylabel('Shifted Energy demand (MWh)')
    # Print out current SOC and power outputs
    print(f'time: {t.strftime("%d.%m.%Y %H:%M")}: Battery SOC: {SOC_traj[k]}')
    print(f'Gas turbine power output: {P_gtg}, Battery power output: {P_bat}, '
          f'Wind turbine power output: {P_wtg}, Total power: {P_total}, Demand: {P_demand[0]}')
    # fig_pred.legend(handles=[plt_nwp, plt_meas, plt_sc, plt_gp])
    # save data
    P_k = ca.horzcat(P_gtg, P_bat, P_wtg, P_total)
    data_save = {'Power output': P_k, 'Power demand': P_demand[0], 'SOC': SOC_traj[k],
                 'Inputs': u_traj[k,:]}
    data_saver.save_trajectories(t, data_save)

    # simulate system
    x_k = ohps.get_next_state(x_k, u_k)

    # save last solution for next iteration
    v_last = multistage_mpc.get_v_middle_fun(v_opt)
    # v_init_next = multistage_mpc.get_initial_guess(v_last, P_wtg, x_k, P_demand)
    P_out_last = P_total
    dE = scheduler.get_E_target_lt(t) - E_tot # difference between generated energy and long-time average
    dE_sched = E_sched-6*E_tot   # difference between generated and scheduled energy
    P_gtg_last = P_gtg
    if plot:
        plt.pause(0.5)
pass