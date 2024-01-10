import datetime

import casadi as ca
import numpy as np

from modules.gp import TimeseriesModel, DataHandler, get_gp_opt
from modules.models import OHPS
from modules.mpc import MultistageMPC, get_mpc_opt
from modules.plotting import TimeseriesPlot

ohps = OHPS()

mpc_opt = get_mpc_opt()
gp_opt = get_gp_opt(dt_pred = mpc_opt['dt'], steps_forward = mpc_opt['N'])
gp = TimeseriesModel(gp_opt)
ohps.setup_integrator(dt=60*mpc_opt['dt'])

multistage_mpc = MultistageMPC(ohps, gp, mpc_opt)

t_start = datetime.datetime(2022, 1, 1)
t_end = datetime.datetime(2022,2,1)
dt = datetime.timedelta(minutes=mpc_opt['dt'])
n_times = int((t_end-t_start)/dt)
times = [t_start + i*dt for i in range(n_times)]

x_traj = ca.DM.zeros(n_times, ohps.nx)
u_traj = ca.DM.zeros(n_times, ohps.nu)
P_traj = ca.DM.zeros(n_times, 5)    # gtg, battery, wtg, total, demand
SOC_traj = ca.DM.zeros(n_times)

data_handler = DataHandler(datetime.datetime(2020,1,1), datetime.datetime(2022,12,31), gp_opt)
x_k = ohps.x0
v_last = None
v_init_next = None

# create plots
plt_power = TimeseriesPlot('Time', 'Power output', #
    ['Gas turbine', 'Battery', 'Wind turbine', 'Total power generation', 'Demand'])
plt_SOC = TimeseriesPlot('Time', 'Battery SOC')
plt_inputs = TimeseriesPlot('Time', 'Control input', ['Gas turbine power', 'Battery current'])

for k, t in enumerate(times):
    # get parameters: predicted wind speed, power demand, initial state
    wind_speeds = [data_handler.get_measurement(t, i) for i in range(multistage_mpc.horizon)] # perfect forecast
    P_wtg = [ohps.wind_turbine.power_curve_fun(ohps.wind_turbine.scale_wind_speed(w)) for w in wind_speeds]
    wind_speeds = ca.vertcat(*wind_speeds)
    train = (k==0) or (t.minute==0)
    wind_speeds_gp, var_gp = gp.predict_trajectory(t, multistage_mpc.horizon, train)

    std_gp = np.sqrt(var_gp)
    P_demand = 1.5*ca.vertcat(*P_wtg)
    P_demand = 8000*ca.DM.ones(multistage_mpc.horizon)

    multistage_mpc.get_optimization_problem(t)

    p = multistage_mpc.get_parameters(x_k, P_demand)

    if v_init_next is None:
        v_init = multistage_mpc.get_initial_guess(v_last, P_wtg, x_k, P_demand)
        # v_init = multistage_mpc.get_initial_guess(None, None, x_k, P_demand)

    v_opt = multistage_mpc.solver(v_init, p)

    u_k = multistage_mpc.get_u_next_fun(v_opt)

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

    plt_inputs.plot(times[:k], u_traj[:k,:])
    plt_power.plot(times[:k], P_traj[:k,:])
    plt_SOC.plot(times[:k], SOC_traj[:k])

    # Print out current SOC and power outputs
    print(f'time: {t.strftime("%d.%m.%Y %H:%M")}: Battery SOC: {SOC_traj[k]}')
    print(f'Gas turbine power output: {P_gtg}, Battery power output: {P_bat}, '
          f'Wind turbine power output: {P_wtg}, Total power: {P_total}, Demand: {P_demand[0]}')
    # simulate system
    x_k = ohps.get_next_state(x_k, u_k)

    # save last solution for next iteration
    v_last = multistage_mpc.get_v_middle_fun(v_opt)
    # v_init_next = multistage_mpc.get_initial_guess(v_last, P_wtg, x_k, P_demand)
    
    # TODO: for simulation: maybe use smaller time scale and vary wind speed for each subinterval 
    # as wind power is not simply a function of the mean wind speed, 
    # possibly account for this uncertainty in gp

pass
# gp_opt = get_gp_opt()
# ohps.setup_integrator(dt=gp_opt['dt_pred'])
# gp = TimeseriesModel(gp_opt)
# mpc_opt = get_mpc_opt()
# mpc = MultistageMPC(ohps, gp, mpc_opt)

# t = datetime.datetime(2022,2,1)
# #means_list, vars_list, parent_nodes, probabilities = mpc.generate_scenario(t, True)
# #optimization_tree = mpc.build_optimization_tree(parent_nodes, probabilities, means_list, vars_list)
# mpc.get_optimization_problem(t, True)
# pass
