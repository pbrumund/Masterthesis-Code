import datetime

import casadi as ca

from modules.mpc import NominalMPC, get_mpc_opt
from modules.models import OHPS
from modules.gp import PriorOnTimeseriesGP as WindPredictionGP
from modules.gp import DataHandler
from modules.gp import get_gp_opt
from modules.plotting import TimeseriesPlot

ohps = OHPS()

mpc_opt = get_mpc_opt()
nominal_mpc = NominalMPC(ohps, mpc_opt)
nominal_mpc.get_optimization_problem()

gp_opt = get_gp_opt(dt_pred = mpc_opt['dt'], steps_forward = mpc_opt['N'])
gp = WindPredictionGP(gp_opt)

t_start = datetime.datetime(2022, 1, 1)
t_end = datetime.datetime(2022,2,1)
dt = datetime.timedelta(minutes=mpc_opt['dt'])
n_times = int((t_end-t_start)/dt)
times = [t_start + i*dt for i in range(n_times)]

x_traj = ca.DM.zeros(n_times, nominal_mpc.nx)
u_traj = ca.DM.zeros(n_times, nominal_mpc.nu)
P_traj = ca.DM.zeros(n_times, 5)    # gtg, battery, wtg, total, demand
SOC_traj = ca.DM.zeros(n_times)

data_handler = DataHandler(datetime.datetime(2020,1,1), datetime.datetime(2022,12,31), gp_opt)
x_k = ohps.x0
v_last = None

# create plots
plt_power = TimeseriesPlot('Time', 'Power output', #
    ['Gas turbine', 'Battery', 'Wind turbine', 'Total power generation', 'Demand'])
plt_SOC = TimeseriesPlot('Time', 'Battery SOC')
plt_inputs = TimeseriesPlot('Time', 'Control input', ['Gas turbine power', 'Battery current'])

for k, t in enumerate(times):
    # get parameters: predicted wind speed, power demand, initial state
    wind_speeds = [data_handler.get_measurement(t, i) for i in range(nominal_mpc.horizon)] # perfect forecast
    P_wtg = [ohps.wind_turbine.power_curve_fun(ohps.wind_turbine.scale_wind_speed(w)) for w in wind_speeds]
    wind_speeds = ca.vertcat(*wind_speeds)
    train = (k==0) or (t.minute==0)
    wind_speeds_gp = gp.predict_trajectory(t, nominal_mpc.horizon, train)[0] # mean predicted by gp
    P_demand = 1.5*ca.vertcat(*P_wtg)
    P_demand = 8000*ca.DM.ones(nominal_mpc.horizon)
    p = nominal_mpc.get_p_fun(x_k, wind_speeds_gp, P_demand)

    # get initial guess
    v_init = nominal_mpc.get_initial_guess(p, v_last)

    # solve optimization problem
    v_opt = nominal_mpc.solver(v_init, p)

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
    v_last = v_opt
    
    # TODO: for simulation: maybe use smaller time scale and vary wind speed for each subinterval 
    # as wind power is not simply a function of the mean wind speed, 
    # possibly account for this uncertainty in gp

pass