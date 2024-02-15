import datetime

import numpy as np

from modules.mpc import get_mpc_opt
from modules.gp import get_gp_opt
from modules.mpc_scoring import get_power_error, get_gtg_power, get_gtg_emissions, DataSaving


def get_unsatisfied_demand_accurate(ohps, P_demand_vec, P_wtg_vec):
    dims = {'Power output': 4, 'Power demand': 1, 'SOC': 1, 'Inputs': 2}
    ds = DataSaving('baseline_controller', get_mpc_opt(), get_gp_opt(), dims)
    data, times = ds.load_trajectories()
    
    import casadi as ca
    ohps.battery.setup_integrator(600)
    x_0 = 0.5
    n_vals = len(P_demand_vec)
    P_unsatisfied = np.zeros(n_vals)
    P_bat_vec = np.zeros(n_vals)
    P_gtg_vec = np.zeros(n_vals)
    x_vec = np.zeros(n_vals)
    SOC_vec = np.zeros(n_vals)
    P_gtg = ca.MX.sym('P_gtg')
    i_bat = ca.MX.sym('P_gtg')
    P_demand = ca.MX.sym('P_demand')
    P_wtg = ca.MX.sym('P_wtg')
    x_bat = ca.MX.sym('x_bat')
    P_total = P_gtg+P_wtg+ohps.get_P_bat(x_bat, i_bat, 0)
    x_bat_next = ohps.battery.get_next_state(x_bat, i_bat)
    SOC_bat_next = ohps.battery.get_SOC_fun(x_bat_next, i_bat)
    J = 100*(P_demand-P_total)**2 - SOC_bat_next
    opt_problem = {
        'x': ca.vertcat(P_gtg, i_bat),
        'f': J,
        'g': x_bat_next,
        'p': ca.vertcat(P_wtg, P_demand, x_bat)
        }
    x_lb = ca.vertcat(0, -32000)
    x_ub = ca.vertcat(32000, 32000)
    g_lb = 0.1
    g_ub = 0.9
    x_init = ca.vertcat(0.8*32000, 0)
    ipopt_opt = {
        'print_level': 0,
        'print_frequency_time': 100
        }
    nlp_opt = {'ipopt': ipopt_opt}
    solver = ca.nlpsol('simple_controller', 'ipopt', opt_problem, nlp_opt)

    x_k = x_0
    for k in range(n_vals-1):
        p_k = ca.vertcat(P_wtg_vec[k], P_demand_vec[k], x_k)
        u_opt = solver(x0=x_init, p=p_k, lbx=x_lb, ubx=x_ub, lbg=g_lb, ubg=g_ub)['x']
        P_bat_opt = ohps.get_P_bat(x_k, u_opt[1], 0)
        P_gtg_vec[k] = u_opt[0]
        P_bat_vec[k] = P_bat_opt
        P_unsatisfied[k] = np.abs(P_demand_vec[k]-P_wtg_vec[k]-u_opt[0]-P_bat_opt)
        x_vec[k] = x_k
        SOC_vec[k] = ohps.battery.get_SOC_fun(x_k, u_opt[1])
        x_k = ohps.battery.get_next_state(x_k, u_opt[1])
        print(f'{k}: {SOC_vec[k]}')
    P_unsatisfied_acc = np.sum(P_unsatisfied) 
    E_unsatisfied_rel = P_unsatisfied_acc/np.sum(P_demand_vec)
    
    return E_unsatisfied_rel, P_gtg_vec, P_bat_vec, SOC_vec

mpc_opt = get_mpc_opt(N=30)
gp_opt = get_gp_opt(dt_pred = mpc_opt['dt'])
# mpc_opt['param']['']

t_start = datetime.datetime(2022,7,1)
t_end = datetime.datetime(2022,9,30)
mpc_opt['t_start'] = t_start
mpc_opt['t_end'] = t_end

run_ids = ['3b0c0d0eb3', '4f0af55c7e', '463cae5633', '1730492995', 'a24a904f3d', 'a91bd79437', 'bee08ddbaf', 'd02bfc4014']

array_dims = {'Power output': 4, 'Power demand': 1, 'SOC': 1, 'Inputs': 2}
dl = DataSaving('nominal_mpc_perfect_forecast', mpc_opt, gp_opt, array_dims, run_ids[0])
data, times = dl.load_trajectories()

P_demand = data['Power demand']
P_wtg = data['Power output'][:,2]
from modules.models import OHPS
ohps = OHPS()
P_error, P_gtg_vec, P_bat_vec, SOC_vec = get_unsatisfied_demand_accurate(ohps, P_demand, P_wtg)
P_gtg_abs = np.sum(P_gtg_vec/6000)
P_total = P_wtg+P_gtg_vec+P_bat_vec
P_generated = P_gtg_vec + P_wtg
P_gtg_rel = P_gtg_abs/np.sum(P_generated/6000)
eta_gtg_vec = np.array([ohps.gtg.eta_fun(P_gtg/32000) for P_gtg in P_gtg_vec]).reshape(-1)
P_in_gtg = np.mean(P_gtg_vec[:-1]/eta_gtg_vec[:-1])
eta_gtg = np.mean(eta_gtg_vec)
print(f'Perfect forecast: {np.sum(np.abs(P_total-P_demand.reshape(-1)))/6000} MWh total error, {100*P_error}% of demand')
print(f'GTG power: {P_gtg_abs}, {P_gtg_rel*100}% of total generated power')
print(f'Mean power demand of GTG: {P_in_gtg}, average efficiency: {eta_gtg}')

for run_id in run_ids:
    error_abs_pf, error_rel_pf = get_power_error('nominal_mpc_perfect_forecast', mpc_opt, gp_opt, run_id)
    P_gtg_abs, P_gtg_rel = get_gtg_power('nominal_mpc_perfect_forecast', mpc_opt, gp_opt, run_id)
    P_in_gtg, eta_gtg = get_gtg_emissions('nominal_mpc_perfect_forecast', mpc_opt, gp_opt, run_id)
    print(run_id)
    print(f'Perfect forecast: {error_abs_pf} MWh total error, {100*error_rel_pf}% of demand')
    print(f'GTG power: {P_gtg_abs}, {P_gtg_rel*100}% of total generated power')
    print(f'Mean power demand of GTG: {P_in_gtg}, average efficiency: {eta_gtg}')

# import matplotlib.pyplot as plt
# plt.figure()
# times = [t_start+i*datetime.timedelta(minutes=10) for i in range(len(P_wtg))]
# plt.plot(times, np.array([P_wtg, P_bat_vec, P_gtg_vec, P_total, P_demand.reshape((-1))]).T)
# plt.figure()
# plt.plot(SOC_vec)
# pass