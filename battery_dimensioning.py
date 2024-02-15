import datetime

import numpy as np
import matplotlib.pyplot as plt

from modules.gp import DataHandler, get_gp_opt
from modules.models import OHPS

def get_unsatisfied_demand(E_bat_max):
    E_bat = np.zeros(n_vals)
    E_bat[0] = E_bat_max
    P_unsatisfied = np.zeros(n_vals)
    P_bat_vec = np.zeros(n_vals)
    E_bat_min = 0
    for i in range(n_vals-1):
        E_bat_next = E_bat[i] + P_diff_max[i]/6 # 10 minute steps, energy in kWh
        E_bat_next = min(E_bat_next, E_bat_max) # battery charge limited
        E_bat_next = max(E_bat_next, E_bat_min)
        P_bat = (E_bat[i]-E_bat_next)*6
        P_bat_vec[i] = P_bat
        P_total = P_bat + P_generated_max[i]
        P_demand_i = P_demand[i]
        P_res = P_demand_i - P_total # if > 0: unable to satisfy demand
        P_unsatisfied[i] = max(P_res, 0)
        E_bat[i+1] = E_bat_next
    P_unsatisfied_acc = np.sum(P_unsatisfied) 
    E_unsatisfied_rel = P_unsatisfied_acc/np.sum(P_demand)
    return E_unsatisfied_rel

def get_unsatisfied_demand_accurate(ohps, P_demand_vec, P_wtg_vec):
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
        'print_level': 0
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

if __name__ == '__main__':
    plt.ion()
    ohps = OHPS()
    ohps.setup_integrator(600)
    t_start_data = datetime.datetime(2020,1,1)
    t_end_data = datetime.datetime(2022,12,31)
    opt = get_gp_opt()
    data_handler = DataHandler(t_start_data, t_end_data, opt)
    times = data_handler.weather_data['times_meas']
    # Only use data from 2020 and 2021, test on 2022 data
    t_start = datetime.datetime(2022,1,1)
    t_end = datetime.datetime(2022,12,31)
    times = times[times>t_start]
    times = times[times<t_end]
    # get power demand and actual wind power
    try:
        wind_power_nwp = np.loadtxt(f'data/wind_power_nwp_{t_start.year}_{t_end.year}.csv')
        wind_power_actual = np.loadtxt(f'data/wind_power_actual_{t_start.year}_{t_end.year}.csv')
    except:
        wind_speeds_nwp = [data_handler.get_NWP(t) for t in times]
        wind_speeds_meas = [data_handler.get_measurement(t) for t in times]
        
        wind_power_nwp = np.array([ohps.get_P_wtg(0, 0, w) for w in wind_speeds_nwp])
        wind_power_actual = np.array([ohps.get_P_wtg(0, 0, w) for w in wind_speeds_meas])

        wind_power_nwp = wind_power_nwp.reshape(-1)
        wind_power_actual = wind_power_actual.reshape(-1)

        np.savetxt(f'data/wind_power_nwp_{t_start.year}_{t_end.year}.csv', wind_power_nwp)
        np.savetxt(f'data/wind_power_actual_{t_start.year}_{t_end.year}.csv', wind_power_actual)
    n_vals = len(wind_power_actual)

    reserve_factor = 0.2
    P_gtg_nom = (1-reserve_factor)*ohps.P_gtg_max
    P_demand = wind_power_nwp + P_gtg_nom
    P_generated_uncompensated = wind_power_actual + P_gtg_nom
    P_generated_max = wind_power_actual + ohps.P_gtg_max
    P_diff_max = P_generated_max - P_demand # maximum charging power/minimum discharging power

    # assume always generating P_diff_max to charge battery if possible
    E_bat = np.zeros(n_vals)
    P_unsatisfied = np.zeros(n_vals)
    P_bat_vec = np.zeros(n_vals)
    E_bat_min = 0
    E_bat_max = 40000 #kWh
    E_bat[0] = E_bat_max
    for i in range(n_vals-1):
        E_bat_next = E_bat[i] + P_diff_max[i]/6 # 10 minute steps, energy in kWh
        E_bat_next = min(E_bat_next, E_bat_max) # battery charge limited
        E_bat_next = max(E_bat_next, E_bat_min)
        P_bat = (E_bat[i]-E_bat_next)*6
        P_bat_vec[i] = P_bat
        P_total = P_bat + P_generated_max[i]
        P_demand_i = P_demand[i]
        P_res = P_demand_i - P_total # if > 0: unable to satisfy demand
        P_unsatisfied[i] = max(P_res, 0)
        E_bat[i+1] = E_bat_next
    P_unsatisfied_acc = np.cumsum(P_unsatisfied/6000) #mWh
    E_unsatisfied_rel = P_unsatisfied_acc/np.cumsum(P_demand/6000)
    
    plt.figure('Stored energy')
    plt.plot(times, E_bat)
    plt.figure('Battery power')
    plt.plot(times, P_bat_vec)
    plt.figure('Total unsatisfied demand')
    plt.plot(times, P_unsatisfied_acc)
    plt.figure('Share of demand that cannot be satisfied')
    plt.plot(times, E_unsatisfied_rel)

    reserve_factor = 0.1
    P_gtg_nom = (1-reserve_factor)*ohps.P_gtg_max
    P_demand = wind_power_nwp + P_gtg_nom
    P_generated_uncompensated = wind_power_actual + P_gtg_nom
    P_generated_max = wind_power_actual + ohps.P_gtg_max
    P_diff_max = P_generated_max - P_demand # maximum charging power/minimum discharging power

    E_bat_max = np.geomspace(1e2,5e5,50)
    P_unsatisfied = np.array([get_unsatisfied_demand(E_bat_max_i) for E_bat_max_i in E_bat_max])
    fig, ax = plt.subplots(2, sharex=True)
    ax[0].semilogx(E_bat_max/1000, P_unsatisfied*100)
    ax[0].set_ylabel('Unsatisfied demand relative to total demand (%)')
    ax[1].semilogx(E_bat_max/1000, 100-P_unsatisfied/get_unsatisfied_demand(0)*100)
    ax[1].set_xlabel('Battery size (MWh)')
    ax[1].set_ylabel('Reduction of unsatisfiable demand (%)')

    reserve_factor = 0.2
    P_gtg_nom = (1-reserve_factor)*ohps.P_gtg_max
    P_demand = wind_power_nwp + P_gtg_nom
    P_generated_uncompensated = wind_power_actual + P_gtg_nom
    P_generated_max = wind_power_actual + ohps.P_gtg_max
    P_diff_max = P_generated_max - P_demand # maximum charging power/minimum discharging power

    E_bat_max = np.geomspace(1e2,5e5,50)
    P_unsatisfied = np.array([get_unsatisfied_demand(E_bat_max_i) for E_bat_max_i in E_bat_max])
    ax[0].semilogx(E_bat_max/1000, P_unsatisfied*100)
    ax[1].semilogx(E_bat_max/1000, 100-P_unsatisfied/get_unsatisfied_demand(0)*100)

    reserve_factor = 0.3
    P_gtg_nom = (1-reserve_factor)*ohps.P_gtg_max
    P_demand = wind_power_nwp + P_gtg_nom
    P_generated_uncompensated = wind_power_actual + P_gtg_nom
    P_generated_max = wind_power_actual + ohps.P_gtg_max
    P_diff_max = P_generated_max - P_demand # maximum charging power/minimum discharging power
    
    E_bat_max = np.geomspace(1e2,1e6,50)
    P_unsatisfied = np.array([get_unsatisfied_demand(E_bat_max_i) for E_bat_max_i in E_bat_max])
    ax[0].semilogx(E_bat_max/1000, P_unsatisfied*100)
    ax[1].semilogx(E_bat_max/1000, 100-P_unsatisfied/get_unsatisfied_demand(0)*100)

    reserve_factor = 0.2
    P_gtg_nom = (1-reserve_factor)*ohps.P_gtg_max
    P_demand = wind_power_nwp + P_gtg_nom

    P_unsatisfied_sim = get_unsatisfied_demand_accurate(ohps, P_demand, wind_power_actual)
    ax[0].scatter([32], [P_unsatisfied_sim[0]*100], color='tab:orange')

    fig.legend([r'90% base load', r'80% base load', r'70% base load', r'Simulation with 80% base load'])

    plt.show()
    pass