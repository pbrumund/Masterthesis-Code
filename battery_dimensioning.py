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


if __name__ == '__main__':
    plt.ion()
    ohps = OHPS()
    t_start_data = datetime.datetime(2020,1,1)
    t_end_data = datetime.datetime(2022,12,31)
    opt = get_gp_opt()
    data_handler = DataHandler(t_start_data, t_end_data, opt)
    times = data_handler.weather_data['times_meas']
    # Only use data from 2020 and 2021, test on 2022 data
    t_start = datetime.datetime(2020,1,1)
    t_end = datetime.datetime(2021,12,31)
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
    
    E_bat_max = np.geomspace(1e2,5e5,50)
    P_unsatisfied = np.array([get_unsatisfied_demand(E_bat_max_i) for E_bat_max_i in E_bat_max])
    ax[0].semilogx(E_bat_max/1000, P_unsatisfied*100)
    ax[1].semilogx(E_bat_max/1000, 100-P_unsatisfied/get_unsatisfied_demand(0)*100)

    fig.legend([r'90% base load', r'80% base load', r'70% base load'])
    plt.show()
    pass