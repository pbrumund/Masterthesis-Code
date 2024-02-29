import datetime

import numpy as np

from modules.mpc import get_mpc_opt
from modules.gp import get_gp_opt
from modules.mpc_scoring import get_power_error, get_gtg_power, get_gtg_emissions, DataSaving


mpc_opt = get_mpc_opt(N=30)
gp_opt = get_gp_opt(dt_pred = mpc_opt['dt'])
# mpc_opt['param']['']

t_start = datetime.datetime(2022,7,1)
t_end = datetime.datetime(2022,9,30)
mpc_opt['t_start'] = t_start
mpc_opt['t_end'] = t_end

run_ids = ['2d6a17a77e', '50e967695b', '215b07c743']
mpc_types = ['nominal_mpc_gp_forecast_shifting', 'nominal_mpc_nwp_forecast_shifting', 'nominal_mpc_perfect_forecast_shifting']
# GP, NWP, PF
array_dims = {'Power output': 4, 'Power demand': 1, 'SOC': 1, 'Inputs': 2}

for run_id, mpc_type in zip(run_ids, mpc_types):
    error_abs_pf, error_rel_pf = get_power_error(mpc_type, mpc_opt, gp_opt, run_id)
    P_gtg_abs, P_gtg_rel = get_gtg_power(mpc_type, mpc_opt, gp_opt, run_id)
    P_in_gtg, eta_gtg = get_gtg_emissions(mpc_type, mpc_opt, gp_opt, run_id)
    print(f'{mpc_type}: {error_abs_pf} MWh total error, {100*error_rel_pf}% of demand')
    print(f'GTG power: {P_gtg_abs}, {P_gtg_rel*100}% of total generated power')
    print(f'Mean power demand of GTG: {P_in_gtg}, average efficiency: {eta_gtg}')

# import matplotlib.pyplot as plt
# plt.figure()
# times = [t_start+i*datetime.timedelta(minutes=10) for i in range(len(P_wtg))]
# plt.plot(times, np.array([P_wtg, P_bat_vec, P_gtg_vec, P_total, P_demand.reshape((-1))]).T)
# plt.figure()
# plt.plot(SOC_vec)
# pass