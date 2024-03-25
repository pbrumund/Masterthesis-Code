import datetime

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from modules.mpc_scoring import DataSaving
from modules.gp import get_gp_opt
from modules.mpc import get_mpc_opt
plt.ion()
t_start_mpc = datetime.datetime(2022,1,1)
t_end_mpc = datetime.datetime(2022,12,31)

mpc_opt = get_mpc_opt(N=36, use_soft_constraints_state=False)
gp_opt = get_gp_opt(dt_pred = mpc_opt['dt'])
array_dims = {'Power output': 4, 'Power demand': 1, 'SOC': 1, 'Inputs': 2}

"""Nominal MPC with perfect forecast"""
dl = DataSaving('nominal_mpc_perfect_forecast', mpc_opt, gp_opt, array_dims)
data_nmpcpf, times_nmpcpf = dl.load_trajectories()
times_nmpcpf = np.array([datetime.datetime.strptime(t[0]+t[1], '%Y-%m-%d%H:%M:%S') for t in times_nmpcpf])
"""Nominal MPC with NWP forecast"""
dl = DataSaving('nominal_mpc_nwp_forecast_without_llc', mpc_opt, gp_opt, array_dims)
data_nmpcnwpf, times_nmpcnwpf = dl.load_trajectories()
times_nmpcnwpf = np.array([datetime.datetime.strptime(t[0]+t[1], '%Y-%m-%d%H:%M:%S') for t in times_nmpcnwpf])
"""Nominal MPC with GP forecast"""
gp_opt = get_gp_opt(dt_pred = mpc_opt['dt'], steps_forward = mpc_opt['N'], verbose = False)
dl = DataSaving('nominal_mpc_gp_forecast_without_llc', mpc_opt, gp_opt, array_dims)
data_nmpcgpf, times_nmpcgpf = dl.load_trajectories()
times_nmpcgpf = np.array([datetime.datetime.strptime(t[0]+t[1], '%Y-%m-%d%H:%M:%S') for t in times_nmpcgpf])
"""Chance constrained MPC"""
dl = DataSaving('chance_constrained_mpc_without_llc_rerun', mpc_opt, gp_opt, array_dims)
data_ccmpc, times_ccmpc = dl.load_trajectories()
times_ccmpc = np.array([datetime.datetime.strptime(t[0]+t[1], '%Y-%m-%d%H:%M:%S') for t in times_ccmpc])
"""Multi-stage MPC with simple scenario generation"""
epsilon = 0.1
std_factor = norm.ppf(1-epsilon)
std_list = (-std_factor, 0, std_factor)
mpc_opt = get_mpc_opt(N=36, std_list_multistage=std_list, use_simple_scenarios=True, dE_min=0, 
                      use_soft_constraints_state=False, include_last_measurement=True)
gp_opt = get_gp_opt(dt_pred = mpc_opt['dt'], steps_forward = mpc_opt['N'], verbose = False)
dl = DataSaving('multi-stage_mpc_without_llc', mpc_opt, gp_opt, array_dims)
data_msmpcsc, times_msmpcsc = dl.load_trajectories()
times_msmpcsc = np.array([datetime.datetime.strptime(t[0]+t[1], '%Y-%m-%d%H:%M:%S') for t in times_msmpcsc])

plt.figure()
plt.plot(times_nmpcpf, data_nmpcpf['Power demand'].reshape(-1), color='black')
plt.plot(times_nmpcpf, data_nmpcpf['Power output'][:,-1].reshape(-1))
plt.plot(times_nmpcnwpf, data_nmpcnwpf['Power output'][:,-1].reshape(-1))
plt.plot(times_nmpcgpf, data_nmpcgpf['Power output'][:,-1].reshape(-1))
plt.plot(times_ccmpc, data_ccmpc['Power output'][:,-1].reshape(-1))
plt.plot(times_msmpcsc, data_msmpcsc['Power output'][:,-1].reshape(-1))
plt.legend(['Perfect forecast', 'NWP', 'GP mean', 'Chance constraints', 'Multi-stage'])
plt.show()
pass