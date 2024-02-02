import datetime

import numpy as np
import matplotlib.pyplot as plt

from modules.gp import get_gp_opt
from modules.mpc import get_mpc_opt
from modules.mpc_scoring import DataSaving

t_start_mpc = datetime.datetime(2022,1,1)
t_end_mpc = datetime.datetime(2022,12,31)

mpc_opt = get_mpc_opt(N=30, t_start=t_start_mpc, t_end=t_end_mpc)
gp_opt = get_gp_opt(dt_pred = mpc_opt['dt'])
array_dims = {'Power output': 4, 'Power demand': 1, 'SOC': 1, 'Inputs': 2}

"""Nominal MPC with perfect forecast"""
dl = DataSaving('nominal_mpc_perfect_forecast', mpc_opt, gp_opt, array_dims)
data_nmpcpf, times_nmpcpf = dl.load_trajectories()

"""Nominal MPC with NWP forecast"""
dl = DataSaving('nominal_mpc_nwp_forecast', mpc_opt, gp_opt, array_dims)
data_nmpcnwpf, times_nmpcnwpf = dl.load_trajectories()

"""Nominal MPC with GP forecast"""
dl = DataSaving('nominal_mpc_gp_forecast', mpc_opt, gp_opt, array_dims)
data_nmpcgpf, times_nmpcgpf = dl.load_trajectories()

"""Chance constrained MPC"""
dl = DataSaving('chance_constrained_mpc', mpc_opt, gp_opt, array_dims)
data_ccmpc, times_ccmpc = dl.load_trajectories()

"""Multi-stage MPC with simple scenario generation"""
mpc_opt['use_simple_scenarios'] = True
dl = DataSaving('multi-stage_mpc', mpc_opt, gp_opt, array_dims)
data_msmpcsc, times_msmpcsc = dl.load_trajectories()

"""Multi-stage MPC with more branches (branching multiple times)"""
mpc_opt['use_simple_scenarios'] = False
dl = DataSaving('multi-stage_mpc', mpc_opt, gp_opt, array_dims)
data_msmpcmb, times_msmpcmb = dl.load_trajectories()

data_list = [data_nmpcpf, data_nmpcnwpf, data_nmpcgpf, data_ccmpc, data_msmpcsc, data_msmpcmb]
times_list = [times_nmpcpf, times_nmpcnwpf, times_nmpcgpf, times_ccmpc, times_msmpcsc, times_msmpcmb]

t_start_plot = datetime.datetime(2022,1,2)
t_end_plot = datetime.datetime(2022,1,3)

fig = plt.figure(layout='coinstrained')
subfigs = fig.subfigures(3,1)
ax_total = subfigs[0].subplots()
axs_subplots_power = subfigs[1].subplots(1, 2, sharex=True)
ax_subplots_wind_soc = subfigs[2].subplots(1, 2, sharex=True)


for data, times in zip(data_list, times_list):
    indices_in_range = np.nonzero(np.logical_and(times>=t_start_plot, times<t_end_plot))
    times_in_range = times[indices_in_range]
    P_total = data['Power output'][indices_in_range, -1]
    P_demand = data['Power demand'][indices_in_range]
    P_gtg = data['Power output'][indices_in_range, 0]
    P_bat = data['Power output'][indices_in_range, 1]
    P_wtg = data['Power output'][indices_in_range, 2]
    SOC = data['SOC'][indices_in_range]
