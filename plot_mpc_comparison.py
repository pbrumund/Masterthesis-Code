import datetime

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import norm

from modules.gp import get_gp_opt, DataHandler
from modules.models import OHPS
from modules.mpc import get_mpc_opt
from modules.mpc_scoring import DataSaving

matplotlib.use('pgf')
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})
# plt.rcParams.update({
#     'text.usetex': True,
# })

t_start_mpc = datetime.datetime(2022,1,1)
t_end_mpc = datetime.datetime(2022,12,31)

mpc_opt = get_mpc_opt(N=36, use_soft_constraints_state=False)
gp_opt = get_gp_opt(dt_pred = mpc_opt['dt'])
array_dims = {'Power output': 4, 'Power demand': 1, 'SOC': 1, 'Inputs': 2}

"""Nominal MPC with perfect forecast"""
dl = DataSaving('nominal_mpc_perfect_forecast', mpc_opt, gp_opt, array_dims)
data_nmpcpf, times_nmpcpf = dl.load_trajectories()

"""Nominal MPC with NWP forecast"""
dl = DataSaving('nominal_mpc_nwp_forecast_without_llc', mpc_opt, gp_opt, array_dims)
data_nmpcnwpf, times_nmpcnwpf = dl.load_trajectories()

"""Nominal MPC with GP forecast"""
gp_opt = get_gp_opt(dt_pred = mpc_opt['dt'], steps_forward = mpc_opt['N'], verbose = False)
dl = DataSaving('nominal_mpc_gp_forecast_without_llc', mpc_opt, gp_opt, array_dims)
data_nmpcgpf, times_nmpcgpf = dl.load_trajectories()

"""Chance constrained MPC"""
mpc_opt['epsilon_chance_constraint'] = 0.05
dl = DataSaving('chance_constrained_mpc_without_llc_direct_rerun', mpc_opt, gp_opt, array_dims)
data_ccmpc, times_ccmpc = dl.load_trajectories()

"""Multi-stage MPC with complex scenario generation"""
epsilon = 0.1
std_factor = norm.ppf(1-epsilon)
std_list = (-std_factor, 0, std_factor)
mpc_opt = get_mpc_opt(N=36, std_list_multistage=std_list, use_simple_scenarios=False, dE_min=5000, 
                      use_soft_constraints_state=False, include_last_measurement=True)
gp_opt = get_gp_opt(dt_pred = mpc_opt['dt'], steps_forward = mpc_opt['N'], verbose = False)
dl = DataSaving('multi-stage_mpc_without_llc', mpc_opt, gp_opt, array_dims)
data_msmpcsc, times_msmpcsc = dl.load_trajectories()

# """Multi-stage MPC with more branches (branching multiple times)"""
# mpc_opt = get_mpc_opt(N=30, std_list_multistage=std_list, use_simple_scenarios=False, dE_min=5000, 
#                       t_start=t_start_mpc, t_end=t_end_mpc)
# dl = DataSaving('multi-stage_mpc', mpc_opt, gp_opt, array_dims)
# data_msmpcmb, times_msmpcmb = dl.load_trajectories()

data_list = [data_nmpcpf, data_nmpcnwpf, data_nmpcgpf, data_ccmpc, data_msmpcsc]#, data_msmpcmb]
times_list = [times_nmpcpf, times_nmpcnwpf, times_nmpcgpf, times_ccmpc, times_msmpcsc]#, times_msmpcmb]

times_start_plot = [(6,24),(7,22),(5,28),(5,27),(9,30),(6,1),(10,4),(1,6),(4,6),(9,6),(2,7),(12,7),(9,12),(2,13),(1,14),(1,17),(3,17),(1,18),(11,18),(1,24),(2,25),(5,27),(3,29),(6,29),(6,24),(2,1)]
for t_start_i in times_start_plot:
    t_start_plot = datetime.datetime(2022,*t_start_i)
    t_end_plot = t_start_plot + datetime.timedelta(days=2)
    #t_end_plot = datetime.datetime(2022,12,8)

    fig = plt.figure(layout='constrained')
    subfigs = fig.subfigures(3,1)
    ax_total = subfigs[0].subplots()
    axs_subplots_power = subfigs[1].subplots(1, 2, sharex=True)
    ax_subplots_wind_soc = subfigs[2].subplots(1, 2, sharex=True)

    plt_labels = ['Perfect forecast', 'NWP', 'GP mean', 'Chance constraints', 
                'Multi-stage']#, 'Multi-stage, multiple branch']
    for i, (data, times) in enumerate(zip(data_list, times_list)):
        times = np.array([datetime.datetime.strptime(t[0]+t[1], '%Y-%m-%d%H:%M:%S') for t in times])
        indices_in_range = np.nonzero(np.logical_and(times>=t_start_plot, times<t_end_plot))
        times_in_range = times[indices_in_range]
        P_total = data['Power output'][indices_in_range, -1].reshape(-1)
        P_demand = data['Power demand'][indices_in_range].reshape(-1)
        P_gtg = data['Power output'][indices_in_range, 0].reshape(-1)
        P_bat = data['Power output'][indices_in_range, 1].reshape(-1)
        P_wtg = data['Power output'][indices_in_range, 2].reshape(-1)
        SOC = data['SOC'][indices_in_range]
        ax_total.plot(times_in_range, P_total/1000, label=plt_labels[i])
        axs_subplots_power[0].plot(times_in_range, P_gtg/1000, label=plt_labels[i])
        axs_subplots_power[1].plot(times_in_range, P_bat/1000, label=plt_labels[i])
        ax_subplots_wind_soc[1].plot(times_in_range, SOC*100, label=plt_labels[i])
    times = [t_start_plot+i*datetime.timedelta(minutes=10) for i in range(len(times_in_range))]
    ax_total.plot(times, P_demand/1000, '--', color='black')
    ax_subplots_wind_soc[0].plot(times, P_wtg/1000, label='Measurement')
    dh = DataHandler(datetime.datetime(2020,1,1), datetime.datetime(2022,12,31), gp_opt)
    wind_speeds_nwp = [dh.get_NWP(t) for t in times]
    ohps = OHPS()
    wind_power_nwp = np.array([ohps.get_P_wtg(0,0,w) for w in wind_speeds_nwp]).reshape(-1)
    ax_subplots_wind_soc[0].plot(times, wind_power_nwp/1000, label='Weather forecast')
    ax_total.set_xlabel('Time')
    ax_total.set_ylabel(r'$P_\mathrm{total} (\mathrm{MW})$')
    # axs_subplots_power[0].set_xlabel('Time')
    # axs_subplots_power[1].set_xlabel('Time')
    axs_subplots_power[0].set_ylabel(r'$P_\mathrm{gtg} (\mathrm{MW})$')
    axs_subplots_power[1].set_ylabel(r'$P_\mathrm{bat} (\mathrm{MW})$')
    ax_subplots_wind_soc[0].set_xlabel('Time (h)')
    ax_subplots_wind_soc[1].set_xlabel('Time (h)')
    ax_subplots_wind_soc[0].set_ylabel(r'$P_\mathrm{wtg} (\mathrm{MW})$')
    ax_subplots_wind_soc[1].set_ylabel(r'$SOC_\mathrm{bat} (\%)$')
    ax_total.grid()
    axs_subplots_power[0].grid()
    axs_subplots_power[1].grid()
    ax_subplots_wind_soc[0].grid()
    ax_subplots_wind_soc[1].grid()
    axs_subplots_power[0].xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H'))
    axs_subplots_power[1].xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H'))
    ax_subplots_wind_soc[0].xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H'))
    ax_subplots_wind_soc[1].xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H'))
    ax_total.margins(x=0)
    axs_subplots_power[0].margins(x=0)
    axs_subplots_power[1].margins(x=0)
    ax_subplots_wind_soc[0].margins(x=0)
    ax_subplots_wind_soc[1].margins(x=0)
    labels_total = plt_labels + ['Demand']
    subfigs[0].legend(labels_total, ncol=3, loc='upper center', bbox_to_anchor=(0.5,1.35))
    handles_subfig, _ = axs_subplots_power[0].get_legend_handles_labels()
    # subfigs[1].legend(handles=handles_subfig, ncol=3, loc='upper center', bbox_to_anchor=(0.5,1.2))
    ax_subplots_wind_soc[0].legend()

    cm = 1/2.54
    fig.set_size_inches(14*cm, 12*cm)
    plt.savefig(f'../Abbildungen/mpc_comparison_{t_start_plot.strftime("%d_%m")}_new.pgf', bbox_inches='tight')

plt.pause(1)
pass