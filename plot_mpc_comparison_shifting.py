import datetime

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import norm

from modules.gp import get_gp_opt, DataHandler
from modules.models import OHPS
from modules.mpc import get_mpc_opt
from modules.mpc_scoring import DataSaving

#matplotlib.use('pgf')
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

array_dims = {'Power output': 4, 'Power demand': 1, 'SOC': 1, 'Inputs': 2}

"""Nominal MPC with perfect forecast"""
mpc_opt = get_mpc_opt(N=30, t_end_sim=datetime.datetime(2022,12,31))
mpc_opt['param']['k_dP'] = 10
mpc_opt['param']['r_s_E'] = 100
mpc_opt['param']['k_bat'] = 0
mpc_opt['use_path_constraints_energy'] = True
mpc_opt['use_soft_constraints_state'] = False
gp_opt = get_gp_opt(dt_pred = mpc_opt['dt'])
dl = DataSaving('nominal_mpc_perfect_forecast_shifting_fixed_demand', mpc_opt, gp_opt, array_dims)
data_nmpcpf, times_nmpcpf = dl.load_trajectories()

"""Nominal MPC with NWP forecast"""
mpc_opt = get_mpc_opt(N=30, t_start_sim=datetime.datetime(2022,1,1), use_soft_constraints_state=False)
mpc_opt['param']['k_dP'] = 10
mpc_opt['param']['r_s_E'] = 100
mpc_opt['param']['k_bat'] = 0
mpc_opt['use_path_constraints_energy'] = True
dl = DataSaving('nominal_mpc_nwp_forecast_shifting_fixed_demand', mpc_opt, gp_opt, array_dims)
data_nmpcnwpf, times_nmpcnwpf = dl.load_trajectories()

"""Nominal MPC with GP forecast"""
mpc_opt = get_mpc_opt(N=30, t_start_sim=datetime.datetime(2022,1,1), use_soft_constraints_state=False)
mpc_opt['param']['k_dP'] = 10
mpc_opt['param']['r_s_E'] = 100
mpc_opt['param']['k_bat'] = 0
mpc_opt['use_path_constraints_energy'] = True
mpc_opt['N_p'] = 8000
gp_opt = get_gp_opt(dt_pred = mpc_opt['dt'], steps_forward = mpc_opt['N'], verbose=False)

gp_opt = get_gp_opt(dt_pred = mpc_opt['dt'], steps_forward = mpc_opt['N'], verbose = False)
dl = DataSaving('nominal_mpc_gp_forecast_shifting_fixed_demand', mpc_opt, gp_opt, array_dims)
data_nmpcgpf, times_nmpcgpf = dl.load_trajectories()

"""Multi-stage MPC with simple scenario generation"""
epsilon = 0.1
std_factor = norm.ppf(1-epsilon)
std_list = (-std_factor, 0, std_factor)
mpc_opt = get_mpc_opt(N=30, std_list_multistage=std_list, use_simple_scenarios=True, dE_min=0, t_start_sim=datetime.datetime(2022,1,1), use_soft_constraints_state=False, include_last_measurement=True)#,  t_start=datetime.datetime(2022,12,6), t_end=datetime.datetime(2022,12,8))
mpc_opt['param']['k_dP'] = 10
mpc_opt['param']['r_s_E'] = 100
mpc_opt['param']['k_bat'] = 0
mpc_opt['use_path_constraints_energy'] = True
gp_opt = get_gp_opt(dt_pred = mpc_opt['dt'], steps_forward = mpc_opt['N'], verbose=False)
dl = DataSaving('multi-stage_mpc_shifting_fixed_demand_keep_tree', mpc_opt, gp_opt, array_dims)
data_msmpcsc, times_msmpcsc = dl.load_trajectories()

data_list = [data_nmpcpf, data_nmpcnwpf, data_nmpcgpf, data_msmpcsc]#, data_msmpcmb]
times_list = [times_nmpcpf, times_nmpcnwpf, times_nmpcgpf, times_msmpcsc]#, times_msmpcmb]

times_start_plot = [(1,5),(6,1),(1,6),(4,6),(2,7),(2,13),(1,14),(1,17),(3,17),(1,18),(1,24),(2,25),(5,27),(3,29),(2,1),(12,7),(10,4),(11,18),(9,6),(9,12),(6,29),(6,24)]
for t_start_i in times_start_plot:
    t_start_plot = datetime.datetime(2022,*t_start_i)
    t_end_plot = t_start_plot + datetime.timedelta(days=1)
    #t_end_plot = datetime.datetime(2022,12,8)

    fig = plt.figure(layout='constrained')
    subfigs = fig.subfigures(3,1)
    ax_total = subfigs[0].subplots()
    axs_subplots_power = subfigs[1].subplots(1, 2, sharex=True)
    ax_subplots_state = subfigs[2].subplots(1, 2, sharex=True)

    plt_labels = ['Perfect forecast', 'NWP', 'GP mean', 'Multi-stage']#, 'Multi-stage, multiple branch']
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
        E_shifted = np.cumsum((data['Power output'][:,-1]-data['Power demand'][:,0])/6000)[indices_in_range]
        ax_total.plot(times_in_range, P_total/1000, label=plt_labels[i])
        axs_subplots_power[0].plot(times_in_range, P_gtg/1000, label=plt_labels[i])
        ax_subplots_state[1].plot(times_in_range, SOC*100, label=plt_labels[i])
        ax_subplots_state[0].plot(times_in_range,E_shifted, label=plt_labels[i])
    times = [t_start_plot+i*datetime.timedelta(minutes=10) for i in range(len(times_in_range))]
    ax_total.plot(times, P_demand/1000, '--', color='black')
    axs_subplots_power[1].plot(times, P_wtg/1000, label='Measurement')
    dh = DataHandler(datetime.datetime(2020,1,1), datetime.datetime(2022,12,31), gp_opt)
    wind_speeds_nwp = [dh.get_NWP(t) for t in times]
    ohps = OHPS()
    wind_power_nwp = np.array([ohps.get_P_wtg(0,0,w) for w in wind_speeds_nwp]).reshape(-1)
    axs_subplots_power[1].plot(times, wind_power_nwp/1000, label='Weather forecast')
    ax_total.set_xlabel('Time')
    ax_total.set_ylabel(r'$P_\mathrm{total} (\mathrm{MW})$')
    # axs_subplots_power[0].set_xlabel('Time')
    # axs_subplots_power[1].set_xlabel('Time')
    axs_subplots_power[0].set_ylabel(r'$P_\mathrm{gtg} (\mathrm{MW})$')
    axs_subplots_power[1].set_ylabel(r'$P_\mathrm{wtg} (\mathrm{MW})$')
    ax_subplots_state[0].set_xlabel('Time (h)')
    ax_subplots_state[1].set_xlabel('Time (h)')
    ax_subplots_state[0].set_ylabel(r'$E_\mathrm{s} (\mathrm{MWh})$')
    ax_subplots_state[1].set_ylabel(r'$SOC_\mathrm{bat} (\%)$')
    ymin, ymax = ax_subplots_state[0].get_ylim()
    ymin = min(ymin, -11)
    ymax = max(ymax, 11)
    ax_subplots_state[0].fill_between(times, ymin, -10, color='red', alpha=0.25)
    ax_subplots_state[0].fill_between(times, 10, ymax, color='red', alpha=0.25)
    ax_total.grid()
    axs_subplots_power[0].grid()
    axs_subplots_power[1].grid()
    ax_subplots_state[0].grid()
    ax_subplots_state[1].grid()
    axs_subplots_power[0].xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H'))
    axs_subplots_power[1].xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H'))
    ax_subplots_state[0].xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H'))
    ax_subplots_state[1].xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H'))
    ax_total.margins(x=0)
    axs_subplots_power[0].margins(x=0)
    axs_subplots_power[1].margins(x=0)
    ax_subplots_state[0].margins(x=0)
    ax_subplots_state[1].margins(x=0)
    ax_subplots_state[0].set_ylim(ymin, ymax)
    labels_total = plt_labels + ['Demand']
    subfigs[0].legend(labels_total, ncol=3, loc='upper center', bbox_to_anchor=(0.5,1.35))
    handles_subfig, _ = axs_subplots_power[0].get_legend_handles_labels()
    # subfigs[1].legend(handles=handles_subfig, ncol=3, loc='upper center', bbox_to_anchor=(0.5,1.2))
    axs_subplots_power[1].legend()

    cm = 1/2.54
    fig.set_size_inches(14*cm, 12*cm)
    plt.savefig(f'../Abbildungen/mpc_shifting_comparison_{t_start_plot.strftime("%d_%m")}.pdf', bbox_inches='tight')

plt.pause(1)
pass