import datetime

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import norm

from modules.gp import TimeseriesModel, get_gp_opt
from modules.models import OHPS
from modules.mpc import MultistageMPC, get_mpc_opt

plt.rcParams.update({
    'text.usetex': True,
})

t_start = datetime.datetime(2022,1,17,4)
steps = 30
epsilon = 0.1
std_factor = norm.ppf(1-epsilon)
std_list = (-std_factor, 0, std_factor)

gp_opt = get_gp_opt(steps_forward=steps)
gp = TimeseriesModel(gp_opt)
dh = gp.data_handler

ohps = OHPS()
ohps.setup_integrator(600)

mpc_opt = get_mpc_opt(N=steps, use_simple_scenarios=False, std_list_multistage = std_list, dE_min=5000)
multistage_mpc = MultistageMPC(ohps, gp, mpc_opt)

times = [t_start + i*datetime.timedelta(minutes=10) for i in range(steps)]

trajectory_nwp = np.array([dh.get_NWP(t) for t in times])
trajectory_meas = np.array([dh.get_measurement(t) for t in times])
gp_mean, gp_var = gp.predict_trajectory(t_start, steps, train=True)

multistage_mpc.get_optimization_problem(t_start, True)
means = [np.array(multistage_mpc.means[-1])]
means_P = [np.array([ohps.get_P_wtg(0,0,w) for w in multistage_mpc.means[-1]]).reshape(-1)]
parents_i = multistage_mpc.parent_nodes[-1]
for i in range(2, multistage_mpc.horizon+1):
    means_parents = np.array(multistage_mpc.means[-(i)])[parents_i]
    mean_wind_power = np.array([ohps.get_P_wtg(0,0,w) for w in means_parents]).reshape(-1)
    parents_i = np.array(multistage_mpc.parent_nodes[-(i)])[parents_i]
    means.append(means_parents)
    means_P.append(mean_wind_power)
means.reverse()
means_P.reverse()
means = np.array(means)
means_P = np.array(means_P)

fig, ax_pred = plt.subplots(2, sharex=True)
plt_sc = ax_pred[0].plot(times, means, color='tab:blue', label='Scenarios')[0]
ax_pred[1].plot(times, means_P, color='tab:blue', label='Scenarios')
P_meas = np.array([ohps.get_P_wtg(0,0,w) for w in trajectory_meas]).reshape(-1)
P_mean = np.array([ohps.get_P_wtg(0,0,w) for w in gp_mean]).reshape(-1)
P_lower = np.array([ohps.get_P_wtg(0,0,w-std_factor*np.sqrt(v)) for w, v in zip(gp_mean, gp_var)]).reshape(-1)
P_upper = np.array([ohps.get_P_wtg(0,0,w+std_factor*np.sqrt(v)) for w, v in zip(gp_mean, gp_var)]).reshape(-1)
plt_meas, = ax_pred[0].plot(times, trajectory_meas, color='tab:green', label='Measurement')
ax_pred[1].plot(times, P_meas, color='tab:green', label='Measurement')
plt_gp, = ax_pred[0].plot(times, gp_mean, color='tab:orange', label=r'GP mean and 80 \% confidence interval')
ax_pred[1].plot(times, P_mean, color='tab:orange', label=r'GP mean and 80 \% confidence interval')
ax_pred[0].fill_between(times, gp_mean-std_factor*np.sqrt(gp_var), 
    gp_mean+std_factor*np.sqrt(gp_var), color='tab:orange', alpha=0.4)
ax_pred[1].fill_between(times, P_lower, P_upper, color='tab:orange', alpha=0.4)
plt_nwp, = ax_pred[0].plot(times, np.array(trajectory_nwp).reshape(-1), color='tab:red', label='NWP')
wind_power_nwp = np.array([ohps.get_P_wtg(0,0,w) for w in trajectory_nwp]).reshape(-1)
ax_pred[1].plot(times, np.array(wind_power_nwp), color='tab:red', label='NWP')
ax_pred[0].set_ylabel('Wind speed (m/s)')
ax_pred[1].set_ylabel('Wind power (kW)')
ax_pred[1].set_xlabel('Time (h)')
ax_pred[0].xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))
ax_pred[1].xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))
ax_pred[0].grid()
ax_pred[1].grid()
ax_pred[0].margins(x=0)
ax_pred[1].margins(x=0)
handles = [plt_gp, plt_sc, plt_meas, plt_nwp]
fig.legend(handles=handles, ncol=4, loc='upper center', bbox_to_anchor=(0.5,1.05))
cm = 1/2.54
fig.set_size_inches(15*cm, 10*cm)
plt.savefig(f'../Abbildungen/mpc_scenarios_{t_start.strftime("%d_%m_%H")}.pdf', bbox_inches='tight')
plt.show()
pass