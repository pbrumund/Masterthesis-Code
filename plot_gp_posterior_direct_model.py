import datetime

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import norm
import torch

from modules.gp import DirectGPEnsemble, get_gp_opt

matplotlib.use('pgf')
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

plt.ion()
# unstable scenario
t_start = datetime.datetime(2022,5,11,5) # 02,24 03,21 5,25 10,7
steps = 36
steps_before = 36
x = np.arange(-steps_before, steps-1, 1)
times = np.array([t_start+xi*datetime.timedelta(minutes=10) for xi in x])
gp_opt = get_gp_opt()
gp = DirectGPEnsemble(gp_opt)
gp_pred, gp_var = gp.predict_trajectory(t_start - datetime.timedelta(minutes=10), steps, include_last_measurement=False)
gp_prior_std = np.sqrt(gp_var)
NWP_pred = [gp.data_handler.get_NWP(t_start, steps=(x[i])) for i,_ in enumerate(times)]
meas = np.array([gp.data_handler.get_measurement(t) for t in times]).reshape(-1)
times_gp = [t_start + i*datetime.timedelta(minutes=10) for i in range(steps)]

fig, axs = plt.subplots(1,2, layout='constrained', sharey=True)
plt_gp, = axs[0].plot(times_gp, gp_pred, color='tab:blue')
plt_meas, = axs[0].plot(times, meas, color='tab:green')
axs[0].fill_between(times_gp, gp_pred-2*np.sqrt(gp_var), gp_pred+2*np.sqrt(gp_var), alpha=0.2, color='tab:blue')
axs[0].fill_between(times_gp, gp_pred-np.sqrt(gp_var), gp_pred+np.sqrt(gp_var), alpha=0.4, color='tab:blue')
plt_nwp, = axs[0].plot(times, np.array(NWP_pred).reshape(-1), color='tab:orange')
axs[0].grid()
axs[0].set_xlabel('Time')
axs[0].xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H'))
axs[0].set_ylabel('Wind speed (m/s)')
fig.legend(handles = [plt_meas, plt_nwp, plt_gp], labels=['Measurement', 'NWP', 'GP prediction'],
           loc='upper center', bbox_to_anchor=(0.5,1.15), ncol=3)

t_start = datetime.datetime(2022,4,19,5) # 02,24 03,21 6,29 4,13
steps = 36
steps_before = 36
x = np.arange(-steps_before, steps-1, 1)
times = np.array([t_start+xi*datetime.timedelta(minutes=10) for xi in x])
gp_opt = get_gp_opt()
gp_pred, gp_var = gp.predict_trajectory(t_start - datetime.timedelta(minutes=10), steps, include_last_measurement=False)
gp_prior_std = np.sqrt(gp_var)
NWP_pred = [gp.data_handler.get_NWP(t_start, steps=(x[i])) for i,_ in enumerate(times)]
meas = np.array([gp.data_handler.get_measurement(t) for t in times]).reshape(-1)
times_gp = [t_start + i*datetime.timedelta(minutes=10) for i in range(steps)]

axs[1].plot(times_gp, gp_pred, color='tab:blue')
axs[1].plot(times, meas, color='tab:green')
axs[1].fill_between(times_gp, gp_pred-2*np.sqrt(gp_var), gp_pred+2*np.sqrt(gp_var), alpha=0.2, color='tab:blue')
axs[1].fill_between(times_gp, gp_pred-np.sqrt(gp_var), gp_pred+np.sqrt(gp_var), alpha=0.4, color='tab:blue')
axs[1].plot(times, np.array(NWP_pred).reshape(-1), color='tab:orange')
axs[1].grid()
axs[1].set_xlabel('Time')
axs[1].xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H'))
# axs[1].set_ylabel('Wind speed (m/s)')
cm = 1/2.54
fig.set_size_inches(15*cm, 7*cm)
plt.savefig('../Abbildungen/gp_posterior_example_direct.pgf', bbox_inches='tight')
#plt.pause(1)

