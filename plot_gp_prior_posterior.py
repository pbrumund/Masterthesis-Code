import datetime

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import norm
import torch

from modules.gp import TimeseriesModel, get_gp_opt

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
x = np.arange(-steps_before, steps-1, .1)
times = np.array([t_start+xi*datetime.timedelta(minutes=10) for xi in x])
gp_opt = get_gp_opt()
gp = TimeseriesModel(gp_opt)
# _, _ = gp.predict_trajectory(t_start, steps)

gp.t_last_train = t_start#  - datetime.timedelta(minutes=10)
gp.gp_timeseries, gp.timeseries_likelihood = gp.get_timeseries_gp(prediction_time=t_start+datetime.timedelta(minutes=10))
gp.train_timeseries_gp()

gp.gp_timeseries.eval()
gp.timeseries_likelihood.eval()

x = torch.from_numpy(x.astype(float))
# x = torch.arange(-36, 30, .1).double().reshape((-1,1))
# x = x+1
gp_pred_y = gp.timeseries_likelihood(gp.gp_timeseries(x))
gp_mean = gp_pred_y.mean
gp_var = np.array([gp.timeseries_likelihood(gp.gp_timeseries(x_i.reshape((-1,1)))).variance.detach().numpy() for x_i in x]).reshape(-1)
prior_mean, prior_var = gp.gp_timeseries.covar_module.get_gp_pred(x.reshape(-1,1))
# add NWP to get predicted value from prediction error
NWP_pred = [gp.data_handler.get_NWP(t_start, steps=(x[i]).numpy()) for i,_ in enumerate(times)]
gp_pred = np.array(NWP_pred).reshape(-1) + gp_mean.reshape((-1,1)).detach().numpy().reshape(-1)
meas = np.array([gp.data_handler.get_measurement(t) for t in times]).reshape(-1)
gp_prior_pred = np.array(NWP_pred).reshape(-1)+prior_mean.numpy().reshape(-1)
gp_prior_std = np.sqrt(prior_var.numpy().reshape(-1))

fig, axs = plt.subplots(1,2, layout='constrained', sharey=True)
plt_gp, = axs[0].plot(times, gp_pred, color='tab:blue')
plt_meas, = axs[0].plot(times, meas, color='tab:green')
plt_gp_prior, = axs[0].plot(times, gp_prior_pred, '--', alpha=0.6, color='tab:gray')
axs[0].fill_between(times, gp_prior_pred-2*gp_prior_std, gp_prior_pred+2*gp_prior_std, alpha=0.05, color='tab:gray')
axs[0].fill_between(times, gp_prior_pred-gp_prior_std, gp_prior_pred+gp_prior_std, alpha=0.1, color='tab:gray')
axs[0].fill_between(times, gp_pred-2*np.sqrt(gp_var), gp_pred+2*np.sqrt(gp_var), alpha=0.2, color='tab:blue')
axs[0].fill_between(times, gp_pred-np.sqrt(gp_var), gp_pred+np.sqrt(gp_var), alpha=0.4, color='tab:blue')
plt_nwp, = axs[0].plot(times, np.array(NWP_pred).reshape(-1), color='tab:orange')
axs[0].grid()
axs[0].set_xlabel('Time')
axs[0].xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H'))
axs[0].set_ylabel('Wind speed (m/s)')
fig.legend(handles = [plt_meas, plt_nwp, plt_gp, plt_gp_prior], labels=['Measurement', 'NWP', 'GP posterior', 'GP prior'],
           loc='upper center', bbox_to_anchor=(0.5,1.15), ncol=4)

t_start = datetime.datetime(2022,4,19,5) # 02,24 03,21 6,29 4,13
steps = 36
steps_before = 36
x = np.arange(-steps_before, steps-1, .1)
times = np.array([t_start+xi*datetime.timedelta(minutes=10) for xi in x])
gp_opt = get_gp_opt()
gp = TimeseriesModel(gp_opt)
# _, _ = gp.predict_trajectory(t_start, steps)

gp.t_last_train = t_start#  - datetime.timedelta(minutes=10)
gp.gp_timeseries, gp.timeseries_likelihood = gp.get_timeseries_gp(prediction_time=t_start+datetime.timedelta(minutes=10))
gp.train_timeseries_gp()

gp.gp_timeseries.eval()
gp.timeseries_likelihood.eval()

x = torch.from_numpy(x.astype(float))
# x = torch.arange(-36, 30, .1).double().reshape((-1,1))
# x = x+1
gp_pred_y = gp.timeseries_likelihood(gp.gp_timeseries(x))
gp_mean = gp_pred_y.mean
gp_var = np.array([gp.timeseries_likelihood(gp.gp_timeseries(x_i.reshape((-1,1)))).variance.detach().numpy() for x_i in x]).reshape(-1)
prior_mean, prior_var = gp.gp_timeseries.covar_module.get_gp_pred(x.reshape(-1,1))
# add NWP to get predicted value from prediction error
NWP_pred = [gp.data_handler.get_NWP(t_start, steps=(x[i]).numpy()) for i,_ in enumerate(times)]
gp_pred = np.array(NWP_pred).reshape(-1) + gp_mean.reshape((-1,1)).detach().numpy().reshape(-1)
meas = np.array([gp.data_handler.get_measurement(t) for t in times]).reshape(-1)
gp_prior_pred = np.array(NWP_pred).reshape(-1)+prior_mean.numpy().reshape(-1)
gp_prior_std = np.sqrt(prior_var.numpy().reshape(-1))
axs[1].plot(times, gp_pred, color='tab:blue')
axs[1].plot(times, meas, color='tab:green')
axs[1].plot(times, gp_prior_pred, '--', alpha=0.6, color='tab:gray')
axs[1].fill_between(times, gp_prior_pred-2*gp_prior_std, gp_prior_pred+2*gp_prior_std, alpha=0.05, color='tab:gray')
axs[1].fill_between(times, gp_prior_pred-gp_prior_std, gp_prior_pred+gp_prior_std, alpha=0.1, color='tab:gray')
axs[1].fill_between(times, gp_pred-2*np.sqrt(gp_var), gp_pred+2*np.sqrt(gp_var), alpha=0.2, color='tab:blue')
axs[1].fill_between(times, gp_pred-np.sqrt(gp_var), gp_pred+np.sqrt(gp_var), alpha=0.4, color='tab:blue')
axs[1].plot(times, np.array(NWP_pred).reshape(-1), color='tab:orange')
axs[1].grid()
axs[1].set_xlabel('Time')
axs[1].xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H'))
# axs[1].set_ylabel('Wind speed (m/s)')
cm = 1/2.54
fig.set_size_inches(15*cm, 7*cm)
plt.savefig('../Abbildungen/gp_posterior_example.pgf', bbox_inches='tight')
plt.pause(1)

# Plot prior
start_time = datetime.datetime(2022,1,1)
steps = 6*24*14
times = [start_time+i*datetime.timedelta(minutes=gp.opt['dt_meas']) for i in range(steps)]

NWP_traj = np.zeros(steps)
mean_traj = np.zeros(steps)
var_traj = np.zeros(steps)
noise_var_traj = np.zeros(steps)
meas_traj = np.zeros(steps)

for i, t in enumerate(times):
    NWP_traj[i] = gp.data_handler.get_NWP(t, 0)
    meas_traj[i] = gp.data_handler.get_measurement(t, 0)
    x = gp.data_handler.generate_features(t, feature='nwp & time').reshape((1,-1))
    mean, var = gp.gp_prior.compiled_predict_y(x)
    mean_f, _ = gp.gp_prior.compiled_predict_f(x)
    mean_traj[i] = mean
    var_traj[i] = var
    noise_var_traj[i] = np.exp(mean_f.numpy()[0,1])**2

gp_pred_traj = NWP_traj + mean_traj
signal_var_traj = var_traj-noise_var_traj
std_traj = np.sqrt(var_traj)


fig = plt.figure()
ax = plt.axes()
ax.plot(times, NWP_traj, color='tab:orange')
ax.plot(times, meas_traj, color='tab:green')
ax.plot(times, gp_pred_traj, color='tab:blue')
ax.fill_between(times, gp_pred_traj-2*std_traj, gp_pred_traj+2*std_traj, color='lightgray')
ax.fill_between(times, gp_pred_traj-std_traj, gp_pred_traj+std_traj, color='tab:gray')
ax.grid()
ax.set_xlabel('Time')
ax.set_ylabel('Wind speed (m/s)')
ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter(r'%d.%m'))
fig.legend(['Weather prediction', 'Actual wind speed', 'GP prediction'], loc='upper center', 
           bbox_to_anchor=(0.5,1.05), ncol=3)
fig.set_size_inches(15*cm, 7*cm)
plt.savefig('../Abbildungen/gp_prior_example.pgf', bbox_inches='tight')

# fig3, ax3, fig4, ax4 = gp.plot_prior_distribution(datetime.datetime(2022,1,1), datetime.datetime(2022,12,31))

pass


