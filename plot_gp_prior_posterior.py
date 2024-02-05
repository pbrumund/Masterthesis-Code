import datetime

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import norm
import torch

from modules.gp import TimeseriesModel, get_gp_opt

plt.rcParams.update({
    'text.usetex': True,
})

t_start = datetime.datetime(2022,1,17,4)
steps = 30
steps_before = 36
x = np.arange(-steps_before-1, steps-1, .1)
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
x = x+1
gp_pred_y = gp.timeseries_likelihood(gp.gp_timeseries(x))
gp_mean = gp_pred_y.mean
gp_var = np.array([gp.timeseries_likelihood(gp.gp_timeseries(x_i.reshape((-1,1)))).variance.detach().numpy() for x_i in x]).reshape(-1)
# add NWP to get predicted value from prediction error
NWP_pred = [gp.data_handler.get_NWP(t) for t in times]
gp_pred = np.array(NWP_pred).reshape(-1) + gp_mean.reshape((-1,1)).detach().numpy().reshape(-1)
meas = np.array([gp.data_handler.get_measurement(t) for t in times]).reshape(-1)

plt.figure()
plt.plot(times, gp_pred, color='tab:blue')
plt.plot(times, meas, color='tab:green')
plt.fill_between(times, gp_pred-2*np.sqrt(gp_var), gp_pred+2*np.sqrt(gp_var), alpha=0.2, color='tab:blue')
plt.fill_between(times, gp_pred-np.sqrt(gp_var), gp_pred+np.sqrt(gp_var), alpha=0.4, color='tab:blue')
plt.plot(times, np.array(NWP_pred).reshape(-1), color='tab:orange')
plt.grid()
plt.xlabel('Time')
plt.ylabel('Wind speed (m/s)')
plt.legend(['Measurement', 'GP posterior', 'NWP'])
plt.show()
pass


