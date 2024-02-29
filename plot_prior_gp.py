import datetime

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from modules.gp import TimeseriesModel, get_gp_opt

plt.ion()
matplotlib.use('pgf')
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})
start_time = datetime.datetime(2022,1,1)
stop_time = datetime.datetime(2022,12,31)

gp_opt = get_gp_opt()
gp = TimeseriesModel(gp_opt)

steps = int((stop_time - start_time).total_seconds()/(60*gp.opt['dt_meas']))
times = [start_time+i*datetime.timedelta(minutes=gp.opt['dt_meas']) for i in range(steps)]

x = gp.data_handler.generate_features(start_time, feature='nwp & time').reshape((1,-1))
n_inputs = x.shape[1]

x_mat = np.zeros((steps, n_inputs))
mean_traj = np.zeros(steps)
std_traj = np.zeros(steps)

for i, t in enumerate(times):
    x = gp.data_handler.generate_features(t, feature='nwp & time').reshape((1,-1))
    x_mat[i,:] = x  
    mean, var = gp.gp_prior.compiled_predict_y(x)
    mean_traj[i] = mean
    std_traj[i] = np.sqrt(var)

# fig1, ax1 = plt.subplots(2, int(np.ceil(n_inputs/2)))
xlabels = ['Mean wind speed (m/s)', 'Difference of gust and mean wind speed (m/s)', r'$\sqrt{CAPE}$ ($\sqrt{\mathrm{H/kg}}$)', 'Temperature (k)', 
            'Relative humidity (%)', 'Air pressure (hPa)', 'Time since release of NWP (h)', 'Month']
# for input_dim in range(n_inputs):
#     ix = input_dim%ax1.shape[1]
#     iy = input_dim//ax1.shape[1]
#     ax1[iy,ix].scatter(x_mat[:, input_dim], mean_traj)
#     ax1[iy,ix].scatter(x_mat[:, input_dim], var_traj)
#     ax1[iy,ix].set_xlabel(xlabels[input_dim])
#     ax1[iy,ix].set_ylabel('predicted error')
stds_norm = [gp.data_handler.weather_data['std'][i] for i in [
    'wind_speed_10m_sh', 'wind_speed_of_gust_diff_sh', 'sqrt_specific_convective_available_potential_energy_sh',
    'air_temperature_2m_sh', 'relative_humidity_2m_sh', 'air_pressure_at_sea_level_sh'
]]
means_norm = [gp.data_handler.weather_data['means'][i] for i in [
    'wind_speed_10m_sh', 'wind_speed_of_gust_diff_sh', 'sqrt_specific_convective_available_potential_energy_sh',
    'air_temperature_2m_sh', 'relative_humidity_2m_sh', 'air_pressure_at_sea_level_sh'
]]
n_bins = 10
means = np.zeros((n_bins, n_inputs-1))
std = np.zeros((n_bins, n_inputs-1))
input_means = np.zeros((n_bins, n_inputs-1))
n_points = x_mat.shape[0]
for input_dim in range(n_inputs-1):#
    i_sort = np.argsort(x_mat[:, input_dim])
    # x_min = min(x_mat[:, input_dim])
    # x_max = max(x_mat[:,input_dim])
    # bin_width = (x_max - x_min)/n_bins
    for bin in range(n_bins):
        inputs = i_sort[int(bin/n_bins*n_points):int((bin+1)/n_bins*n_points)]
        # inputs = [i for i in range(steps) 
        #             if x_mat[i, input_dim] >= x_min+bin*bin_width 
        #             and x_mat[i, input_dim] <= x_min + (bin+1)*bin_width]
        means_in_bin = mean_traj[inputs]
        std_in_bin = std_traj[inputs]
        means[bin, input_dim] = np.mean(means_in_bin)
        std[bin, input_dim] = np.mean(std_in_bin)
        input_means[bin, input_dim] = np.mean(x_mat[inputs, input_dim])
means_time = np.zeros(12)
std_time = np.zeros(12)
for month in range(1,13):
    means_in_month = [mean_traj[i] for i, t in enumerate(times) if t.month==month]
    std_in_month = [std_traj[i] for i, t in enumerate(times) if t.month==month]
    means_time[month-1] = np.mean(means_in_month)
    std_time[month-1] = np.mean(std_in_month)

fig, ax = plt.subplots(4, 2, sharey=False, layout='constrained')
for input_dim in range(n_inputs-1):
    ix = input_dim%ax.shape[1]
    iy = input_dim//ax.shape[1]
    if input_dim <= 5:
        input_means_i = input_means[:, input_dim]*stds_norm[input_dim]+means_norm[input_dim]
        if input_dim == 4: input_means_i = input_means_i*100
        if input_dim == 5: input_means_i = input_means_i/100
    elif input_dim == 7:
        input_means_i = (input_means[:, input_dim]-365-366)/24
    else:
        input_means_i = input_means[:,input_dim]
    input_means_i = input_means_i[np.logical_not(np.isnan(std[:,input_dim]))]    
    means_i = means[:, input_dim][np.logical_not(np.isnan(std[:,input_dim]))]
    std_i = std[:, input_dim][np.logical_not(np.isnan(std[:,input_dim]))]
    ax[iy,ix].plot((input_means_i), means_i)
    ax[iy,ix].plot(input_means_i, std_i)
    ax[iy,ix].set_xlabel(xlabels[input_dim])
    ax[iy,ix].grid()
ax[-1,-1].plot(np.arange(1,13), means_time)
ax[-1,-1].plot(np.arange(1,13), std_time)
ax[-1,-1].set_xlabel('Month')
ax[-1,-1].grid()
ax[-1,-1].set_xticks([2,4,6,8,10,12])
ax[0,0].set_ylabel('Predicted error (m/s)')
ax[1,0].set_ylabel('Predicted error (m/s)')
ax[2,0].set_ylabel('Predicted error (m/s)')
ax[3,0].set_ylabel('Predicted error (m/s)')
fig.legend(['Mean', 'Standard deviation'], loc='upper center', bbox_to_anchor=(0.5,1.05), ncol=2)
cm = 1/2.54
fig.set_size_inches(15*cm, 18*cm)
plt.savefig('../Abbildungen/prior_gp_marginal.pgf', bbox_inches='tight')
plt.show()