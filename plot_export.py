import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('pgf')
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})
#plt.ion()
cm = 1/2.54

"""Error mean and variance over prediction"""
x_wind_speed = np.loadtxt('../Abbildungen/data_analysis/error_wind_speed_x.csv')
mean_wind_speed = np.loadtxt('../Abbildungen/data_analysis/error_wind_speed_mean.csv')
std_wind_speed = np.loadtxt('../Abbildungen/data_analysis/error_wind_speed_std.csv')

x_wind_gust = np.loadtxt('../Abbildungen/data_analysis/error_gust_x.csv')
mean_wind_gust = np.loadtxt('../Abbildungen/data_analysis/error_gust_mean.csv')
std_wind_gust = np.loadtxt('../Abbildungen/data_analysis/error_gust_std.csv')

x_wind_gust_diff = np.loadtxt('../Abbildungen/data_analysis/error_gust_diff_x.csv')
mean_wind_gust_diff = np.loadtxt('../Abbildungen/data_analysis/error_gust_diff_mean.csv')
std_wind_gust_diff = np.loadtxt('../Abbildungen/data_analysis/error_gust_diff_std.csv')

fig, axs = plt.subplots(1, 3, layout="constrained", sharey=True)
ax = axs[0]
ax.plot(x_wind_speed, mean_wind_speed, label='Mean')
ax.plot(x_wind_speed, std_wind_speed, label='Standard deviation')
ax.set_xlabel('Mean wind speed (m/s)')
ax.set_ylabel('Prediction error (m/s)')
ax.grid()
ax = axs[1]
ax.plot(x_wind_gust, mean_wind_gust, label='Mean')
ax.plot(x_wind_gust, std_wind_gust, label='Standard deviation')
ax.set_xlabel('Gust wind speed (m/s)')
# ax.set_ylabel('Prediction error (m/s)')
ax.grid()
ax = axs[2]
ax.plot(x_wind_gust_diff, mean_wind_gust_diff, label='Mean')
ax.plot(x_wind_gust_diff, std_wind_gust_diff, label='Standard deviation')
ax.set_xlabel('Difference of gust\nand mean wind speed')
# ax.set_ylabel('Prediction error (m/s)')
ax.grid()
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=2, bbox_to_anchor=(.5,1.17))
fig.set_size_inches(12*cm, 5*cm)
# plt.tight_layout()
# plt.subplots_adjust(top=0.7)
# plt.show()
plt.savefig('../Abbildungen/error_over_wind_speed.pgf', bbox_inches='tight')

# Other NWP values
x_p = np.loadtxt('../Abbildungen/data_analysis/error_pressure_x.csv')
mean_p = np.loadtxt('../Abbildungen/data_analysis/error_pressure_mean.csv')
std_p = np.loadtxt('../Abbildungen/data_analysis/error_pressure_std.csv')

x_t = np.loadtxt('../Abbildungen/data_analysis/error_temperature_x.csv')
mean_t = np.loadtxt('../Abbildungen/data_analysis/error_temperature_mean.csv')
std_t = np.loadtxt('../Abbildungen/data_analysis/error_temperature_std.csv')

x_h = np.loadtxt('../Abbildungen/data_analysis/error_humidity_x.csv')
mean_h = np.loadtxt('../Abbildungen/data_analysis/error_humidity_mean.csv')
std_h = np.loadtxt('../Abbildungen/data_analysis/error_humidity_std.csv')

x_c = np.loadtxt('../Abbildungen/data_analysis/error_sqrt_cape_x.csv')
mean_c = np.loadtxt('../Abbildungen/data_analysis/error_sqrt_cape_mean.csv')
std_c = np.loadtxt('../Abbildungen/data_analysis/error_sqrt_cape_std.csv')

fig, axs = plt.subplots(2, 2, layout="constrained", sharey=True)
ax = axs[0,0]
ax.plot(x_p/100, mean_p, label='Mean')
ax.plot(x_p/100, std_p, label='Standard deviation')
ax.set_xlabel('Predicted air pressure (hPa)')
ax.set_ylabel('Prediction error (m/s)')
ax.grid()
ax = axs[0,1]
ax.plot(x_t, mean_t, label='Mean')
ax.plot(x_t, std_t, label='Standard deviation')
ax.set_xlabel('Predicted temperature (K)')
# ax.set_ylabel('Prediction error (m/s)')
ax.grid()
ax = axs[1,0]
ax.plot(100*x_h, mean_h, label='Mean')
ax.plot(100*x_h, std_h, label='Standard deviation')
ax.set_xlabel('Predicted relative humidity (\%)')
ax.set_ylabel('Prediction error (m/s)')
ax.grid()
ax = axs[1,1]
ax.plot(x_c, mean_c, label='Mean')
ax.plot(x_c, std_c, label='Standard deviation')
ax.set_xlabel(r'Predicted $\sqrt{\mathrm{CAPE}}$ ($\sqrt{\mathrm{J/kg}}$)')
# ax.set_ylabel('Prediction error (m/s)')
ax.grid()
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=2, bbox_to_anchor=(.5,1.1))
fig.set_size_inches(12*cm, 10*cm)
# plt.tight_layout()
# plt.subplots_adjust(top=0.7)
# plt.show()
plt.savefig('../Abbildungen/error_over_NWP.pgf', bbox_inches='tight')



"""Error mean and variance over time"""
x_month = np.loadtxt('../Abbildungen/data_analysis/error_month_x.csv')
mean_month = np.loadtxt('../Abbildungen/data_analysis/error_month_mean.csv')
std_month = np.loadtxt('../Abbildungen/data_analysis/error_month_std.csv')

x_hour = np.loadtxt('../Abbildungen/data_analysis/error_hour_x.csv')
mean_hour = np.loadtxt('../Abbildungen/data_analysis/error_hour_mean.csv')
std_hour = np.loadtxt('../Abbildungen/data_analysis/error_hour_std.csv')

x_step = np.loadtxt('../Abbildungen/data_analysis/error_steps_x.csv')
mean_step = np.loadtxt('../Abbildungen/data_analysis/error_steps_mean.csv')
std_step = np.loadtxt('../Abbildungen/data_analysis/error_steps_std.csv')

fig, axs = plt.subplots(1, 3, layout="constrained", sharey=True)
ax = axs[0]
ax.plot(x_month, mean_month, label='Mean')
ax.plot(x_month, std_month, label='Standard deviation')
ax.set_xlabel('Month')
ax.set_ylabel('Prediction error (m/s)')
ax.grid()
ax = axs[1]
ax.plot(x_hour, mean_hour, label='Mean')
ax.plot(x_hour, std_hour, label='Standard deviation')
ax.set_xlabel('Hour')
#ax.set_ylabel('Prediction error (m/s)')
ax.grid()
ax = axs[2]
ax.plot(x_step, mean_step, label='Mean')
ax.plot(x_step, std_step, label='Standard deviation')
ax.set_xlabel('Hours since release of NWP')
#ax.set_ylabel('Prediction error (m/s)')
ax.grid()
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=2, bbox_to_anchor=(.5,1.2))
fig.set_size_inches(12*cm, 4*cm)
# plt.tight_layout()
# plt.subplots_adjust(top=0.7)
# plt.show()
plt.savefig('../Abbildungen/error_over_time.pgf', bbox_inches='tight')


"""Error covariance over time"""
cov_time = np.loadtxt('../Abbildungen/data_analysis/cov_over_steps.csv')
x_time = np.arange(len(cov_time))/6

fig = plt.figure()
ax = plt.axes()
ax.plot(x_time, cov_time)
ax.grid()
plt.xlabel('Time difference (h)')
plt.ylabel('Error covariance ($\mathrm{(m/s)}^2$)')
fig.set_size_inches(12*cm, 6*cm)
plt.savefig('../Abbildungen/covariance_over_time.pgf', bbox_inches='tight')


"""Error correlation over time for different CAPE and pressure"""
cov_cape = np.loadtxt('../Abbildungen/data_analysis/cov_over_steps_cape.csv')
sqrt_cape = np.loadtxt('../Abbildungen/data_analysis/cape.csv')
sqrt_cape_sorted = np.sort(sqrt_cape)
n_vals = len(sqrt_cape)
bins = cov_cape.shape[0]
cape_lb = [sqrt_cape_sorted[int(i/bins*n_vals)] for i in range(bins)]
cape_ub = [sqrt_cape_sorted[int((i+1)/bins*n_vals)-1] for i in range(bins)]

# labels_cape = [f'${cape_lb[i]**2:.3g}\leq CAPE/J\cdot kg^{"{"}-1{"}"}\leq{cape_ub[i]**2:.3g}$' for i in range(bins)]

steps = cov_cape.shape[1]
x_cape = np.array([np.arange(steps)/6]*bins)

cov_p = np.loadtxt('../Abbildungen/data_analysis/cov_over_steps_pressure.csv')
p = np.loadtxt('../Abbildungen/data_analysis/pressure.csv')
p_sorted = np.sort(p/100)
n_vals = len(p)
bins = cov_p.shape[0]
p_lb = [p_sorted[int(i/bins*n_vals)] for i in range(bins)]
p_ub = [p_sorted[int((i+1)/bins*n_vals)-1] for i in range(bins)]

labels_p = [f'${int(p_lb[i])}\leq p/hPa\leq{int(p_ub[i])}$' for i in range(bins)]
labels_fig = ['lower third', 'middle third', 'upper third']
steps = cov_p.shape[1]
x_p = np.array([np.arange(steps)/6]*bins)

fig, axs = plt.subplots(1, 2, layout='constrained', sharey=True)
axs[0].plot(x_cape.T, cov_cape.T)#, label=labels_cape)
axs[0].set_xlabel('Time difference (h)')
axs[0].set_ylabel('Error correlation')
axs[0].set_title('CAPE')
# axs[0].legend(loc='upper center', bbox_to_anchor=(.5,2))
axs[0].grid()
axs[1].plot(x_p.T, cov_p.T)#, label=labels_p)
axs[1].set_xlabel('Time difference (h)')
axs[1].set_title('Pressure')
# axs[1].set_ylabel('Error correlation')
# axs[1].legend(loc='upper center', bbox_to_anchor=(.5,2))
axs[1].grid()
fig.legend(labels=labels_fig, loc='upper center', bbox_to_anchor=(.5, 1.17), ncol=3)
fig.set_size_inches(12*cm, 6*cm)
plt.savefig('../Abbildungen/correlation_over_time_NWP.pgf', bbox_inches='tight')

"""Histograms for conditional distribution"""
from scipy.stats import norm
predictions = np.loadtxt('../Abbildungen/data_analysis/wind_predictions.csv')
measurements = np.loadtxt('../Abbildungen/data_analysis/wind_measurements.csv')

errors = measurements - predictions

wind_predicted_range = (2,3)
indices_in_range = [i for i, v in enumerate(predictions) 
                    if v > wind_predicted_range[0] and v <= wind_predicted_range[1]]
errors_in_range = errors[indices_in_range]
measurements_in_range = measurements[indices_in_range]
mean_error = np.mean(errors_in_range)
std_error = np.std(errors_in_range)
norm_1_error = norm(mean_error, std_error)
mean_meas = np.mean(measurements_in_range)
std_meas = np.std(measurements_in_range)
norm_1_meas = norm(mean_meas, std_meas)

bins = 50
fig, axs = plt.subplots(2, layout='constrained')
labels = []
axs[1].hist(errors_in_range, bins=bins, density=True, histtype='step')
axs[1].set_xlabel('Prediction error')
axs[0].hist(measurements_in_range, bins=bins, density=True, histtype='step')
axs[0].set_xlabel('Measured wind speed')
labels.append(fr'''${wind_predicted_range[0]}\,\mathrm{"{"}m/s{"}"}\leq v_\mathrm{"{"}wind,MF{"}"}\leq {wind_predicted_range[1]}\,\mathrm{"{"}m/s{"}"}$''')

wind_predicted_range = (4,5)
indices_in_range = [i for i, v in enumerate(predictions) 
                    if v > wind_predicted_range[0] and v <= wind_predicted_range[1]]
errors_in_range = errors[indices_in_range]
measurements_in_range = measurements[indices_in_range]
mean_error = np.mean(errors_in_range)
std_error = np.std(errors_in_range)
norm_2_error = norm(mean_error, std_error)
mean_meas = np.mean(measurements_in_range)
std_meas = np.std(measurements_in_range)
norm_2_meas = norm(mean_meas, std_meas)


axs[1].hist(errors_in_range, bins=bins, density=True, histtype='step')
axs[1].set_xlabel('Prediction error')
axs[0].hist(measurements_in_range, bins=bins, density=True, histtype='step')
axs[0].set_xlabel('Measured wind speed')
labels.append(fr'''${wind_predicted_range[0]}\,\mathrm{"{"}m/s{"}"}\leq v_\mathrm{"{"}wind,MF{"}"}\leq {wind_predicted_range[1]}\,\mathrm{"{"}m/s{"}"}$''')


wind_predicted_range = (10,11)
indices_in_range = [i for i, v in enumerate(predictions) 
                    if v > wind_predicted_range[0] and v <= wind_predicted_range[1]]
errors_in_range = errors[indices_in_range]
measurements_in_range = measurements[indices_in_range]
mean_error = np.mean(errors_in_range)
std_error = np.std(errors_in_range)
norm_3_error = norm(mean_error, std_error)
mean_meas = np.mean(measurements_in_range)
std_meas = np.std(measurements_in_range)
norm_3_meas = norm(mean_meas, std_meas)

axs[1].hist(errors_in_range, bins=bins, density=True, histtype='step')
axs[1].set_xlabel('Prediction error')
axs[0].hist(measurements_in_range, bins=bins, density=True, histtype='step')
axs[0].set_xlabel('Measured wind speed')
axs[0].grid()
axs[1].grid()
axs[1].set_xlim(-6,6)
labels.append(fr'''${wind_predicted_range[0]}\,\mathrm{"{"}m/s{"}"}\leq v_\mathrm{"{"}wind,MF{"}"}\leq {wind_predicted_range[1]}\,\mathrm{"{"}m/s{"}"}$''')
fig.legend(labels, loc='upper center', bbox_to_anchor = (.5, 1.15), ncol=2)
# add plots for normal distribution
# measurements
xmin_meas, xmax_meas = axs[0].get_xlim()
x_meas = np.linspace(xmin_meas, xmax_meas, 200)
axs[0].plot(x_meas, norm_1_meas.pdf(x_meas), '--', color='tab:blue', alpha=0.5)
axs[0].plot(x_meas, norm_2_meas.pdf(x_meas), '--', color='tab:orange', alpha=0.5)
axs[0].plot(x_meas, norm_3_meas.pdf(x_meas), '--', color='tab:green', alpha=0.5)
xmin_err, xmax_err = axs[1].get_xlim()
x_err = np.linspace(xmin_err, xmax_err, 200)
axs[1].plot(x_err, norm_1_error.pdf(x_err), '--', color='tab:blue', alpha=0.5)
axs[1].plot(x_err, norm_2_error.pdf(x_err), '--', color='tab:orange', alpha=0.5)
axs[1].plot(x_err, norm_3_error.pdf(x_err), '--', color='tab:green', alpha=0.5)

fig.set_size_inches(12*cm, 10*cm)
plt.savefig('../Abbildungen/conditional_distribution_hist.pgf', bbox_inches='tight')


"""Histogram for error and power distribution"""
predictions = np.loadtxt('../Abbildungen/data_analysis/wind_predictions.csv')
measurements = np.loadtxt('../Abbildungen/data_analysis/wind_measurements.csv')

errors = measurements - predictions

fig, axs = plt.subplots(2, layout='constrained')
axs[0].hist(measurements, bins=50, density=True, histtype='step')
axs[0].set_xlabel('Wind speed (m/s)')
axs[1].hist(errors, bins=250, density=True, histtype='step')
axs[1].set_xlabel('Prediction error (m/s)')
axs[1].set_xlim(-7.5,7.5)
axs[0].grid()
axs[1].grid()

# Plot distributions
from scipy.stats import norm, weibull_min
mean_err = np.mean(errors)
std_error = np.std(errors)
xmin, xmax = axs[1].get_xlim()
x_err = np.linspace(xmin, xmax, 200)
norm_err = norm.pdf(x_err, loc=mean_err, scale=std_error)
axs[1].plot(x_err, norm_err, '--', color='tab:blue', alpha=0.5)

# weibull_dist = weibull_min(1)
c, loc, scale = weibull_min.fit(measurements, loc=0) #scale=9.22, c=2.12
weibull_dist = weibull_min(c, loc=loc, scale=scale)
_, xmax = axs[0].get_xlim()
x_meas = np.linspace(0, xmax, 200)
axs[0].plot(x_meas, weibull_dist.pdf(x_meas), '--', color='tab:blue', alpha=0.5)

fig.set_size_inches(10*cm, 6*cm)
plt.savefig('../Abbildungen/hist_distribution.pgf', bbox_inches='tight')

# scoring
from modules.gp.fileloading import load_weather_data
from modules.gp import get_gp_opt
from modules.gp.scoring import (get_interval_score, get_mae, get_posterior_trajectories, get_rmse, 
    get_RE, get_trajectory_gp_prior, get_trajectory_measured, get_trajectory_nwp, 
    get_direct_model_trajectories, get_simple_timeseries_traj)

opt = get_gp_opt(n_z=200, max_epochs_second_training=10, epochs_timeseries_retrain=500, 
                    epochs_timeseries_first_train=500)
weather_data = load_weather_data(opt['t_start'], opt['t_end'])

try:
    trajectory_measured = np.loadtxt('modules/gp/scoring/trajectory_meas.csv')
except:
    trajectory_measured = get_trajectory_measured(weather_data, opt)
    np.savetxt('modules/gp/scoring/trajectory_meas.csv', trajectory_measured)
try:
    trajectory_nwp = np.loadtxt('modules/gp/scoring/trajectory_nwp.csv')
except:
    trajectory_nwp = get_trajectory_nwp(weather_data, opt)
    np.savetxt('modules/gp/scoring/trajectory_nwp.csv', trajectory_nwp)
try:
    trajectory_gp_prior = np.loadtxt('modules/gp/scoring/trajectory_gp_prior_heteroscedastic_200.csv')
    var_gp_prior = np.loadtxt('modules/gp/scoring/var_gp_prior_heteroscedastic_200.csv')
except:
    trajectory_gp_prior, var_gp_prior = get_trajectory_gp_prior(weather_data, opt)
    np.savetxt('modules/gp/scoring/trajectory_gp_prior.csv', trajectory_gp_prior)
    np.savetxt('modules/gp/scoring/var_gp_prior.csv', var_gp_prior)
rmse_nwp = get_rmse(trajectory_measured, trajectory_nwp)
mae_nwp = get_mae(trajectory_measured, trajectory_nwp)

rmse_gp_prior = get_rmse(trajectory_measured, trajectory_gp_prior)
mae_gp_prior = get_mae(trajectory_measured, trajectory_gp_prior)

alpha_vec = np.linspace(0.01,1,100)
re_gp_prior = [get_RE(alpha, trajectory_measured, trajectory_gp_prior, var_gp_prior)
                for alpha in alpha_vec]
int_score_gp_prior = [get_interval_score(alpha, trajectory_measured, trajectory_gp_prior, var_gp_prior)
                for alpha in alpha_vec]

percent_in_interval_gp_prior = np.array(re_gp_prior) + (1-alpha_vec)    

print(f'RMSE of NWP: {rmse_nwp}, MAE of NWP: {mae_nwp}')
print(f'RMSE of GP: {rmse_gp_prior}, MAE of GP: {mae_gp_prior}')

fig, axs = plt.subplots(1, 3, layout='constrained')
ax = axs[0]
ax.plot(alpha_vec, re_gp_prior)
ax.set_xlabel(r'$\alpha$')
ax.set_ylabel('Reliability evaluation')
ax.grid()
ax = axs[1]
ax.plot(alpha_vec, int_score_gp_prior)
ax.set_xlabel(r'$\alpha$')
ax.set_ylabel('Interval score')
ax.grid()
ax = axs[2]
ax.plot(1-alpha_vec, percent_in_interval_gp_prior)
ax.plot(alpha_vec, alpha_vec, '--', color='tab:gray', alpha=0.75, lw=0.5)
ax.set_xlabel(r'1-$\alpha$')
ax.set_ylabel(r'Actual proportion in 1-$\alpha$-interval')
ax.grid()
fig.set_size_inches(15*cm, 7*cm)
plt.savefig('../Abbildungen/nwp_gp_scoring.pgf', bbox_inches='tight')

# Homoscedastic vs heteroscedastic
try:
    trajectory_measured = np.loadtxt('modules/gp/scoring/trajectory_meas.csv')
except:
    trajectory_measured = get_trajectory_measured(weather_data, opt)
    np.savetxt('modules/gp/scoring/trajectory_meas.csv', trajectory_measured)
try:
    trajectory_nwp = np.loadtxt('modules/gp/scoring/trajectory_nwp.csv')
except:
    trajectory_nwp = get_trajectory_nwp(weather_data, opt)
    np.savetxt('modules/gp/scoring/trajectory_nwp.csv', trajectory_nwp)
try:
    trajectory_gp_prior = np.loadtxt('modules/gp/scoring/trajectory_gp_prior_heteroscedastic_200.csv')
    var_gp_prior = np.loadtxt('modules/gp/scoring/var_gp_prior_heteroscedastic_200.csv')
    trajectory_gp_prior_homoscedastic = np.loadtxt('modules/gp/scoring/trajectory_gp_prior_homoscedastic.csv')
    var_gp_prior_homoscedastic = np.loadtxt('modules/gp/scoring/var_gp_prior_homoscedastic.csv')
except:
    trajectory_gp_prior, var_gp_prior = get_trajectory_gp_prior(weather_data, opt)
    np.savetxt('modules/gp/scoring/trajectory_gp_prior.csv', trajectory_gp_prior)
    np.savetxt('modules/gp/scoring/var_gp_prior.csv', var_gp_prior)
rmse_nwp = get_rmse(trajectory_measured, trajectory_nwp)
mae_nwp = get_mae(trajectory_measured, trajectory_nwp)

rmse_gp_prior_homoscedastic = get_rmse(trajectory_measured, trajectory_gp_prior_homoscedastic)
mae_gp_prior_homoscedastic = get_mae(trajectory_measured, trajectory_gp_prior_homoscedastic)

alpha_vec = np.linspace(0.01,1,100)
re_gp_prior_homoscedastic = [get_RE(alpha, trajectory_measured, trajectory_gp_prior_homoscedastic, var_gp_prior_homoscedastic)
                for alpha in alpha_vec]
int_score_gp_prior_homoscedastic = [get_interval_score(alpha, trajectory_measured, trajectory_gp_prior_homoscedastic, var_gp_prior_homoscedastic)
                for alpha in alpha_vec]

percent_in_interval_gp_prior = np.array(re_gp_prior) + (1-alpha_vec)    
percent_in_interval_gp_prior_homoscedastic = np.array(re_gp_prior_homoscedastic) + (1-alpha_vec)    

print(f'RMSE of NWP: {rmse_nwp}, MAE of NWP: {mae_nwp}')
print(f'RMSE of GP: {rmse_gp_prior}, MAE of GP: {mae_gp_prior}')
print(f'RMSE of homoscedastic GP: {rmse_gp_prior_homoscedastic}, MAE of GP: {mae_gp_prior_homoscedastic}')

fig, axs = plt.subplots(1, 3, layout='constrained')
ax = axs[0]
ax.plot(alpha_vec, re_gp_prior)
ax.plot(alpha_vec, re_gp_prior_homoscedastic)
ax.set_xlabel(r'$\alpha$')
ax.set_ylabel('Reliability evaluation')
ax.grid()
ax = axs[1]
ax.plot(alpha_vec, int_score_gp_prior)
ax.plot(alpha_vec, int_score_gp_prior_homoscedastic)
ax.set_xlabel(r'$\alpha$')
ax.set_ylabel('Interval score')
ax.grid()
ax = axs[2]
ax.plot(1-alpha_vec, percent_in_interval_gp_prior)
ax.plot(1-alpha_vec, percent_in_interval_gp_prior_homoscedastic)
ax.plot(alpha_vec, alpha_vec, '--', color='tab:gray', alpha=0.75, lw=0.5)
ax.set_xlabel(r'1-$\alpha$')
ax.set_ylabel(r'Actual proportion in 1-$\alpha$-interval')
ax.grid()
fig.legend(['Heteroscedastic GP', 'Homoscedastic GP'], loc='upper center', bbox_to_anchor = (0.5,1.15), ncol=2)
fig.set_size_inches(15*cm, 7*cm)
plt.savefig('../Abbildungen/nwp_gp_scoring_homoscedastic_heteroscedastic.pgf', bbox_inches='tight')

steps_forward = opt['steps_forward']
rmse_post = np.zeros(steps_forward)
mae_post = np.zeros(steps_forward)
re_post = np.zeros(steps_forward)
score_post = np.zeros(steps_forward)

trajectories_mean_post, trajectories_var_post = get_posterior_trajectories(opt)
# trajectories_mean_post = np.loadtxt('gp/scoring/trajectories_mean_post.csv')
# trajectories_var_post = np.loadtxt('gp/scoring/trajectories_var_post.csv')
# first dimension: time of prediction, second dimension: number of steps forward
# n_points = len(trajectory_measured)
alpha = 0.1
for i in range(opt['steps_forward']):
    trajectory_post = trajectories_mean_post[:,i]
    var_post = trajectories_var_post[:,i]
    n_points = len(trajectory_post)
    rmse_post[i] = get_rmse(trajectory_measured[i:n_points+i], trajectory_post)
    mae_post[i] = get_mae(trajectory_measured[i:n_points+i], trajectory_post)
    re_post[i] = get_RE(alpha, trajectory_measured[i:n_points+i], trajectory_post, var_post)
    score_post[i] = get_interval_score(alpha, trajectory_measured[i:n_points+i], trajectory_post, var_post)
# steps = np.arange(1, steps_forward+1)
# plt.figure()
# plt.plot(steps, rmse_post)
# plt.xlabel('Number of steps predicted forward')
# plt.ylabel('RMSE of prediction')
# plt.figure()
# plt.plot(steps, mae_post)
# plt.xlabel('Number of steps predicted forward')
# plt.ylabel('MAE of prediction')
# plt.figure()
# plt.plot(steps, re_post)
# plt.xlabel('Number of steps predicted forward')
# plt.ylabel('RE of prediction interval for alpha=0.1')
# plt.figure()
# plt.plot(steps, score_post)
# plt.xlabel('Number of steps predicted forward')
# plt.ylabel('Interval score of prediction interval for alpha=0.1')

rmse_direct = np.zeros(steps_forward)
mae_direct = np.zeros(steps_forward)
re_direct = np.zeros(steps_forward)
score_direct = np.zeros(steps_forward)

trajectories_mean_direct, trajectories_var_direct = get_direct_model_trajectories(opt)
# trajectories_mean_post = np.loadtxt('gp/scoring/trajectories_mean_post.csv')
# trajectories_var_post = np.loadtxt('gp/scoring/trajectories_var_post.csv')
# first dimension: time of prediction, second dimension: number of steps forward
# n_points = len(trajectory_measured)
alpha = 0.1
for i in range(opt['steps_forward']):
    trajectory_direct = trajectories_mean_direct[:,i]
    var_direct = trajectories_var_direct[:,i]
    n_points = len(trajectory_direct)
    rmse_direct[i] = get_rmse(trajectory_measured[i:n_points+i], trajectory_direct)
    mae_direct[i] = get_mae(trajectory_measured[i:n_points+i], trajectory_direct)
    re_direct[i] = get_RE(alpha, trajectory_measured[i:n_points+i], trajectory_direct, var_direct)
    score_direct[i] = get_interval_score(alpha, trajectory_measured[i:n_points+i], trajectory_direct, var_direct)
hours = np.arange(0,len(rmse_direct)/6, 1/6)+1/6
fig, axs = plt.subplots(2,2,layout='constrained', sharex=True)
ax = axs[0,0]
ax.plot(hours, rmse_post, label='Time series model')
ax.plot(hours, rmse_direct, label='Direct model')
# ax.set_xlabel('Time predicted ahead (h)')
ax.set_ylabel('RMSE')
ax.grid()
ax = axs[0,1]
ax.plot(hours, mae_post, label='Time series model')
ax.plot(hours, mae_direct, label='Direct model')
# ax.set_xlabel('Time predicted ahead (h)')
ax.set_ylabel('MAE')
ax.grid()
ax = axs[1,0]
ax.plot(hours, score_post, label='Time series model')
ax.plot(hours, score_direct, label='Direct model')
ax.set_xlabel('Time predicted ahead (h)')
ax.set_ylabel('Interval score')
ax.grid()
ax = axs[1,1]
ax.plot(hours, re_post, label='Time series model')
ax.plot(hours, re_direct, label='Direct model')
ax.set_xlabel('Time predicted ahead (h)')
ax.set_ylabel('Reliability evaluation')
ax.grid()
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=2, bbox_to_anchor=(.5,1.1))
fig.set_size_inches(12*cm, 10*cm)
plt.savefig('../Abbildungen/timeseries_vs_direct.pgf', bbox_inches='tight')

"""Time series with GP prior vs simple time series"""
steps_forward = opt['steps_forward']
rmse_simple = np.zeros(steps_forward-1)
mae_simple = np.zeros(steps_forward-1)
re_simple = np.zeros(steps_forward-1)
score_simple = np.zeros(steps_forward-1)

trajectories_mean_simple, trajectories_var_simple = get_simple_timeseries_traj(opt)

alpha = 0.1
for i in range(opt['steps_forward']-1):
    trajectory_simple = trajectories_mean_simple[:,i+1]
    var_simple = trajectories_var_simple[:,i+1]
    n_points = len(trajectory_simple)
    rmse_simple[i] = get_rmse(trajectory_measured[i:n_points+i], trajectory_simple)
    mae_simple[i] = get_mae(trajectory_measured[i:n_points+i], trajectory_simple)
    re_simple[i] = get_RE(alpha, trajectory_measured[i:n_points+i], trajectory_simple, var_simple)
    score_simple[i] = get_interval_score(alpha, trajectory_measured[i:n_points+i], trajectory_simple, var_simple)

fig, axs = plt.subplots(2,2,layout='constrained', sharex=True)
ax = axs[0,0]
ax.plot(hours[:-1], rmse_post[:-1], label='GP Prior')
ax.plot(hours[:-1], rmse_simple, label='Simple prior')
# ax.set_xlabel('Time predicted ahead (h)')
ax.set_ylabel('RMSE')
ax.grid()
ax = axs[0,1]
ax.plot(hours[:-1], mae_post[:-1], label='GP Prior')
ax.plot(hours[:-1], mae_simple, label='Simple prior')
# ax.set_xlabel('Time predicted ahead (h)')
ax.set_ylabel('MAE')
ax.grid()
ax = axs[1,0]
ax.plot(hours[:-1], score_post[:-1], label='GP Prior')
ax.plot(hours[:-1], score_simple, label='Simple prior')
ax.set_xlabel('Time predicted ahead (h)')
ax.set_ylabel('Interval score')
ax.grid()
ax = axs[1,1]
ax.plot(hours[:-1], re_post[:-1], label='GP Prior')
ax.plot(hours[:-1], re_simple, label='Simple prior')
ax.set_xlabel('Time predicted ahead (h)')
ax.set_ylabel('Reliability evaluation')
ax.grid()
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=2, bbox_to_anchor=(.5,1.1))
fig.set_size_inches(12*cm, 10*cm)
plt.savefig('../Abbildungen/timeseries_gp_vs_simple.pgf', bbox_inches='tight')
"""Power curve"""
# from modules.models.ohps import OHPS
# ohps = OHPS()
# wind_speeds = np.linspace(0,30,100)
# power_outputs = np.array([ohps.wind_turbine.power_curve_fun(w) for w in wind_speeds]).reshape(-1)
# fig = plt.figure()
# ax = plt.axes()
# ax.plot(wind_speeds, power_outputs)
# ax.set_xlabel('Wind speed at hub height (m/s)')
# ax.set_ylabel('Power output (kW)')
# fig.set_size_inches(10*cm,7*cm)
# plt.savefig('../Abbildungen/power_curve.pgf', bbox_inches='tight')


pass
