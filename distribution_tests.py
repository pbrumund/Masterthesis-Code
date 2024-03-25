import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import normaltest, shapiro
# import statsmodels.api as sm

# """Compare https://rowannicholls.github.io/python/statistics/distributions/tests_for_normality.html"""

# def test_normality_shapiro(x, alpha):
#     stat, p = shapiro(x)
#     if p > alpha:
#         return True
#     return False

# def test_normality_dagostino(x, alpha):
#     try:
#         stat, p = normaltest(x)
#     except:
#         return None
#     if p > alpha:
#         return True
#     return False

# def test_normality_lilliefors(x, alpha):
#     stat, p = sm.stats.diagnostic.lilliefors(x)
#     if p > alpha:
#         return True
#     return False

# predictions = np.loadtxt('../Abbildungen/data_analysis/wind_predictions.csv')
# measurements = np.loadtxt('../Abbildungen/data_analysis/wind_measurements.csv')

# errors = measurements - predictions

# # select values only in 6 h intervals to reduce correlation
# n_vals = len(errors)
# step = 3*6
# i_select = np.arange(0, n_vals, step)
# errors = errors[i_select]
# predictions = predictions[i_select]
# measurements = measurements[i_select]

# interval_borders = np.arange(0,20,19)

# alpha = 0.05
# for i, (lb, ub) in enumerate(zip(interval_borders[:-1], interval_borders[1:])):
#     indices_in_range = np.where(np.logical_and(predictions>=lb, predictions<ub))
#     errors_in_range = errors[indices_in_range]
#     measurements_in_range = measurements[indices_in_range]
#     result_shapiro = test_normality_shapiro(measurements_in_range, alpha)
#     result_dagostino = test_normality_dagostino(measurements_in_range, alpha)
#     result_lilliefors = test_normality_lilliefors(measurements_in_range, alpha)
#     print(f'Wind speeds between {lb} and {ub}: {len(errors_in_range)}')
#     print(f'Shapiro-Wilk: {result_shapiro}')
#     print(f"D'Agostino-Pearson: {result_dagostino}")
#     print(f'Lilliefors: {result_lilliefors}')
#     if i%10==0:
#         plt.figure()
#         plt.hist(errors_in_range, bins=50)

def get_RE(alpha, errors, var_predicted):
    # From Kou paper
    a = norm.ppf(1-0.5*alpha) # Number of ± standard deviations for 1-alpha-interval
    i_in_interval = [i for i, error in enumerate(errors) if np.abs(error)<a*(np.sqrt(var_predicted))]
    p_in_interval = len(i_in_interval)/len(errors)
    return p_in_interval - (1-alpha)

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

alpha_vec = np.linspace(0.01,1,100)
re = [get_RE(alpha, errors-mean_err, std_error**2) for alpha in alpha_vec]
plt.figure()
plt.plot(alpha_vec, re)
plt.xlabel('alpha')
plt.ylabel('RE')
plt.show()
pass

    
