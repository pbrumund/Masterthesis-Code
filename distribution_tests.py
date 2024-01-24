import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import normaltest, shapiro
import statsmodels.api as sm

"""Compare https://rowannicholls.github.io/python/statistics/distributions/tests_for_normality.html"""

def test_normality_shapiro(x, alpha):
    stat, p = shapiro(x)
    if p > alpha:
        return True
    return False

def test_normality_dagostino(x, alpha):
    try:
        stat, p = normaltest(x)
    except:
        return None
    if p > alpha:
        return True
    return False

def test_normality_lilliefors(x, alpha):
    stat, p = sm.stats.diagnostic.lilliefors(x)
    if p > alpha:
        return True
    return False

predictions = np.loadtxt('../Abbildungen/data_analysis/wind_predictions.csv')
measurements = np.loadtxt('../Abbildungen/data_analysis/wind_measurements.csv')

errors = measurements - predictions

# select values only in 6 h intervals to reduce correlation
n_vals = len(errors)
step = 3*6
i_select = np.arange(0, n_vals, step)
errors = errors[i_select]
predictions = predictions[i_select]
measurements = measurements[i_select]

interval_borders = np.arange(0,20,19)

alpha = 0.05
for i, (lb, ub) in enumerate(zip(interval_borders[:-1], interval_borders[1:])):
    indices_in_range = np.where(np.logical_and(predictions>=lb, predictions<ub))
    errors_in_range = errors[indices_in_range]
    measurements_in_range = measurements[indices_in_range]
    result_shapiro = test_normality_shapiro(measurements_in_range, alpha)
    result_dagostino = test_normality_dagostino(measurements_in_range, alpha)
    result_lilliefors = test_normality_lilliefors(measurements_in_range, alpha)
    print(f'Wind speeds between {lb} and {ub}: {len(errors_in_range)}')
    print(f'Shapiro-Wilk: {result_shapiro}')
    print(f"D'Agostino-Pearson: {result_dagostino}")
    print(f'Lilliefors: {result_lilliefors}')
    if i%10==0:
        plt.figure()
        plt.hist(errors_in_range, bins=50)
plt.show()
pass

    
