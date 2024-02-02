import numpy as np

loads = np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
efficiencies = np.array([0.7525, 0.8325, 0.8875, 0.93, 0.9525, 0.9725, 0.9875, 1])
loads = np.array([0.25, 0.305, .369, .4385, .533, .627, .689, .762, 1])
efficiencies = np.array([0.3467, .37, .39, .41, .428, .443, .45, .458, .484])
coeffs = np.polyfit(loads, efficiencies, 2)
print(coeffs)