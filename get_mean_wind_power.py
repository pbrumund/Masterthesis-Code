import datetime

import numpy as np
import matplotlib.pyplot as plt

from modules.models import OHPS
from modules.gp import DataHandler, get_gp_opt

ohps = OHPS()
opt = get_gp_opt()
dh = DataHandler(datetime.datetime(2012,1,1), datetime.datetime(2021,12,31), opt)

times = [datetime.datetime(2012,1,1)+i*datetime.timedelta(minutes=10) for i in range((3*366+7*365)*24*6-1)]
means = []

for month in range(1,12+1):
    times_in_month = [t for t in times if t.month == month]
    wind_speeds = [dh.get_measurement(t) for t in times_in_month]
    wind_power = [ohps.get_P_wtg(0,0,w) for w in wind_speeds]
    mean_wind_power = np.mean(wind_power)
    means.append(mean_wind_power)
    print(f'Month: {month}, mean wind power: {mean_wind_power:.2g}')

plt.figure()
plt.plot(np.arange(1,13), np.array(means)/1000)

dh = DataHandler(datetime.datetime(2020,1,1), datetime.datetime(2022,12,31), opt)
times = [datetime.datetime(2022,1,1)+i*datetime.timedelta(minutes=10) for i in range((365)*24*6-1)]
means = []

for month in range(1,12+1):
    times_in_month = [t for t in times if t.month == month]
    wind_speeds = [dh.get_measurement(t) for t in times_in_month]
    wind_power = [ohps.get_P_wtg(0,0,w) for w in wind_speeds]
    mean_wind_power = np.mean(wind_power)
    means.append(mean_wind_power)
    print(f'Month: {month}, mean wind power: {mean_wind_power:.2g}')

plt.plot(np.arange(1,13), np.array(means)/1000)
plt.legend(['2012-2021', '2022'])    
plt.show()