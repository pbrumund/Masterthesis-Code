import numpy as np
import matplotlib.pyplot as plt
import csv
import datetime

path = '../Winddaten/table.csv'
with open(path, 'r') as f:
    reader = csv.reader(f, delimiter=';')
    headers = next(reader)
    wind_speeds = np.array(list(reader))[:-1,3].astype(float)
t0 = datetime.datetime(2020,1,1)
dt = datetime.timedelta(minutes=10)
times = np.array([t0+n*dt for n in range(len(wind_speeds))])
plt.plot(times, wind_speeds)
plt.show()
