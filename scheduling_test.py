import datetime

import matplotlib.pyplot as plt
import casadi as ca

from modules.mpc import DayAheadScheduler
from modules.models import OHPS
from modules.gp import DataHandler, get_gp_opt

plt.ion()

gp_opt = get_gp_opt()
dh = DataHandler(datetime.datetime(2020,1,1), datetime.datetime(2022,12,31), gp_opt)
ohps = OHPS()
scheduler = DayAheadScheduler(ohps, dh)

t = datetime.datetime(2022,12,8,0)
times = [t + i*datetime.timedelta(hours=1) for i in range(24)]

P_min = 16000
E_target = 24*40000
P_demand, v_opt, p = scheduler.solve_problem(t, P_min, E_target, 0.5)
P_gtg_opt = v_opt[:24]
I_bat_opt = v_opt[24:2*24]
X_bat_opt = v_opt[2*24:-1]
P_wtg = p[:24]
P_bat_opt = scheduler.P_bat_fun(v_opt)

plt.figure()
plt.plot(times, P_demand/1000)
plt.plot(times, P_gtg_opt/1000)
plt.plot(times, P_wtg/1000)
plt.plot(times, P_bat_opt/1000)
plt.plot(times, P_min/1000*ca.DM.ones(24), '--', color='black')
plt.xlabel('Time')
plt.ylabel('Scheduled power output (MW)')
plt.legend(['Total scheduled power', 'GTG', 'WTG', 'Battery'])

plt.figure()
plt.plot(times, ca.cumsum(P_demand/1000))
plt.plot(times, ca.DM.ones(24)*E_target/1000, '--', color='black')

times = times + [t+datetime.timedelta(hours=24)]
plt.figure()
plt.plot(times, 1-X_bat_opt)
plt.xlabel('Time')
plt.ylabel('SOC')
plt.show()
pass