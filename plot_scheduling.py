import datetime

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import casadi as ca

from modules.gp import DataHandler, get_gp_opt
from modules.models import OHPS
from modules.mpc import DayAheadScheduler, get_mpc_opt


matplotlib.use('pgf')
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

ohps = OHPS()
gp_opt = get_gp_opt()
dh = DataHandler(datetime.datetime(2020,1,1), datetime.datetime(2022,12,31), gp_opt)
mpc_opt = get_mpc_opt()
scheduler = DayAheadScheduler(ohps, dh, mpc_opt)

t1 = datetime.datetime(2022,1,1,0)
_ = scheduler.get_P_demand(t1, 0.5, 0)
sol1 = np.array(scheduler.P_sched).reshape(-1)
P_gtg1 = scheduler.P_gtg_fun(scheduler.v_opt)
P_bat1 = scheduler.P_bat_fun(scheduler.v_opt)
P_wtg1 = scheduler.P_wtg_NWP
P_out1 = scheduler.P_sched
# sol1 = 2*sol1 - scheduler.get_P_demand_mean(t1)
sol1 = np.concatenate([[0], sol1])

t2 = t1+datetime.timedelta(hours=6)
E1 = np.sum(sol1[:7])
_ = scheduler.get_P_demand(t2, 0.5, 6*scheduler.get_P_demand_mean(t1)-E1)
sol2 = np.array(scheduler.P_sched).reshape(-1)
P_gtg2 = scheduler.P_gtg_fun(scheduler.v_opt)
P_bat2 = scheduler.P_bat_fun(scheduler.v_opt)
P_wtg2 = scheduler.P_wtg_NWP
P_out2 = scheduler.P_sched
# sol2 = 2*sol2 - scheduler.get_P_demand_mean(t1)
sol2 = np.concatenate([[0], sol2])

t3 = t1+datetime.timedelta(hours=12)
E2 = np.sum(sol2[:7]) + E1
_ = scheduler.get_P_demand(t2, 0.5, 12*scheduler.get_P_demand_mean(t1)-E2)
sol3 = np.array(scheduler.P_sched).reshape(-1)
P_gtg3 = scheduler.P_gtg_fun(scheduler.v_opt)
P_bat3 = scheduler.P_bat_fun(scheduler.v_opt)
P_wtg3 = scheduler.P_wtg_NWP
P_out3 = scheduler.P_sched
# sol3 = 2*sol3 - scheduler.get_P_demand_mean(t1)
sol3 = np.concatenate([[0], sol3])

t4 = t1+datetime.timedelta(hours=18)
E3 = np.sum(sol3[:7]) + E2
_ = scheduler.get_P_demand(t3, 0.5, 18*scheduler.get_P_demand_mean(t1)-E3)
sol4 = np.array(scheduler.P_sched).reshape(-1)
P_gtg4 = scheduler.P_gtg_fun(scheduler.v_opt)
P_bat4 = scheduler.P_bat_fun(scheduler.v_opt)
P_wtg4 = scheduler.P_wtg_NWP
P_out4 = scheduler.P_sched
# sol4 = 2*sol4 - scheduler.get_P_demand_mean(t1)
sol4 = np.concatenate([[0], sol4])


fig, ax = plt.subplots()
ax.plot(np.arange(0,25+18), np.arange(0,25+18)*scheduler.get_P_demand_mean(t1)/1000, 
         '--', color='black', alpha=0.7)
ax.plot(np.arange(0,7), np.cumsum(sol1/1000)[:7], color='tab:blue')
ax.plot(np.arange(6,25), np.cumsum(sol1/1000)[6:], '--', color='tab:blue', alpha=0.75)
ax.plot(np.arange(6,13), np.cumsum(sol2/1000)[:7]+E1/1000, color='tab:blue')
ax.plot(np.arange(12,31), np.cumsum(sol2/1000)[6:]+E1/1000, '--', color='tab:blue', alpha=0.75)
ax.plot(np.arange(12,19), np.cumsum(sol3/1000)[:7]+E2/1000, color='tab:blue')
ax.plot(np.arange(18,37), np.cumsum(sol3/1000)[6:]+E2/1000, '--', color='tab:blue', alpha=0.75)
ax.plot(np.arange(18,25), np.cumsum(sol4/1000)[:7]+E3/1000, color='tab:blue')
ax.plot(np.arange(24,43), np.cumsum(sol4/1000)[6:]+E3/1000, '--', color='tab:blue', alpha=0.75)
ax.set_xlabel('Time (h)')
ax.set_ylabel('Scheduled energy (MWh)')
ax.grid()
cm = 1/2.54
fig.set_size_inches(15*cm, 7*cm)
plt.savefig('../Abbildungen/scheduling_acc.pgf', bbox_inches='tight')

P_out = ca.vertcat(P_out1[:6], P_out2[:6], P_out3[:6], P_out4[:7])
P_gtg = ca.vertcat(P_gtg1[:6], P_gtg2[:6], P_gtg3[:6], P_gtg4[:7])
P_wtg = ca.vertcat(P_wtg1[:6], P_wtg2[:6], P_wtg3[:6], P_wtg4[:7])
P_bat = ca.vertcat(P_bat1[:6], P_bat2[:6], P_bat3[:6], P_bat4[:7])
fig, ax = plt.subplots()
plt_gtg, = ax.plot(np.arange(25), P_gtg/1000)
plt_bat, = ax.plot(np.arange(25), P_bat/1000)
plt_wtg, = ax.plot(np.arange(25), P_wtg/1000)
plt_tot, = ax.plot(np.arange(25), P_out/1000)
ax.set_xlabel('Time (h)')
ax.set_ylabel('Power output (MW)')
ax.set_xlim((0,24))
ax.grid()
fig.legend(handles=[plt_wtg, plt_gtg, plt_bat, plt_tot], 
           labels=['Wind turbine', 'Gas turbine', 'Battery', 'Total power output'],
           loc='upper center', ncol=4, bbox_to_anchor=(0.5,1.05))
fig.set_size_inches(15*cm, 7*cm)
plt.savefig('../Abbildungen/scheduling_power.pgf', bbox_inches='tight')
pass