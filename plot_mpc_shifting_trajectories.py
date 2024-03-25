import datetime

import casadi as ca
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec

from modules.mpc import NominalMPCLoadShifting, DayAheadScheduler, get_mpc_opt
from modules.models import OHPS
from modules.gp import DataHandler, get_gp_opt
from modules.plotting import TimeseriesPlot
from modules.mpc_scoring import DataSaving

matplotlib.use('pgf')
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

ohps = OHPS(N_p=8000)

mpc_opt = get_mpc_opt(N=30)
mpc_opt['param']['k_dP'] = 10
mpc_opt['param']['r_s_E'] = 10
mpc_opt['param']['k_bat'] = 0
mpc_opt['use_path_constraints_energy'] = True
mpc_opt['use_soft_constraints_state'] = False
nominal_mpc = NominalMPCLoadShifting(ohps, mpc_opt)

nominal_mpc.get_optimization_problem()

gp_opt = get_gp_opt(dt_pred = mpc_opt['dt'])
# gp = WindPredictionGP(gp_opt)



dt = datetime.timedelta(minutes=mpc_opt['dt'])
data_handler = DataHandler(datetime.datetime(2020,1,1), datetime.datetime(2022,12,31), gp_opt)
x_k = ohps.x0
P_gtg_last = ohps.gtg.bounds['ubu']
P_out_last = 45000
v_last = None

dE = 0
E_tot = 0
E_sched = 0
dE_sched = 0
E_target_lt = np.array([])
scheduler = DayAheadScheduler(ohps, data_handler, mpc_opt)


t = datetime.datetime(2022,9,12,18)
t_start = t
times = [t+i*datetime.timedelta(minutes=10) for i in range(30)]
# get parameters: predicted wind speed, power demand, initial state
wind_speeds = [data_handler.get_measurement(t, i) for i in range(nominal_mpc.horizon)] # perfect forecast
wind_speeds = ca.vertcat(*wind_speeds)
P_demand = scheduler.get_P_demand(t, x_k, dE)
E_sched += P_demand[0]
E_target = ca.sum1(P_demand) + dE_sched # total scheduled demand plus compensation for previously not satisfied demand
# P_demand = 8000*ca.DM.ones(nominal_mpc.horizon)
p = nominal_mpc.get_p_fun(x_k, P_gtg_last, P_out_last, wind_speeds, 16000, P_demand, 0, 10000)

# get initial guess
v_init = nominal_mpc.get_initial_guess(p, v_last)

# solve optimization problem
v_opt = nominal_mpc.solver(v_init, p)

E_tot_traj = ca.vertcat(0, nominal_mpc.E_tot_fun(v_opt, p))

fig, (ax1, ax2) = plt.subplots(2, sharex=True, layout='constrained')
ax2.plot(times+[times[-1]+datetime.timedelta(minutes=10)], E_tot_traj/1000-ca.vertcat(0,ca.cumsum(P_demand/6000))-dE_sched/6000)
plt_demand, = ax1.plot(times, P_demand/1000, '--', color='black')
plt_total, = ax1.plot(times, nominal_mpc.P_out_fun(v_opt, p)/1000)
ax1.fill_between(times+[times[-1]+datetime.timedelta(minutes=15)], 15, 16, color='red', alpha=0.75)
ax2.fill_between(times+[times[-1]+datetime.timedelta(minutes=15)], -10, -11, color='red', alpha=0.5)
ax2.fill_between(times+[times[-1]+datetime.timedelta(minutes=15)], 10, 11, color='red', alpha=0.5)
ax2.fill_between([times[-1]+datetime.timedelta(minutes=10), times[-1]+datetime.timedelta(minutes=15)], 0, -10, color='red', alpha=0.5)
ax2.set_xlabel('Time')
ax1.set_ylabel('Power output (MW)')
ax2.set_ylabel('Shifted demand (MWh)')
ax1.grid()
ax2.grid()
ax2.set_xlim([times[0], times[-1]+datetime.timedelta(minutes=15)])
ax2.set_ylim([-11,11])
ax1.set_ylim([15,50])
ax2.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))
cm = 1/2.54
fig.set_size_inches(12*cm,10*cm)

plt.savefig('../Abbildungen/mpc_shifting_example.pgf', bbox_inches='tight')


t = datetime.datetime(2022,3,17,2)
t_start = t
times = [t+i*datetime.timedelta(minutes=10) for i in range(30)]
# get parameters: predicted wind speed, power demand, initial state
wind_speeds = [data_handler.get_measurement(t, i) for i in range(nominal_mpc.horizon)] # perfect forecast
wind_speeds = ca.vertcat(*wind_speeds)
P_demand = scheduler.get_P_demand(t, x_k, dE)
E_sched += P_demand[0]
E_target = ca.sum1(P_demand) + dE_sched # total scheduled demand plus compensation for previously not satisfied demand
# P_demand = 8000*ca.DM.ones(nominal_mpc.horizon)
p = nominal_mpc.get_p_fun(x_k, P_gtg_last, P_out_last, wind_speeds, 16000, P_demand, 0, 10000)

# get initial guess
v_init = nominal_mpc.get_initial_guess(p, v_last)

# solve optimization problem
v_opt = nominal_mpc.solver(v_init, p)

E_tot_traj = ca.vertcat(0, nominal_mpc.E_tot_fun(v_opt, p))

fig = plt.figure(figsize=(8, 6)) 
gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1]) 
ax2 = plt.subplot(gs[1])
ax1 = plt.subplot(gs[0], sharex=ax2)

# fig, (ax1, ax2) = plt.subplots(2, sharex=True, layout='constrained', height_ratios=[2,1])
ax2.plot(times+[times[-1]+datetime.timedelta(minutes=10)], E_tot_traj/1000-ca.vertcat(0,ca.cumsum(P_demand/6000))-dE_sched/6000)
plt_demand, = ax1.plot(times, P_demand/1000, '--', color='black')
plt_total, = ax1.plot(times, nominal_mpc.P_out_fun(v_opt, p)/1000)
u_opt = nominal_mpc.get_u_from_v_fun(v_opt)
x_opt = nominal_mpc.get_x_from_v_fun(v_opt)
u_k = u_opt[0,:]
w_k = data_handler.get_measurement(t)
P_gtg = ohps.get_P_gtg(x_k, u_k, w_k)
P_bat = ohps.get_P_bat(x_k, u_k, w_k)
P_wtg = ohps.get_P_wtg(x_k, u_k, w_k)

plt_wtg, = ax1.plot(times, np.array([ohps.get_P_wtg(0,0,wind_speeds[i])/1000 for i in range(30)]).reshape(-1))
plt_gtg, = ax1.plot(times, np.array([ohps.get_P_gtg(x_opt[i], u_opt[i,:], 0)/1000 for i in range(30)]).reshape(-1))
plt_bat, = ax1.plot(times, np.array([ohps.get_P_bat(x_opt[i], u_opt[i,:], 0)/1000 for i in range(30)]).reshape(-1))
P_total = P_gtg + P_bat + P_wtg
E_tot += P_total/6
dE_sched = E_sched - 6*E_tot
P_gtg_last = P_gtg
P_out_last = P_total
x_k = ohps.get_next_state(x_k, u_k)

# for k in range(3):
#     t = t+datetime.timedelta(minutes=10)
#     times = [t+i*datetime.timedelta(minutes=10) for i in range(30)]
#     # get parameters: predicted wind speed, power demand, initial state
#     wind_speeds = [data_handler.get_measurement(t, i) for i in range(nominal_mpc.horizon)] # perfect forecast
#     wind_speeds = ca.vertcat(*wind_speeds)
#     P_demand = scheduler.get_P_demand(t, x_k, dE)
#     E_sched += P_demand[0]
#     E_target = ca.sum1(P_demand) + dE_sched # total scheduled demand plus compensation for previously not satisfied demand
#     # P_demand = 8000*ca.DM.ones(nominal_mpc.horizon)
#     p = nominal_mpc.get_p_fun(x_k, P_gtg_last, P_out_last, wind_speeds, E_target, 16000, ca.cumsum(P_demand), 6*50000)

#     # get initial guess
#     v_init = nominal_mpc.get_initial_guess(p, v_last)

#     # solve optimization problem
#     v_opt = nominal_mpc.solver(v_init, p)
#     E_tot_traj = ca.vertcat(0, nominal_mpc.E_tot_fun(v_opt, p))
#     ax2.plot(times+[times[-1]+datetime.timedelta(minutes=10)], E_tot_traj/1000-ca.vertcat(0,ca.cumsum(P_demand/6000))-dE_sched/6000)
#     ax1.plot(times, nominal_mpc.P_out_fun(v_opt, p)/1000)
#     u_opt = nominal_mpc.get_u_from_v_fun(v_opt)
#     u_k = u_opt[0,:]
#     w_k = data_handler.get_measurement(t)
#     P_gtg = ohps.get_P_gtg(x_k, u_k, w_k)
#     P_bat = ohps.get_P_bat(x_k, u_k, w_k)
#     P_wtg = ohps.get_P_wtg(x_k, u_k, w_k)
#     P_total = P_gtg + P_bat + P_wtg
#     E_tot += P_total/6
#     dE_sched = E_sched - 6*E_tot
#     P_gtg_last = P_gtg
#     P_out_last = P_total
#     x_k = ohps.get_next_state(x_k, u_k)

#ax2.set_xlim([t_start, t_start+30*datetime.timedelta(minutes=10)])
ax2.set_xlabel('Time')
ax1.set_ylabel('Power output (MW)')
ax2.set_ylabel('Shifted demand (MWh)')
ax1.grid()
ax2.grid()
plt.setp(ax1.get_xticklabels(), visible=False)
fig.legend(handles=[plt_wtg, plt_gtg, plt_bat, plt_total, plt_demand], 
           labels=[r'$P_\mathrm{wtg}$', r'P$_\mathrm{gtg}$', r'P$_\mathrm{bat}$', r'P$_\mathrm{out}$', r'P$_\mathrm{demand}$'],
           loc='upper center', ncol=5, bbox_to_anchor=(0.5,1))
ax2.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))
cm = 1/2.54
fig.set_size_inches(12*cm,10*cm)
plt.show()
plt.savefig('../Abbildungen/mpc_shifting_example_1703.pgf', bbox_inches='tight')
pass
