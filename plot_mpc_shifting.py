import datetime

import numpy as np
import matplotlib.pyplot as plt
import casadi as ca

from modules.gp import get_gp_opt
from modules.mpc import get_mpc_opt
from modules.mpc_scoring import DataSaving


mpc_opt = get_mpc_opt(N=30)
t_start = datetime.datetime(2022,1,1)
t_end = datetime.datetime(2022,12,31)
mpc_opt['t_start_sim'] = t_start
mpc_opt['t_end_sim'] = t_end
mpc_opt['param']['k_dP'] = 10
mpc_opt['param']['r_s_E'] = 100
mpc_opt['param']['k_bat'] = 0
mpc_opt['use_path_constraints_energy'] = True
mpc_opt['use_soft_constraints_state'] = False
gp_opt = get_gp_opt(dt_pred = mpc_opt['dt'])

dims = {'Power output': 4, 'Power demand': 1, 'SOC': 1, 'Inputs': 2}
data_saver = DataSaving('nominal_mpc_perfect_forecast_shifting', mpc_opt, gp_opt, dims)
data, times = data_saver.load_trajectories()
plt.figure()
plt.plot(data['Power output'][:,0])
plt.plot(data['Power output'][:,1])
plt.plot(data['Power output'][:,2])
plt.plot(data['Power output'][:,3])
P_traj = data['Power output']
P_demand = data['Power demand']
plt.figure()
plt.plot(ca.cumsum(P_traj[:,-1]/6000)-ca.cumsum(P_demand[:]/6000))
plt.show()