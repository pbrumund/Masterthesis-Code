import datetime

import casadi as ca

from modules.mpc import NominalMPC, get_mpc_opt
from modules.models import OHPS
from modules.gp import PriorOnTimeseriesGP as WindPredictionGP
from modules.gp import get_gp_opt

mpc_opt = get_mpc_opt()
ohps = OHPS()
nominal_mpc = NominalMPC(ohps, mpc_opt)
nominal_mpc.get_optimization_problem()

gp_opt = get_gp_opt(dt_pred = mpc_opt['dt'])
gp = WindPredictionGP(gp_opt)

t_start = datetime.datetime(2022, 1, 1)
t_end = datetime.datetime(2022,2,1)
dt = datetime.timedelta(minutes=mpc_opt['dt'])
n_times = int((t_end-t_start)/dt)
times = [t_start + i*dt for i in range(n_times)]

x_k = ohps.x0

for k, t in enumerate(times):
    # TODO: solve optimization problem, simulate system
    # TODO: for simulation: maybe use smaller time scale and vary wind speed for each subinterval 
    # as wind power is not simply a function of the mean wind speed, 
    # possibly account for this uncertainty in gp
    pass
pass