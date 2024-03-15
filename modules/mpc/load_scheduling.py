# Schedule load for entire day ahead, using NWP to predict wind power
# TODO: Implement load shifting for closed-loop controller

import datetime

import casadi as ca
import numpy as np

from ..models import OHPS
from ..gp import DataHandler
from ..mpc import get_nlp_opt

class DayAheadScheduler:
    def __init__(self, ohps: OHPS, dh: DataHandler, mpc_opt) -> None:
        self.ohps = ohps
        self.dh = dh
        self.steps = 24
        self.t_last_sched = None
        self.mpc_opt = mpc_opt
        self.get_optimization_problem()
        self.mean_P_wtg = {
                1:22000,
                2:22000,
                3:20000,
                4:18000,
                5:16000,
                6:16000,
                7:13000,
                8:15000,
                9:19000,
                10:20000,
                11:22000,
                12:22000
            }
        t_start = mpc_opt['t_start_sim']
        t_end = mpc_opt['t_end_sim']
        dt = datetime.timedelta(hours=1)
        n_times = int((t_end-t_start)/dt)
        times = [t_start + i*dt for i in range(n_times)]
        self.times_E_target = [(t-t_start).total_seconds() for t in times]
        self.t_start = t_start
        self.E_target_mean = np.cumsum([self.get_P_demand_mean(t) for t in times])

    def get_optimization_problem(self):
        # Optimization variables
        steps = self.steps # 1 hour steps as NWP is in 1 h steps
        P_gtg = ca.MX.sym('P_gtg', steps)
        P_gtg_lb = ca.DM.zeros(steps)
        P_gtg_ub = ca.DM.ones(steps)*self.ohps.P_gtg_max
        I_bat = ca.MX.sym('I_bat', steps)
        I_bat_lb = ca.DM.ones(steps)*self.ohps.battery.lbu
        I_bat_ub = ca.DM.ones(steps)*self.ohps.battery.ubu
        X_bat = ca.MX.sym('X_bat', steps+1)
        X_bat_lb = ca.DM.ones(steps+1)*self.ohps.battery.lbx
        X_bat_ub = ca.DM.ones(steps+1)*self.ohps.battery.ubx
        # Might not start within bounds, so remove constraint for initial value
        X_bat_lb[0] = -ca.inf
        X_bat_ub[0] = ca.inf
        # Soft constraint on total produced energy
        s_E = ca.MX.sym('s_E')
        s_E_lb = 0
        s_E_ub = ca.inf
        self.v = ca.vertcat(P_gtg, I_bat, X_bat, s_E)
        self.v_lb = ca.vertcat(P_gtg_lb, I_bat_lb, X_bat_lb, s_E_lb)
        self.v_ub = ca.vertcat(P_gtg_ub, I_bat_ub, X_bat_ub, s_E_ub)
        
        # Parameters
        P_wtg = ca.MX.sym('P_wtg', steps)
        x0 = ca.MX.sym('x0') # Initial state
        P_min = ca.MX.sym('P_min') # Minimum power output at each step
        # Hard constraint possible if P_min is below maximum GTG output
        E_target = ca.MX.sym('P_target') # Desired total energy output, 
        # might change depending on whether the last target was met
        self.p = ca.vertcat(P_wtg, x0, P_min, E_target)

        # Variables for cost function, constraints
        load = P_gtg/self.ohps.P_gtg_max
        eta_gtg = ca.vertcat(*[self.ohps.gtg.eta_fun(load[i]) for i in range(steps)])
        eta_gtg = ca.if_else(load>1e-3, eta_gtg, self.ohps.gtg.eta_fun(1))
        P_bat = ca.vertcat(*[self.ohps.get_P_bat(X_bat[i], I_bat[i], 0) for i in range(steps)])
        P_out = P_gtg + P_wtg + P_bat
        E_tot = ca.sum1(P_out)
        SOC_fin = self.ohps.get_SOC_bat(X_bat[-1], 0, 0)
        P_in_gtg = ca.if_else(load>1e-4, P_gtg/eta_gtg, 0)
        # constraints
        # Equality constraints: battery dynamics
        X_next = X_bat[:-1] + I_bat/self.ohps.battery.N_p
        g_eq = ca.vertcat(X_bat[0]-x0, X_bat[1:] - X_next)
        g_eq_lb = ca.DM.zeros(steps+1)
        g_eq_ub = ca.DM.zeros(steps+1)
        # Inequality constraints: Minimum power ouput and total generated energy
        # P_out >= P_min
        g_ineq_P = P_min - P_out
        g_ineq_P_lb = -ca.inf*ca.DM.ones(steps)
        g_ineq_P_ub = ca.DM.zeros(steps)
        # E_tot + s_E >= E_target
        g_ineq_E = E_target - E_tot - s_E
        g_ineq_E_lb = -ca.inf
        g_ineq_E_ub = 0
        # All constraints
        self.g = ca.vertcat(g_eq, g_ineq_P, g_ineq_E)
        self.g_lb = ca.vertcat(g_eq_lb, g_ineq_P_lb, g_ineq_E_lb)
        self.g_ub = ca.vertcat(g_eq_ub, g_ineq_P_ub, g_ineq_E_ub)

        # Cost function
        self.J = ca.sum1(P_in_gtg) + 0*ca.sumsqr(eta_gtg-self.ohps.gtg.eta_fun(1)) + 100*s_E + 0.001*ca.sumsqr(I_bat) - SOC_fin #+ #0.0001*ca.sumsqr((P_out[1:]-P_out[:-1]))

        # NLP
        self.nlp = {
            'x': self.v,
            'p': self.p,
            'f': self.J,
            'g': self.g
        }
    
        self.solver = ca.nlpsol('one_day_load_scheduler', 'ipopt', self.nlp)
        self.P_out_fun = ca.Function('get_P_sched', [self.v, self.p], [P_out])
        self.P_bat_fun = ca.Function('P_bat', [self.v], [P_bat])
        self.P_in_gtg_fun = ca.Function('P_in_gtg', [self.v], [P_in_gtg])

    def solve_problem(self, t, P_min, E_target, x0):
        times = [t + i*datetime.timedelta(hours=1) for i in range(self.steps)]
        v_wind_NWP = [self.dh.get_NWP(t) for t in times]
        P_wtg_NWP = ca.vertcat(*[self.ohps.get_P_wtg(0, 0, v) for v in v_wind_NWP])
        p = ca.vertcat(P_wtg_NWP, x0, P_min, E_target)
        v_init = ca.vertcat(0*self.ohps.P_gtg_max*ca.DM.ones(self.steps), ca.DM.zeros(self.steps), 
                            x0*ca.DM.ones(self.steps+1), E_target)
        sol = self.solver(
            x0=v_init, p=p, lbx=self.v_lb, ubx=self.v_ub, lbg=self.g_lb, ubg=self.g_ub)
        
        v_opt = sol['x']
        P_sched = self.P_out_fun(v_opt, p)
        return P_sched, v_opt, p
        
    def get_P_demand(self, t, x0, dE, P_min=16000, E_target=None):
        if self.t_last_sched is None or t.hour%6==0 and t.minute==0:
            # solve scheduling problem for next 24 hours
            times = [t + i*datetime.timedelta(hours=1) for i in range(self.steps)]
            v_wind_NWP = [self.dh.get_NWP(t) for t in times]
            P_wtg_NWP = ca.vertcat(*[self.ohps.get_P_wtg(0, 0, v) for v in v_wind_NWP])
            if E_target is None:
                E_target = self.get_E_target(t)
            E_target += dE
            p = ca.vertcat(P_wtg_NWP, x0, P_min, E_target)
            v_init = ca.vertcat(0*self.ohps.P_gtg_max*ca.DM.ones(self.steps), ca.DM.zeros(self.steps), 
                                x0*ca.DM.ones(self.steps+1), E_target)
            sol = self.solver(
                x0=v_init, p=p, lbx=self.v_lb, ubx=self.v_ub, lbg=self.g_lb, ubg=self.g_ub)
            
            v_opt = sol['x']
            self.P_sched = self.P_out_fun(v_opt, p)
            self.t_last_sched = t
            self.times_sched = [(t_i-t).total_seconds() for t_i in times]
        times_mpc = [t + i*datetime.timedelta(minutes=self.mpc_opt['dt']) 
                     for i in range(self.mpc_opt['N'])]
        dt_mpc = [(t_i-self.t_last_sched).total_seconds() for t_i in times_mpc]
        return ca.interp1d(self.times_sched, self.P_sched, dt_mpc)
    
    def get_P_demand_mean(self, t):
            month = t.month            
            return 0.75*self.ohps.P_gtg_max + self.mean_P_wtg[month]
    
    def get_E_target(self, t):
        times = [t+i*datetime.timedelta(hours=1) for i in range(24)]
        return sum([self.get_P_demand_mean(t_i) for t_i in times])
    
    def get_E_target_lt(self, t):
        dt = (t+datetime.timedelta(minutes=self.mpc_opt['dt'])-self.t_start).total_seconds()
        return np.interp(dt, self.times_E_target, self.E_target_mean)