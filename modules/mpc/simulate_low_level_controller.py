import datetime

import casadi as ca
import numpy as np

from modules.models import OHPS
from modules.gp import DataHandler

class LowLevelController:
    def __init__(self, ohps: OHPS, dh: DataHandler, mpc_opt) -> None:
        self.ohps = ohps
        self.dh = dh
        self.n_intervals = 1
        self.ohps.battery.setup_integrator(dt=60*mpc_opt['dt']/self.n_intervals)
        self.get_optimization_problem()
    def get_optimization_problem(self):
        # Assume constant P_demand and P_gtg, mimimize tracking error
        w = ca.MX.sym('w', self.n_intervals) # Parameter
        i = ca.MX.sym('i', self.n_intervals) # Optimization variable
        P_gtg = ca.MX.sym('P_gtg', self.n_intervals) # Optimization variable
        x = ca.MX.sym('x', self.n_intervals+1) # Optimization variable, including x0
        x0 = ca.MX.sym('x0') # Parameter

        P_demand = ca.MX.sym('P_demand') # Parameter
        P_gtg_last = ca.MX.sym('P_gtg_last')  # Parameter
        P_gtg_init = ca.MX.sym('P_gtg_init')
        i_init = ca.MX.sym('i_init')
        s_P = ca.MX.sym('s_P') # Parameter
        P_wtg = [self.ohps.get_P_wtg(x[j], ca.vertcat(P_gtg, i[j]), w[j]) for j in range(self.n_intervals)]
        P_bat = [self.ohps.get_P_bat(x[j], ca.vertcat(P_gtg, i[j]), w[j]) for j in range(self.n_intervals)]

        P_wtg = ca.vertcat(*P_wtg)
        P_bat = ca.vertcat(*P_bat)

        f = 100*ca.sumsqr(P_demand - P_gtg - P_wtg - P_bat - s_P) + 0.01*ca.sumsqr(P_gtg-P_gtg_init) #+ 0.01*ca.sumsqr(i-i_init)
        v = ca.vertcat(x, i, P_gtg)
        v_lb = ca.vertcat(self.ohps.battery.bounds['lbx']*ca.DM.ones(self.n_intervals+1),
                          self.ohps.battery.bounds['lbu']*ca.DM.ones(self.n_intervals),
                          self.ohps.gtg.bounds['lbu']*ca.DM.ones(self.n_intervals))
        v_ub = ca.vertcat(self.ohps.battery.bounds['ubx']*ca.DM.ones(self.n_intervals+1),
                          self.ohps.battery.bounds['ubu']*ca.DM.ones(self.n_intervals),
                          self.ohps.gtg.bounds['ubu']*ca.DM.ones(self.n_intervals))
        p = ca.vertcat(w, x0, P_demand, P_gtg_init, P_gtg_last, s_P, i_init)
        # ODE constraint
        x_next = [self.ohps.battery.get_next_state(x[j], i[j]) for j in range(self.n_intervals)]
        x_next = ca.vertcat(*x_next)
        g_ode = ca.vertcat(x_next-x[1:])
        g_x0 = x0 - x[0]
        g = ca.vertcat(g_ode, g_x0)
        g_lb = ca.DM.zeros(self.n_intervals+1)
        g_ub = ca.DM.zeros(self.n_intervals+1)
        g_P = 16000 - (P_wtg + P_bat + P_gtg)
        g_P_lb = -ca.inf*ca.DM.ones(self.n_intervals)
        g_P_ub = ca.DM.zeros(self.n_intervals)
        g = ca.vertcat(g, g_P)
        g_lb = ca.vertcat(g_lb, g_P_lb)
        g_ub = ca.vertcat(g_ub, g_P_ub)
        i_init = ca.MX.sym('i_init')
        x_init = [x0]
        for j in range(self.n_intervals):
            x_init.append(self.ohps.battery.get_next_state(x_init[j], i_init))
        v_init = ca.vertcat(*x_init, i_init*ca.DM.ones(self.n_intervals), P_gtg_init*ca.DM.ones(self.n_intervals))
        nlp = {
            'x': v,
            'f': f,
            'g': g,
            'p': p
        }

        ipopt_opt = {
        'print_level': 5,
        'tol': 1e-10,
        'compl_inf_tol': 1e-6
        }
        nlp_opt = {'ipopt': ipopt_opt}
        self._solver = ca.nlpsol('llc', 'ipopt', nlp, nlp_opt)
        self.solver = ca.Function('llc', [i_init, p], 
                                  [self._solver(x0=v_init, p=p, lbx=v_lb, ubx=v_ub, lbg=g_lb, ubg=g_ub)['x'], 
                                   self._solver(x0=v_init, p=p, lbx=v_lb, ubx=v_ub, lbg=g_lb, ubg=g_ub)['f'],
                                   self._solver(x0=v_init, p=p, lbx=v_lb, ubx=v_ub, lbg=g_lb, ubg=g_ub)['g']])
        self.v_init_fun = ca.Function('v_init', [p, i_init], [v_init])
        self.v_lb = v_lb
        self.v_ub = v_ub
        self.g_lb = g_lb
        self.g_ub = g_ub
        # lower GTG output if battery is full
    def simulate(self, t, x_k, u_k, s_P_k, P_demand, P_gtg_last=25600):
        P_gtg = u_k[0]
        i_init = u_k[1]
        w = self.dh.get_measurement(t)
        p = ca.vertcat(w, x_k, P_demand, P_gtg, P_gtg_last, s_P_k, i_init)
        v_init = self.v_init_fun(p, i_init)
        sol = self._solver(x0=v_init, p=p, lbx=self.v_lb, ubx=self.v_ub, lbg=self.g_lb, ubg=self.g_ub)
        v_opt = sol['x']
        f_opt = sol['f']
        g_opt = sol['g']
        # v_opt, f_opt, g_opt = self.solver(i_init, p)
        P_bat = self.ohps.get_P_bat(x_k, v_opt[self.n_intervals+1],0)
        P_wtg = self.ohps.get_P_wtg(x_k, u_k, w)
        P_gtg_opt = v_opt[2*self.n_intervals+1:]
        if P_demand - P_gtg_opt - P_wtg - P_bat - s_P_k < -10:
            print('Was ist hier los??????')
        x_opt = v_opt[:self.n_intervals+1]
        i_opt = v_opt[self.n_intervals+1:2*self.n_intervals+1]
        
        return i_opt[0], P_gtg_opt[0], x_opt[-1]