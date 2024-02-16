import casadi as ca

from .nominal_mpc import NominalMPC
from ..models import OHPS
from .get_mpc_opt import get_nlp_opt

class NominalMPCLoadShifting(NominalMPC):
    def get_optimization_variables(self):
        """
        Get optimization variables: Inputs, states, s_P for soft constraint on power
        Also returns bounds on optimization variables including state constraints
        """
        # U_mat = (u_0, u_1, ..., u_N-1)
        U_mat = ca.MX.sym('U', self.horizon, self.nu)   # system inputs
        U_lb = ca.DM.ones(self.horizon)@self.ohps.lbu.T
        U_ub = ca.DM.ones(self.horizon)@self.ohps.ubu.T
        # X_mat = (x_1, x_2, ..., x_N)
        X_mat = ca.MX.sym('X', self.horizon, self.nx)   # state trajectory for multiple shooting
        X_lb = ca.DM.ones(self.horizon)@self.ohps.lbx.T
        X_ub = ca.DM.ones(self.horizon)@self.ohps.ubx.T
        # Soft constraint on total energy
        s_E = ca.MX.sym('s_E')
        s_E_lb = 0
        s_E_ub = ca.inf
        s_x = ca.MX.sym('s_xl', self.horizon) # for x + s_x >= x_lb_sc, x - s_x <= x_ub_sc
        s_x_lb = ca.DM.zeros(self.horizon)
        s_x_ub = ca.inf*ca.DM.ones(self.horizon)
        v = ca.vertcat(U_mat.reshape((-1,1)), X_mat.reshape((-1,1)), s_E, s_x)
        
        self.get_u_from_v_fun = ca.Function('get_u_from_v', [v], [U_mat], ['v'], ['U_mat'])
        self.get_x_from_v_fun = ca.Function('get_x_from_v', [v], [X_mat], ['v'], ['X_mat'])
        self.get_s_from_v_fun = ca.Function('get_s_from_v', [v], [s_E], ['v'], ['s_P'])
        self.get_s_x_from_v_fun = ca.Function('get_s_x_from_v', [v], [s_x], ['v'], ['s_xl'])
        self.get_v_fun = ca.Function('get_v', [U_mat, X_mat, s_E, s_x], [v], 
                                     ['U_mat', 'X_mat', 's_P', 's_x'], ['v'])

        v_lb = self.get_v_fun(U_lb, X_lb, s_E_lb, s_x_lb)
        v_ub = self.get_v_fun(U_ub, X_ub, s_E_ub, s_x_ub)
        return v, v_lb, v_ub

    def get_optimization_parameters(self):
        """
        get symbolic variables for the parameters of the optimizaton problem:
        initial state, predicted wind speed and power demand
        """
        x0 = ca.MX.sym('x0', self.nx)
        P_gtg_0 = ca.MX.sym('P_gtg_0') # to penalize large changes of the gas turbine power output
        P_out_0 = ca.MX.sym('P_out_0') # to penalize large changes in total power output
        wind_speeds = ca.MX.sym('wind_speed', self.horizon)
        E_target = ca.MX.sym('E_target')
        P_min = ca.MX.sym('P_min')
        p = ca.vertcat(x0, P_gtg_0, P_out_0, wind_speeds, E_target, P_min)
        self.get_x0_fun = ca.Function('get_x0', [p], [x0], ['p'], ['x0'])
        self.get_P_gtg_0_fun = ca.Function('get_P_gtg_0', [p], [P_gtg_0], ['p'], ['P_gtg_0'])
        self.get_P_out_last_fun = ca.Function('get_P_out_0', [p], [P_out_0], ['p'], ['P_out_0'])
        self.get_wind_speed_fun = ca.Function('get_wind_speed', [p], [wind_speeds], 
                                              ['p'], ['wind_speeds'])
        self.get_E_target_fun = ca.Function('get_E_target', [p], [E_target], ['p'], ['P_demand'])
        self.get_P_min_fun = ca.Function('get_P_min', [p], [P_min], ['p'], ['P_min'])
        self.get_p_fun = ca.Function('get_p', [x0, P_gtg_0, P_out_0, wind_speeds, E_target, P_min], [p], 
                                     ['x0', 'P_gtg_last', 'P_out_last', 'wind_speeds', 'E_target', 'P_min'], ['p'])
        return p

    def stage_cost(self, state, input, P_gtg_last, P_out, P_out_last, s_x = None):
        # P_gtg_last = None
        x_gtg = self.ohps.get_x_gtg(state)
        u_gtg = self.ohps.get_u_gtg(input)
        P_gtg = self.ohps.gtg.get_power_output(x_gtg, u_gtg, None)
        load = P_gtg/self.ohps.P_gtg_max
        eta_gtg = self.ohps.gtg.eta_fun(load)
        eta_max = self.param['eta_gtg_max']
        J_gtg_P = self.param['k_gtg_P']*load
        # J_gtg_eta = self.param['k_gtg_eta']*(eta_gtg-eta_max)**2
        J_gtg_eta = ca.if_else(load<1e-4, 0, self.param['k_gtg_eta']*(eta_gtg-eta_max)**2)
        J_gtg_fuel = ca.if_else(load<1e-3, 0, self.param['k_gtg_fuel']*load/eta_gtg)
        J_gtg= J_gtg_P + J_gtg_eta + J_gtg_fuel
                
        x_bat = self.ohps.get_x_bat(state)
        u_bat = self.ohps.get_u_bat(input)
        SOC = self.ohps.battery.get_SOC_fun(x_bat, u_bat)
        J_bat = -self.param['k_bat']*SOC
        J_u = input.T@self.param['R_input']@input
        if self.opt['use_soft_constraints_state']:
            J_s_x = self.param['r_s_x']*s_x
        else:
            J_s_x = 0
        # J_gtg_dP = self.param['k_gtg_dP']*ca.log(100*ca.fabs(u_gtg)/self.ohps.gtg.bounds['ubu']+1)
        # J_gtg += J_gtg_dP
        if P_gtg_last is not None:
            J_gtg += self.param['k_gtg_dP']*((P_gtg-P_gtg_last)/self.ohps.P_gtg_max)**2
        if P_out_last is not None:
            J_dP = self.param['k_dP']*((P_out-P_out_last)
                /(self.ohps.P_gtg_max+self.ohps.P_wtg_max))**2
        # functions for individual terms for tuning parameters
        self.J_gtg_i = J_gtg
        self.J_gtg_P_i = J_gtg_P
        self.J_gtg_eta_i = J_gtg_eta
        self.J_gtg_dP_i = self.param['k_gtg_dP']*((P_gtg-P_gtg_last)/self.ohps.P_gtg_max)**2
        self.J_dP_i = J_dP
        self.J_bat_i = J_bat
        self.J_u_i = J_u
        self.J_s_x_i = J_s_x
        return J_gtg + J_bat + J_u + J_dP + J_s_x
    
    def cost_function(self, v, p):
        """Return the cost function depending on the optimization variable and the parameters"""
        # get states and inputs from optimization variables
        X_mat = self.get_x_from_v_fun(v)
        U_mat = self.get_u_from_v_fun(v)
        s_x = self.get_s_x_from_v_fun(v)
        # get initial state from parameters
        x0 = self.get_x0_fun(p)
        P_gtg_last = self.get_P_gtg_0_fun(p)
        P_out_last = self.get_P_out_last_fun(p)
        wind_speeds = self.get_wind_speed_fun(p)
        J = 0
        self.J_gtg = self.J_gtg_P = self.J_gtg_eta = self.J_gtg_dP = self.J_bat = self.J_u = self.J_s_P = self.J_s_x = 0
        for i in range(self.horizon):
            if i == 0:
                x_i = x0
            else:
                x_i = X_mat[i-1,:]
            u_i = U_mat[i,:].T
            s_x_i = s_x[i]
            P_gtg_i = self.ohps.get_P_gtg(x_i, u_i, wind_speeds[i])
            P_bat_i = self.ohps.get_P_bat(x_i, u_i, wind_speeds[i])
            P_wtg_i = self.ohps.get_P_wtg(x_i, u_i, wind_speeds[i])
            P_out_i = P_gtg_i + P_wtg_i + P_bat_i
            J += self.stage_cost(x_i, u_i, P_gtg_last, P_out_i, P_out_last, s_x_i)
            # Parameter tuning
            self.J_gtg += self.J_gtg_i
            self.J_gtg_P += self.J_gtg_P_i
            self.J_gtg_eta += self.J_gtg_eta_i
            self.J_gtg_dP += self.J_gtg_dP_i
            self.J_bat += self.J_bat_i
            self.J_u += self.J_u_i
            self.J_s_x += self.J_s_x_i
            # J += self.param['k_gtg_dP']*ca.log(100*ca.fabs(P_gtg-P_gtg_last)/self.param['P_gtg_max']+1)
            P_gtg_last = P_gtg_i
            P_out_last = P_out_i
            # TODO: maybe integrate cost function over interval instead of adding terms at discretization points
        # If needed, add terminal cost term here
        X_last = X_mat[-1,:]
        SOC_bat_last = self.ohps.get_SOC_bat(X_last, 0, 0)
        J_bat = -self.param['k_bat_final']*SOC_bat_last
        self.J_bat += J_bat
        J += J_bat
        s_E = self.get_s_from_v_fun(v)
        E_target = self.get_E_target_fun(p)
        J_dE = self.param['r_s_E']*s_E/E_target
        J += J_dE
        self.J_s_E = J_dE
        # Parameter tuning
        self.J_fun = ca.Function('J', [v, p], [J])
        self.J_gtg_fun = ca.Function('J_gtg', [v, p], [self.J_gtg])
        self.J_gtg_P_fun = ca.Function('J_gtg_P', [v, p], [self.J_gtg_P])
        self.J_gtg_eta_fun = ca.Function('J_gtg_eta', [v, p], [self.J_gtg_eta])
        self.J_gtg_dP_fun = ca.Function('J_gtg_dP', [v, p], [self.J_gtg_dP])
        self.J_bat_fun = ca.Function('J_bat', [v, p], [self.J_bat])
        self.J_u_fun = ca.Function('J_u', [v, p], [self.J_u])
        self.J_s_E_fun = ca.Function('J_s_P', [v, p], [self.J_s_E])
        self.J_s_x_fun = ca.Function('J_s_x', [v, p], [self.J_s_x])
        return J
    def get_constraints(self, v, p):
        """
        Return additional constraints depending on the optimization variables and parameters
        System dynamics, power constraint
        """
        # get states and inputs from optimization variables
        X_mat = self.get_x_from_v_fun(v)
        U_mat = self.get_u_from_v_fun(v)
        s_E = self.get_s_from_v_fun(v)
        # get initial state, wind speed and power demand from parameters
        x0 = self.get_x0_fun(p)
        wind_speeds = self.get_wind_speed_fun(p)
        P_min = self.get_P_min_fun(p)
        E_target = self.get_E_target_fun(p)
        
        g = []
        g_lb = []
        g_ub = []
        P_sum = 0
        for i in range(self.horizon):
            #X_mat[i,:] = x_i+1, U_mat[i,:] = u_i
            if i == 0:
                x_i = x0
            else:
                x_i = X_mat[i-1,:]
            u_i = U_mat[i,:]
            x_next = self.ohps.get_next_state(x_i, u_i)
            # System dynamics
            # x_k+1 - f(x_k, u_k) = 0^
            g_state = X_mat[i,:].T - x_next
            g_state_lb = ca.DM.zeros(self.nx)
            g_state_ub = ca.DM.zeros(self.nx)
            g.append(g_state)
            g_lb.append(g_state_lb)
            g_ub.append(g_state_ub)
            # Power constraint
            # P_min - (P_gtg + P_bat + P_wtg) <= 0
            P_gtg = self.ohps.get_P_gtg(x_i, u_i, wind_speeds[i])
            P_bat = self.ohps.get_P_bat(x_i, u_i, wind_speeds[i])
            P_wtg = self.ohps.get_P_wtg(x_i, u_i, wind_speeds[i])
            P_sum += P_gtg + P_wtg + P_bat
            g_demand = P_min - P_gtg - P_bat - P_wtg
            g_demand_lb = -ca.inf
            g_demand_ub = 0
            g.append(g_demand)
            g_lb.append(g_demand_lb)
            g_ub.append(g_demand_ub)
            if self.opt['use_soft_constraints_state']:
                s_x = self.get_s_x_from_v_fun(v)
                sc_backoff = 0.05
                x_lb_sc = self.ohps.lbx + sc_backoff
                x_ub_sc = self.ohps.ubx - sc_backoff
                g_x_lb = x_lb_sc - x_next - s_x[i]
                g_x_ub = x_next - x_ub_sc - s_x[i]
                g.append(g_x_lb)
                g_lb.append(-ca.inf)
                g_ub.append(0)
                g.append(g_x_ub)
                g_lb.append(-ca.inf)
                g_ub.append(0)
        # terminal constraint: E_target - sum(P_out) - s_E <= 0 
        g_E = E_target - P_sum - s_E
        g_E_lb = -ca.inf
        g_E_ub = 0
        g.append(g_E)
        g_lb.append(g_E_lb)
        g_ub.append(g_E_ub)
        g = ca.vertcat(*g)
        g_lb = ca.vertcat(*g_lb)
        g_ub = ca.vertcat(*g_ub)
        self.g_fun = ca.Function('constraints', [v, p], [g], ['v', 'p'], ['g']) 
        return g, g_lb, g_ub
    
    def get_initial_guess(self, p, v_last = None):
        if v_last is not None:
            U_last = self.get_u_from_v_fun(v_last)
            X_last = self.get_x_from_v_fun(v_last)
            s_E_init = self.get_s_from_v_fun(v_last) + self.ohps.P_gtg_max + self.ohps.P_wtg_max
            # shift states and inputs, repeat last value
            U_init = ca.vertcat(U_last[1:,:], U_last[-1,:])
            X_init = ca.vertcat(X_last[1:,:], X_last[-1,:])
            s_x_init = ca.DM.zeros(self.horizon)
            return self.get_v_fun(U_init, X_init, s_E_init, s_x_init)
        # No last solution provided, guess zeros for inputs and initial state for X
        # U_init = ca.horzcat(P_gtg_init, I_bat_init)
        U_init = ca.DM.zeros(self.horizon, self.nu)
        X_init = ca.DM.ones(self.horizon)@self.ohps.x0.T
        # s_P_init = P_demand - self.ohps.gtg.bounds['ubu']*ca.DM.ones(self.horizon)
        # s_P_init = 1/3*self.get_P_demand_fun(p)
        s_E_init = self.horizon*(self.ohps.P_gtg_max + self.ohps.P_wtg_max)
        s_x_init = ca.DM.zeros(self.horizon)
        return self.get_v_fun(U_init, X_init, s_E_init, s_x_init)

