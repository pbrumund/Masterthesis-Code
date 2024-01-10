import casadi as ca

from .mpc_class import MPC
from ..models import OHPS

class NominalMPC(MPC):
    def __init__(self, ohps: OHPS, opt):
        self.ohps = ohps
        self.ohps.setup_integrator(dt=60*opt['dt'])
        self.horizon = opt['N']
        self.sampling_frequency = opt['dt']
        self.nx = self.ohps.nx
        self.nu = self.ohps.nu
        self.param = opt['param']

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
        s_P = ca.MX.sym('s_P', self.horizon)    # for soft power constraints
        s_P_lb = ca.DM.zeros(self.horizon)
        s_P_ub = ca.inf*ca.DM.ones(self.horizon)
        
        v = ca.vertcat(U_mat.reshape((-1,1)), X_mat.reshape((-1,1)), s_P)   # Vector for optimization problem
        
        self.get_u_from_v_fun = ca.Function('get_u_from_v', [v], [U_mat], ['v'], ['U_mat'])
        self.get_x_from_v_fun = ca.Function('get_x_from_v', [v], [X_mat], ['v'], ['X_mat'])
        self.get_s_from_v_fun = ca.Function('get_s_from_v', [v], [s_P], ['v'], ['s_P'])
        self.get_v_fun = ca.Function('get_v', [U_mat, X_mat, s_P], [v], 
                                     ['U_mat', 'X_mat', 's_P'], ['v'])

        v_lb = self.get_v_fun(U_lb, X_lb, s_P_lb)
        v_ub = self.get_v_fun(U_ub, X_ub, s_P_ub)
        return v, v_lb, v_ub

    def get_optimization_parameters(self):
        """
        get symbolic variables for the parameters of the optimizaton problem:
        initial state, predicted wind speed and power demand
        """
        x0 = ca.MX.sym('x0', self.nx)
        P_gtg_0 = ca.MX.sym('P_gtg_0') # to penalize large changes of the gas turbine power output
        wind_speeds = ca.MX.sym('wind_speed', self.horizon)
        P_demand = ca.MX.sym('P_demand', self.horizon)
        p = ca.vertcat(x0, P_gtg_0, wind_speeds, P_demand)
        self.get_x0_fun = ca.Function('get_x0', [p], [x0], ['p'], ['x0'])
        self.get_P_gtg_0_fun = ca.Function('get_P_gtg_0', [p], [P_gtg_0], ['p'], ['P_gtg_0'])
        self.get_wind_speed_fun = ca.Function('get_wind_speed', [p], [wind_speeds], 
                                              ['p'], ['wind_speeds'])
        self.get_P_demand_fun = ca.Function('get_P_demand', [p], [P_demand], ['p'], ['P_demand'])
        self.get_p_fun = ca.Function('get_p', [x0, P_gtg_0, wind_speeds, P_demand], [p], 
                                     ['x0', 'P_gtg_last', 'wind_speeds', 'P_demand'], ['p'])
        return p

    def stage_cost(self, state, input, s_P, P_gtg_last=None):
        # P_gtg_last = None
        alpha_1 = self.param['alpha_1']
        alpha_2 = self.param['alpha_2']
        x_gtg = self.ohps.get_x_gtg(state)
        u_gtg = self.ohps.get_u_gtg(input)
        P_gtg = self.ohps.gtg.get_power_output(x_gtg, u_gtg, None)
        load = P_gtg/self.param['P_gtg_max']
        eta_gtg = ca.if_else(load>0.01, alpha_1*load**2 + alpha_2*load, 0)  # TODO: add efficiency to model 
        eta_max = self.param['eta_gtg_max']
        J_gtg = self.param['k_gtg_eta']*(eta_gtg-eta_max)**2 + self.param['k_gtg_P']*load
        x_bat = self.ohps.get_x_bat(state)
        u_bat = self.ohps.get_u_bat(input)
        SOC = self.ohps.battery.get_SOC_fun(x_bat, u_bat)
        J_bat = -self.param['k_bat']*SOC
        J_u = input.T@self.param['R_input']@input
        J_s_P = s_P.T*self.param['r_s_P']*s_P/(8000+4500)
        # if P_gtg_last is not None:
        #     J_gtg += self.param['k_gtg_dP']*ca.fabs(P_gtg-P_gtg_last)/self.param['P_gtg_max']
        return J_gtg+J_bat+J_u+J_s_P
    
    def cost_function(self, v, p):
        """Return the cost function depending on the optimization variable and the parameters"""
        # get states and inputs from optimization variables
        X_mat = self.get_x_from_v_fun(v)
        U_mat = self.get_u_from_v_fun(v)
        s_P = self.get_s_from_v_fun(v)
        # get initial state from parameters
        x0 = self.get_x0_fun(p)
        P_gtg_last = self.get_P_gtg_0_fun(p)
        J = 0
        for i in range(self.horizon):
            if i == 0:
                x_i = x0
            else:
                x_i = X_mat[i-1,:]
            u_i = U_mat[i,:].T
            s_P_i = s_P[i,:]
            J += self.stage_cost(x_i, u_i, s_P_i, P_gtg_last)
            P_gtg = u_i[0]
            J += self.param['k_gtg_dP']*ca.fabs(P_gtg-P_gtg_last)/self.param['P_gtg_max']
            P_gtg_last = P_gtg#self.ohps.get_P_gtg(x_i, u_i, 0)
            # TODO: maybe integrate cost function over interval instead of adding terms at discretization points
        # If needed, add terminal cost term here
        return J
    def get_constraints(self, v, p):
        """
        Return additional constraints depending on the optimization variables and parameters
        System dynamics, power constraint
        """
        # get states and inputs from optimization variables
        X_mat = self.get_x_from_v_fun(v)
        U_mat = self.get_u_from_v_fun(v)
        s_P = self.get_s_from_v_fun(v)
        # get initial state, wind speed and power demand from parameters
        x0 = self.get_x0_fun(p)
        wind_speeds = self.get_wind_speed_fun(p)
        P_demand = self.get_P_demand_fun(p)
        
        g = []
        g_lb = []
        g_ub = []
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
            # TODO: possibly test implicit RK method for discretisation of dynamic model
            g_state = X_mat[i,:] - x_next
            g_state_lb = ca.DM.zeros(self.nx)
            g_state_ub = ca.DM.zeros(self.nx)
            g.append(g_state)
            g_lb.append(g_state_lb)
            g_ub.append(g_state_ub)
            # Power constraint
            # P_demand - (P_gtg + P_bat + P_wtg + s_P) <= 0
            P_gtg = self.ohps.get_P_gtg(x_i, u_i, wind_speeds[i])
            P_bat = self.ohps.get_P_bat(x_i, u_i, wind_speeds[i])
            P_wtg = self.ohps.get_P_wtg(x_i, u_i, wind_speeds[i])
            g_demand = P_demand[i] - P_gtg - P_bat - P_wtg - s_P[i]
            g_demand_lb = -ca.inf
            g_demand_ub = 0
            g.append(g_demand)
            g_lb.append(g_demand_lb)
            g_ub.append(g_demand_ub)

        g = ca.vertcat(*g)
        g_lb = ca.vertcat(*g_lb)
        g_ub = ca.vertcat(*g_ub)
        self.g_fun = ca.Function('constraints', [v, p], [g], ['v', 'p'], ['g']) 
        return g, g_lb, g_ub
    

    def get_optimization_problem(self):
        v, v_lb, v_ub = self.get_optimization_variables()
        p = self.get_optimization_parameters()
        J = self.cost_function(v, p)
        g, g_lb, g_ub = self.get_constraints(v, p)
        nlp = {
            'x': v,
            'f': J,
            'g': g,
            'p': p
        }
        nlp_opt = self.get_nlp_opt()
        self._solver = ca.nlpsol('mpc_solver', 'ipopt', nlp, nlp_opt)
        v_init = ca.MX.sym('v_init', v.shape)

        self.v = v; self.v_lb = v_lb; self.v_ub  = v_ub
        self.g = g; self.g_lb = g_lb; self.g_ub = g_ub
        self.solver = ca.Function(
            'mpc', [v_init, p], 
            [self._solver(x0=v_init, p=p, lbx=v_lb, ubx=v_ub, lbg=g_lb, ubg=g_ub)['x']],
            ['v_init', 'p'], ['v_opt'])
    
    def get_initial_guess(self, p, v_last = None):
        if v_last is not None:
            U_last = self.get_u_from_v_fun(v_last)
            X_last = self.get_x_from_v_fun(v_last)
            s_P_last = self.get_s_from_v_fun(v_last)
            # shift states and inputs, repeat last value
            U_init = ca.vertcat(U_last[1:,:], U_last[-1,:])
            X_init = ca.vertcat(X_last[1:,:], X_last[-1,:])
            P_demand = self.get_P_demand_fun(p)
            s_P_init = ca.vertcat(s_P_last[1:,:], P_demand[-1])
            return self.get_v_fun(U_init, X_init, s_P_init)
        # No last solution provided, guess zeros for inputs and initial state for X
        P_gtg_init = 0.8*self.ohps.gtg.bounds['ubu']*ca.DM.ones(self.horizon)
        I_bat_init = ca.DM.zeros(self.horizon)
        U_init = ca.horzcat(P_gtg_init, I_bat_init)
        # U_init = ca.DM.zeros(self.horizon, self.nu)
        X_init = ca.DM.ones(self.horizon)@self.ohps.x0.T
        P_demand = self.get_P_demand_fun(p)
        # s_P_init = P_demand - self.ohps.gtg.bounds['ubu']*ca.DM.ones(self.horizon)
        # s_P_init = 1/3*self.get_P_demand_fun(p)
        s_P_init = ca.DM.zeros(self.horizon)
        return self.get_v_fun(U_init, X_init, s_P_init)

