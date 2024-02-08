import casadi as ca

from modules.models import OHPS
from .nominal_mpc import NominalMPC
from scipy.stats import norm


class ChanceConstrainedMPC(NominalMPC):
    """
    Keep all functions from nominal mpc but use chance constraints for power constraint
    It is also possible to use the same class and just change the wind predictions
    """
    def __init__(self, ohps: OHPS, opt):
        super().__init__(ohps, opt)
        epsilon = opt['epsilon_chance_constraint']
        self.back_off_factor = norm.ppf(1-epsilon)  # Phi^-1(1-epsilon)

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
        # x_lb_sc - x - s_x <=0, x - s_x - x_ub_sc <= 0
        s_x = ca.MX.sym('s_xl', self.horizon) # for x + s_x >= x_lb_sc, x - s_x <= x_ub_sc
        s_x_lb = ca.DM.zeros(self.horizon)
        s_x_ub = ca.inf*ca.DM.ones(self.horizon)
        v = ca.vertcat(U_mat.reshape((-1,1)), X_mat.reshape((-1,1)), s_P, s_x)   # Vector for optimization problem
        
        self.get_u_from_v_fun = ca.Function('get_u_from_v', [v], [U_mat], ['v'], ['U_mat'])
        self.get_x_from_v_fun = ca.Function('get_x_from_v', [v], [X_mat], ['v'], ['X_mat'])
        self.get_s_from_v_fun = ca.Function('get_s_P_from_v', [v], [s_P], ['v'], ['s_P'])
        self.get_s_x_from_v_fun = ca.Function('get_s_x_from_v', [v], [s_x], ['v'], ['s_xl'])
        self.get_v_fun = ca.Function('get_v', [U_mat, X_mat, s_P, s_x], [v], 
                                     ['U_mat', 'X_mat', 's_P', 's_x'], ['v'])

        v_lb = self.get_v_fun(U_lb, X_lb, s_P_lb, s_x_lb)
        v_ub = self.get_v_fun(U_ub, X_ub, s_P_ub, s_x_ub)
        return v, v_lb, v_ub
    
    def get_optimization_parameters(self):
        """
        get symbolic variables for the parameters of the optimizaton problem:
        initial state, predicted wind speed and power demand
        """
        x0 = ca.MX.sym('x0', self.nx)
        P_gtg_0 = ca.MX.sym('P_gtg_0') # to penalize large changes of the gas turbine power output
        wind_speeds = ca.MX.sym('wind_speed', self.horizon)
        wind_speed_std = ca.MX.sym('std', self.horizon)
        P_demand = ca.MX.sym('P_demand', self.horizon)
        p = ca.vertcat(x0, P_gtg_0, wind_speeds, wind_speed_std, P_demand)
        self.get_x0_fun = ca.Function('get_x0', [p], [x0], ['p'], ['x0'])
        self.get_P_gtg_0_fun = ca.Function('get_P_gtg_0', [p], [P_gtg_0], ['p'], ['P_gtg_0'])
        self.get_wind_speed_fun = ca.Function('get_wind_speed', [p], [wind_speeds], 
                                              ['p'], ['wind_speeds'])
        self.get_std_fun = ca.Function('get_std', [p], [wind_speed_std], ['p'], ['std'])
        self.get_P_demand_fun = ca.Function('get_P_demand', [p], [P_demand], ['p'], ['P_demand'])
        self.get_p_fun = ca.Function('get_p', [x0, P_gtg_0, wind_speeds, wind_speed_std, P_demand], [p], 
                                     ['x0', 'P_gtg_last', 'wind_speeds', 'standard deviation', 'P_demand'], ['p'])
        return p
    
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
        wind_speeds_std = self.get_std_fun(p)
        # Chance constraints: assume worst case wind speed
        wind_speeds_backoff = wind_speeds - self.back_off_factor*wind_speeds_std
        # check for possibility of wind speeds above cut-out
        wind_speeds_upper = wind_speeds + self.back_off_factor*wind_speeds_std
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
            # x_k+1 - f(x_k, u_k) = 0
            g_state = X_mat[i,:] - x_next
            g_state_lb = 0
            g_state_ub = 0
            g.append(g_state)
            g_lb.append(g_state_lb)
            g_ub.append(g_state_ub)
            # Power constraint
            # P_demand - (P_gtg + P_bat + P_wtg + s_P) <= 0
            P_gtg = self.ohps.get_P_gtg(x_i, u_i, wind_speeds[i])
            P_bat = self.ohps.get_P_bat(x_i, u_i, wind_speeds[i])
            P_wtg_backoff = self.ohps.get_P_wtg(x_i, u_i, wind_speeds_backoff[i])
            # assume there are no scenarios where the wind speed can be above cut-out or below rated speed
            # If probability of wind speed above cut-out is greater than epsilon, assume wind power of 0
            P_wtg_chance_constraint = ca.if_else(
                wind_speeds_upper[i]>self.ohps.wind_turbine.cut_out_speed,
                0, P_wtg_backoff)
            g_demand = P_demand[i] - P_gtg - P_bat - P_wtg_chance_constraint - s_P[i]
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
        
        g = ca.vertcat(*g)
        g_lb = ca.vertcat(*g_lb)
        g_ub = ca.vertcat(*g_ub)
        self.g_fun = ca.Function('constraints', [v, p], [g], ['v', 'p'], ['g']) 
        return g, g_lb, g_ub
