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

        g = ca.vertcat(*g)
        g_lb = ca.vertcat(*g_lb)
        g_ub = ca.vertcat(*g_ub)
        self.g_fun = ca.Function('constraints', [v, p], [g], ['v', 'p'], ['g']) 
        return g, g_lb, g_ub
