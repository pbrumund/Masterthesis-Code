import casadi as ca
from scipy.stats import norm

from .mpc_class import MPC
from ..models import OHPS
from ..gp import PriorOnTimeseriesGP

class MultistageMPC(MPC):
    def __init__(self, ohps: OHPS, gp: PriorOnTimeseriesGP, opt) -> None:
        self.ohps = ohps
        self.gp = gp
        self.horizion = opt['N']
        self.sampling_frequency = opt['dt']
        self.opt = opt
    # TODO: maybe build scenario tree dynamically depending on the size of the uncertainty, 
    # i.e. branch if the uncertainty exceeds a certain threshold.
    # This should use the uncertainty of the wind turbine power output since it depends on the
    # gradient of the power curve, i.e. at high wind speeds above the rated wind speed and below 
    # the cut-off, the power output is constant so branching would be unnecessary
    # Maybe use different trees for scenario generation (branching every time) 
    # and control (branching only when necessary)
    # Maybe also use prediction intervals for chance constraints 
    # to use in combination with multi-stage MPC
    # TODO: maybe class for nodes that generates the necessary symbolics to build tree
    pass
    def generate_scenario(self, start_time, train=False):
        certainty_horizon = self.opt['certainty_horizon']   # minimum steps before branching
        robust_horizon = self.opt['robust_horizon'] # maximum step for branching
        max_scenarios = self.opt['max_scenarios']   # maximum number of branches
        branching_interval = self.opt['branching_interval'] # only branch every braching_interval steps
        std_list = (-1,0,1) # number of standard deviations for scenario generation
        dP_min = self.opt['dP_mean'] # minimum wind power difference at +-1 std for branching

        mean_init, var_init = self.gp.predict_trajectory(start_time, self.horizon, train)

        means_i = [mean_init]
        vars_i = [var_init]
        

class TreeNode:
    # TODO: Methods for branching and getting single successor, likelihood
    def __init__(self, ohps: OHPS, time_index, node_index, opt, predecessor=None):
        self.ohps = ohps
        self.time_index = time_index
        self.node_index = node_index
        self.predecessor = predecessor
        if self.predecessor is None:
            self.is_root_node = True
        self.opt = opt
        if self.opt['use_chance_constraints_multistage']:
            epsilon = self.opt['epsilon_chance_constraint']
            self.back_off_factor = norm.ppf(1-epsilon)
    def get_optimization_variables(self):
        self.u = ca.MX.sym(f'u_{self.time_index}_{self.node_index}', self.ohps.nu)
        self.u_lb = self.ohps.lbu
        self.u_ub = self.ohps.ubu
        self.x = ca.MX.sym(f'x_{self.time_index}_{self.node_index}', self.ohps.nx)
        self.x_lb = self.ohps.lbx
        self.x_ub = self.ohps.ubx
        self.s_P = ca.MX.sym(f's_P_{self.time_index}_{self.node_index}')
        self.s_P_lb = 0
        self.s_P_ub = ca.inf
        self.v = ca.vertcat(self.u, self.x, self.s_P)
        self.v_lb = ca.vertcat(self.u_lb, self.x_lb, self.s_P_lb)
        self.v_ub = ca.vertcat(self.u_ub, self.x_ub, self.s_P_ub)
        self.get_v_fun = ca.Function(f'get_v_{self.time_index}_{self.node_index}', 
                                     [self.x, self.u, self.s_P], [self.v])
        self.get_u_fun = ca.Function(f'get_u_{self.time_index}_{self.node_index}', 
                                     [self.v], [self.u])
        self.get_x_fun = ca.Function(f'get_x_{self.time_index}_{self.node_index}', 
                                     [self.v], [self.x])
        self.get_s_P_fun = ca.Function(f'get_s_P_{self.time_index}_{self.node_index}', 
                                     [self.v], [self.s_P])
        
    def get_parameters(self):
        self.param = []
        if self.is_root_node:
            self.x0 = ca.MX.sym('x0', self.ohps.nx)
            self.param.append(self.x0)
        self.wind_speed = ca.MX.sym(f'wind_speed_{self.time_index}_{self.node_index}')
        self.param.append(self.wind_speed)
        if self.opt['use_chance_constraints_multistage']:
            self.std = ca.MX.sym(f'std_{self.time_index}_{self.node_index}')
            self.param.append(self.std)
        self.P_demand = ca.MX.sym(f'P_demand_{self.time_index}_{self.node_index}')
        self.param.append(self.P_demand)
        self.p = ca.vertcat(*self.param)
        self.get_p_fun = ca.Function(f'get_p_{self.time_index}_{self.node_index}', self.param, [self.p])

    def get_constraints(self):
        # system dynamics, power 
        # (non-anticipativity already contained in system dynamics since only one control input u for predecessor) 
        self.constraints = []
        if self.is_root_node:
            x_i_k = self.x0 # initial condition
        else:
            x_i_k = self.ohps.get_next_state(self.predecessor.x, self.predecessor.u) # system dynamics
        state_constraint = x_i_k - self.x
        self.constraints.append(state_constraint)
        # Use chance constraints if wanted
        if self.opt['use_chance_constraints_multistage']:
            wind_speed = self.wind_speed - self.back_off_factor*self.std
        else:
            wind_speed = self.wind_speed
        P_gtg = self.ohps.get_P_gtg(self.x, self.u, wind_speed)
        P_bat = self.ohps.get_P_bat(self.x, self.u, wind_speed)
        P_wtg_backoff = self.ohps.get_P_wtg(self.x, self.u, wind_speed)
        g_demand = self.P_demand - P_gtg - P_bat - P_wtg_backoff - self.s_P
        self.constraints.append(g_demand)
        self.g = ca.vertcat(*self.constraints)

    def cost_function(self):
        alpha_1 = self.opt['param']['alpha_1']
        alpha_2 = self.opt['param']['alpha_2']
        x_gtg = self.ohps.get_x_gtg(self.x)
        u_gtg = self.ohps.get_u_gtg(self.u)
        P_gtg = self.ohps.gtg.get_power_output(x_gtg, u_gtg, None)
        load = P_gtg/self.opt['param']['P_gtg_max']
        eta_gtg = ca.if_else(load>0.01, alpha_1*load**2 + alpha_2*load, 0)  # TODO: add efficiency to model 
        eta_max = self.opt['param']['eta_gtg_max']
        J_gtg = self.opt['param']['k_gtg_eta']*(eta_gtg-eta_max)**2 + self.opt['param']['k_gtg_P']*P_gtg
        x_bat = self.ohps.get_x_bat(self.x)
        u_bat = self.ohps.get_u_bat(self.u)
        SOC = self.ohps.battery.get_SOC_fun(x_bat, u_bat)
        J_bat = -self.opt['param']['k_bat']*SOC
        J_u = self.u.T@self.opt['param']['R_input']@self.u
        J_s_P = self.s_P.T*self.param['r_s_P']*self.s_P
        self.J = J_gtg+J_bat+J_u+J_s_P
