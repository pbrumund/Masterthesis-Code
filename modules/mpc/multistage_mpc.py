import casadi as ca
import numpy as np
from scipy.stats import norm

from .mpc_class import MPC
from ..models import OHPS
from ..gp import TimeseriesModel

class MultistageMPC(MPC):
    def __init__(self, ohps: OHPS, gp: TimeseriesModel, opt) -> None:
        self.ohps = ohps
        self.gp = gp
        self.horizon = opt['N']
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
        norm_factor = sum(norm.pdf(x) for x in std_list) # to keep sum of probabilities at 1
        dP_min = self.opt['dP_min'] # minimum wind power difference at +-1 std for branching

        mean_init, var_init = self.gp.predict_trajectory(start_time, self.horizon, train)

        means_i = [mean_init] # all trajectories at step i (contains entire trajectory)
        vars_i = [var_init]
        
        means_par = []  # values at each step i 
        vars_par = []

        means_list = []
        vars_list = []

        probabilities = [[1]]
        parent_nodes = [[None]] # keep track of parents for tree construction

        pseudo_times_i = [np.array([])]  # to keep pseudo measurement times and vectors
        pseudo_measurements_i = [np.array([])]
        for i in range(self.horizon):
            means_list.append([mean[i] for mean in means_i])
            vars_list.append([var[i] for var in vars_i])
            if (i < certainty_horizon or i >= robust_horizon 
                or (i-certainty_horizon)%branching_interval != 0 or len(means_i) >= max_scenarios):
                # do not branch at this timestep
                for mean_traj in means_i: means_par.append(mean_traj[i])
                for var_traj in vars_i: vars_par.append(var_traj[i])
                parent_nodes.append([k for k, _ in enumerate(parent_nodes[i])])
                probabilities.append(probabilities[i])
                continue
            pseudo_times_next = []  # values for next iteration
            pseudo_measurements_next = []
            means_i_next = [] # Trajectories for next iteration
            vars_i_next = []
            parent_nodes_next = []
            probabilities_next = []
            for k, (mean, var, pseudo_times, pseudo_measurements) in enumerate(zip(
                    means_i, vars_i, pseudo_times_i, pseudo_measurements_i)):
                means_par.append(mean[i]) # keep mean/variance of current scenario even if branching
                vars_par.append(var[i])
                # Check if power difference large enough
                P_upper = self.ohps.wind_turbine.power_curve_fun(
                    self.ohps.wind_turbine.scale_wind_speed(mean[i]+np.sqrt(var[i])))
                P_lower = self.ohps.wind_turbine.power_curve_fun(
                    self.ohps.wind_turbine.scale_wind_speed(mean[i]-np.sqrt(var[i])))
                if np.abs(P_upper-P_lower) < dP_min:
                    # Keep trajectories, do not branch this scenario
                    pseudo_times_next.append(pseudo_times)
                    pseudo_measurements_next.append(pseudo_measurements)
                    means_i_next.append(mean)
                    vars_i_next.append(var)
                    parent_nodes_next.append(k)
                    probabilities_next.append(probabilities[i][k])
                    continue
                # add measurement at timestep i
                pseudo_times_new = np.append(pseudo_times, i)
                for x in std_list:
                    # add scenarios for +- 1 std and mean
                    w = mean[i] + x*np.sqrt(var[i])
                    pseudo_measurements_new = np.append(pseudo_measurements, w)
                    pseudo_gp = self.gp.get_pseudo_timeseries_gp(
                        start_time, pseudo_measurements_new, pseudo_times_new)
                    mean_new, var_new = self.gp.predict_trajectory(
                        start_time, self.horizon, pseudo_gp=pseudo_gp)
                    pseudo_times_next.append(pseudo_times_new)
                    pseudo_measurements_next.append(pseudo_measurements_new)
                    parent_nodes_next.append(k)
                    probabilities_next.append(probabilities[i][k]*norm.pdf(x)/norm_factor)
                    means_i_next.append(mean_new)
                    vars_i_next.append(var_new)
            means_i = means_i_next
            vars_i = vars_i_next
            pseudo_measurements_i = pseudo_measurements_next
            pseudo_times_i = pseudo_times_next
            probabilities.append(probabilities_next)
            parent_nodes.append(parent_nodes_next)
        # parent_nodes[0] = [None]
        return means_list, vars_list, parent_nodes[:-1], probabilities[:-1]
    
    def build_optimization_tree(self, parent_nodes, probabilities, means=None, vars=None):
        nodes = [
            [TreeNode(ohps=self.ohps, time_index=0, node_index=0, opt=self.opt, predecessor=None,
                      wind_pred=(means[0][0], vars[0][0]), probability=probabilities[0][0])]]
        for i in range(1, self.horizon):
            nodes_i = [TreeNode(
                ohps=self.ohps, time_index=i, node_index=k, opt=self.opt, 
                probability=probabilities[i][k], predecessor=nodes[i-1][k_predecessor], 
                wind_pred = (mean, var)) 
                for k, (k_predecessor, mean, var) in enumerate(zip(parent_nodes[i], means[i], vars[i]))]
            nodes.append(nodes_i)
        return nodes

    def get_optimization_problem(self, start_time, train=False):
        means, vars, parent_nodes, probabilities = self.generate_scenario(start_time, train)
        nodes = self.build_optimization_tree(parent_nodes, probabilities, means, vars)
        self.nodes = nodes
        # nodes is nested list with time as first index and node index as second index
        nodes_flattened = []
        middle_nodes = []
        for nodes_i in nodes:
            nodes_flattened.extend(nodes_i)
            middle_nodes.append(nodes_i[len(nodes_i)//2])
        # get optimization variables, constraints, parameters
        v = [node.v for node in nodes_flattened]
        v_vec = ca.vertcat(*v)
        self.v = v_vec
        v_lb = ca.vertcat(*[node.v_lb for node in nodes_flattened])
        v_ub = ca.vertcat(*[node.v_ub for node in nodes_flattened])
        self.get_u_next_fun = ca.Function('get_u_next', [v_vec], [nodes[0][0].u])
        v_middle = [node.v for node in middle_nodes] # for generating initial guess
        self.get_v_middle_fun = ca.Function('get_v_middle', [v_vec], v_middle)
        g = ca.vertcat(*[node.g for node in nodes_flattened])
        g_lb = ca.vertcat(*[node.g_lb for node in nodes_flattened])
        g_ub = ca.vertcat(*[node.g_ub for node in nodes_flattened])
        p = ca.vertcat(*[node.p for node in nodes_flattened])
        J = sum(node.J for node in nodes_flattened)
        nlp = {'f': J, 'x': v_vec, 'g': g, 'p': p}
        nlp_opt = self.get_nlp_opt()
        _solver = ca.nlpsol('multistage_mpc', 'ipopt', nlp, nlp_opt)
        v_init = ca.MX.sym('v_init', v_vec.shape)
        self.solver = ca.Function(
            'mpc', [v_init, p], 
            [_solver(x0=v_init, p=p, lbx=v_lb, ubx=v_ub, lbg=g_lb, ubg=g_ub)['x']],
            ['v_init', 'p'], ['v_opt'])
        

    def get_initial_guess(self, v_last = None, wind_power_vec = None, x0 = None, 
                          P_demand = None):
        if v_last is not None:
            # use last solution to generate initial guess
            v_last = list(v_last[1:])
            v_last.append(v_last[-1])
            v_init = []
            for v_i, nodes_i in zip(v_last, self.nodes):
                for node in nodes_i:
                    v_init.append(v_i)
            return ca.vertcat(*v_init)
        # Build initial solution
        v_init = []
        u_init_list = []
        x_init_list = []
        if wind_power_vec is None:
            wind_power_vec = np.zeros(self.horizon)
        
        if P_demand is None:
            P_demand = 10000*np.ones(self.horizon)
        for i, nodes_i in enumerate(self.nodes):
            for node in nodes_i:
                u_init = ca.DM.zeros(self.ohps.nu)
                if i == 0:
                    x_init = x0
                else:
                    x_init = self.ohps.get_next_state(x_init_list[i-1], u_init_list[i-1])
                w = wind_power_vec[i]
                P_gtg = self.ohps.get_P_gtg(x_init, u_init, w)
                P_bat = self.ohps.get_P_bat(x_init, u_init, w)
                P_wtg = self.ohps.get_P_wtg(x_init, u_init, w)
                s_P_init = P_demand[i] - P_gtg - P_bat - P_wtg
                v_init_i_k = node.get_v_fun(x_init, u_init, s_P_init)
                v_init.append(v_init_i_k)
            x_init_list.append(x_init)
            u_init_list.append(u_init)
        return ca.vertcat(*v_init)
                
    def get_parameters(self, x0, P_demand):
        p = []
        for i, nodes_i in enumerate(self.nodes):
            for node in nodes_i:
                p_i_k = []
                if i == 0:
                    p_i_k.append(x0)
                p_i_k.append(P_demand[i])
                p_i_k = node.get_p_fun(*p_i_k)
                p.append(p_i_k)
        return ca.vertcat(*p)
        




            
class TreeNode:
    # TODO: Methods for branching and getting single successor, likelihood
    def __init__(self, ohps: OHPS, time_index, node_index, opt, probability, 
                 predecessor=None, wind_pred=None):
        self.ohps = ohps
        self.time_index = time_index
        self.node_index = node_index
        self.predecessor = predecessor
        if self.predecessor is None:
            self.is_root_node = True
        else:
            self.is_root_node = False
        self.opt = opt
        if self.opt['use_chance_constraints_multistage']:
            epsilon = self.opt['epsilon_chance_constraint']
            self.back_off_factor = norm.ppf(1-epsilon)
        self.wind_pred = wind_pred
        self.probability = probability
        self.setup()

    def setup(self):
        self.get_optimization_variables()
        self.get_parameters()
        self.get_constraints()
        self.get_cost_function()

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
        if self.wind_pred is None:
            self.wind_speed = ca.MX.sym(f'wind_speed_{self.time_index}_{self.node_index}')
            self.param.append(self.wind_speed)
            if self.opt['use_chance_constraints_multistage']:
                self.wind_std = ca.MX.sym(f'std_{self.time_index}_{self.node_index}')
                self.param.append(self.wind_std)
        self.P_demand = ca.MX.sym(f'P_demand_{self.time_index}_{self.node_index}')
        self.param.append(self.P_demand)
        self.p = ca.vertcat(*self.param)
        self.get_p_fun = ca.Function(f'get_p_{self.time_index}_{self.node_index}', self.param, [self.p])

    def get_constraints(self):
        # system dynamics, power 
        # (non-anticipativity already contained in system dynamics since only one control input u for predecessor) 
        self.constraints = []
        g_lb = []
        g_ub = []
        if self.is_root_node:
            x_i_k = self.x0 # initial condition
        else:
            x_i_k = self.ohps.get_next_state(self.predecessor.x, self.predecessor.u) # system dynamics
        state_constraint = x_i_k - self.x
        state_constraint_lb = ca.DM.zeros(self.ohps.nx)
        state_constraint_ub = ca.DM.zeros(self.ohps.nx)
        self.constraints.append(state_constraint)
        g_lb.append(state_constraint_lb)
        g_ub.append(state_constraint_ub)
        # Use chance constraints if wanted
        if self.wind_pred is not None:
            wind_speed = self.wind_pred[0] # use numerical value
        else:
            wind_speed = self.wind_speed # use symbolic
        if self.opt['use_chance_constraints_multistage']:
            if self.wind_pred is not None:
                wind_std = self.wind_pred[1]
            else:
                wind_std = self.wind_std
            wind_speed = wind_speed - self.back_off_factor*wind_std
        P_gtg = self.ohps.get_P_gtg(self.x, self.u, wind_speed)
        P_bat = self.ohps.get_P_bat(self.x, self.u, wind_speed)
        P_wtg_backoff = self.ohps.get_P_wtg(self.x, self.u, wind_speed)
        g_demand = self.P_demand - P_gtg - P_bat - P_wtg_backoff - self.s_P
        g_demand_lb = -ca.inf
        g_demand_ub = 0
        self.constraints.append(g_demand)
        g_lb.append(g_demand_lb)
        g_ub.append(g_demand_ub)
        self.g_lb = ca.vertcat(*g_lb)
        self.g_ub = ca.vertcat(*g_ub)
        self.g = ca.vertcat(*self.constraints)

    def get_cost_function(self):
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
        J_s_P = self.s_P.T@self.opt['param']['r_s_P']@self.s_P
        self.J = (J_gtg+J_bat+J_u+J_s_P)*self.probability
        

