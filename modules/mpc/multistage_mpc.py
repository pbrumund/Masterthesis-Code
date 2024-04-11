import casadi as ca
import numpy as np
from scipy.stats import norm
import datetime

from .mpc_class import MPC
from .get_mpc_opt import get_nlp_opt
from ..models import OHPS
from ..gp import TimeseriesModel

class MultistageMPC(MPC):
    def __init__(self, ohps: OHPS, gp: TimeseriesModel, opt) -> None:
        self.ohps = ohps
        self.gp = gp
        self.horizon = opt['N']
        self.sampling_frequency = opt['dt']
        self.opt = opt
        self.nodes = None

    def generate_scenario(self, start_time, train=False, first_value_known=True):
        """
        Use mean of GP posterior, adding pseudo measurements if power uncertainty gets too large
        Does not work very well as scenarios converge towards prior mean for large horizons
        """
        certainty_horizon = self.opt['certainty_horizon']   # minimum steps before branching
        robust_horizon = self.opt['robust_horizon'] # maximum step for branching
        max_scenarios = self.opt['max_scenarios']   # maximum number of branches
        branching_interval = self.opt['branching_interval'] # only branch every braching_interval steps
        std_list = self.opt['std_list_multistage'] # number of standard deviations for scenario generation
        norm_factor = sum(norm.pdf(x) for x in std_list) # to keep sum of probabilities at 1
        dE_min = self.opt['dE_min'] # minimum wind power difference at +-1 std for branching

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

        power_uncertainty_acc = [[0]]
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
                P_lower = [self.ohps.get_P_wtg(0,0,mean[i]+std_list[0]+var[i]) 
                                               for mean, var in zip(means_i, vars_i)]
                P_upper = [self.ohps.get_P_wtg(0,0,mean[i]+std_list[2]+var[i]) 
                                               for mean, var in zip(means_i, vars_i)]
                power_uncertainties_acc_next = [power_uncertainty_acc_i_k+np.abs(P_upper[k]-P_lower[k])
                    for k, power_uncertainty_acc_i_k in enumerate(power_uncertainty_acc[i])]
                power_uncertainty_acc.append(power_uncertainties_acc_next)
                continue
            pseudo_times_next = []  # values for next iteration
            pseudo_measurements_next = []
            means_i_next = [] # Trajectories for next iteration
            vars_i_next = []
            parent_nodes_next = []
            probabilities_next = []
            power_uncertainties_acc_next = []
            for k, (mean, var, pseudo_times, pseudo_measurements) in enumerate(zip(
                    means_i, vars_i, pseudo_times_i, pseudo_measurements_i)):
                means_par.append(mean[i]) # keep mean/variance of current scenario even if branching
                vars_par.append(var[i])
                # Check if power difference large enough
                P_upper = self.ohps.n_wind_turbines*self.ohps.wind_turbine.power_curve_fun(
                    self.ohps.wind_turbine.scale_wind_speed(mean[i]+np.sqrt(var[i])))
                P_lower = self.ohps.n_wind_turbines*self.ohps.wind_turbine.power_curve_fun(
                    self.ohps.wind_turbine.scale_wind_speed(mean[i]-np.sqrt(var[i])))
                if power_uncertainty_acc[i][k]+np.abs(P_upper-P_lower)*self.sampling_frequency/60 < dE_min:
                    # Keep trajectories, do not branch this scenario
                    pseudo_times_next.append(pseudo_times)
                    pseudo_measurements_next.append(pseudo_measurements)
                    means_i_next.append(mean)
                    vars_i_next.append(var)
                    parent_nodes_next.append(k)
                    probabilities_next.append(probabilities[i][k])
                    power_uncertainties_acc_next.append(power_uncertainty_acc[i][k]+
                                                        np.abs(P_upper-P_lower)*self.sampling_frequency/60)# MWh
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
                    power_uncertainties_acc_next.append(0)
            means_i = means_i_next
            vars_i = vars_i_next
            pseudo_measurements_i = pseudo_measurements_next
            pseudo_times_i = pseudo_times_next
            probabilities.append(probabilities_next)
            parent_nodes.append(parent_nodes_next)
            power_uncertainty_acc.append(power_uncertainties_acc_next)
        # parent_nodes[0] = [None]
        return means_list, vars_list, parent_nodes[:-1], probabilities[:-1]
    
    def generate_scenario_new(self, start_time, train=False, first_value_known=True):
        """Use average, best and worst case scenarios but keep track of trajectory with 
        pseudo-measurements, branch if difference in generated energy gets too large"""
        max_scenarios = self.opt['max_scenarios']   # maximum number of branches
        std_list = self.opt['std_list_multistage'] # number of standard deviations for scenario generation
        norm_factor = sum(norm.pdf(x) for x in std_list) # to keep sum of probabilities at 1
        dE_min = self.opt['dE_min'] # minimum accumulated wind power difference between min and max trajectory for branching
        include_last_measurement = self.opt.get('include_last_measurement')
        mean_init, var_init = self.gp.predict_trajectory(start_time, self.horizon, train, include_last_measurement=include_last_measurement)

        trajectory_means = [mean_init]
        trajectory_vars = [var_init]

        control_means = [mean_init]
        control_vars = [var_init]

        means_list = [[mean_init[0]]]
        vars_list = [[var_init[0]]] # is not used

        probabilities = [[1]]
        parent_nodes = [[None]] # keep track of parents for tree construction

        pseudo_measurements = [np.empty((0,len(std_list)))] # to keep pseudo measurement times and vectors

        probabilities_last = [1]
        power_uncertainty_acc = [0]
        for i in range(1, self.horizon):
            means_list_i = []
            vars_list_i = []
            probabilities_list_i = []
            parent_nodes_i = []

            control_means_next = []
            control_vars_next = []
            trajectory_means_next = []
            trajectory_vars_next = []

            power_uncertainty_next = []
            pseudo_measurements_next = []
            for k, (mean_gp, var_gp) in enumerate(zip(trajectory_means, trajectory_vars)):
                # iterate over current gp means (1 if not branched, 3 after first branch etc.)
                pseudo_measurements_i_k = (
                    control_means[k][i]+np.array(std_list).reshape((1,-1))*np.sqrt(control_vars[k][i]))
                # pseudo_measurements[k] = np.concatenate([pseudo_measurements[k], 
                #     pseudo_measurements_i_k], axis=0)
                pseudo_measurements_k = np.concatenate([pseudo_measurements[k], 
                    pseudo_measurements_i_k], axis=0)
                P_lower = self.ohps.get_P_wtg(0,0,mean_gp[i]+std_list[0]*np.sqrt(var_gp[i]))
                P_upper = self.ohps.get_P_wtg(0,0,mean_gp[i]+std_list[-1]*np.sqrt(var_gp[i]))
                power_uncertainty_acc[k] += np.abs(P_upper-P_lower)*self.sampling_frequency/60
                if power_uncertainty_acc[k] < dE_min or len(control_means) >= max_scenarios:
                    # do not branch, use control means (keep constant difference)
                    means_list_i.append(control_means[k][i])
                    vars_list_i.append(control_vars[k][i])
                    parent_nodes_i.append(k)
                    probabilities_list_i.append(probabilities_last[k])
                    control_means_next.append(control_means[k])
                    control_vars_next.append(control_vars[k])
                    trajectory_means_next.append(trajectory_means[k])
                    trajectory_vars_next.append(trajectory_vars[k])
                    pseudo_measurements_next.append(pseudo_measurements_k)
                    power_uncertainty_next.append(power_uncertainty_acc[k])
                    continue
                # branch
                parent_nodes_i.extend([k]*len(std_list))
                means_list_i.extend(
                    [mean_gp[i]+x*np.sqrt(var_gp[i]) for x in std_list]
                )
                vars_list_i.extend([var_gp[i]]*len(std_list))
                probability_i_k = probabilities_last[k]
                probabilities_list_i.extend(
                    [probability_i_k*norm.pdf(x)/norm_factor for x in std_list]
                )
                control_means_next.extend([trajectory_means[k]+x*np.sqrt(trajectory_vars)[k] 
                                            for x in std_list])
                control_vars_next.extend([trajectory_vars[k]]*len(std_list))
                for j in range(pseudo_measurements_k.shape[1]):
                    pseudo_gp_x = np.arange(1,i+1)
                    pseudo_gp_y = pseudo_measurements_k[:,j]
                    pseudo_gp = self.gp.get_pseudo_timeseries_gp(
                        start_time, pseudo_gp_y, pseudo_gp_x)
                    pseudo_gp_mean, pseudo_gp_var = self.gp.predict_trajectory(
                        start_time, self.horizon, pseudo_gp=pseudo_gp
                    )
                    trajectory_means_next.append(pseudo_gp_mean)
                    trajectory_vars_next.append(pseudo_gp_var)
                pseudo_measurements_next.extend([pseudo_measurements_k]*len(std_list))
                power_uncertainty_next.extend([0]*len(std_list))
            control_means = control_means_next
            control_vars = control_vars_next
            trajectory_means = trajectory_means_next
            trajectory_vars = trajectory_vars_next
            parent_nodes.append(parent_nodes_i)
            probabilities.append(probabilities_list_i)
            probabilities_last = probabilities_list_i
            pseudo_measurements = pseudo_measurements_next
            power_uncertainty_acc = power_uncertainty_next  
            means_list.append(means_list_i)
            vars_list.append(vars_list_i)          
        return means_list, vars_list, parent_nodes, probabilities
    
    def generate_scenario_simple(self, start_time, train=False):
        """Only branch once using +-a standard deviations as scenarios,
        adapt certainty horizon based on accumulated power uncertainty"""
        include_last_measurement = self.opt.get('include_last_measurement')
        std_list = self.opt['std_list_multistage'] # number of standard deviations for scenario generation
        norm_factor = sum(norm.pdf(x) for x in std_list) # to keep sum of probabilities at 1
        dE_min = self.opt['dE_min'] # minimum wind power difference at +-1 std for branching

        mean_init, var_init = self.gp.predict_trajectory(start_time, self.horizon, train, include_last_measurement=include_last_measurement)

        means_list = [[mean_init[0]]]
        vars_list = [[var_init[0]]] # is not used

        probabilities = [[1]]
        parent_nodes = [[None]] # keep track of parents for tree construction

        has_branched = False

        power_uncertainty_acc = 0
        for i in range(1, self.horizon):
            if has_branched:
                parent_nodes.append(list(np.arange(len(std_list))))
                means_list.append(
                    [mean_init[i]+x*np.sqrt(var_init[i]) for x in std_list]
                )
                vars_list.append([var_init[i]]*len(std_list))
                probabilities.append(
                    [norm.pdf(x)/norm_factor for x in std_list]
                )
                continue
            P_upper = self.ohps.get_P_wtg(0,0,mean_init[i]+std_list[0]*np.sqrt(var_init[i]))
            P_lower = self.ohps.get_P_wtg(0,0,mean_init[i]-std_list[0]*np.sqrt(var_init[i]))
            power_uncertainty_acc += np.abs(P_upper-P_lower)*self.sampling_frequency/60
            if power_uncertainty_acc >= dE_min:
                # branch
                has_branched = True
                parent_nodes.append([0]*len(std_list))
                means_list.append(
                    [mean_init[i]+x*np.sqrt(var_init[i]) for x in std_list]
                )
                vars_list.append([var_init[i]]*len(std_list))
                probabilities.append(
                    [norm.pdf(x)/norm_factor for x in std_list]
                )
                continue
            # do not branch yet
            parent_nodes.append([0])
            means_list.append([mean_init[i]])
            vars_list.append([var_init[i]])
            probabilities.append([1])
        return means_list, vars_list, parent_nodes, probabilities

    def build_optimization_tree(self, parent_nodes, probabilities, means=None, vars=None):
        nodes = [
            [TreeNode(ohps=self.ohps, time_index=0, node_index=0, opt=self.opt, predecessor=None,
                probability=probabilities[0][0])]]
        for i in range(1, self.horizon):
            nodes_i = [TreeNode(
                ohps=self.ohps, time_index=i, node_index=k, opt=self.opt, 
                probability=probabilities[i][k], predecessor=nodes[i-1][k_predecessor]) 
                for k, k_predecessor in enumerate(parent_nodes[i])]
            nodes.append(nodes_i)
        return nodes

    def get_optimization_problem(self, start_time, train=False, keep_tree=False):
        if self.opt['use_simple_scenarios']:
            means, vars, parent_nodes, probabilities = self.generate_scenario_simple(start_time, train)
        else:
            means, vars, parent_nodes, probabilities = self.generate_scenario_new(start_time, train)
        if self.opt.get('use_old_scenarios'):
            means, vars, parent_nodes, probabilities = self.generate_scenario(start_time, train)
        self.means = means
        self.vars = vars
        # test if structure of tree has changed to not have to build the tree again
        if self.nodes is not None:
            can_keep_tree =  np.all([len(means[i])==len(self.nodes[i]) for i in range(self.horizon)]) 
        else:
            can_keep_tree = False
        if keep_tree and can_keep_tree:
            return
        self.parent_nodes = parent_nodes
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
        self.get_s_P_next_fun = ca.Function('get_s_P_next', [v_vec], [nodes[0][0].s_P])
        v_middle = [node.v for node in middle_nodes] # for generating initial guess
        self.get_v_middle_fun = ca.Function('get_v_middle', [v_vec], v_middle)
        g = ca.vertcat(*[node.g for node in nodes_flattened])
        g_lb = ca.vertcat(*[node.g_lb for node in nodes_flattened])
        g_ub = ca.vertcat(*[node.g_ub for node in nodes_flattened])
        p = ca.vertcat(*[node.p for node in nodes_flattened])
        J = sum(node.J for node in nodes_flattened)
        nlp = {'f': J, 'x': v_vec, 'g': g, 'p': p}
        nlp_opt = get_nlp_opt()
        _solver = ca.nlpsol('multistage_mpc', 'ipopt', nlp, nlp_opt)
        v_init = ca.MX.sym('v_init', v_vec.shape)
        self.solver = ca.Function(
            'mpc', [v_init, p], 
            [_solver(x0=v_init, p=p, lbx=v_lb, ubx=v_ub, lbg=g_lb, ubg=g_ub)['x']],
            ['v_init', 'p'], ['v_opt'])
        
        self.J_gtg = sum(node.J_gtg*node.probability for node in nodes_flattened)
        self.J_gtg_P = sum(node.J_gtg_P*node.probability for node in nodes_flattened)
        self.J_gtg_eta = sum(node.J_gtg_eta*node.probability for node in nodes_flattened)
        self.J_gtg_dP = sum(node.J_gtg_dP*node.probability for node in nodes_flattened)
        self.J_bat = sum(node.J_bat*node.probability for node in nodes_flattened)
        self.J_s_P = sum(node.J_s_P*node.probability for node in nodes_flattened)
        self.J_s_x = sum(node.J_s_x*node.probability for node in nodes_flattened)
        self.J_u = sum(node.J_u*node.probability for node in nodes_flattened)
        self.J_fun = ca.Function('J', [v_vec, p], [J])
        self.J_gtg_fun = ca.Function('J_gtg', [v_vec, p], [self.J_gtg])
        self.J_gtg_P_fun = ca.Function('J_gtg_P', [v_vec, p], [self.J_gtg_P])
        self.J_gtg_eta_fun = ca.Function('J_gtg_eta', [v_vec, p], [self.J_gtg_eta])
        self.J_gtg_dP_fun = ca.Function('J_gtg_dP', [v_vec, p], [self.J_gtg_dP])
        self.J_bat_fun = ca.Function('J_bat', [v_vec, p], [self.J_bat])
        self.J_u_fun = ca.Function('J_u', [v_vec, p], [self.J_u])
        self.J_s_P_fun = ca.Function('J_s_P', [v_vec, p], [self.J_s_P])
        self.J_s_x_fun = ca.Function('J_s_x', [v_vec, p], [self.J_s_x])
        
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
                s_x_init = 0
                v_init_i_k = node.get_v_fun(x_init, u_init, s_P_init, s_x_init)
                v_init.append(v_init_i_k)
            x_init_list.append(x_init)
            u_init_list.append(u_init)
        return ca.vertcat(*v_init)
                
    def get_parameters(self, x0, P_gtg_last, P_demand, wind_predictions):
        p = []
        for i, nodes_i in enumerate(self.nodes):
            for k, node in enumerate(nodes_i):
                p_i_k = []
                if i == 0:
                    p_i_k.append(x0)
                    p_i_k.append(P_gtg_last)
                p_i_k.append(wind_predictions[i][k])
                p_i_k.append(P_demand[i])
                p_i_k = node.get_p_fun(*p_i_k)
                p.append(p_i_k)
        return ca.vertcat(*p)


class TreeNode:
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
        if time_index == opt['N']-1:
            self.is_leaf_node = True
        else:
            self.is_leaf_node = False
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
        if self.is_root_node: # No state constraint on root node for feasibility
            self.x_lb = -ca.inf*ca.DM.ones(self.ohps.nx)
            self.x_ub = ca.inf*ca.DM.ones(self.ohps.nx)
        self.s_P = ca.MX.sym(f's_P_{self.time_index}_{self.node_index}')
        self.s_P_lb = -ca.inf
        self.s_P_ub = ca.inf
        self.s_x = ca.MX.sym(f's_x_{self.time_index}_{self.node_index}')
        self.s_x_lb = 0
        self.s_x_ub = ca.inf
        self.v = ca.vertcat(self.u, self.x, self.s_P, self.s_x)
        self.v_lb = ca.vertcat(self.u_lb, self.x_lb, self.s_P_lb, self.s_x_lb)
        self.v_ub = ca.vertcat(self.u_ub, self.x_ub, self.s_P_ub, self.s_x_ub)
        self.get_v_fun = ca.Function(f'get_v_{self.time_index}_{self.node_index}', 
                                     [self.x, self.u, self.s_P, self.s_x], [self.v])
        self.get_u_fun = ca.Function(f'get_u_{self.time_index}_{self.node_index}', 
                                     [self.v], [self.u])
        self.get_x_fun = ca.Function(f'get_x_{self.time_index}_{self.node_index}', 
                                     [self.v], [self.x])
        self.get_s_P_fun = ca.Function(f'get_s_P_{self.time_index}_{self.node_index}', 
                                     [self.v], [self.s_P])
        self.get_s_x_fun = ca.Function(f'get_s_x_{self.time_index}_{self.node_index}',
                                     [self.v], [self.s_x])
        
    def get_parameters(self):
        self.param = []
        if self.is_root_node:
            self.x0 = ca.MX.sym('x0', self.ohps.nx)
            self.param.append(self.x0)
            self.P_gtg_last = ca.MX.sym('P_gtg_last')
            self.param.append(self.P_gtg_last)
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
        if self.is_leaf_node: # State constraints for successor of root node
            x_next = self.ohps.get_next_state(self.x, self.u)
            x_next_lb = self.ohps.lbx
            x_next_ub = self.ohps.ubx
            self.constraints.append(x_next)
            g_lb.append(x_next_lb)
            g_ub.append(x_next_ub)
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
        P_wtg = self.ohps.get_P_wtg(self.x, self.u, wind_speed)
        g_demand = self.P_demand - P_gtg - P_bat - P_wtg - self.s_P
        g_demand_lb = 0
        g_demand_ub = 0
        self.constraints.append(g_demand)
        g_lb.append(g_demand_lb)
        g_ub.append(g_demand_ub)
        if self.opt['use_soft_constraints_state'] and not self.is_root_node:
            x_next = self.ohps.get_next_state(self.x, self.u)
            sc_backoff = 0.05
            x_lb_sc = self.ohps.lbx + sc_backoff
            x_ub_sc = self.ohps.ubx - sc_backoff
            g_x_lb = x_lb_sc - x_next - self.s_x
            g_x_ub = x_next - x_ub_sc - self.s_x
            self.constraints.append(g_x_lb)
            g_lb.append(-ca.inf)
            g_ub.append(0)
            self.constraints.append(g_x_ub)
            g_lb.append(-ca.inf)
            g_ub.append(0)
        self.g_lb = ca.vertcat(*g_lb)
        self.g_ub = ca.vertcat(*g_ub)
        self.g = ca.vertcat(*self.constraints)

    def get_cost_function(self):
        alpha_1 = self.opt['param']['alpha_1']
        alpha_2 = self.opt['param']['alpha_2']
        x_gtg = self.ohps.get_x_gtg(self.x)
        u_gtg = self.ohps.get_u_gtg(self.u)
        self.P_gtg = self.ohps.gtg.get_power_output(x_gtg, u_gtg, None)
        load = self.P_gtg/self.ohps.P_gtg_max
        self.J_gtg_P = self.opt['param']['k_gtg_P']*load
        eta_gtg = ca.if_else(load>0.01, alpha_1*load**2 + alpha_2*load, 0) 
        eta_max = self.opt['param']['eta_gtg_max']
        self.J_gtg_eta = self.opt['param']['k_gtg_eta']*(eta_gtg-eta_max)**2
        if self.is_root_node:
            P_gtg_last = self.P_gtg_last
        else:
            P_gtg_last = self.predecessor.P_gtg
        self.J_gtg_dP = (self.opt['param']['k_gtg_dP']*
                    ((self.P_gtg-P_gtg_last)/self.ohps.P_gtg_max)**2)
        J_gtg_fuel = self.opt['param']['k_gtg_fuel']*load/self.ohps.gtg.eta_fun(load)
        self.J_gtg = self.J_gtg_P + self.J_gtg_eta + self.J_gtg_dP + J_gtg_fuel
        x_bat = self.ohps.get_x_bat(self.x)
        u_bat = self.ohps.get_u_bat(self.u)
        SOC = self.ohps.battery.get_SOC_fun(x_bat, u_bat)
        self.J_bat = -self.opt['param']['k_bat']*SOC
        self.J_u = self.u.T@self.opt['param']['R_input']@self.u
        self.J_s_P = (self.opt['param']['r_s_P']*(self.s_P)**2/(self.ohps.P_wtg_max+self.ohps.P_gtg_max)
                 *1/(self.time_index+1)**2)
        if self.opt['use_soft_constraints_state']:
            self.J_s_x = self.opt['param']['r_s_x']*self.s_x
        else:
            self.J_s_x = 0
        self.J = (self.J_gtg+self.J_bat+self.J_u+self.J_s_P+self.J_s_x)*self.probability
        
        

