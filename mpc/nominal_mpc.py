import casadi as ca
from mpc_class import MPC

class NominalMPC(MPC):
    def __init__(self, ohps, opt):
        self.ohps = ohps
        self.horizon = opt['N']
        self.sampling_frequency = opt['dt']
        self.nx = self.ohps.nx
        self.nu = self.ohps.nu

    def get_optimization_variables(self):
        U_mat = ca.MX.sym('U', self.horizon, self.nu)   # system inputs
        X_mat = ca.MX.sym('X', self.horizon, self.nx)   # state trajectory for multiple shooting
        s_P = ca.MX.sym('s_P', self.horizon)    # for soft power constraints
        v = ca.vertcat(U_mat.reshape((-1)), X_mat.reshape((-1)), s_P)
        self.get_u_from_v_fun = ca.Function('get_u_from_v', [v], [U_mat])
        self.get_x_from_v_fun = ca.Function('get_x_from_v', [v], [X_mat])
        self.get_s_from_v_fun = ca.Function('get_s_from_v', [v], [s_P])


    def cost_function(self, state, input, output):
        alpha_1 = self.param['alpha_1']
        alpha_2 = self.param['alpha_2']
        load = output[0]/self.param['P_gtg_max']
        eta_gtg = ca.if_else(load>0.01, alpha_1*load**2 + alpha_2*load, 0)
        eta_max = self.param['eta_gtg_max']
        J_gtg = self.param['k_gtg_eta']*(eta_gtg-eta_max)**2 + self.param['k_gtg_P']*output[0]
        J_bat = -self.param['k_gtg_eta']*state[2]
        J_u = input.T*self.param['R_input']*input
        return J_gtg+J_bat+J_u


    


# def nominal_mpc(ohps, opt):
#     # optimization variables: u, x, s_p for soft constraints
#     # constraints: Power constraints, state and input constraints
#     # equality constraints for state because of multiple shooting
#     # initial condition
#     # for load shifting terminal constraint
#     pass