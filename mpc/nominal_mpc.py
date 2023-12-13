import casadi as ca

def cost_function(state, input, output, param):
    # Term for gas turbine
    alpha_1 = param['alpha_1']
    alpha_2 = param['alpha_2']
    load = output[0]/param['P_gtg_max']
    eta_gtg = ca.if_else(load>0.01, alpha_1*load**2 + alpha_2*load, 0)
    J_gtg = param['j_gtg_eta']

def get_constraints(state_traj, input_traj, output_traj):
    pass


def nominal_mpc():