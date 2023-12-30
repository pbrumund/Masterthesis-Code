import casadi as ca
def get_mpc_opt(method='nominal', **kwargs):
    opt = {
        'N': 60,    # Prediction horizon (number of steps)
        'dt': 5,    # Discretization interval in minutes
        'epsilon_chance_constraint': 0.1 # 90% certainty
    }
    param = {
        'alpha_1': -0.4,
        'alpha_2': 0.8,
        'P_gtg_max': 10000,
        'eta_gtg_max': 0.4,
        'k_gtg_eta': 5000,
        'k_gtg_P': 1,
        'k_bat': 500,
        'R_input': 0*ca.DM.eye(2),
        'r_s_P': 100
    }
    opt['param'] = param
    opt.update(kwargs)
    return opt