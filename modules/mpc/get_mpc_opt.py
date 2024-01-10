import casadi as ca
def get_mpc_opt(method='nominal', **kwargs):
    opt = {
        'N': 60,    # Prediction horizon (number of steps)
        'dt': 10,    # Discretization interval in minutes
        'epsilon_chance_constraint': 0.1, # 90% certainty
        'certainty_horizon': 0,
        'robust_horizon': 10,
        'max_scenarios': 9,
        'branching_interval': 2,
        'dP_min': 2000,
        'use_chance_constraints_multistage': True,
    }
    param = {
        'alpha_1': -0.4,
        'alpha_2': 0.8,
        'P_gtg_max': 10000,
        'eta_gtg_max': 0.4,
        'k_gtg_eta': 1,
        'k_gtg_P': 1,
        'k_bat': 0.5,
        'R_input': 0*ca.DM.eye(2),
        'r_s_P': 10,
        'k_gtg_dP': 0.02
    }
    opt['param'] = param
    opt.update(kwargs)
    return opt