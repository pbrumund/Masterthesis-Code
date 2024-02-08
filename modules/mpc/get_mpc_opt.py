import casadi as ca
def get_mpc_opt(method='nominal', **kwargs):
    opt = {
        'N': 60,    # Prediction horizon (number of steps)
        'dt': 10,    # Discretization interval in minutes
        'epsilon_chance_constraint': 0.1, # 90% certainty
        'certainty_horizon': 1,
        'robust_horizon': 30,
        'max_scenarios': 9,
        'branching_interval': 1,
        'dE_min': 5000,
        'use_chance_constraints_multistage': False,
        'std_list_multistage': (-1,0,1),
        'use_simple_scenarios': False,
        'use_soft_constraints_state': False
    }
    param = {
        'alpha_1': -0.4,
        'alpha_2': 0.8,
        'P_gtg_max': 32000,
        'eta_gtg_max': 0.506,
        'k_gtg_eta': 10,
        'k_gtg_P': 10,
        'k_gtg_fuel': 0,
        'k_bat': .5,#.1,#.5,
        'R_input': ca.diag([0,1e-8]),# 5e-6
        'r_s_P': 100,
        'r_s_x': 10000,
        'k_gtg_dP': 5,#5,#.5,
        'k_bat_final': 0#500
    }
    opt['param'] = param
    opt.update(kwargs)
    return opt

def get_nlp_opt(**kwargs):
    nlp_opt = {}
    ipopt_opt = {
        'print_frequency_time': 1,
        'print_frequency_iter': 10,
        'print_level': 5,
        'max_iter': 1000,
        'linear_solver': 'mumps',
        # 'tol': 0.0001,
        # 'dual_inf_tol': 10,
        # 'compl_inf_tol': .001,
    }
    ipopt_opt.update(kwargs)
    nlp_opt['ipopt'] = ipopt_opt
    return nlp_opt