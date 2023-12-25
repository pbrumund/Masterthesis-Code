def get_mpc_opt(method='nominal'):
    opt = {
        'N': 60,    # Prediction horizon (number of steps)
        'dt': 5     # Discretization interval in minutes
    }