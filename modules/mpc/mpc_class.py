class MPC():
    """General functions for all MPC methods"""
    def get_nlp_opt(self, **kwargs):
        nlp_opt = {}
        nlp_opt['ipopt'] = {
            'print_frequency_time': 1,
            'print_level': 5,
            'max_iter': 1000,
            'linear_solver': 'mumps',
            'tol': 0.2
        }
        return nlp_opt