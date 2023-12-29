class MPC():
    """General functions for all MPC methods"""
    def get_nlp_opt(self, **kwargs):
        nlp_opt = {}
        nlp_opt['ipopt'] = {
            'print_frequency_time': 2,
            'print_level': 1,
            'max_iter': 1000
        }
        return nlp_opt