class MPC():
    """General functions for all MPC methods"""
    def get_nlp_opt(self, **kwargs):
        nlp_opt = {}
        nlp_opt['ipopt'] = {
        }
        return nlp_opt