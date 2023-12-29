from hilo_mpc import Kernel
import casadi as ca
import numpy as np
from typing import List

class HeteroscedasticKernel(Kernel):
    def __init__(self, kernel, noise_gp, active_dims=None):
        super().__init__(active_dims=active_dims)
        self._kernel = kernel
        self._gp = noise_gp
        # self.hyperparameters = self._kernel.hyperparameters

    @property
    def hyperparameters(self):
        return self._kernel.hyperparameters
    
    def get_covariance_function(self, x: ca.SX, x_bar: ca.SX, active_dims: np.ndarray) -> ca.Function:
        k_mean = self._kernel(x, x_bar)
        mean_noise_log, _ = self._gp.predict(x_bar)
        var = ca.exp(mean_noise_log)*self._gp.mean_var

        hyperparameters = [parameter.SX for parameter in self.hyperparameters]
        hyperparameter_names = [parameter.name for parameter in self.hyperparameters]

        condition = ca.logic_all(ca.eq(x, x_bar))
        # condition = ca.le(ca.norm_2(x-x_bar),1e-9)
        K_out = ca.if_else(condition, k_mean+var, k_mean)
        K_sum = ca.Function(
            'covariance',
            [x, x_bar, *hyperparameters],
            [K_out],
            ['x', 'x_bar', *hyperparameter_names],
            ['covariance']
        )
        return K_sum