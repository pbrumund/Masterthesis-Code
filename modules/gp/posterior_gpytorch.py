from typing import Optional, Tuple
from gpytorch.constraints import Interval
from gpytorch.kernels import Kernel
from gpytorch.means import Mean
import gpytorch
import torch
import numpy as np
import tensorflow as tf
from gpytorch.priors import Prior

class TimeseriesGP(gpytorch.models.ExactGP):
    def __init__(self, train_inputs, train_targets, likelihood, nwp_gp, get_x_fun, 
                 cashing=True, gp_predictions = None):
        super().__init__(train_inputs, train_targets, likelihood)
        # self.mean_module = gpytorch.means.ZeroMean()
        self.mean_module = NWPMean(nwp_gp, get_x_fun, cashing, gp_predictions)
        # self.inner_kernel = gpytorch.kernels.MaternKernel(nu=0.5)
        self.inner_kernel = GammaExponentialKernel(gamma=0.5)
        # self.inner_kernel = gpytorch.kernels.ScaleKernel(
        #     gpytorch.kernels.MaternKernel(nu=0.5)
        # )
        self.covar_module = PosteriorKernel(
            self.inner_kernel, nwp_gp, get_x_fun, cashing, gp_predictions)
        self.get_x_fun = get_x_fun
        self.nwp_gp = nwp_gp
        self.cashing = cashing
    def forward(self, x):
        mean_x = self.mean_module(x)
        cov_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, cov_x)

class PosteriorKernel(Kernel):
    def __init__(self, kernel, nwp_gp, get_x_fun, cashing=True, gp_predictions=None):
        """Scale kernel by product of the predicted standard deviations at the two inputs"""
        super().__init__()
        self.kernel = kernel
        self.gp = nwp_gp
        self.get_x_fun = get_x_fun
        if gp_predictions is not None:
            self.gp_predictions = gp_predictions
        else:
            self.gp_predictions = {}
        self.cashing = cashing

    def get_gp_pred(self, X):
        # Speed up GP by saving output of prior GP and saving it for future computations (training or predicting)
        mean = []
        var = []
        for i, x_i in enumerate(X):
            x_hash = hash(x_i.numpy().tostring())
            out = self.gp_predictions.get(x_hash)
            if out is not None:
                mean.append(out[0])
                var.append(out[1])
                continue
            x_in = self.get_x_fun(x_i)
            mean_i, var_i = self.gp.compiled_predict_y(x_in)
            mean_i = tf.reshape(mean_i, (-1))
            var_i = tf.reshape(var_i, (-1))
            mean.append(mean_i)
            var.append(var_i)
            self.gp_predictions[x_hash] = (mean_i, var_i)
        mean = tf.convert_to_tensor(mean)
        var = tf.convert_to_tensor(var)
        return mean, var 
    
    def forward(self, x1, x2, **params):
        k_inner = self.kernel.forward(x1, x2,**params)
        if self.cashing:
            _, var1 = self.get_gp_pred(x1)
            _, var2 = self.get_gp_pred(x2)
        else:
            _, var1 = self.gp.compiled_predict_y(self.get_x_fun(x1))
            _, var2 = self.gp.compiled_predict_y(self.get_x_fun(x2))
        var_mat = tf.linalg.matmul(var1, tf.transpose(var2)).numpy()
        k = torch.mul(torch.from_numpy(np.sqrt(var_mat)), k_inner)
        return k
    
class NWPMean(Mean):
    def __init__(self, nwp_gp, get_x_fun, cashing=True, gp_predictions = None):
        super().__init__()
        self.gp = nwp_gp
        self.get_x_fun = get_x_fun
        self.cashing = cashing
        if gp_predictions is not None:
            self.gp_predictions = gp_predictions
        else:
            self.gp_predictions = {}

    def get_gp_pred(self, X):
        # Speed up GP by saving output of prior GP and saving it for future computations
        mean = []
        var = []
        for i, x_i in enumerate(X):
            x_hash = hash(x_i.numpy().tostring())
            out = self.gp_predictions.get(x_hash)
            if out is not None:
                # use cashed predictions of GP
                mean.append(out[0])
                var.append(out[1])
                continue
            x_in = self.get_x_fun(x_i)
            mean_i, var_i = self.gp.compiled_predict_y(x_in)
            mean_i = tf.reshape(mean_i, (-1))
            var_i = tf.reshape(var_i, (-1))
            mean.append(mean_i)
            var.append(var_i)
            self.gp_predictions[x_hash] = (mean_i, var_i)
        mean = tf.convert_to_tensor(mean)
        var = tf.convert_to_tensor(var)
        return mean, var

    def forward(self, x):
        if self.cashing:
            mean, _ = self.get_gp_pred(x)
        else:
            x_in = self.get_x_fun(x)
            mean, _ = self.gp.compiled_predict_y(x_in)
        return torch.from_numpy(mean.numpy()).reshape((-1))

class GammaExponentialKernel(Kernel):
    """
    Gamma exponential covariance function k(r) = exp(-(r/l)^gamma) with 0<gamma<=2
    Compare Rasmussen & Williams p. 86
    Parameter alpha handled as in Rational Quadratic kernel from GPytorch
    """
    has_lengthscale = True
    def __init__(self, gamma=0.5, **kwargs):
        super().__init__(**kwargs)
        self.register_parameter(name='raw_gamma', 
                                parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1)))
        gamma_constraint = gpytorch.constraints.Interval(0.499,0.501)
        self.register_constraint(param_name='raw_gamma', constraint=gamma_constraint)
        self.gamma = gamma
    def forward(self, x1, x2, diag=False, **params):
        x1_ = x1.div(self.lengthscale)
        x2_ = x2.div(self.lengthscale)
        distance = self.covar_dist(x1_, x2_, diag=diag, **params)
        gamma = self.gamma
        for _ in range(1, len(distance.shape) - len(self.batch_shape)):
            gamma = gamma.unsqueeze(-1)
        return torch.exp(-distance.pow(gamma))
    @property
    def gamma(self):
        return self.raw_gamma_constraint.transform(self.raw_gamma)

    @gamma.setter
    def gamma(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_lengthscale)
        self.initialize(raw_gamma=self.raw_gamma_constraint.inverse_transform(value))