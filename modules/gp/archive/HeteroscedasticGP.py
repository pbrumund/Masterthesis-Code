from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple, TypeVar, Union
import warnings
import random

import casadi as ca
import numpy as np
from scipy import stats

from hilo_mpc.modules.machine_learning.base import LearningBase
from hilo_mpc.modules.machine_learning.gp.likelihood import Likelihood
from hilo_mpc.modules.machine_learning.gp.mean import Mean
from hilo_mpc.modules.machine_learning.gp.kernel import Kernel
from hilo_mpc.modules.machine_learning.gp.inference import Inference
from hilo_mpc.util.machine_learning import Parameter, Hyperparameter, register_hyperparameters
from hilo_mpc import GaussianProcess
from HeteroscedasticKernel import HeteroscedasticKernel

Numeric = Union[int, float]
Values = Union[Numeric, np.ndarray]
Bound = Union[Numeric, str, Tuple[Numeric], Tuple[Numeric, Numeric]]
Bounds = Dict[str, Union[str, Tuple[Numeric, Numeric]]]
Array = TypeVar('Array', np.ndarray, ca.DM, ca.SX, ca.MX)
Param = TypeVar('Param', bound=Parameter)
Inf = TypeVar('Inf', bound=Inference)
Lik = TypeVar('Lik', bound=Likelihood)
Mu = TypeVar('Mu', bound=Mean)
Cov = TypeVar('Cov', bound=Kernel)

class MostLikelyHeteroscedasticGP (GaussianProcess):
    def __init__(
            self,
            features: Union[str, list[str]],
            labels: Union[str, list[str]],
            inference: Optional[Union[str, Inf]] = None,
            likelihood: Optional[Union[str, Lik]] = None,
            mean: Optional[Mu] = None,
            kernel: Optional[Cov] = None,
            kernel_mean: Optional[Cov] = None,
            kernel_var: Optional[Cov] = None,
            noise_variance: Numeric = 1.,
            hyperprior: Optional[str] = None,  # Maybe we should be able to access other hyperpriors from here as well
            id: Optional[str] = None,
            name: Optional[str] = None,
            solver: Optional[str] = None,
            solver_options: Optional[dict] = None,  # TODO: Switch to mapping?
            **kwargs
    ):
        self._gp_log_var = GaussianProcess(features, labels, inference, likelihood, mean, kernel_var, 
                                           noise_variance, hyperprior, id, name, solver, solver_options)
        heteroscedasic_kernel = HeteroscedasticKernel(kernel_mean, self._gp_log_var, active_dims=kernel_mean.active_dims)
        self._gp_mean = GaussianProcess(features, labels, inference, likelihood, mean, heteroscedasic_kernel, 
                                        noise_variance, hyperprior, id, name, solver, solver_options)
        
    def _sparsify_training_data(self):
        n_sample = 100
        n_sample = min(n_sample, len(self._y_train))
        rand_indices = random.sample(range(len(self._y_train)), n_sample)
        self._X_train_mean = self._X_train[:, rand_indices]
        self._y_train_mean = np.reshape(self._y_train[rand_indices], (1,-1))

    def _estimate_variance(self):
        # normalize x for every dimension
        X = self._X_train
        y = self._y_train
        n_inputs = X.shape[0]
        X_norm = np.zeros(X.shape)
        var_x = np.zeros(n_inputs)
        mean_x = np.zeros(n_inputs)
        for i in range(n_inputs):
            mean_i = np.mean(X[i,:])
            var_i = np.var(X[i,:])
            if var_i != 0:
                X_norm[i,:] = (X[i,:])/np.sqrt(var_i)
                var_x[i] = var_i
            else:
                X_norm[i,:] = X[i,:]
                var_x[i] = 1
        
        #draw random points
        n_points = 100
        n_points = min(n_points, len(y))
        rand_indices = random.sample(range(len(y)), n_points)

        #find n_closest closest neighbors
        n_closest = 50
        n_closest = min(n_closest, len(y))
        X_mean = np.zeros((n_inputs, n_points))
        y_var = np.zeros(n_points)
        for i in range(n_points):
            index = rand_indices[i]
            x_i = X_norm[:,index]
            y_i = y[index]
            X_others = np.delete(X_norm,index,1)
            y_others = np.delete(y, index)
            distances = [np.linalg.norm(x_i - X_others[:,k]) for k in range(X_others.shape[1])]
            d_sort = np.argsort(distances)[:n_closest]
            X_closest = np.array([X_others[:,k] for k in d_sort]).T
            y_closest = np.array([y_others[k] for k in d_sort])
            X_mean[:,i] = np.multiply(np.mean(X_closest, axis=1), np.sqrt(var_x))
            y_var[i] = np.var(y_closest)

        self._X_train_var = X_mean
        self._mean_var = np.mean(y_var)
        self._y_train_var = np.reshape(np.log(y_var/self._mean_var), (1,-1))
        self._gp_log_var.mean_var = self._mean_var

        
    def set_training_data(self, X: np.ndarray, y: np.ndarray) -> None:
        self._X_train = X
        self._y_train = y
        self._sparsify_training_data()
        self._gp_mean.set_training_data(self._X_train_mean, self._y_train_mean)
        self._estimate_variance()
        self._gp_log_var.set_training_data(self._X_train_var, self._y_train_var)
        
    def setup(self):
        self._gp_log_var.setup()
        self._gp_log_var.fit_model()
        print('Finished setting up and training log noise GP')
        self._gp_mean.setup()
        print('Finished setup')

    def fit_model(self) -> None:
        self._gp_log_var.fit_model()
        print('Finished training log noise GP')
        self._gp_mean.fit_model()
        print('Finished training')
    
    def predict(self, X_query: Array, noise_free: bool = False) -> (Array, Array):
        mean, var = self._gp_mean.predict(X_query, noise_free=True)
        var_add = np.exp(self._gp_log_var.predict(X_query)[0])*self._mean_var
        var += var_add
        return mean, var

    def plot(self, X_query: Array, backend: Optional[str] = None, **kwargs) -> None:
        self._gp_mean.plot(X_query, backend)
        self._gp_log_var.plot(X_query, backend)