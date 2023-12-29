from typing import Optional, Tuple
import gpytorch
from gpytorch.constraints import Interval
from gpytorch.priors import Prior
import torch
from utils import sparsify_training_data
class ExactGP(gpytorch.models.ExactGP):
        def __init__(self, X_train, y_train, likelihood, opt):
            super(ExactGP, self).__init__(X_train, y_train, likelihood)
            n_dims = X_train.shape[1]
            if opt['label'] == 'measurement' and isinstance(likelihood, gpytorch.likelihoods.FixedNoiseGaussianLikelihood):
                 self.mean_module = NWPMean(opt['order'])
            else:
                self.mean_module = gpytorch.means.ZeroMean()
            
            self.covar_module = gpytorch.kernels.ScaleKernel(
                  gpytorch.kernels.RQKernel(ard_num_dims=n_dims))
            self.covar_module.base_kernel.lengthscale = torch.tensor([.1]*n_dims)

        def forward(self, x):
              mean_x = self.mean_module(x)
              cov_x = self.covar_module(x)
              return gpytorch.distributions.MultivariateNormal(mean_x, cov_x)

class SparseGP(gpytorch.models.ExactGP):
    def __init__(self, X_train, y_train, likelihood, opt):
        super(SparseGP, self).__init__(X_train, y_train, likelihood)
        n_dims = X_train.shape[1]
        if opt['label'] == 'measurement':
                self.mean_module = NWPMean(opt['order'])
        else:
            self.mean_module = gpytorch.means.ConstantMean()
        self.base_covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_num_dims=n_dims))
        self.covar_module = gpytorch.kernels.InducingPointKernel(
                self.base_covar_module,
                inducing_points=sparsify_training_data(X_train, y_train, opt['n_z'])[0].clone(),
                likelihood=likelihood
        )
        # self.covar_module.base_kernel.lengthscale = torch.tensor([5]*n_dims)

    def forward(self, x):
        mean_x = self.mean_module(x)
        cov_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, cov_x)

class SKIPGP(gpytorch.models.ExactGP):
    def __init__(self, X_train, y_train, likelihood, opt):
        super(SKIPGP, self).__init__(X_train, y_train, likelihood)
        n_dims = X_train.shape[1]
        if opt['label'] == 'measurement':
                self.mean_module = NWPMean(opt['order'])
        else:
            self.mean_module = gpytorch.means.ConstantMean()
        self.base_covar_module = gpytorch.kernels.RBFKernel()
        self.covar_module = gpytorch.kernels.ProductStructureKernel(
             gpytorch.kernels.ScaleKernel(
                  gpytorch.kernels.GridInterpolationKernel(
                       self.base_covar_module, grid_size=100, num_dims=1)
             ), num_dims=n_dims
        )
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        cov_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, cov_x)
    
class HeteroscedasticGP(gpytorch.models.ExactGP):
    def __init__(self, X_train, y_train, likelihood, noise_gp, opt):
        super(HeteroscedasticGP, self).__init__(X_train, y_train, likelihood)
        n_dims = X_train.shape[1]
        if opt['label'] == 'measurement':
                self.mean_module = NWPMean(opt['order'])
        else:
            self.mean_module = gpytorch.means.ConstantMean()
        self.bas_covar_module = gpytorch.kernels.ScaleKernel(
                  gpytorch.kernels.RQKernel(ard_num_dims=n_dims))
        
        self.noise_gp = noise_gp
        self.covar_module = HeteroscedasticKernel(
             kernel=self.bas_covar_module, gp=self.noise_gp, X_train = X_train)

    def forward(self, x):
        mean_x = self.mean_module(x)
        cov_x = self.covar_module(x)
        if x==self.train_inputs:
             print('x_in is X_train')

        return gpytorch.distributions.MultivariateNormal(mean_x, cov_x)

class NWPMean(gpytorch.means.Mean):
    def __init__(self, order):
        super().__init__()
        self.order = order

    def forward(self, x):
         return x[:,self.order]
    
class HeteroscedasticKernel(gpytorch.kernels.Kernel):
    def __init__(self, kernel, gp, X_train, **kwargs):
        super().__init__(**kwargs)
        self.kernel = kernel
        self.gp = gp
        self.X_train = X_train
        self.noise_train = gp(X_train).mean


    def forward(self, x1, x2, **params):
        k =  self.kernel.forward(x1, x2, **params)
        if torch.equal(x1, self.X_train) and torch.equal(x2, self.X_train):
             # print('Training inputs')
             k += torch.diag(self.noise_train)
        elif torch.isequal(x1, x2):
             k += self.gp(x1).mean
        return k
