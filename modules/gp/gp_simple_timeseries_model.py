import numpy as np
import random
import datetime
import gpytorch
import torch
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt

from multiprocessing import Pool

from .data_handling import DataHandler
from .posterior_gpytorch import SimpleTimeseriesGP
from .gp_timeseries_model import WindPredictionGP

class SimpleTimeseriesModel(WindPredictionGP):
    """Generate training data for the prior GP using NWP as inputs"""
    def __init__(self, opt):
        super().__init__(opt)
        self.timeseries_gp_param = None
        self.t_last_train = None

    def get_timeseries_gp(self, prediction_time):
        """Set up a GP to predict wind speeds based on previous measurements, 
        using a simple constant prior"""
        n_last = 36
        if self.t_last_train is not None:
            i_shift = (prediction_time-self.t_last_train).total_seconds()/(self.opt['dt_meas']*60)
        else:
            i_shift = 0
        X_train = torch.arange(start=-n_last+i_shift, end=i_shift).double().reshape((-1,1))
        y_train = torch.zeros(n_last).double()
        self.X_train_timeseries = X_train
        self.y_train_timeseries = y_train
        t_start = prediction_time - n_last*datetime.timedelta(minutes=self.opt['dt_meas'])
        times = [t_start + i*datetime.timedelta(minutes=self.opt['dt_meas']) for i in range(n_last)]

        for i, t in enumerate(times):
            measurement = self.data_handler.get_measurement(t, 0)
            prediction = self.data_handler.get_NWP(t, 0)
            y_train[i] = measurement - prediction
        if self.timeseries_gp_param is None:
            self.first_train = True
        else:
            self.first_train = False
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        get_x_fun = lambda x: self._get_in(self.t_last_train, x)
        gp_timeseries = SimpleTimeseriesGP(X_train, y_train, likelihood)
        # initialize hyperparameters
        if self.first_train:
            gp_timeseries.covar_module.base_kernel.lengthscale = 20
            gp_timeseries.covar_module.outputscale = 2
            # gp_timeseries.covar_module.kernel.gamma = 0.5  
            likelihood.noise = 1e-2
        else:
            gp_timeseries.covar_module.base_kernel.lengthscale = self.timeseries_gp_param[0]
            gp_timeseries.covar_module.outputscale = self.timeseries_gp_param[1]
            # gp_timeseries.covar_module.kernel.gamma = self.timeseries_gp_param[1]  
            likelihood.noise = self.timeseries_gp_param[2] 
        return gp_timeseries, likelihood

    def train_timeseries_gp(self):
        """Optimize length scale of kernel and noise"""
        self.gp_timeseries.train()
        self.timeseries_likelihood.train()
        # Different learning rates for length and variance scale
        # optimizer_l = torch.optim.Adam(
        #     [self.gp_timeseries.covar_module.kernel.base_kernel.raw_lengthscale,
        #      self.timeseries_likelihood.raw_noise], lr=0.1
        # )
        # optimizer_sigma = torch.optim.Adam(
        #     [self.gp_timeseries.covar_module.kernel.raw_outputscale], lr=0.1
        # )
        optimizer = torch.optim.Adam(
            self.gp_timeseries.parameters(), lr=0.2)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(
            self.timeseries_likelihood, self.gp_timeseries)
        if self.first_train:
            epochs = self.opt['epochs_timeseries_first_train']
        else:
            epochs = self.opt['epochs_timeseries_retrain']
        for i in range(epochs):
            optimizer.zero_grad()
            # optimizer_l.zero_grad()
            # optimizer_sigma.zero_grad()
            output = self.gp_timeseries(self.X_train_timeseries)
            loss = -mll(output, self.y_train_timeseries)
            loss.backward()
            param_vals = [
                self.gp_timeseries.covar_module.base_kernel.lengthscale.item(),
                # self.gp_timeseries.covar_module.kernel.gamma.item(),
                self.gp_timeseries.covar_module.outputscale.item(),
                self.timeseries_likelihood.noise.item()
            ]
            if self.opt['verbose']:
                print(f'Epoch {i+1}: '
                    f'l: {param_vals[0]}, '
                    f'signal variance: {param_vals[1]}, '
                    # f'gamma: {param_vals[1]}, '
                    f'noise: {param_vals[2]}, '
                    f'loss: {loss.item()}')
            optimizer.step()
            # optimizer_l.step()
            # optimizer_sigma.step()
        self.timeseries_gp_param = param_vals

    def reset_timeseries_gp_hyperparameters(self):
        if hasattr(self, 'gp_timeseries'):
            self.gp_timeseries.covar_module.base_kernel.lengthscale = 10
            self.gp_timeseries.covar_module.outputscale = 1    
            self.timeseries_likelihood.noise = 1e-2
            # self.gp_timeseries.covar_module.kernel.gamma = 0.5
    
    def predict_trajectory(self, start_time, steps, train=False, pseudo_gp = None, 
                           include_last_measurement=True):
        """
        Set up the timeseries GP and train if required, then predict a number of steps ahead
        returns mean and variance as numpy arrays
        """
        if include_last_measurement:
            # include last measurement in prediction for exact first value by shifting time and indices
            start_time_gp = start_time-datetime.timedelta(minutes=self.opt['dt_meas'])
        else:
            start_time_gp = start_time
        start_time_gp = start_time
        dt = 0  # if start time is not multiple of 10 min, difference to last previous multiple to shift indices#
        if start_time_gp.minute%self.opt['dt_meas'] != 0:
            dt = start_time_gp.minute%self.opt['dt_meas']
            start_time_gp = start_time_gp.replace(
                minute=start_time_gp.minute//self.opt['dt_meas']*self.opt['dt_meas'])
        if pseudo_gp is None:
            if train or self.t_last_train is None:
                self.gp_predictions = None
                self.t_last_train = start_time_gp
                self.gp_timeseries, self.timeseries_likelihood = self.get_timeseries_gp(
                    prediction_time=start_time_gp)
                self.reset_timeseries_gp_hyperparameters()
                self.train_timeseries_gp()
            else:
                self.gp_timeseries, self.timeseries_likelihood = self.get_timeseries_gp(
                prediction_time=start_time_gp)
            gp_timeseries = self.gp_timeseries
        else:
            gp_timeseries = pseudo_gp
        self.gp_timeseries.eval()
        self.timeseries_likelihood.eval()
        # shift inputs by 1 for each 10 minute step after training so 0 stays training time
        i_shift = (start_time_gp-self.t_last_train).total_seconds()/(self.opt['dt_meas']*60)
        dt_factor = self.opt['dt_pred']/self.opt['dt_meas']
        # input: number of 10 minute steps since last training, 
        # non-integer possible for times that are not multiples of 10 minutes
        x = np.arange(steps)*dt_factor+i_shift+dt/self.opt['dt_meas']
        x = torch.from_numpy(x.reshape((-1,1)).astype(float))
        if include_last_measurement:
            x = x-1
        gp_pred_y = self.timeseries_likelihood(gp_timeseries(x))
        gp_mean, gp_var = gp_pred_y.mean, gp_pred_y.variance
        if include_last_measurement:
            gp_mean[0] = self.data_handler.generate_labels(start_time, steps_ahead=0)
            gp_var[0] = 0
        # add NWP to get predicted value from prediction error
        NWP_pred = [self.data_handler.get_NWP(start_time, i*dt_factor) for i in range(steps)]
        gp_pred = np.array(NWP_pred).reshape((-1,1)) + gp_mean.reshape((-1,1)).detach().numpy()
        
        return gp_pred[:,0], gp_var.detach().numpy()
    

    def plot_posterior(self, start_time, steps, train=False):
        gp_pred_traj, gp_var = self.predict_trajectory(start_time, steps, train)
        times = [start_time+i*datetime.timedelta(minutes=self.opt['dt_pred']) for i in range(steps)]
        
        NWP_traj = np.zeros(steps)
        meas_traj = np.zeros(steps)

        for i, t in enumerate(times):
                NWP_traj[i] = self.data_handler.get_NWP(
                    start_time, i*self.opt['dt_pred']/self.opt['dt_meas'])
                meas_traj[i] = self.data_handler.get_measurement(t, 0)

        std_traj = np.sqrt(gp_var)

        plt.figure()
        plt.plot(times, NWP_traj, color='tab:orange')
        plt.plot(times, meas_traj, color='tab:green')
        plt.plot(times, gp_pred_traj, color='tab:blue')
        plt.fill_between(times, gp_pred_traj-2*std_traj, gp_pred_traj+2*std_traj, color='lightgray')
        plt.fill_between(times, gp_pred_traj-std_traj, gp_pred_traj+std_traj, color='tab:gray')
        plt.xlabel('time')
        plt.ylabel('wind speed')
        plt.legend(['Weather prediction', 'Actual wind speed', 'Timeseries GP prediction'])
        # plt.figure()
        # plt.plot(times, std_traj)
        # plt.xlabel('time')
        # plt.ylabel('predicted uncertainty')

    
# if __name__ == '__main__':
#     from get_gp_opt import get_gp_opt
#     opt = get_gp_opt(n_z = 200, cashing=True)
    
#     gp = TimeseriesModel(opt)
#     t_start_predict = datetime.datetime(2022,8,1,1)
#     steps = 120
#     # gp.plot_prior(t_start_predict, steps)
#     gp.plot_posterior(t_start_predict, steps, train=True)
#     for i in range(1,10):
#         gp.plot_posterior(t_start_predict + i*datetime.timedelta(minutes=5), steps, train=False)
#     # t_start_predict = datetime.datetime(2022,1,1,1)
#     # steps = 60
#     # gp.plot_prior(t_start_predict, steps)
#     # gp.plot_posterior(t_start_predict, steps, train=False)
#     # gp.plot_prior_distribution(datetime.datetime(2022,1,1), datetime.datetime(2022,12,31))
#     # gp.plot_prior(datetime.datetime(2022,1,1), 365*24*6-1)
#     plt.show()
#     pass