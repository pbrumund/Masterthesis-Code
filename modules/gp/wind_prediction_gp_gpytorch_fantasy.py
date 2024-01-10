from gpflow.base import TensorType
import numpy as np
import random
import datetime
import gpytorch
import torch
import gpflow as gpf
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import sklearn

from multiprocessing import Pool

import utils
from fileloading import load_weather_data
from posterior_kernel import PosteriorKernel, PosteriorMean
from posterior_gpytorch import TimeseriesGP


class WindPredictionGP:
    def __init__(self, opt):
        t_start = opt['t_start']
        t_end = opt['t_end']
        self.weather_data = load_weather_data(t_start, t_end)
        self.opt = opt

    def get_training_data(self, opt):
        filename_X = opt['filename_x']# f'gp\\training_data\X_train_prior.txt'
        filename_y = opt['filename_y']# f'gp\\training_data\y_train_prior.txt'
        try:
            # If training data has been generated before, load it from file
            X_train = np.loadtxt(filename_X)
            y_train = np.loadtxt(filename_y).reshape((-1,1))
            print('loaded data from file')
        except:
            # generate training data
            print('generating data')
            start_datetime = self.opt['start_date_train']
            end_datetime = self.opt['end_date_train']
            n_points = int((end_datetime-start_datetime)/datetime.timedelta(minutes=10))
            times = [start_datetime + i*datetime.timedelta(minutes=10) for i in range(n_points)]
            n_x = utils.generate_features(
                self.weather_data, start_datetime, 1, opt['input_feature'], 0).shape[0]
            steps_ahead = opt['steps_ahead']
            
            args_X = [(self.weather_data, time, 0, opt['input_feature'], steps_ahead) 
                        for time in times]# for steps_ahead in range(1,max_steps_ahead)]
            args_y = [(self.weather_data, time, opt['label'], steps_ahead)
                        for time in times]# for steps_ahead in range(1,max_steps_ahead)]
            with Pool(processes=12) as pool:
                X_train = pool.starmap(utils.generate_features, args_X, chunksize=1000)
                print('finished generating X_train')
                y_train = pool.starmap(utils.generate_labels, args_y, chunksize=1000)
                print('finished generating y_train')
                X_train = np.array(X_train).reshape((n_points, n_x))
                y_train = np.array(y_train).reshape((n_points, 1))

            
            # Save to file
            np.savetxt(filename_X, X_train)
            np.savetxt(filename_y, y_train)
        return X_train, y_train

class TimeseriesModel(WindPredictionGP):
    """Generate training data for the prior GP using NWP as inputs"""
    def __init__(self, opt):
        super().__init__(opt)
        self.X_train, self.y_train = self.get_training_data_prior()
        self.gp_prior, is_trained = self.get_prior_gp(self.X_train, self.y_train)
        if not is_trained:
            self.train_prior_gp(self.X_train, self.y_train)
        self.timeseries_gp_param = None


    def get_training_data_prior(self):
        opt = {}
        opt['filename_x'] = f'gp\\training_data\X_train_prior.txt'
        opt['filename_y'] = f'gp\\training_data\y_train_prior.txt'
        opt['input_feature'] = 'nwp & time'
        opt['label'] = 'error'
        opt['steps_ahead'] = 0

        return super().get_training_data(opt)  
        
    def get_prior_gp(self, X_train=None, y_train=None):
        """Get a heteroscedastic SVGP model to predict the prediction error/residuals 
        and uncertainty based on NWP values"""
        self.filename_gp = f'gp\models\gp_prior_{self.opt["n_z"]}'
        try:
            gp_prior = tf.saved_model.load(self.filename_gp)
            print('loaded gp from file')
            return gp_prior, True
        except:
            if X_train is None:
                raise RuntimeError('tried to fit gp without providing training data')
        print(f'training gp for prior mean and variance')
        # X_train, y_train = self.get_training_data_prior()
        n_inputs = X_train.shape[1]
        n_samples = X_train.shape[0]

        likelihood = gpf.likelihoods.HeteroskedasticTFPConditional(
            scale_transform=tfp.bijectors.Exp())
        
        kernels_nwp_mean = gpf.kernels.Sum([
            gpf.kernels.RationalQuadratic(lengthscales=[1], active_dims=[i]) for i in range(n_inputs-1)
        ])
        kernels_nwp_var = gpf.kernels.Sum([
            gpf.kernels.RationalQuadratic(lengthscales=[.3], active_dims=[i]) for i in range(n_inputs-1)
        ])

        # kernels_nwp_var = gpf.kernels.RationalQuadratic(
        #     lengthscales=[.1]*(n_inputs-1), active_dims=range(n_inputs-1))

        kernel_mean = (
            # gpf.kernels.RationalQuadratic(
            #     lengthscales=[1]*(n_inputs-1), active_dims = range(n_inputs-1)) 
            kernels_nwp_mean
            + gpf.kernels.Periodic(
                gpf.kernels.SquaredExponential(active_dims=[n_inputs-1]), period=365) 
            + gpf.kernels.Periodic(
                gpf.kernels.SquaredExponential(active_dims=[n_inputs-1]), period=1))
        gpf.set_trainable(kernel_mean.submodules[6].period, False)
        gpf.set_trainable(kernel_mean.submodules[7].period, False)
        kernel_var = (
            # gpf.kernels.RationalQuadratic(
            #    lengthscales=[.1]*(n_inputs-1), active_dims = range(n_inputs-1)) 
            kernels_nwp_var
            + gpf.kernels.Periodic(
                gpf.kernels.SquaredExponential(active_dims=[n_inputs-1]), period=365) 
            + gpf.kernels.Periodic(
                gpf.kernels.SquaredExponential(active_dims=[n_inputs-1]), period=1))
        gpf.set_trainable(kernel_var.submodules[6].period, False)
        gpf.set_trainable(kernel_var.submodules[7].period, False)
        kernel = gpf.kernels.SeparateIndependent(
            [
                kernel_mean,
                kernel_var
            ]
        )
       
        A = np.zeros((n_inputs, 2))
        b = np.zeros((2))
        
        class LinearMeanNWP(gpf.functions.Linear):
            def __call__(self, X: TensorType):
                return tf.tensordot(X[...,:-1], self.A[:-1,:], [[-1], [0]]) + self.b

        class LinearMeanNWPMean(gpf.functions.Linear):
            def __call__(self, X: TensorType):
                return tf.tensordot(X[...,0,:-1], self.A[:-1,1], [[-1], [0]]) + self.b
        
        # mean = LinearMeanNWP(A, b)
        # mean = LinearMeanNWPMean(A, b)
        mean = gpf.functions.Constant(np.zeros(2))
        # mean = gpf.functions.Zero()
        
        n_z = self.opt['n_z']
        i_Z1 = random.sample(range(n_samples), n_z)
        i_Z2 = random.sample(range(n_samples), n_z)
        Z1 = X_train[i_Z1, :]
        Z2 = X_train[i_Z2, :]

        inducing_variable = gpf.inducing_variables.SeparateIndependentInducingVariables(
            [
                gpf.inducing_variables.InducingPoints(Z1),  
                gpf.inducing_variables.InducingPoints(Z2), 
            ]
        )

        # _, var_0 = utils.estimate_variance(X_train, y_train, indices=i_Z2)
        # q_mu_mean = tf.zeros((n_z, 1))
        # q_mu_var = tf.convert_to_tensor(var_0, dtype=float)
        # q_mu = tf.concat([q_mu_mean, q_mu_var], axis=1)
        # self.q_mu_init = q_mu
        
        gp_prior = gpf.models.SVGP(
            kernel=kernel, 
            likelihood=likelihood, 
            inducing_variable=inducing_variable,
            num_latent_gps=likelihood.latent_dim,
            mean_function=mean,
            # q_mu=q_mu
            # q_diag=True
            )
        return gp_prior, False
        
    def get_training_step(self, X_train, y_train):
        loss_fn = self.gp_prior.training_loss_closure(
            (X_train, y_train))
        variational_vars = [(self.gp_prior.q_mu, self.gp_prior.q_sqrt)]
        natgrad_opt = gpf.optimizers.NaturalGradient(gamma=0.1)

        adam_vars = self.gp_prior.trainable_variables
        adam_opt = tf.optimizers.Adam(0.1)

        config = gpf.config.Config(jitter=1e-4)
        with gpf.config.as_context(config):
            @tf.function
            def training_step():
                natgrad_opt.minimize(loss_fn, variational_vars)
                adam_opt.minimize(loss_fn, adam_vars)

        return training_step, loss_fn

    def train_prior_gp(self, X_train, y_train):
        """Train the GP"""
        # Train on subset of data
        reselect_data = self.opt['reselect_data']
        n_inputs = X_train.shape[1]
        n_samples = X_train.shape[0]
        if reselect_data:
            n_train_1 = int(X_train.shape[0]/100)
        else:
            n_train_1 = int(X_train.shape[0]/10)
            training_subset = random.sample(range(n_samples), n_train_1)
            loss_fn = self.gp_prior.training_loss_closure(
                (X_train[training_subset,:], y_train[training_subset,:]))

        gpf.utilities.set_trainable(self.gp_prior.q_mu, False)
        gpf.utilities.set_trainable(self.gp_prior.q_sqrt, False)

        config = gpf.config.Config(jitter=1e-4)
        with gpf.config.as_context(config):
            if reselect_data:
                training_subset = random.sample(range(n_samples), n_train_1)
                training_step, loss_fn = self.get_training_step(
                    X_train[training_subset,:], y_train[training_subset,:])
            else:
                @tf.function
                def training_step():
                    natgrad_opt.minimize(loss_fn, variational_vars)
                    adam_opt.minimize(loss_fn, adam_vars)
            
            max_epochs = self.opt['epochs_first_training']
            loss_lb = self.opt['loss_lb']
            
            for i in range(max_epochs+1):
                try:
                    training_step()
                except:
                    raise RuntimeError('Failed to train model')
                if loss_fn().numpy() < loss_lb:
                    break
                if self.opt['verbose']:#and i%20==0:
                    print(f"Epoch {i} - Loss: {loss_fn().numpy() : .4f}")
                if (i+1)%10==0 and reselect_data:
                    training_subset = random.sample(range(n_samples), n_train_1)
                    training_step, loss_fn = self.get_training_step(
                        X_train[training_subset,:], y_train[training_subset,:])
                
        
        # Second training on full data set
        variational_vars = [(self.gp_prior.q_mu, self.gp_prior.q_sqrt)]
        natgrad_opt = gpf.optimizers.NaturalGradient(gamma=0.5)

        adam_vars = self.gp_prior.trainable_variables
        adam_opt = tf.optimizers.Adam(0.1)
        loss_fn = self.gp_prior.training_loss_closure((X_train, y_train))
        

        config = gpf.config.Config(jitter=1e-2)
        with gpf.config.as_context(config):
            @tf.function
            def training_step():
                natgrad_opt.minimize(loss_fn, variational_vars)
                adam_opt.minimize(loss_fn, adam_vars)
            
            max_epochs = self.opt['max_epochs_second_training']
            loss_lb = self.opt['loss_lb']
            for i in range(max_epochs+1):
                try:
                    training_step()
                except:
                    print('Likelihood is nan')
                    raise RuntimeError('Failed to train model')
                if loss_fn().numpy() < loss_lb:
                    break
                if self.opt['verbose']:# and i%20==0:
                    print(f"Epoch {i} - Loss: {loss_fn().numpy() : .4f}")

        # save gp
        self.gp_prior.compiled_predict_f = tf.function(
            lambda x: self.gp_prior.predict_f(x),
            input_signature=[tf.TensorSpec(shape=[None, n_inputs], dtype=tf.float64)]
        )
        self.gp_prior.compiled_predict_y = tf.function(
            lambda x: self.gp_prior.predict_y(x),
            input_signature=[tf.TensorSpec(shape=[None, n_inputs], dtype=tf.float64)]
        )

        tf.saved_model.save(self.gp_prior, self.filename_gp)

    def _get_in(self, time, steps):
        """get the inputs of the first GP for a given timestamp or vector, used by tensorflow model"""
        if steps.shape[0] != 1:
            # recursively get the input for each step
            x = []
            for i in range(steps.shape[0]):
                x.append(self._get_in(time, steps[i]).reshape(-1))
            return tf.convert_to_tensor(np.array(x))

        s = tf.get_static_value(steps, partial=True)
        steps = int(s[0])
        
        if steps < 0:
            # time before current time, training
            t = time+steps*datetime.timedelta(minutes=10)
            x = utils.generate_features(
                self.weather_data, t, feature='nwp & time', steps_ahead=0).reshape((1,-1))
        else:
            # time after current time, predicting
            x = utils.generate_features(
                self.weather_data, time, feature='nwp & time', steps_ahead=steps).reshape((1,-1))
        return x
    
    def get_timeseries_gp(self, prediction_time):
        """Set up a GP to predict wind speeds based on previous measurements, 
        using the GP based on NWP data as a prior"""
        n_last = self.opt['n_last']
        X_train = torch.arange(start=-n_last, end=0).double().reshape((-1,1))
        y_train = torch.zeros(n_last).double()
        self.X_train_timeseries = X_train
        self.y_train_timeseries = y_train
        t_start = prediction_time - n_last*datetime.timedelta(minutes=10)
        times = [t_start + i*datetime.timedelta(minutes=10) for i in range(n_last)]

        for i, t in enumerate(times):
            measurement = utils.get_wind_value(self.weather_data, t, 0)
            prediction = utils.get_NWP(self.weather_data, t, 0)
            y_train[i] = measurement - prediction
        if self.timeseries_gp_param is None:
            self.first_train = True
        else:
            self.first_train = False
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        get_x_fun = lambda x: self._get_in(prediction_time, x)
        gp_posterior = TimeseriesGP(
            X_train, y_train, likelihood, self.gp_prior, get_x_fun, cashing=self.opt['cashing'])
        # initialize hyperparameters
        if self.first_train:
            gp_posterior.covar_module.kernel.base_kernel.lengthscale = 20
            gp_posterior.covar_module.kernel.outputscale = 1    
            likelihood.noise = 1e-2
        else:
            gp_posterior.covar_module.kernel.base_kernel.lengthscale = self.timeseries_gp_param[0]
            gp_posterior.covar_module.kernel.outputscale = self.timeseries_gp_param[1]   
            likelihood.noise = self.timeseries_gp_param[2]
        self.t_start_timeseries = prediction_time
        return gp_posterior, likelihood

    def train_timeseries_gp(self):
        self.gp_timeseries.train()
        self.timeseries_likelihood.train()
        # Different learning rates for length and variance scale
        optimizer_l = torch.optim.Adam(
            [self.gp_timeseries.covar_module.kernel.base_kernel.raw_lengthscale,
             self.timeseries_likelihood.raw_noise], lr=0.1
        )
        optimizer_sigma = torch.optim.Adam(
            [self.gp_timeseries.covar_module.kernel.raw_outputscale], lr=0.1
        )
        # optimizer = torch.optim.Adam(
        #     self.gp_timeseries.parameters(), lr=0.1)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(
            self.timeseries_likelihood, self.gp_timeseries)
        if self.first_train:
            epochs = self.opt['epochs_timeseries_first_train']
        else:
            epochs = self.opt['epochs_timeseries_retrain']
        for i in range(epochs):
            # optimizer.zero_grad()
            optimizer_l.zero_grad()
            optimizer_sigma.zero_grad()
            output = self.gp_timeseries(self.X_train_timeseries)
            loss = -mll(output, self.y_train_timeseries)
            loss.backward()
            param_vals = [
                self.gp_timeseries.covar_module.kernel.base_kernel.lengthscale.item(),
                self.gp_timeseries.covar_module.kernel.outputscale.item(),
                self.timeseries_likelihood.noise.item()
            ]
            print(f'Epoch {i+1}: '
                  f'l: {param_vals[0]}, '
                  f'variance scale: {param_vals[1]}, '
                  f'noise: {param_vals[2]}, '
                  f'loss: {loss.item()}')
            # optimizer.step()
            optimizer_l.step()
            optimizer_sigma.step()
        self.timeseries_gp_param = param_vals
        
    
    def predict_trajectory(self, start_time, steps, train=False):
        """Set up the timeseries GP and train if required, otherwise add new training inputs,
          then predict a number of steps ahead"""
        if train:
            self.gp_timeseries, self.timeseries_likelihood = self.get_timeseries_gp(
                prediction_time=start_time)
            self.train_timeseries_gp()
            # timeseries_gp = self.gp_timeseries
        else:
            x_new = (start_time-self.t_start_timeseries).total_seconds()/(opt['dt_meas']*60) - 1
            y_new = utils.generate_labels(
                self.weather_data, start_time-datetime.timedelta(minutes=opt['dt_meas']), 
                steps_ahead=0)
            x_new = torch.tensor([x_new]).reshape((-1,1))
            y_new = torch.tensor([y_new])
            self.gp_timeseries = self.gp_timeseries.get_fantasy_model(x_new, y_new)
            
        self.gp_timeseries.eval()
        self.timeseries_likelihood.eval()
        x = np.arange(steps).reshape((-1,1)).astype(float)
        x = torch.from_numpy(x)
        gp_pred_y = self.timeseries_likelihood(self.gp_timeseries(x))
        gp_mean, gp_var = gp_pred_y.mean, gp_pred_y.variance
        # gp_mean, gp_var = self.gp_timeseries.predict_y(x)
        NWP_pred = [utils.get_NWP(self.weather_data, start_time, i) for i in range(steps)]
        gp_pred = np.array(NWP_pred).reshape((-1,1)) + gp_mean.reshape((-1,1)).detach().numpy()
        return gp_pred[:,0], gp_var.detach().numpy()
    
    def plot_prior(self, start_time, steps):
        times = [start_time+i*datetime.timedelta(minutes=10) for i in range(steps)]

        NWP_traj = np.zeros(steps)
        mean_traj = np.zeros(steps)
        var_traj = np.zeros(steps)
        noise_var_traj = np.zeros(steps)
        meas_traj = np.zeros(steps)

        for i, t in enumerate(times):
            NWP_traj[i] = utils.get_NWP(self.weather_data, t, 0)
            meas_traj[i] = utils.get_wind_value(self.weather_data, t, 0)
            x = utils.generate_features(self.weather_data, t, feature='nwp & time').reshape((1,-1))
            mean, var = self.gp_prior.compiled_predict_y(x)
            mean_f, _ = self.gp_prior.compiled_predict_f(x)
            mean_traj[i] = mean
            var_traj[i] = var
            noise_var_traj[i] = np.exp(mean_f.numpy()[0,1])**2

        gp_pred_traj = NWP_traj + mean_traj
        signal_var_traj = var_traj-noise_var_traj
        std_traj = np.sqrt(var_traj)
        

        plt.figure()
        plt.plot(times, NWP_traj, color='tab:orange')
        plt.plot(times, meas_traj, color='tab:green')
        plt.plot(times, gp_pred_traj, color='tab:blue')
        plt.fill_between(times, gp_pred_traj-2*std_traj, gp_pred_traj+2*std_traj, color='lightgray')
        plt.fill_between(times, gp_pred_traj-std_traj, gp_pred_traj+std_traj, color='tab:gray')
        plt.xlabel('time')
        plt.ylabel('wind speed')
        plt.legend(['Weather prediction', 'Actual wind speed', 'Prior GP prediction'])
        plt.figure()
        plt.plot(times, std_traj)
        plt.plot(times, np.sqrt(signal_var_traj))
        plt.plot(times, np.sqrt(noise_var_traj))
        plt.xlabel('time')
        plt.ylabel('predicted uncertainty')
        plt.legend(['total uncertainty', 'uncertainty of mean', 'estimated heteroscedastic noise'])

    def plot_prior_distribution(self, start_time, stop_time):
        steps = int((stop_time - start_time).total_seconds()/(60*10))
        times = [start_time+i*datetime.timedelta(minutes=10) for i in range(steps)]

        x = utils.generate_features(self.weather_data, start_time, feature='nwp & time').reshape((1,-1))
        n_inputs = x.shape[1]

        x_mat = np.zeros((steps, n_inputs))
        mean_traj = np.zeros(steps)
        var_traj = np.zeros(steps)

        for i, t in enumerate(times):
            x = utils.generate_features(self.weather_data, t, feature='nwp & time').reshape((1,-1))
            x_mat[i,:] = x  
            mean, var = self.gp_prior.compiled_predict_y(x)
            mean_traj[i] = mean
            var_traj[i] = np.sqrt(var)
        
        fig, ax = plt.subplots(2, int(np.ceil(n_inputs/2)))
        xlabels = ['wind prediction', 'wind speed of gust', 'sqrt(CAPE)', 'temperature', 
                   'humidity', 'pressure', 'time']
        for input_dim in range(n_inputs):
            ix = input_dim%ax.shape[1]
            iy = input_dim//ax.shape[1]
            ax[iy,ix].scatter(x_mat[:, input_dim], mean_traj)
            ax[iy,ix].scatter(x_mat[:, input_dim], var_traj)
            ax[iy,ix].set_xlabel(xlabels[input_dim])
            ax[iy,ix].set_ylabel('predicted error')

        n_bins = 50
        means = np.zeros((n_bins, n_inputs))
        vars = np.zeros((n_bins, n_inputs))
        input_means = np.zeros((n_bins, n_inputs))
        for input_dim in range(n_inputs):
            x_min = min(x_mat[:, input_dim])
            x_max = max(x_mat[:,input_dim])
            bin_width = (x_max - x_min)/n_bins
            for bin in range(n_bins):
                inputs = [i for i in range(steps) 
                          if x_mat[i, input_dim] >= x_min+bin*bin_width 
                          and x_mat[i, input_dim] <= x_min + (bin+1)*bin_width]
                means_in_bin = mean_traj[inputs]
                vars_in_bin = var_traj[inputs]
                means[bin, input_dim] = np.mean(means_in_bin)
                vars[bin, input_dim] = np.mean(vars_in_bin)
                input_means[bin, input_dim] = np.mean(x_mat[inputs, input_dim])
        fig, ax = plt.subplots(2, int(np.ceil(n_inputs/2)))
        for input_dim in range(n_inputs):
            ix = input_dim%ax.shape[1]
            iy = input_dim//ax.shape[1]
            ax[iy,ix].plot(input_means[:,input_dim], means[:,input_dim])
            ax[iy,ix].plot(input_means[:,input_dim], vars[:,input_dim])
            ax[iy,ix].set_xlabel(xlabels[input_dim])
            ax[iy,ix].set_ylabel('predicted error')

    def plot_posterior(self, start_time, steps, train=False):
        gp_pred_traj, gp_var = self.predict_trajectory(start_time, steps, train)
        times = [start_time+i*datetime.timedelta(minutes=10) for i in range(steps)]
        
        NWP_traj = np.zeros(steps)
        meas_traj = np.zeros(steps)

        for i, t in enumerate(times):
                NWP_traj[i] = utils.get_NWP(self.weather_data, t, 0)
                meas_traj[i] = utils.get_wind_value(self.weather_data, t, 0)

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
        plt.figure()
        plt.plot(times, std_traj)
        plt.xlabel('time')
        plt.ylabel('predicted uncertainty')

    
if __name__ == '__main__':
    from get_gp_opt import get_gp_opt
    opt = get_gp_opt(n_z = 200, cashing=True)
    
    gp = TimeseriesModel(opt)
    t_start_predict = datetime.datetime(2022,8,1,1)
    steps = 60
    # gp.plot_prior(t_start_predict, steps)
    gp.plot_posterior(t_start_predict, steps, train=True)
    for i in range(1,10):
        gp.plot_posterior(t_start_predict + i*datetime.timedelta(minutes=10), steps, train=False)
    # t_start_predict = datetime.datetime(2022,1,1,1)
    # steps = 60
    # gp.plot_prior(t_start_predict, steps)
    # gp.plot_posterior(t_start_predict, steps, train=False)
    # gp.plot_prior_distribution(datetime.datetime(2022,1,1), datetime.datetime(2022,12,31))
    # gp.plot_prior(datetime.datetime(2022,1,1), 365*24*6-1)
    plt.show()
    pass