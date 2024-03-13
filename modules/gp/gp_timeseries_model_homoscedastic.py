import numpy as np
import random
import datetime
import gpytorch
import torch
import gpflow as gpf
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt

from multiprocessing import Pool

from .data_handling import DataHandler
from .posterior_gpytorch import TimeseriesGP
from .gp_timeseries_model import TimeseriesModel


class WindPredictionGP:
    def __init__(self, opt):
        t_start = opt['t_start']
        t_end = opt['t_end']
        self.data_handler = DataHandler(t_start, t_end, opt)
        self.opt = opt

    def get_training_data(self, opt):
        filename_X = opt['filename_x']# f'gp\\training_data\X_train_prior.txt'
        filename_y = opt['filename_y']# f'gp\\training_data\y_train_prior.txt'
        try:
            # If training data has been generated before, load it from file
            X_train = np.loadtxt(filename_X)
            y_train = np.loadtxt(filename_y).reshape((-1,1))
            if self.opt['verbose']:
                print('loaded data from file')
        except:
            # generate training data
            if self.opt['verbose']:
                print('generating data')
            start_datetime = self.opt['start_date_train']
            if '3 steps' in opt['input_feature']:
                start_datetime += datetime.timedelta(hours=1)
            end_datetime = self.opt['end_date_train']
            n_points = int((end_datetime-start_datetime)/datetime.timedelta(minutes=self.opt['dt_meas']))
            n_last = opt.get('n_last') # last measurements for autoregressive models
            if n_last is None: n_last = 0
            steps_ahead = opt['steps_ahead']
            times = [start_datetime + i*datetime.timedelta(minutes=self.opt['dt_meas']) 
                     for i in range(steps_ahead+n_last, n_points)]
            # n_points = len(times)
            n_x = self.data_handler.generate_features(
                times[0], n_last, opt['input_feature'], 0).shape[0]
            
            args_X = [(time, n_last, opt['input_feature'], steps_ahead) 
                        for time in times]# for steps_ahead in range(1,max_steps_ahead)]
            args_y = [(time, opt['label'], steps_ahead)
                        for time in times]# for steps_ahead in range(1,max_steps_ahead)]
            if opt['steps_ahead'] == 0:
                # add training data for 6 hour horizon
                args_X = args_X + [(time, n_last, opt['input_feature'], 36) 
                        for time in times]
                args_y = args_y + [(time, opt['label'], 36) for time in times]
            with Pool(processes=12) as pool:
                X_train = pool.starmap(self.data_handler.generate_features, args_X, chunksize=1000)
                if self.opt['verbose']:
                    print('finished generating X_train')
                y_train = pool.starmap(self.data_handler.generate_labels, args_y, chunksize=1000)
                n_points = len(y_train)
                if self.opt['verbose']:
                    print('finished generating y_train')
                X_train = np.array(X_train).reshape((n_points, n_x))
                y_train = np.array(y_train).reshape((n_points, 1))

            
            # Save to file
            np.savetxt(filename_X, X_train)
            np.savetxt(filename_y, y_train)
        return X_train, y_train

class HomoscedasticTimeseriesModel(TimeseriesModel):
    """Generate training data for the prior GP using NWP as inputs"""
    def __init__(self, opt):
        super().__init__(opt)
        self.X_train, self.y_train = self.get_training_data_prior()
        self.gp_prior, is_trained = self.get_prior_gp(self.X_train, self.y_train)
        if not is_trained:
            self.train_prior_gp(self.X_train, self.y_train)
        self.timeseries_gp_param = None
        self.gp_predictions = None
        self.t_last_train = None


    def get_training_data_prior(self):
        opt = {}
        opt['filename_x'] = f'modules/gp/training_data/X_train_prior.txt'
        opt['filename_y'] = f'modules/gp/training_data/y_train_prior.txt'
        opt['input_feature'] = 'nwp & time'
        opt['label'] = 'error'
        opt['steps_ahead'] = 0

        return super().get_training_data(opt)  
        
    def get_prior_gp(self, X_train=None, y_train=None):
        """
        Get a heteroscedastic SVGP model to predict the prediction error/residuals 
        and uncertainty based on NWP values
        see https://gpflow.github.io/GPflow/develop/notebooks/advanced/heteroskedastic.html
        """
        self.filename_gp = f'modules/gp/models/gp_prior_{self.opt["n_z"]}_homoscedastic_only_nwp'
        try:
            gp_prior = tf.saved_model.load(self.filename_gp)
            if self.opt['verbose']:
                print('loaded gp from file')
            return gp_prior, True
        except:
            if X_train is None:
                raise RuntimeError('tried to fit gp without providing training data')
        if self.opt['verbose']:
            print(f'training gp for prior mean and variance')

        n_inputs = X_train.shape[1]
        n_samples = X_train.shape[0]

        likelihood = gpf.likelihoods.Gaussian()
        # kernels_nwp_mean = gpf.kernels.SquaredExponential(
        #     lengthscales=[1]*(n_inputs-1), active_dims=[i for i in range(n_inputs-1)])
        kernels_nwp_mean = gpf.kernels.Sum([
            gpf.kernels.RationalQuadratic(lengthscales=[1], active_dims=[i]) for i in range(n_inputs-2)
        ])
        

        kernel_mean = (
            kernels_nwp_mean
            # + gpf.kernels.SquaredExponential(active_dims=[n_inputs-2])
            # + gpf.kernels.Periodic(
            #     gpf.kernels.SquaredExponential(active_dims=[n_inputs-1]), period=365) 
            # + gpf.kernels.Periodic(
            #     gpf.kernels.SquaredExponential(active_dims=[n_inputs-1]), period=1)
            )
        # gpf.set_trainable(kernel_mean.submodules[7].period, False)
        # gpf.set_trainable(kernel_mean.submodules[7+1].period, False)
        
        mean = gpf.functions.Constant(np.zeros(1))
        
        n_z = self.opt['n_z']
        i_Z1 = random.sample(range(n_samples), n_z)
        Z1 = X_train[i_Z1, :]

        # Homoscedastic model for comparison
        gp_prior = gpf.models.SVGP(kernel=kernel_mean, likelihood=likelihood,
            inducing_variable=Z1, mean_function=mean)
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
        # Train on subset of data to speed up training and avoid numerical instability
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
            
            for i in range(max_epochs):
                try:
                    training_step()
                except:
                    raise RuntimeError('Failed to train model')
                if loss_fn().numpy() < loss_lb:
                    break
                if self.opt['verbose']:# and i%20==0:
                    print(f"Epoch {i} - Loss: {loss_fn().numpy() : .4f}")
                if (i+1)%10==0 and reselect_data:# or i>150:
                    training_subset = random.sample(range(n_samples), n_train_1)
                    training_step, loss_fn = self.get_training_step(
                        X_train[training_subset,:], y_train[training_subset,:])
                
        
        # Second training on full data set
        variational_vars = [(self.gp_prior.q_mu, self.gp_prior.q_sqrt)]
        natgrad_opt = gpf.optimizers.NaturalGradient(gamma=0.5)

        adam_vars = self.gp_prior.trainable_variables
        adam_opt = tf.optimizers.Adam(0.1)
        loss_fn = self.gp_prior.training_loss_closure((X_train, y_train))
        

        config = gpf.config.Config(jitter=1e-3)
        with gpf.config.as_context(config):
            @tf.function
            def training_step():
                natgrad_opt.minimize(loss_fn, variational_vars)
                adam_opt.minimize(loss_fn, adam_vars)
            
            max_epochs = self.opt['max_epochs_second_training']
            loss_lb = self.opt['loss_lb']
            for i in range(max_epochs):
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

    