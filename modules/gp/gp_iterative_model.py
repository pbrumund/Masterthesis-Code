import numpy as np
import gpflow as gpf
import tensorflow as tf
import tensorflow_probability as tfp
import datetime
import random
import matplotlib.pyplot as plt
from multiprocessing import Pool

from .data_handling import DataHandler
from .gp_timeseries_model import WindPredictionGP

class IterativeGPModel(WindPredictionGP):
    def __init__(self, opt):
        super().__init__(opt)
        self.steps_ahead = 1
        self.order = opt['iterative_model_order']
        self.filename_gp = f'modules/gp/models/gp_iterative'
        self.gp = self.load_gp_model()
        if self.gp is None:
            super().__init__(opt) # load data handler
            self.X_train, self.y_train = self.get_training_data()
            self.gp, is_trained = self.get_gp_model(self.X_train, self.y_train)
            if not is_trained:
                self.train_gp(self.X_train, self.y_train)

    def get_training_data(self):
        opt = {}
        opt['filename_x'] = f'modules/gp/training_data/X_train_iterative.txt'
        opt['filename_y'] = f'modules/gp/training_data/y_train_iterative.txt'
        opt['input_feature'] = 'error & nwp & time'
        opt['label'] = 'error'
        opt['steps_ahead'] = 1
        opt['n_last'] = self.order
        return super().get_training_data(opt) 

    def load_gp_model(self):
        try:
            gp = tf.saved_model.load(self.filename_gp)
            if self.opt['verbose']:
                print('loaded gp from file')
            return gp
        except:
            return None
        
    def get_gp_model(self, X_train=None, y_train=None):
        """
        Get a heteroscedastic SVGP model to predict the prediction error/residuals 
        and uncertainty based on NWP values and recent measurements
        see https://gpflow.github.io/GPflow/develop/notebooks/advanced/heteroskedastic.html
        """
        try:
            gp = tf.saved_model.load(self.filename_gp)
            print('loaded gp from file')
            return gp, True
        except:
            if X_train is None:
                raise RuntimeError('tried to fit gp without providing training data')#
        if self.opt['verbose']:
            print(f'training gp for {self.steps_ahead} step prediction')

        n_inputs = X_train.shape[1]
        n_samples = X_train.shape[0]
        n_nwp_inputs = n_inputs - self.order
        n_measurement_inputs = self.order

        likelihood = gpf.likelihoods.HeteroskedasticTFPConditional(
            scale_transform=tfp.bijectors.Exp())
        
        # Kernel for last measurements
        # kernel_measurements_mean = gpf.kernels.Sum(
        #     [gpf.kernels.SquaredExponential(lengthscales=[1], active_dims=[i]) 
        #      for i in range(0, n_measurement_inputs)]
        # )
        # kernel_measurements_mean = gpf.kernels.SquaredExponential(
        #     lengthscales=[1]*n_measurement_inputs)

        # kernel_measurements_var = gpf.kernels.Sum(
        #     [gpf.kernels.SquaredExponential(lengthscales=[.3], active_dims=[i]) 
        #      for i in range(0, n_measurement_inputs)]
        # )
        # kernel_measurements_var = gpf.kernels.SquaredExponential(
        #     lengthscales=[1]*n_measurement_inputs)
        
        # kernel_nwp_mean = gpf.kernels.Sum(
        #     [gpf.kernels.SquaredExponential(lengthscales=[1], active_dims=[i]) 
        #      for i in range(n_measurement_inputs, n_inputs-1)]
        # )
        # kernel_nwp_mean = gpf.kernels.SquaredExponential(
        #     lengthscales=[1]*n_nwp_inputs)
        
        # kernel_nwp_var = gpf.kernels.Sum(
        #     [gpf.kernels.SquaredExponential(lengthscales=[.3], active_dims=[i]) 
        #      for i in range(n_measurement_inputs, n_inputs-1)]
        # )

        # kernel_nwp_var = gpf.kernels.SquaredExponential(
        #     lengthscales=[1]*n_nwp_inputs)
        kernel_mean_se = gpf.kernels.RationalQuadratic(
            lengthscales=[1]*(n_inputs-1), active_dims = range(n_inputs-1)
        )
        kernel_var_se = gpf.kernels.RationalQuadratic(
            lengthscales=[1]*(n_inputs-1), active_dims = range(n_inputs-1)
        )
        kernel_mean = (
            kernel_mean_se
            # kernel_measurements_mean
            # * (kernel_nwp_mean
            + gpf.kernels.Periodic(
                gpf.kernels.SquaredExponential(active_dims=[n_inputs-1]), period=365) 
            # + gpf.kernels.Periodic(
            #     gpf.kernels.SquaredExponential(active_dims=[n_inputs-1]), period=1))
            )
        # gpf.set_trainable(kernel_mean.submodules[9].period, False)
        # gpf.set_trainable(kernel_mean.submodules[10].period, False)
        kernel_var = (
            kernel_var_se
            # kernel_measurements_var
            # * (kernel_nwp_var
            + gpf.kernels.Periodic(
                gpf.kernels.SquaredExponential(active_dims=[n_inputs-1]), period=365) 
            # + gpf.kernels.Periodic(
            #     gpf.kernels.SquaredExponential(active_dims=[n_inputs-1]), period=1)
            )
        # gpf.set_trainable(kernel_var.submodules[9].period, False)
        # gpf.set_trainable(kernel_var.submodules[10].period, False)
        kernel = gpf.kernels.SeparateIndependent(
            [
                kernel_mean,
                kernel_var
            ]
        )
        
        mean = gpf.functions.Constant(np.zeros(2))
        
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

        gp = gpf.models.SVGP(
            kernel=kernel, 
            likelihood=likelihood, 
            inducing_variable=inducing_variable,
            num_latent_gps=likelihood.latent_dim,
            mean_function=mean,
            )
        return gp, False
    
    def get_training_step(self, X_train, y_train):
        loss_fn = self.gp.training_loss_closure(
            (X_train, y_train))
        variational_vars = [(self.gp.q_mu, self.gp.q_sqrt)]
        natgrad_opt = gpf.optimizers.NaturalGradient(gamma=0.1)

        adam_vars = self.gp.trainable_variables
        adam_opt = tf.optimizers.Adam(0.1)

        config = gpf.config.Config(jitter=1e-2)
        with gpf.config.as_context(config):
            @tf.function
            def training_step():
                natgrad_opt.minimize(loss_fn, variational_vars)
                adam_opt.minimize(loss_fn, adam_vars)

        return training_step, loss_fn

    def train_gp(self, X_train, y_train):
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
            loss_fn = self.gp.training_loss_closure(
                (X_train[training_subset,:], y_train[training_subset,:]))

        gpf.utilities.set_trainable(self.gp.q_mu, False)
        gpf.utilities.set_trainable(self.gp.q_sqrt, False)

        config = gpf.config.Config(jitter=1e-2)
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
        variational_vars = [(self.gp.q_mu, self.gp.q_sqrt)]
        natgrad_opt = gpf.optimizers.NaturalGradient(gamma=0.5)

        adam_vars = self.gp.trainable_variables
        adam_opt = tf.optimizers.Adam(0.1)
        loss_fn = self.gp.training_loss_closure((X_train, y_train))
        

        config = gpf.config.Config(jitter=1e-2)
        with gpf.config.as_context(config):
            @tf.function
            def training_step():
                natgrad_opt.minimize(loss_fn, variational_vars)
                adam_opt.minimize(loss_fn, adam_vars)
            
            max_epochs = self.opt['max_epochs_second_training']
            loss_lb = self.opt['loss_lb']
            min_loss = 1e12
            for i in range(max_epochs+1):
                try:
                    training_step()
                except:
                    print('Likelihood is nan')
                    raise RuntimeError('Failed to train model')
                if loss_fn().numpy() < loss_lb or loss_fn().numpy()>1.01*min_loss:
                    break
                if self.opt['verbose']:# and i%20==0:
                    print(f"Epoch {i} - Loss: {loss_fn().numpy() : .4f}")
                if loss_fn().numpy() < min_loss: min_loss = loss_fn().numpy()

        # save gp
        self.gp.compiled_predict_f = tf.function(
            lambda x: self.gp.predict_f(x),
            input_signature=[tf.TensorSpec(shape=[None, n_inputs], dtype=tf.float64)]
        )
        self.gp.compiled_predict_y = tf.function(
            lambda x: self.gp.predict_y(x),
            input_signature=[tf.TensorSpec(shape=[None, n_inputs], dtype=tf.float64)]
        )

        tf.saved_model.save(self.gp, self.filename_gp)

    def get_input_cov(self, input_cov, new_cov, new_var):
        """ get the input covariance matrix for the next step 
        by shifting the last values and concatenating with the new covariances """
        n_uncertain_in = self.opt['order']
        n_in = input_cov.shape[0]
        n_pad = n_in-n_uncertain_in
        cov_old = input_cov[1:n_uncertain_in,1:n_uncertain_in]
        # x = np.append(x, var)[1:, 1:]
        # var_new = new_cov[-1]
        input_cov = np.block([
            [cov_old, new_cov[1:n_uncertain_in].reshape(-1,1), np.zeros((n_uncertain_in-1, n_pad))],
            [new_cov[1:n_uncertain_in].reshape(1,-1), new_var, np.zeros((1, n_pad))], 
            [np.zeros((n_pad, n_in))]])

        #input_uncertainty[:n_uncertain_inputs] = x
        return input_cov


    def get_input_cov(self, input_cov, new_cov, new_var):
        """get the input covariance matrix for the next step 
        by shifting the last values and concatenating with the new covariances"""
        n_uncertain_in = self.order
        n_in = input_cov.shape[0]
        n_pad = n_in-n_uncertain_in
        cov_old = input_cov[1:n_uncertain_in,1:n_uncertain_in]
        # x = np.append(x, var)[1:, 1:]
        # var_new = new_cov[-1]
        input_cov = np.block([
            [cov_old, new_cov[1:n_uncertain_in].reshape(-1,1), np.zeros((n_uncertain_in-1, n_pad))],
            [new_cov[1:n_uncertain_in].reshape(1,-1), new_var, np.zeros((1, n_pad))], 
            [np.zeros((n_pad, n_in))]])

        #input_uncertainty[:n_uncertain_inputs] = x
        return input_cov

    def predict_uncertain_inputs(self, x, input_cov):
        """propagate input uncertainty through monte carlo samples"""
        n_samples = self.opt['n_samples_mc']
        n_dimensions = x.shape[1]
        outputs = np.zeros(n_samples)
        inputs = np.zeros((n_samples, n_dimensions))
        rng = np.random.default_rng()
        for i in range(n_samples):
            x_i = rng.multivariate_normal(mean=x.reshape(-1), cov=input_cov).reshape((1,-1))
            # x_i = np.random.normal(size=x.shape, loc=x, scale=np.sqrt(input_cov))
            inputs[i,:] = x_i
            mean_i, var_i = self.gp.compiled_predict_y(x_i)
            y_i = np.random.normal(loc=mean_i, scale=np.sqrt(var_i))
            outputs[i] = y_i
        # approximate the output distribution as a joint gaussian described by the mean, variance 
        # and covariance between the uncertain inputs and the output
        mean = np.mean(outputs)
        var = np.var(outputs)
        cov = np.array([np.cov(inputs[:,i], outputs)[0,1] for i in range(n_dimensions)])
        # cov = np.cov(outputs, inputs.T)[0,:]
        return mean, var, cov


    def predict_trajectory(self, t_start, steps_ahead, include_last_measurement=True):
        mean_traj = np.zeros(steps_ahead)
        var_traj = np.zeros(steps_ahead)
        
        x_0 = self.data_handler.generate_features(
            t_start, self.order, 'error & nwp & time', 1).reshape((1,-1))
        x = x_0
        n_dim = x.shape[1]
        input_cov = np.zeros((n_dim, n_dim))

        for i in range(steps_ahead):
            print(f'steps ahead: {i+1}')
            mean, var, cov = self.predict_uncertain_inputs(x, input_cov)
            input_cov = self.get_input_cov(input_cov, cov, var)
            mean_traj[i] = mean
            var_traj[i] = var
            x_new = self.data_handler.generate_features(t_start, n_last=self.order, 
                feature='error & nwp & time', steps_ahead=i+2)
            x_new[:self.order] = np.append(x[1:self.order], mean)
            x = x_new.reshape((1,-1))
            
        NWP_preds = np.array([self.data_handler.get_NWP(t_start, i+1) for i in range(steps_ahead)])
        gp_pred = mean_traj + NWP_preds
        if include_last_measurement:
            gp_pred = np.insert(gp_pred, 0, self.data_handler.get_measurement(t_start))[:-1]
            var_traj = np.insert(var_traj, 0, 0)[:-1]
        return gp_pred, var_traj

