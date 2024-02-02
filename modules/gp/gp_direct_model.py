import numpy as np
import gpflow as gpf
import tensorflow as tf
import tensorflow_probability as tfp
import datetime
import random
import matplotlib.pyplot as plt
from multiprocessing import Pool

# import utils
# from fileloading import load_weather_data
from .data_handling import DataHandler
from .gp_timeseries_model import WindPredictionGP

class DirectGP(WindPredictionGP):
    def __init__(self, steps_ahead, opt):
        self.opt = opt
        self.steps_ahead = steps_ahead
        self.order = opt['direct_model_order']
        self.filename_gp = f'modules/gp/models/direct_model/gp_direct_{self.steps_ahead}'
        self.gp = self.load_gp_model()
        if self.gp is None:
            super().__init__(opt) # load data handler
            self.X_train, self.y_train = self.get_training_data()
            self.gp, is_trained = self.get_gp_model(self.X_train, self.y_train)
            if not is_trained:
                self.train_gp(self.X_train, self.y_train)

    def get_training_data(self):
        opt = {}
        opt['filename_x'] = f'modules/gp/training_data/X_train_direct_{self.steps_ahead}_steps.txt'
        opt['filename_y'] = f'modules/gp/training_data/y_train_direct_{self.steps_ahead}_steps.txt'
        opt['input_feature'] = 'error & nwp'
        opt['label'] = 'error'
        opt['steps_ahead'] = self.steps_ahead
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
        kernel_mean_se = gpf.kernels.SquaredExponential(
            lengthscales=[1]*(n_inputs-1), active_dims = range(n_inputs-1)
        )
        kernel_var_se = gpf.kernels.SquaredExponential(
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

        gp_direct = gpf.models.SVGP(
            kernel=kernel, 
            likelihood=likelihood, 
            inducing_variable=inducing_variable,
            num_latent_gps=likelihood.latent_dim,
            mean_function=mean,
            )
        return gp_direct, False
    
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

class DirectGPEnsemble(WindPredictionGP):
    def __init__(self, opt):
        super().__init__(opt)
        self.get_models()
        self.model_lut = {}

    def get_models(self):
        self.models = {}
        self.steps_list = (1,2,3,4,5,6,7,8,10,15,20,25,30,40,60)
        for step in self.steps_list:
            direct_gp = DirectGP(step, self.opt)
            self.models[step] = direct_gp
    
    def find_model(self, steps_scaled):
        model_steps = self.model_lut.get(steps_scaled)
        if model_steps is not None: return model_steps
        if steps_scaled >= self.steps_list[-1]: return self.steps_list[-1]
        model_steps = steps_scaled
        while not model_steps in self.steps_list:
            model_steps += self.opt['dt_pred']/self.opt['dt_meas']
        self.model_lut[steps_scaled] = int(model_steps)
        return int(model_steps)
    
    def predict_trajectory(self, start_time, steps, include_last_measurement=True):
        # dt = 0  # if start time is not multiple of 10 min, difference to last previous multiple to shift indices
        # if start_time.minute%self.opt['dt_meas'] != 0:
        #     dt = start_time.minute%self.opt['dt_meas']
        #     start_time = start_time.replace(
        #         minute=start_time.minute//self.opt['dt_meas']*self.opt['dt_meas'])
        scale = self.opt['dt_pred']/self.opt['dt_meas']
        mean_traj = np.zeros(steps)
        var_traj = np.zeros(steps)
        if include_last_measurement:
            start=0
        else:
            start=1
        for step in range(start, steps+start):
            if step==0:
                mean_traj[step] = self.data_handler.get_measurement(start_time)
                var_traj[step] = 0
                continue
            step_scaled = step*scale
            model_steps = self.find_model(step_scaled)
            model = self.models[model_steps]
            timesteps_shift = model_steps - step_scaled
            t_inputs = start_time - timesteps_shift*datetime.timedelta(minutes=self.opt['dt_meas'])
            gp_in = self.data_handler.generate_features(time=t_inputs,
                                                        n_last=model.order,
                                                        feature='error & nwp',
                                                        steps_ahead=model_steps
                                                        ).reshape((1,-1))
            mean, var = model.gp.compiled_predict_y(gp_in)
            mean = mean + self.data_handler.get_NWP(start_time, step_scaled)
            mean_traj[step-start] = mean
            var_traj[step-start] = var
        return mean_traj, var_traj




