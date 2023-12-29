from gpflow.base import TensorType
import numpy as np
import random
import datetime
import gpytorch
import gpflow as gpf
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import sklearn

from multiprocessing import Pool

import utils
from fileloading import load_weather_data
from posterior_kernel import PosteriorKernel, PosteriorMean


def get_training_data_prior(weather_data, opt):
    filename_X = f'gp\\training_data\X_train_prior.txt'
    filename_y = f'gp\\training_data\y_train_prior.txt'
    try:
        # If training data has been generated before, load it from file
        X_train = np.loadtxt(filename_X)
        y_train = np.loadtxt(filename_y).reshape((-1,1))
        print('loaded data from file')
    except:
        # generate training data
        print('generating data')
        start_datetime = opt['start_date_train']
        end_datetime = opt['end_date_train']
        n_points = int((end_datetime-start_datetime)/datetime.timedelta(minutes=10))
        times = [start_datetime + i*datetime.timedelta(minutes=10) for i in range(n_points)]
        n_x = utils.generate_features(
            weather_data, start_datetime, 1, 'nwp & time', 0).shape[0]
        steps_ahead = 1
        
        args_X = [(weather_data, time, 0, 'nwp & time', steps_ahead) 
                    for time in times]# for steps_ahead in range(1,max_steps_ahead)]
        args_y = [(weather_data, time, 'error', steps_ahead)
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

def get_prior_gp(weather_data, opt):
    filename_gp = f'gp\models\gp_prior_{opt["n_z"]}'
    try:
        gp = tf.saved_model.load(filename_gp)
        print('loaded gp from file')
        return gp
    except:
        pass
    print(f'training gp for prior mean and variance')
    X_train, y_train = get_training_data_prior(weather_data, opt)
    n_inputs = X_train.shape[1]
    n_samples = X_train.shape[0]

    likelihood = gpf.likelihoods.HeteroskedasticTFPConditional(scale_transform=tfp.bijectors.Exp())

    kernel_mean = gpf.kernels.RationalQuadratic(
        lengthscales=[.1]*(n_inputs-1), active_dims = range(n_inputs-1)) + gpf.kernels.Periodic(
            gpf.kernels.SquaredExponential(active_dims=[n_inputs-1]), period=365) + gpf.kernels.Periodic(
            gpf.kernels.SquaredExponential(active_dims=[n_inputs-1]), period=1)
    gpf.set_trainable(kernel_mean.submodules[1].period, False)
    gpf.set_trainable(kernel_mean.submodules[2].period, False)
    kernel_var = gpf.kernels.RationalQuadratic(
        lengthscales=[.1]*(n_inputs-1), active_dims = range(n_inputs-1)) + gpf.kernels.Periodic(
            gpf.kernels.SquaredExponential(active_dims=[n_inputs-1]), period=365) + gpf.kernels.Periodic(
            gpf.kernels.SquaredExponential(active_dims=[n_inputs-1]), period=1)
    gpf.set_trainable(kernel_var.submodules[1].period, False)
    gpf.set_trainable(kernel_var.submodules[2].period, False)
    kernel = gpf.kernels.SeparateIndependent(
        [
            kernel_mean,
            kernel_var
        ]
    )
    # A1 = gpf.Parameter(tf.zeros((n_inputs-1,2)))
    # A2 = gpf.Parameter(tf.zeros((1,2)), trainable=False)
    # A = tf.concat([A1, A2], 0)
    # A = gpf.Parameter(A)
    # A = tf.concat([tf.zeros((n_inputs-1,2)),tf.constant(tf.zeros((1,2)))],0)
    A = np.zeros((n_inputs, 2))
    b = np.zeros((2))
    # A = gpf.Parameter(A)
    class LinearMeanNWP(gpf.functions.Linear):
        def __call__(self, X: TensorType):
            return tf.tensordot(X[...,:-1], self.A[:-1,:], [[-1], [0]]) + self.b
    # b = gpf.Parameter(b)
    mean = LinearMeanNWP(A, b)
    # mean = gpf.functions.Constant(np.zeros(2))
    #mean = gpf.functions.Zero()
    
    n_z = opt['n_z']
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
        mean_function=mean)
    # Train on subset of data
    n_train_1 = int(X_train.shape[0]/10)
    training_subset = random.sample(range(n_samples), n_train_1)
    loss_fn = gp.training_loss_closure((X_train[training_subset,:], y_train[training_subset,:]))

    gpf.utilities.set_trainable(gp.q_mu, False)
    gpf.utilities.set_trainable(gp.q_sqrt, False)

    variational_vars = [(gp.q_mu, gp.q_sqrt)]
    natgrad_opt = gpf.optimizers.NaturalGradient(gamma=0.1)

    adam_vars = gp.trainable_variables
    adam_opt = tf.optimizers.Adam(0.1)
    
    config = gpf.config.Config(jitter=1e-2)
    with gpf.config.as_context(config):
        @tf.function
        def training_step():
            natgrad_opt.minimize(loss_fn, variational_vars)
            adam_opt.minimize(loss_fn, adam_vars)
        
        max_epochs = opt['epochs_first_training']
        loss_lb = opt['loss_lb']
        for i in range(max_epochs+1):
            try:
                training_step()
            except:
                raise RuntimeError('Failed to train model')
            if loss_fn().numpy() < loss_lb:
                break
            if opt['verbose']:#and i%20==0:
                print(f"Epoch {i} - Loss: {loss_fn().numpy() : .4f}")
    
    # Second training on full data set
    loss_fn = gp.training_loss_closure((X_train, y_train))
    natgrad_opt = gpf.optimizers.NaturalGradient(gamma=0.1)
    adam_opt = tf.optimizers.Adam(0.01)

    config = gpf.config.Config(jitter=1e-2)
    with gpf.config.as_context(config):
        @tf.function
        def training_step():
            natgrad_opt.minimize(loss_fn, variational_vars)
            adam_opt.minimize(loss_fn, adam_vars)
        
        max_epochs = opt['max_epochs_second_training']
        loss_lb = opt['loss_lb']
        for i in range(max_epochs+1):
            try:
                training_step()
            except:
                raise RuntimeError('Failed to train model')
            if loss_fn().numpy() < loss_lb:
                break
            if opt['verbose']:# and i%20==0:
                print(f"Epoch {i} - Loss: {loss_fn().numpy() : .4f}")

    # save gp
    gp.compiled_predict_f = tf.function(
        lambda x: gp.predict_f(x),
        input_signature=[tf.TensorSpec(shape=[None, n_inputs], dtype=tf.float64)]
    )
    gp.compiled_predict_y = tf.function(
        lambda x: gp.predict_y(x),
        input_signature=[tf.TensorSpec(shape=[None, n_inputs], dtype=tf.float64)]
    )

    tf.saved_model.save(gp, filename_gp)
    return gp

    
def get_in(weather_data, time, steps):
    if steps.shape[0] != 1:
        x = []
        for i in range(steps.shape[0]):
            x.append(get_in(weather_data, time, steps[i]).reshape(-1))
        return tf.convert_to_tensor(np.array(x))
    # print(steps)
    s = tf.get_static_value(steps, partial=True)
    steps = int(s[0])
    # except:
    #     print('na toll')
    #     return utils.generate_features(weather_data, time, feature='nwp & time', steps_ahead=0).reshape((1,-1))
    if steps < 0:
        t = time+steps*datetime.timedelta(minutes=10)
        x = utils.generate_features(weather_data, t, feature='nwp & time', steps_ahead=0).reshape((1,-1))
    else:
        x = utils.generate_features(weather_data, time, feature='nwp & time', steps_ahead=steps).reshape((1,-1))
    return x

def get_timeseries_gp(weather_data, time, gp_prior, opt):
    n_last = opt['n_last']
    X_train = np.arange(start=-n_last, stop=0)
    y_train = np.zeros(n_last)
    t_start = time - n_last*datetime.timedelta(minutes=10)
    times = [t_start + i*datetime.timedelta(minutes=10) for i in range(n_last)]

    for i, t in enumerate(times):
        measurement = utils.get_wind_value(weather_data, t, 0)
        prediction = utils.get_NWP(weather_data, t, 0)
        y_train[i] = measurement - prediction
    X_train = X_train.reshape((-1,1)).astype(float)
    y_train = y_train.reshape((-1,1))
    
    kernel = gpf.kernels.Matern12(lengthscales=[30], variance=1)
    likelihood = gpf.likelihoods.Gaussian(variance=1e-3)
    # gpf.set_trainable(kernel.variance, False)
    get_x_fun = lambda x: get_in(weather_data, time, x)
    kernel_posterior = PosteriorKernel(kernel, gp_prior, get_x_fun)
    mean_posterior = PosteriorMean(gp_prior, get_x_fun)
    gp_posterior = gpf.models.GPR(
        (X_train, y_train),
        kernel=kernel_posterior,
        mean_function=mean_posterior,
        likelihood=likelihood
    )
    
    if opt['train_posterior']:
        train_vars = (gp_posterior.trainable_variables[0], gp_posterior.trainable_variables[1], gp_posterior.trainable_variables[3])
        adam_opt = tf.optimizers.Adam(0.5)
        loss_fn = gp_posterior.training_loss
        config = gpf.config.Config(jitter=1e-1)
        with gpf.config.as_context(config):
            for i in range(100):       
                adam_opt.minimize(loss_fn, train_vars)
                print(f'iteration {i}, a: {train_vars[0].numpy()}, l: {train_vars[1].numpy()}, noise:{train_vars[2].numpy()}, loss: {loss_fn().numpy()}')
        # opt = gpf.optimizers.Scipy()
        # opt.minimize(gp_posterior.training_loss, train_vars)

    return gp_posterior, gp_posterior.trainable_parameters

def predict_trajectory(weather_data, time, gp_prior, opt):
    gp_posterior, _ = get_timeseries_gp(weather_data, time, gp_prior, opt)
    n_next = opt['steps_forward']
    x = np.arange(n_next).reshape((-1,1)).astype(float)
    gp_mean, gp_var = gp_posterior.predict_y(x)
    NWP_pred = [utils.get_NWP(weather_data, time, i) for i in range(n_next)]
    gp_pred = np.array(NWP_pred).reshape((-1,1)) + gp_mean
    return gp_pred, gp_var

def plot_prior(weather_data, gp_prior, time_range):
    start_time = time_range[0]
    end_time = time_range[1]
    dt = end_time-start_time
    n_steps = int(dt.total_seconds()/600)

    times = [start_time+i*datetime.timedelta(minutes=10) for i in range(n_steps)]
    NWP_traj = np.zeros(n_steps)
    mean_traj = np.zeros(n_steps)
    var_traj = np.zeros(n_steps)
    meas_traj = np.zeros(n_steps)

    for i, t in enumerate(times):
        NWP_traj[i] = utils.get_NWP(weather_data, t, 0)
        meas_traj[i] = utils.get_wind_value(weather_data, t, 0)
        x = utils.generate_features(weather_data, t, feature='nwp & time').reshape((1,-1))
        mean, var = gp_prior.compiled_predict_y(x)
        mean_traj[i] = mean
        var_traj[i] = var

    gp_pred_traj = NWP_traj + mean_traj
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
    plt.xlabel('time')
    plt.ylabel('predicted uncertainty')
    # plt.show()

def plot_posterior(times, gp_pred_traj, gp_var):
    gp_pred_traj = gp_pred_traj.numpy()[:,0]
    gp_var = gp_var.numpy()[:,0]
    n_steps = len(times)
    NWP_traj = np.zeros(n_steps)
    meas_traj = np.zeros(n_steps)

    for i, t in enumerate(times):
            NWP_traj[i] = utils.get_NWP(weather_data, t, 0)
            meas_traj[i] = utils.get_wind_value(weather_data, t, 0)

    std_traj = np.sqrt(gp_var)

    plt.figure()
    plt.plot(times, NWP_traj, color='tab:orange')
    plt.plot(times, meas_traj, color='tab:green')
    plt.plot(times, gp_pred_traj, color='tab:blue')
    plt.fill_between(times, gp_pred_traj-2*std_traj, gp_pred_traj+2*std_traj, color='lightgray')
    plt.fill_between(times, gp_pred_traj-std_traj, gp_pred_traj+std_traj, color='tab:gray')
    plt.xlabel('time')
    plt.ylabel('wind speed')
    plt.legend(['Weather prediction', 'Actual wind speed', 'Posterior GP prediction'])
    plt.figure()
    plt.plot(times, std_traj)
    plt.xlabel('time')
    plt.ylabel('predicted uncertainty')
    # plt.show()

            


if __name__ == "__main__":
    start_time = datetime.datetime(2020,1,1)
    end_time = datetime.datetime(2022,12,31,23,50)
    end_time_train = datetime.datetime(2021,12,31)

    opt = {'start_date_train': start_time,
           'end_date_train': end_time_train,
           'n_z': 500,
           'epochs_first_training': 100,
           'max_epochs_second_training': 50,
           'loss_lb': 0.5,
           'verbose': True,
           'n_last': 500,
           'train_posterior': True,
           'steps_forward': 100
           }
    
    weather_data = load_weather_data(start_time, end_time)
    gp_prior = get_prior_gp(weather_data, opt)


    t = datetime.datetime(2022,1,1)
    # gp_timeseries = get_timeseries_gp(weather_data, time=t, gp_prior=gp_prior, opt=opt)
    t1 = datetime.datetime(2020,1,1)
    t2 = datetime.datetime(2022,12,31)
    # t2 = t + opt['steps_forward']*datetime.timedelta(minutes=10)
    plot_prior(weather_data, gp_prior, [t1, t2])

    times = [t + i*datetime.timedelta(minutes=10) for i in range(opt['steps_forward'])]
    pred_traj, var_traj = predict_trajectory(weather_data, t, gp_prior, opt)
    plot_posterior(times, pred_traj, var_traj)

    
    plt.show()
    pass