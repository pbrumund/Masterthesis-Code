import numpy as np
import gpflow as gpf
import tensorflow as tf
import tensorflow_probability as tfp
import datetime
import random
import matplotlib.pyplot as plt
from multiprocessing import Pool

import utils
from fileloading import load_weather_data

def get_training_data(weather_data, opt):
    # generate X_train and y_train
    order = opt['order']
    # n_samples = opt['n_samples']
    input_feature = opt['input_feature']
    label = opt['label']
    max_steps_ahead = opt['steps_ahead']
    filename_X = f'gp\\training_data\X_train_{order}_last_{input_feature}_{label}_1-{max_steps_ahead}step.txt'
    filename_y = f'gp\\training_data\y_train_{order}_last_{input_feature}_{label}_1-{max_steps_ahead}step.txt'
    try:
        # If training data has been generated before, load it from file
        X_train = np.loadtxt(filename_X)
        y_train = np.loadtxt(filename_y).reshape((-1,1))
        print('loaded data from file')
    except:
        # generate training data
        print('generating data')
        end_datetime = opt['end_date_train']
        start_datetime = weather_data['times_meas'][order]
        n_points = int((end_datetime-start_datetime)/datetime.timedelta(minutes=10))
        times = [start_datetime + i*datetime.timedelta(minutes=10) for i in range(n_points)]
        n_x = utils.generate_features(
            weather_data, start_datetime, order, input_feature, 0).shape[0]
        steps_ahead = 1
        
        if opt['multithread']:
            args_X = [(weather_data, time, order, input_feature, steps_ahead) 
                      for time in times]# for steps_ahead in range(1,max_steps_ahead)]
            args_y = [(weather_data, time, label, steps_ahead)
                      for time in times]# for steps_ahead in range(1,max_steps_ahead)]
            with Pool(processes=12) as pool:
                X_train = pool.starmap(utils.generate_features, args_X, chunksize=1000)
                print('finished generating X_train')
                y_train = pool.starmap(utils.generate_labels, args_y, chunksize=1000)
                print('finished generating y_train')
                X_train = np.array(X_train).reshape((n_points, n_x))
                y_train = np.array(y_train).reshape((n_points, 1))

        else:
            X_train = np.zeros((n_points, n_x))
            y_train = np.zeros((n_points, 1))
            for i, time in enumerate(times):
                for steps_ahead in range(1, max_steps_ahead):
                    x = utils.generate_features(weather_data, time, order, input_feature, steps_ahead)
                    y = utils.generate_labels(weather_data, time, label, steps_ahead)
                    X_train[i,:] = x
                    y_train[i,:] = y
                if (i+1)%int(n_points/20)==0:
                    print(f'{int((i+1)/int(n_points/20)*5)}% done')
        # Save to file
        np.savetxt(filename_X, X_train)
        np.savetxt(filename_y, y_train)
    return X_train, y_train

def get_gp(weather_data, opt):
    order = opt['order']
    input_feature = opt['input_feature']
    label = opt['label']
    steps_ahead = opt['steps_ahead']

    filename_gp = f'gp\gp_iterative_{order}_last_{input_feature.replace(" ", "_")}_{label}'
    # load from file if model has been trained before
    try:
        gp = tf.saved_model.load(filename_gp)
        print('loaded gp from file')
        return gp
    except:
        pass
    # train gp if no file has been found
    print(f'training gp for {steps_ahead} steps ahead')
    X_train, y_train = get_training_data(weather_data, opt)
    n_inputs = X_train.shape[1]
    n_samples = X_train.shape[0]



    likelihood = gpf.likelihoods.HeteroskedasticTFPConditional(scale_transform=tfp.bijectors.Exp())

    kernel_mean = gpf.kernels.SquaredExponential(lengthscales=[.1]*n_inputs) #+ gpf.kernels.Linear() + gpf.kernels.White()
    kernel_var = gpf.kernels.SquaredExponential(lengthscales=[.1]*n_inputs) #+ gpf.kernels.Linear() + gpf.kernels.White()
    kernel = gpf.kernels.SeparateIndependent(
        [
            kernel_mean,
            kernel_var
        ]
    )
    
    n_z = opt['n_z']
    i_Z1 = random.sample(range(n_samples), n_z)
    i_Z2 = random.sample(range(n_samples), n_z)
    Z1 = X_train[i_Z1, :]
    Z2 = X_train[i_Z2, :]

    inducing_variable = gpf.inducing_variables.SeparateIndependentInducingVariables(
        [
            gpf.inducing_variables.InducingPoints(Z1),  # This is U1 = f1(Z1)
            gpf.inducing_variables.InducingPoints(Z2),  # This is U2 = f2(Z2)
        ]
    )

    gp = gpf.models.SVGP(
        kernel=kernel, 
        likelihood=likelihood, 
        inducing_variable=inducing_variable,
        num_latent_gps=likelihood.latent_dim)
    # Train on subset of data
    n_train_1 = int(X_train.shape[0]/10)
    training_subset = random.sample(range(n_samples), n_train_1)
    loss_fn = gp.training_loss_closure((X_train[training_subset,:], y_train[training_subset,:]))

    gpf.utilities.set_trainable(gp.q_mu, False)
    gpf.utilities.set_trainable(gp.q_sqrt, False)

    variational_vars = [(gp.q_mu, gp.q_sqrt)]
    natgrad_opt = gpf.optimizers.NaturalGradient(gamma=0.1)

    adam_vars = gp.trainable_variables
    adam_opt = tf.optimizers.Adam(0.01)

    config = gpf.config.Config(jitter=1e-2)
    with gpf.config.as_context(config):
        @tf.function
        def training_step():
            natgrad_opt.minimize(loss_fn, variational_vars)
            adam_opt.minimize(loss_fn, adam_vars)
        
        max_epochs = opt['epochs_first_training']
        loss_lb = opt['loss_lb']

        for i in range(10):
            natgrad_opt.minimize(loss_fn, variational_vars)
            print(f"Epoch {i} - Loss: {loss_fn().numpy() : .4f}")

    natgrad_opt = gpf.optimizers.NaturalGradient(gamma=0.01)

    config = gpf.config.Config(jitter=1e-2)
    with gpf.config.as_context(config):
        for i in range(max_epochs+1):
            try:
                training_step()
            except:
                print('Likelihood is nan')
                raise RuntimeError('Failed to train model')
            if loss_fn().numpy() < loss_lb:
                break
            if True:#opt['verbose'] and i%20==0:
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
                print('Likelihood is nan')
                raise RuntimeError('Failed to train model')
            if loss_fn().numpy() < loss_lb:
                break
            if True:#opt['verbose'] and i%20==0:
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

def get_input_cov(input_cov, new_cov, new_var, opt):
    # get the input covariance matrix for the next step 
    # by shifting the last values and concatenating with the new covariances
    n_uncertain_in = opt['order']
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

def predict_uncertain_inputs(gp, x, input_cov, opt):
    # propagate input uncertainty through monte carlo samples
    n_samples = opt['n_samples_mc']
    n_dimensions = x.shape[1]
    outputs = np.zeros(n_samples)
    inputs = np.zeros((n_samples, n_dimensions))
    rng = np.random.default_rng()
    for i in range(n_samples):
        x_i = rng.multivariate_normal(mean=x.reshape(-1), cov=input_cov).reshape((1,-1))
        # x_i = np.random.normal(size=x.shape, loc=x, scale=np.sqrt(input_cov))
        inputs[i,:] = x_i
        mean_i, var_i = gp.compiled_predict_y(x_i)
        y_i = np.random.normal(loc=mean_i, scale=np.sqrt(var_i))
        outputs[i] = y_i
    # approximate the output distribution as a joint gaussian described by the mean, variance 
    # and covariance between the uncertain inputs and the output
    mean = np.mean(outputs)
    var = np.var(outputs)
    cov = np.array([np.cov(inputs[:,i], outputs)[0,1] for i in range(n_dimensions)])
    # cov = np.cov(outputs, inputs.T)[0,:]
    return mean, var, cov


def predict_trajectory(weather_data, t_start, steps_ahead, gp, opt):
    mean_traj = np.zeros(steps_ahead)
    var_traj = np.zeros(steps_ahead)
    
    x_0 = utils.generate_features(
        weather_data, t_start, opt['order'], opt['input_feature'], 1).reshape((1,-1))
    x = x_0
    n_dim = x.shape[1]
    input_cov = np.zeros((n_dim, n_dim))

    for i in range(steps_ahead):
        print(f'steps ahead: {i+1}')
        mean, var, cov = predict_uncertain_inputs(gp, x, input_cov, opt)
        input_cov = get_input_cov(input_cov, cov, var, opt)
        mean_traj[i] = mean
        var_traj[i] = var
        x = utils.get_new_input(weather_data, t_start, x, mean, i+2, opt)

    gp_pred = np.zeros(steps_ahead)
    for i in range(steps_ahead):
        if opt['label'] == 'error':
            prediction_NWP_i = utils.get_NWP(weather_data, t_start, i+1)
            gp_pred[i] = prediction_NWP_i + mean_traj[i]
        elif opt['label'] == 'measurement':
            gp_pred[i] = mean_traj[i]
        elif opt['label'] == 'change of wind speed':
            wind_now = utils.get_wind_value(weather_data, t_start, 0)
            gp_pred[i] = wind_now + mean_traj[i]

    return gp_pred, var_traj

if __name__ == "__main__":
    random.seed(1)
    # Data loading
    start_time = datetime.datetime(2020,1,1)
    end_time = datetime.datetime(2022,12,31)
    end_time_train = datetime.datetime(2021,12,31)

    n_last = 5
    input_feature = 'error & nwp'
    label = 'error'
    print(input_feature)
    print(label)
    print(f'n_last = {n_last}')

    opt = {'end_date_train': end_time_train,
           'order': n_last,
           'input_feature': input_feature,
           'label': label,
           'n_z': 1000,
           'epochs_first_training': 100,
           'max_epochs_second_training': 100,
           'loss_lb': 10,
           'verbose': True,
           'steps_ahead': 1,
           'multithread': True,
           'n_samples_mc': 100}
    
    weather_data = load_weather_data(start_time, end_time)

    success = False
    while not success:
        try:
            gp = get_gp(weather_data, opt)
            success = True
        except RuntimeError:
            print('failed to train model, reducing number of inducing variables')
            opt['n_z'] = int(opt['n_z']/2)
            print(f'Number of inducing variables: {opt["n_z"]}')
    
    t = datetime.datetime(2022,2,1,5,10)
    steps_ahead = 60

    nwp_pred = np.zeros(steps_ahead)
    wind_meas = np.zeros(steps_ahead)

    times = []
    for i in range(steps_ahead):
        prediction_NWP = utils.get_NWP(weather_data, time=t, steps=i+1)
        nwp_pred[i] = prediction_NWP
        wind_meas[i] = utils.get_wind_value(weather_data, t, i+1)
        times.append(t + (i+1)*datetime.timedelta(minutes=10))

    gp_traj, var_traj = predict_trajectory(weather_data, t, steps_ahead, gp, opt)

    plt.figure()
    plt.plot(times, wind_meas, color='g')
    plt.plot(times, nwp_pred)
    plt.plot(times, gp_traj)
    plt.plot(times, gp_traj + 1*np.sqrt(var_traj), color='k')
    plt.plot(times, gp_traj - 1*np.sqrt(var_traj), color='k')
    plt.plot(times, gp_traj + 2*np.sqrt(var_traj), color='tab:gray')
    plt.plot(times, gp_traj - 2*np.sqrt(var_traj), color='tab:gray')
    plt.show()
    pass