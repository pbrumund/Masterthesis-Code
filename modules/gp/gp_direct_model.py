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
    order = opt['order']
    # n_samples = opt['n_samples']
    input_feature = opt['input_feature']
    label = opt['label']
    steps_ahead = opt['steps_ahead']
    filename_X = f'gp\\training_data\X_train_{order}_last_{input_feature}_{label}_{steps_ahead}step.txt'
    filename_y = f'gp\\training_data\y_train_{order}_last_{input_feature}_{label}_{steps_ahead}step.txt'
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
            weather_data, start_datetime, order, input_feature, steps_ahead).shape[0]
        
        if opt['multithread']:
            args_X = [(weather_data, time, order, input_feature, steps_ahead) for time in times]
            args_y = [(weather_data, time, label, steps_ahead) for time in times]
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

    filename_gp = f'gp\gp_direct_{order}_last_{input_feature.replace(" ", "_")}_{label}_{steps_ahead}step'
    try:
        gp = tf.saved_model.load(filename_gp)
        print('loaded gp from file')
        return gp
    except:
        pass
    print(f'training gp for {steps_ahead} steps ahead')
    X_train, y_train = get_training_data(weather_data, opt)
    n_inputs = X_train.shape[1]
    n_samples = X_train.shape[0]

    likelihood = gpf.likelihoods.HeteroskedasticTFPConditional(scale_transform=tfp.bijectors.Exp())

    kernel_mean = gpf.kernels.SquaredExponential(lengthscales=[.1]*n_inputs) + gpf.kernels.Linear()
    kernel_var = gpf.kernels.SquaredExponential(lengthscales=[.1]*n_inputs) + gpf.kernels.Linear()
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
    adam_opt = tf.optimizers.Adam(0.1)
    
    config = gpf.config.Config(jitter=1e-3)
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
                print('Likelihood is nan')
                raise RuntimeError('Failed to train model')
            if loss_fn().numpy() < loss_lb:
                break
            if True:#opt['verbose'] and i%20==0:
                print(f"Epoch {i} - Loss: {loss_fn().numpy() : .4f}")
    
    # Second training on full data set
    loss_fn = gp.training_loss_closure((X_train, y_train))
    natgrad_opt = gpf.optimizers.NaturalGradient(gamma=0.1)
    adam_opt = tf.optimizers.Adam(0.1)

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

if __name__ == "__main__":
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
           'n_z': 4000,
           'epochs_first_training': 100,
           'max_epochs_second_training': 100,
           'loss_lb': 10,
           'verbose': True,
           'steps_ahead': 1,
           'multithread': True}
    
    weather_data = load_weather_data(start_time, end_time)
    for steps in range(1,60):
        print(f'getting model for {steps} steps ahead')
        opt_i = opt.copy()
        opt_i['steps_ahead'] = steps
        # X_train, y_train = get_training_data(weather_data, opt)
        success = False
        while not success:
            try:
                gp = get_gp(weather_data, opt_i)
                success = True
            except RuntimeError:
                # Failed training, use half the number of inducing variables
                print('failed to train model, reducing number of inducing variables')
                opt_i['n_z'] = int(opt_i['n_z']/2)
                print(f'Number of inducing variables: {opt_i["n_z"]}')
    pass