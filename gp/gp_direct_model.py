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
    filename_X = f'X_train_{order}_last_{input_feature}_{label}_{steps_ahead}step.txt'
    filename_y = f'y_train_{order}_last_{input_feature}_{label}_{steps_ahead}step.txt'
    try:
        # If training data has been generated before, load it from file
        X_train = np.loadtxt(filename_X)
        y_train = np.loadtxt(filename_y)
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




def get_iterative_model(weather_data, opt):
    load = False
    order = opt['order']
    n_samples = opt['n_samples']
    input_feature = opt['input_feature']
    label = opt['label']
    steps_ahead = steps_ahead
    if load:
        X_train = np.loadtxt(f'X_train_{steps_ahead}steps.txt')
        y_train = np.loadtxt(f'y_train_{steps_ahead}steps.txt').reshape((-1,1))
    else:
        X_train, y_train = utils.sample_random_inputs(
            weather_data, n_samples, order, input_feature, label, steps_ahead=steps_ahead)
    n_inputs = X_train.shape[1]

    likelihood = gpf.likelihoods.HeteroskedasticTFPConditional(scale_transform=tfp.bijectors.Exp())
    if 'time' in input_feature:
        kernel_mean = (gpf.kernels.SquaredExponential(
            # lengthscales=[5]*(n_inputs-1), active_dims = list(range(n_inputs-1)))
                lengthscales=[5]*(order), active_dims = list(range(order)))
            * gpf.kernels.SquaredExponential(
                lengthscales=[5]*(n_inputs-1-order), active_dims = list(range(order, n_inputs-1)))
            # * gpf.kernels.SquaredExponential(active_dims=[n_inputs-1]))
            # + gpf.kernels.White()
        )
        kernel_var = (gpf.kernels.SquaredExponential(
            # lengthscales=[1]*(n_inputs-1), active_dims = list(range(n_inputs-1))) + 
            lengthscales=[5]*(order), active_dims = list(range(order)))
            * gpf.kernels.SquaredExponential(
                lengthscales=[5]*(n_inputs-1-order), active_dims = list(range(order, n_inputs-1)))
            * gpf.kernels.SquaredExponential(active_dims=[n_inputs-1])
            # + gpf.kernels.White()
            )
    else:
        kernel_mean = gpf.kernels.SquaredExponential(lengthscales=[1]*n_inputs)
        kernel_var = gpf.kernels.SquaredExponential(lengthscales=[1]*n_inputs)
    kernel = gpf.kernels.SeparateIndependent(
        [
            kernel_mean,
            kernel_var
        ]
    )
    
    n_z1 = opt['n_z1']
    n_z2 = opt['n_z2']
    if load:
        Z1 = np.loadtxt('Z1.txt')
        Z2 = np.loadtxt('Z2.txt')
    else:
        Z1, _ = utils.sample_random_inputs(
            weather_data, n_z1, order, input_feature, label, max_steps_ahead=opt['steps_ahead'])
        Z2, _ = utils.sample_random_inputs(
            weather_data, n_z1, order, input_feature, label, max_steps_ahead=opt['steps_ahead'])
    
    # noise_add = np.random.normal(scale=0.01, size=Z1.shape)
    # noise_add[:,-1] = 0
    # Z1 = Z1 + noise_add
    # noise_add = np.random.normal(scale=0.01, size=Z1.shape)
    # noise_add[:,-1] = 0
    # Z2 = Z2 + noise_add

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
    loss_fn = gp.training_loss_closure((X_train[:n_train_1,:], y_train[:n_train_1,:]))

    gpf.utilities.set_trainable(gp.q_mu, False)
    gpf.utilities.set_trainable(gp.q_sqrt, False)

    variational_vars = [(gp.q_mu, gp.q_sqrt)]
    natgrad_opt = gpf.optimizers.NaturalGradient(gamma=0.1)

    adam_vars = gp.trainable_variables
    adam_opt = tf.optimizers.Adam(0.1)

    @tf.function
    def training_step():
        print(loss_fn())
        natgrad_opt.minimize(loss_fn, variational_vars)
        print(loss_fn())
        adam_opt.minimize(loss_fn, adam_vars)
        print(loss_fn())
    
    max_epochs = opt['epochs_first_training']
    loss_lb = opt['loss_lb']
    for i in range(max_epochs+1):
        try:
            training_step()
        except ValueError:
            print('Likelihood is nan')
            loss_fn = gp.training_loss_closure(
                (X_train[:n_train_1,:], 
                 y_train[:n_train_1,:]+np.random.normal(size=(n_train_1,1), scale=0.1))
                 )
        if loss_fn().numpy() < loss_lb:
            break
        if True:#opt['verbose'] and i%20==0:
            print(f"Epoch {i} - Loss: {loss_fn().numpy() : .4f}")
    
    # Second training on full data set
    loss_fn = gp.training_loss_closure((X_train, y_train))

    # gpf.utilities.set_trainable(gp.q_mu, False)
    # gpf.utilities.set_trainable(gp.q_sqrt, False)

    # variational_vars = [(gp.q_mu, gp.q_sqrt)]
    natgrad_opt = gpf.optimizers.NaturalGradient(gamma=0.1)

    # adam_vars = gp.trainable_variables
    adam_opt = tf.optimizers.Adam(0.1)

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
            loss_fn = gp.training_loss_closure((
                X_train, 
                y_train+np.random.normal(size=y_train.shape, scale=0.1))
                )
        # print(adam_vars[2])
        # print(adam_vars[5])
        if loss_fn().numpy() < loss_lb:
            break
        if True:#opt['verbose'] and i%20==0:
            print(f"Epoch {i} - Loss: {loss_fn().numpy() : .4f}")
            
    
    return gp

if __name__ == "__main__":
    start_time = datetime.datetime(2020,1,1)
    end_time = datetime.datetime(2022,12,31)
    end_time_train = datetime.datetime(2021,12,31)

    n_last = 4
    input_feature = 'error & nwp'
    label = 'error'
    print(input_feature)
    print(label)
    print(f'n_last = {n_last}')

    opt = {'end_date_train': end_time_train,
           'order': n_last,
           'input_feature': input_feature,
           'label': label,
           'n_z1': 2000,
           'n_z2': 200,
           'epochs_first_training': 100,
           'max_epochs_second_training': 100,
           'loss_lb': 10,
           'verbose': True,
           'steps_ahead': 2,
           'multithread': True}
    
    weather_data = load_weather_data(start_time, end_time)
    X_train, y_train = get_training_data(weather_data, opt)
    pass