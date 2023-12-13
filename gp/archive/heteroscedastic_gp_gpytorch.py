import numpy as np
import gpytorch
import torch
import datetime
import random
import matplotlib.pyplot as plt
from multiprocessing import Pool

import utils
from fileloading import load_weather_data
from gpytorch_models import ExactGP, SparseGP, SKIPGP, HeteroscedasticGP

def get_training_data(weather_data, opt):
    # generate X_train and y_train
    order = opt['order']
    # n_samples = opt['n_samples']
    input_feature = opt['input_feature']
    label = opt['label']
    max_steps_ahead = opt['steps_ahead']
    filename_X = f"gp\\training_data\X_train_{order}_last_{input_feature}_{label}_1-{max_steps_ahead}step.txt"
    filename_y = f"gp\\training_data\y_train_{order}_last_{input_feature}_{label}_1-{max_steps_ahead}step.txt"
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
    
    # train gp if no file has been found
    print(f'training gp for {steps_ahead} steps ahead')
    X_train_full, y_train_full = get_training_data(weather_data, opt)
    X_train_sparse, y_train_sparse = utils.sparsify_training_data(X_train_full, y_train_full, 10000)
    y_train = y_train_sparse.reshape((-1,))
    X_train, y_train = torch.from_numpy(X_train_sparse).float(), torch.from_numpy(y_train).float()
    n_inputs = X_train.shape[1]
    n_samples = X_train.shape[0]
    # set up noise GP
    # X_train_noise, y_train_noise = utils.estimate_variance(X_train_sparse, y_train_sparse, 1000)
    X_train_noise = np.loadtxt('X_train_noise_gpytorch.csv')
    y_train_noise = np.loadtxt('y_train_noise_gpytorch.csv')
    y_train_noise = 1e2*y_train_noise.reshape((-1,))
    X_train_noise, y_train_noise = torch.from_numpy(X_train_noise).float(), torch.from_numpy(y_train_noise).float()

    likelihood_noise = gpytorch.likelihoods.GaussianLikelihood()
    likelihood_noise.noise = 1e-2
    gp_noise = ExactGP(X_train_noise, y_train_noise, likelihood_noise, opt)
    gp_noise.covar_module.outputscale = 1e4
    gp_noise.train()
    likelihood_noise.train()

    optimizer = torch.optim.Adam(gp_noise.parameters(), lr=opt['learning_rate'])
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood_noise, gp_noise)
    # train noise gp
    
    with gpytorch.settings.cholesky_jitter(1e-3):
        for i in range(opt['epochs']):
            optimizer.zero_grad()
            if isinstance(gp_noise, SKIPGP):
                with gpytorch.settings.use_toeplitz(False), gpytorch.settings.max_root_decomposition_size(50):
                    output = gp(X_train)
                    loss = -mll(output, y_train)
                    loss.backward()
            else:
                output = gp_noise(X_train_noise)
                loss = -mll(output, y_train_noise)
                loss.backward()
            print(f'Iteration {i}: loss = {loss.mean().item()}')
            optimizer.step()
            print(gp_noise.covar_module.base_kernel.lengthscale.detach().numpy())
            print(likelihood_noise.noise.detach().numpy(), gp_noise.covar_module.outputscale.detach().numpy())
            if i > 0:
                if loss.item()/previous_loss > 0.999 or gp_noise.covar_module.base_kernel.lengthscale.detach().numpy()[0,0]<0.5:
                    break
                if loss.item() < opt['loss_lb']:
                    break
            previous_loss = loss.item()

    # train()

    # estimate variance at training points
    gp_noise.eval()
    likelihood_noise.eval()
    noise = np.zeros(X_train.shape[0])
    # with torch.no_grad():
    #     for i in range(X_train.shape[0]):
    #         noise[i] = gp_noise(X_train[i:i+1]).mean.numpy()/1e2
    # noise = torch.from_numpy(np.exp(noise)).float()
    noise = torch.from_numpy(np.loadtxt('noise_values.csv')).float()
    #set up mean gp
    likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=noise)
    # likelihood.noise = 5

    gp = ExactGP(X_train, y_train, likelihood, opt)
    gp.covar_module.outputscale = 2

    gp.train()
    likelihood.train()

    optimizer = torch.optim.Adam(gp.parameters(), lr=opt['learning_rate'])
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp)

    with gpytorch.settings.cholesky_jitter(1e-2):
        for i in range(opt['epochs']):
            optimizer.zero_grad()
            if isinstance(gp, SKIPGP):
                with gpytorch.settings.use_toeplitz(False), gpytorch.settings.max_root_decomposition_size(50):
                    output = gp(X_train)
                    loss = -mll(output, y_train)
                    loss.backward()
            else:
                output = gp(X_train)
                loss = -mll(output, y_train)
                loss.backward()
            print(f'Iteration {i}: loss = {loss.mean().item()}')
            optimizer.step()
            # if i > 0:
            #     if loss.item()/previous_loss > 0.999:
            #         break
            previous_loss = loss.item()
    # with gpytorch.settings.use_toeplitz(False):
    # train()
    # gp.eval()
    # likelihood.eval()
    return gp, likelihood, gp_noise

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

def predict_uncertain_inputs(gp, likelihood, gp_noise, x, input_cov, opt):
    # propagate input uncertainty through monte carlo samples
    n_samples = opt['n_samples_mc']
    n_dimensions = x.shape[1]
    outputs = np.zeros(n_samples)
    inputs = np.zeros((n_samples, n_dimensions))
    rng = np.random.default_rng()
    
    gp.eval()
    likelihood.eval()
    # torch.no_grad()
    for i in range(n_samples):
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            if isinstance(gp, SKIPGP):
                with gpytorch.settings.use_toeplitz(True), gpytorch.settings.max_root_decomposition_size(50):#, gpytorch.settings.max_preconditioner_size(10):
                    x_i = rng.multivariate_normal(mean=x.reshape(-1), cov=input_cov).reshape((1,-1))
                    inputs[i,:] = x_i
                    y_pred = likelihood(gp(torch.from_numpy(x_i).float()))
                    mean_i, var_i = y_pred.mean.detach().numpy(), y_pred.variance.detach().numpy()
                    y_i = np.random.normal(loc=mean_i, scale=np.sqrt(var_i))
                    outputs[i] = y_i
            else:
                x_i = rng.multivariate_normal(mean=x.reshape(-1), cov=input_cov).reshape((1,-1))
                inputs[i,:] = x_i
                y_pred = gp(torch.from_numpy(x_i).float())
                mean_i, var_i = y_pred.mean.detach().numpy(), y_pred.variance.detach().numpy()
                var_i += np.exp(gp_noise(torch.tensor(x_i).float()).mean.numpy()/1e2)
                y_i = np.random.normal(loc=mean_i, scale=np.sqrt(var_i))
                outputs[i] = y_i
    # approximate the output distribution as a joint gaussian described by the mean, variance 
    # and covariance between the uncertain inputs and the output
    mean = np.mean(outputs)
    var = np.var(outputs)
    cov = np.array([np.cov(inputs[:,i], outputs)[0,1] for i in range(n_dimensions)])
    # cov = np.cov(outputs, inputs.T)[0,:]
    return mean, var, cov


def predict_trajectory(weather_data, t_start, steps_ahead, gp, likelihood, gp_noise, opt):
    mean_traj = np.zeros(steps_ahead)
    var_traj = np.zeros(steps_ahead)
    
    x_0 = utils.generate_features(
        weather_data, t_start, opt['order'], opt['input_feature'], 1).reshape((1,-1))
    x = x_0
    n_dim = x.shape[1]
    input_cov = np.zeros((n_dim, n_dim))

    for i in range(steps_ahead):
        print(f'steps ahead: {i+1}')
        mean, var, cov = predict_uncertain_inputs(gp, likelihood, gp_noise, x, input_cov, opt)
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
           'n_z': 2000,
           'epochs': 100,
           'learning_rate': 0.7,
           'max_epochs_second_training': 100,
           'loss_lb': 10,
           'verbose': True,
           'steps_ahead': 1,
           'multithread': True,
           'n_samples_mc': 100}
    
    weather_data = load_weather_data(start_time, end_time)

    gp, likelihood, gp_noise = get_gp(weather_data, opt)
    
    
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

    gp_traj, var_traj = predict_trajectory(weather_data, t, steps_ahead, gp, likelihood, gp_noise, opt)

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