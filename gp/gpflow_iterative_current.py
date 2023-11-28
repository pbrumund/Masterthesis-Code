import numpy as np
import gpflow as gpf
import tensorflow as tf
import tensorflow_probability as tfp
import datetime
import random
import matplotlib.pyplot as plt

import utils
from fileloading import load_weather_data


def get_iterative_model(weather_data, opt):
    order = opt['order']
    n_samples = opt['n_samples']
    input_feature = opt['input_feature']
    label = opt['label']

    if 'time' in input_feature:
        X_train, y_train = utils.sample_random_inputs(
            weather_data, n_samples, order, input_feature, label, max_steps_ahead=opt['steps_ahead'])
    else:
        X_train, y_train = utils.sample_random_inputs(
            weather_data, n_samples, order, input_feature, label, steps_ahead=1)
    n_inputs = X_train.shape[1]

    likelihood = gpf.likelihoods.HeteroskedasticTFPConditional(scale_transform=tfp.bijectors.Exp())
    if 'time' in input_feature:
        kernel_mean = (
            gpf.kernels.SquaredExponential(lengthscales=[5]*(n_inputs-1), 
                                           active_dims = list(range(n_inputs-1)))
            * gpf.kernels.SquaredExponential(active_dims=[n_inputs-1], lengthscales=[10])
        )
        kernel_var = (
            gpf.kernels.SquaredExponential(
                lengthscales=[5]*(n_inputs-1), active_dims = list(range(n_inputs-1))) 
            * gpf.kernels.SquaredExponential(active_dims=[n_inputs-1], lengthscales=[5]))
    else:
        kernel_mean = gpf.kernels.SquaredExponential(lengthscales=[1]*n_inputs)
        kernel_var = gpf.kernels.SquaredExponential(lengthscales=[1]*n_inputs)
    kernel = gpf.kernels.SeparateIndependent(
        [
            kernel_mean,
            kernel_var
        ]
    )
    
    n_z = opt['n_z']
    Z1, _ = utils.sample_random_inputs(
        weather_data, n_z, n_last, input_feature, label, steps_ahead=1)
    Z2, _ = utils.sample_random_inputs(
        weather_data, n_z, n_last, input_feature, label, steps_ahead=1)
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
    
    loss_fn = gp.training_loss_closure((X_train, y_train))

    gpf.utilities.set_trainable(gp.q_mu, False)
    gpf.utilities.set_trainable(gp.q_sqrt, False)

    variational_vars = [(gp.q_mu, gp.q_sqrt)]
    natgrad_opt = gpf.optimizers.NaturalGradient(gamma=0.05)

    adam_vars = gp.trainable_variables
    adam_opt = tf.optimizers.Adam(0.1)

    @tf.function
    def training_step():
        try:
            natgrad_opt.minimize(loss_fn, variational_vars)
        finally:
            adam_opt.minimize(loss_fn, adam_vars)
    
    max_epochs = opt['max_epochs']
    loss_lb = opt['loss_lb']
    for i in range(max_epochs+1):
        try:
            training_step()
        except:
            #variational_vars[0][0] += np.random.normal(size=variational_vars[0][0].shape, scale=1e-3)
            #variational_vars[0][1] += np.random.normal(size=variational_vars[0][1].shape, scale=1e-3)
            print('Graph execution error occured, adding random noise to variational variables')
        if loss_fn().numpy() < loss_lb:
            break
        if opt['verbose'] and i%10==0:
            print(f"Epoch {i} - Loss: {loss_fn().numpy() : .4f}")
    
    return gp

def get_input_uncertainty(input_uncertainty, var):
    # n_uncertain_inputs = min(opt['order'], steps-1)
    input_uncertainty = np.append(input_uncertainty, var)[1:]
    return input_uncertainty

def get_new_input(weather_data, time, x_last, new_prediction, steps_ahead, opt):
    x_last = x_last.T
    input_feature = opt['input_feature']
    if 'time' in input_feature:
        k = x_last[-1] + 1
        x_last = x_last[:-1] 
    if 'nwp' in input_feature:
        cape = utils.get_NWP(weather_data, time, steps_ahead, 'specific_convective_available_potential_energy')
        sqrt_cape = np.sqrt(cape)
        temperature = utils.get_NWP(weather_data, time, steps_ahead, 'air_temperature_2m')
        wind_prediction_at_step = utils.get_NWP(weather_data, time, steps_ahead, 'wind_speed_10m')
        x_NWP = np.array([wind_prediction_at_step, sqrt_cape, temperature])
        x_last = x_last[:-4]
    x_last = x_last[1:]
    label = opt['label']
    prediction_NWP = utils.get_NWP(weather_data, time, steps_ahead-1, 'wind_speed_10m')
    if label == 'error':
        wind_predicted = new_prediction + prediction_NWP
    elif label == 'measurement':
        wind_predicted = new_prediction
    elif label == 'change of wind speed':
        wind_now = utils.get_wind_value(weather_data, time, 0, 'wind_speed_10m')
        wind_predicted = wind_now + new_prediction
    else:
        raise ValueError('Unknown value for label')
    if 'error' in input_feature:
        x_new = wind_predicted - prediction_NWP
    elif 'measurement' in input_feature:
        x_new = wind_predicted
    x_out = np.append(x_last, x_new)
    if 'error' in input_feature:
        x_out = np.append(x_out, wind_predicted)
    if 'nwp' in input_feature:
        x_out = np.append(x_out, x_NWP)
    if 'time' in input_feature:
        x_out = np.append(x_out, k)
    return x_out.reshape((1,-1))

def predict(gp, input, input_uncertainty):
    gp_mean, gp_var = gp.predict_y(input)
    return gp_mean, gp_var

def predict_trajectory(weather_data, t_start, steps_ahead, gp, opt):
    mean_traj = np.zeros(steps_ahead)
    var_traj = np.zeros(steps_ahead)
    input_uncertainty = np.zeros(opt['order'])
    # Simple model ignoring input uncertainty:
    x_0 = utils.generate_features(weather_data, t_start, opt['order'], opt['input_feature'], 1).reshape((1,-1))
    x = x_0
    for i in range(steps_ahead):
        gp_mean, gp_var = gp.predict_y(x)
        x = get_new_input(weather_data, t_start, x, gp_mean, i+2, opt)
        mean_traj[i] = gp_mean
        var_traj[i] = gp_var
    gp_pred = np.zeros(steps_ahead)
    for i in range(steps_ahead):
        if label == 'error':
            prediction_NWP_i = utils.get_NWP(weather_data, t_start, i+1)
            gp_pred[i] = prediction_NWP_i + mean_traj[i]
        elif label == 'measurement':
            gp_pred[i] = mean_traj[i]
        elif label == 'change of wind speed':
            wind_now = utils.get_wind_value(weather_data, t_start, 0)
            gp_pred[i] = wind_now + mean_traj[i]

    return gp_pred, var_traj
        

if __name__ == "__main__":
    random.seed(10)
    # Data loading
    start_time = datetime.datetime(2020,1,1)
    end_time = datetime.datetime(2022,12,31)

    n_samples = 2000
    n_last = 3
    input_feature = 'error & nwp & time'
    label = 'error'
    print(input_feature)
    print(label)
    print(f'n_last = {n_last}')

    opt = {'n_samples': n_samples,
           'order': n_last,
           'input_feature': input_feature,
           'label': label,
           'n_z': 250,
           'max_epochs': 1000,
           'loss_lb': 10,
           'verbose': True,
           'steps_ahead': 60}
    
    weather_data = load_weather_data(start_time, end_time)

    gp = get_iterative_model(weather_data, opt)
    
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
    plt.plot(times, gp_traj + 2*np.sqrt(var_traj), color='k')
    plt.plot(times, gp_traj - 2*np.sqrt(var_traj), color='k')
    plt.show()
    pass
    # weather_data = add_variance(weather_data)
    
    # n_measurements = len(weather_data['times_meas'])

    