import numpy as np
import gpflow as gpf
import tensorflow as tf
import tensorflow_probability as tfp
import datetime
import random
import matplotlib.pyplot as plt

from fileloading import load_weather_data
import utils

def generate_features(weather_data, time, n_last=3, feature='error', steps_ahead = 1):
    measurements = np.zeros(n_last)
    wind_predictions = np.zeros(n_last)

    start_time = time - datetime.timedelta(minutes=(n_last-1)*10)
    for i in range(n_last):
        measurements[i] = utils.get_wind_value(weather_data, start_time, i)
        wind_predictions[i] = utils.get_NWP(weather_data, start_time, i, 'wind_speed_10m')
    
    cape = utils.get_NWP(weather_data, time, steps_ahead, 'specific_convective_available_potential_energy')
    temperature = utils.get_NWP(weather_data, time, steps_ahead, 'air_temperature_2m')
    wind_prediction_at_step = utils.get_NWP(weather_data, time, steps_ahead, 'wind_speed_10m')
    errors = measurements - wind_predictions

    # Normalize inputs
    # measurements = (measurements-weather_data['mean_meas'])/weather_data['std_meas']
    # wind_predictions = (wind_predictions-weather_data['mean_pred'])/weather_data['std_pred']
    # errors = (errors-weather_data['mean_error'])/weather_data['std_error']
    sqrt_cape = np.sqrt(cape)
    # sqrt_cape = (sqrt_cape-weather_data['mean_cape'])/weather_data['std_cape']
    # temperature = (temperature-weather_data['mean_temp'])/weather_data['std_temp']

    if feature == 'error':
        return errors
    elif feature == 'measurement':
        return measurements
    elif feature == 'measurement & error':
        return np.concatenate([measurements, errors])
    elif feature == 'prediction & error':
        return np.concatenate([wind_predictions, errors])
    elif feature == 'prediction & measurement':
        return np.concatenate([wind_predictions, measurements])
    elif feature == 'error & nwp':
        return np.concatenate([errors, [measurements[-1], wind_prediction_at_step, sqrt_cape, temperature]])
    elif feature == 'error & nwp & time':
        return np.concatenate([errors, [measurements[-1], wind_prediction_at_step, sqrt_cape, temperature, steps_ahead]])
    elif feature == 'measurement & nwp':
        return np.concatenate([measurements, wind_predictions, [wind_prediction_at_step, sqrt_cape, temperature]])
    elif feature == 'measurement & nwp & time':
        return np.concatenate([measurements, wind_predictions, [wind_prediction_at_step, sqrt_cape, temperature, steps_ahead]])
    else:
        raise ValueError("Unknown value for feature")

def generate_labels(weather_data, time, label='error', steps_ahead = 1):
    measurement = utils.get_wind_value(weather_data, time, steps=steps_ahead)
    if label == 'error':
        prediction = utils.get_NWP(weather_data, time, steps=steps_ahead)
        return measurement - prediction
    elif label == 'measurement':
        return measurement
    elif label == 'change of wind speed':
        current_measurement = utils.get_wind_value(weather_data, time, steps=0)
        return measurement-current_measurement
    else:
        raise ValueError("Unknown value for feature")

def sample_random_inputs(weather_data, n_samples, n_features, feature='error', label='error', steps_ahead = 1, max_steps_ahead = 1):
    n_data_points = len(weather_data['times_meas'])
    sample_indices = random.sample(range(n_features, n_data_points-steps_ahead), n_samples)
    if max_steps_ahead > 1:
        steps = [random.randint(1, max_steps_ahead) for i in range(n_samples)]
    else:
        steps = steps_ahead*np.ones(n_samples)

    features = [generate_features(weather_data, weather_data['times_meas'][sample_indices[i]], n_features, feature, steps[i]) for i in range(len(sample_indices))]
    labels = [generate_labels(weather_data, weather_data['times_meas'][i], label, steps_ahead) for i in sample_indices]

    X = np.array(features).T
    y = np.array(labels).reshape((1,-1))

    return X, y

def sparsify_training_data(X, y, M):
        n_sample = M
        n_sample = min(n_sample, len(y))
        rand_indices = random.sample(range(len(y)), n_sample)
        X_train_mean = X[rand_indices,:]
        y_train_mean = np.reshape(y[rand_indices], (-1,1))
        return X_train_mean, y_train_mean

def generate_multi_step_training_data(weather_data, max_steps):
    X = np.zeros()
    for i in range(max_steps):
        pass

def estimate_variance(X, y, M):
    # normalize x for every dimension
    n_inputs = X.shape[1]
    X_norm = np.zeros(X.shape)
    var_x = np.zeros(n_inputs)
    mean_x = np.zeros(n_inputs)
    for i in range(n_inputs):
        mean_i = np.mean(X[:,i])
        var_i = np.var(X[:,i])
        if var_i != 0:
            X_norm[:,i] = (X[:,i])/np.sqrt(var_i)
            var_x[i] = var_i
        else:
            X_norm[:,i] = X[:,i]
            var_x[i] = 1
    
    #draw random points
    n_points = M
    n_points = min(n_points, len(y))
    rand_indices = random.sample(range(len(y)), n_points)

    #find n_closest closest neighbors
    n_closest = 50
    n_closest = min(n_closest, len(y))
    X_mean = np.zeros((n_points, n_inputs))
    y_var = np.zeros(n_points)
    for i in range(n_points):
        index = rand_indices[i]
        x_i = X_norm[index,:]
        y_i = y[index]
        X_others = np.delete(X_norm,index,0)
        y_others = np.delete(y, index)
        distances = [np.linalg.norm(x_i - X_others[k,:]) for k in range(X_others.shape[0])]
        d_sort = np.argsort(distances)[:n_closest]
        X_closest = np.array([X_others[k,:] for k in d_sort]).T
        y_closest = np.array([y_others[k] for k in d_sort])
        X_mean[i,:] = np.multiply(np.mean(X_closest, axis=1), np.sqrt(var_x))
        y_var[i] = np.var(y_closest)

    X_train_var = X_mean
    mean_var = np.mean(y_var)
    y_train_var = np.reshape(np.log(y_var), (-1,1))
    return X_train_var, y_train_var
        
def add_variance(weather_data):
    prediction_errors = np.array([utils.get_wind_value(weather_data, t, 0) - utils.get_NWP(weather_data, t, 0) for t in weather_data['times_meas']])
    weather_data['mean_error'] = np.mean(prediction_errors)
    weather_data['std_error'] = np.sqrt(np.var(prediction_errors))
    weather_data['mean_temp'] = np.mean(weather_data['air_temperature_2m_sh'])
    weather_data['std_temp'] = np.sqrt(np.var(weather_data['air_temperature_2m_sh'])) 
    weather_data['mean_cape'] = np.mean(np.sqrt(weather_data['specific_convective_available_potential_energy_sh']))
    weather_data['std_cape'] = np.sqrt(np.var(np.sqrt(weather_data['specific_convective_available_potential_energy_sh'])))
    weather_data['mean_pred'] = np.mean(weather_data['wind_speed_10m_sh'])
    weather_data['std_pred'] = np.sqrt(np.var(weather_data['wind_speed_10m_sh']))
    weather_data['mean_meas'] = np.mean(weather_data['meas'])
    weather_data['std_meas'] = np.sqrt(np.var(weather_data['meas']))
    return weather_data
        

if __name__ == "__main__":

    # Data loading
    start_time = datetime.datetime(2020,1,1)
    end_time = datetime.datetime(2022,12,31)

    n_samples = 25000
    n_last = 3
    input_feature = 'measurement & nwp & time'
    label = 'change of wind speed'
    steps_ahead = 60
    print(input_feature)
    print(label)
    print(f'n_last = {n_last}, steps_ahead = {steps_ahead}')

    weather_data = load_weather_data(start_time, end_time)
    # weather_data = add_variance(weather_data)
    
    # X_train, y_train = sample_random_inputs(weather_data, n_samples, n_features, input_feature, label, max_steps_ahead=60)
    n_measurements = len(weather_data['times_meas'])

    def gen_features(index_list):   #TODO: Normalize features
        X_train = np.array([])
        y_train = np.array([])
        
        for i, index in enumerate(index_list):
            if i%int(len(index_list)/10)==0:
                print(int(100*i/len(index_list)))
            if 'time' in input_feature:
                # dt = int(np.random.exponential(20))
                dt = random.randint(1,steps_ahead)
            else:
                dt = steps_ahead
            time = weather_data['times_meas'][index]
            x_i = generate_features(weather_data, time, n_last=n_last, feature=input_feature, steps_ahead=dt)
            n_features = len(x_i)
            X_train = np.append(X_train, x_i)
            y_train = np.append(y_train, generate_labels(weather_data, time, label, steps_ahead))
        X_train = np.reshape(X_train, (-1,n_features))
        y_train = np.reshape(y_train, (-1,1))
        return X_train, y_train

    def get_random_data(n_points):
        n_sample = min(n_points, n_measurements)
        max_index = int(2*n_measurements/3)
        rand_indices = random.sample(range(n_last,max_index), n_sample)
        return gen_features(rand_indices)
    
    # X_train = np.array([])
    # y_train = np.array([]) 
    # for index in range(n_last,n_measurements-steps_ahead):#  in weather_data['times_meas'][n_last-1:-steps_ahead]:
    #     time = weather_data['times_meas'][index]
    #     x_i = generate_features(weather_data, time, n_last=n_last, feature=input_feature, steps_ahead=steps_ahead)  # TODO: first choose random data, then generate features
    #     n_features = len(x_i)
    #     X_train = np.append(X_train, x_i)
    #     y_train = np.append(y_train, generate_labels(weather_data, time, label, steps_ahead))
    # X_train = np.reshape(X_train, (-1,n_features))

    M = 250
    X_train, y_train = get_random_data(n_samples)
    #X_train_mean, y_train_mean = sparsify_training_data(X_train, y_train, M)
    
    Z1, _ = sparsify_training_data(X_train, y_train, M)  # get random inputs # get mean inputs of random clusters
    Z2, _ = sparsify_training_data(X_train, y_train, M)

    noise_add = np.random.normal(scale=0.01, size=Z1.shape)
    noise_add[:,-1] = 0
    Z1 = Z1 + noise_add
    noise_add = np.random.normal(scale=0.01, size=Z1.shape)
    noise_add[:,-1] = 0
    Z2 = Z2 + noise_add


    # https://gpflow.github.io/GPflow/develop/notebooks/advanced/heteroskedastic.html#Heteroskedastic-Regression

    likelihood = gpf.likelihoods.HeteroskedasticTFPConditional(
        distribution_class=tfp.distributions.Normal,  # Gaussian Likelihood
        scale_transform=tfp.bijectors.Exp(),  # Exponential Transform
    )
    n_inputs = X_train.shape[1]
    kernel_mean = (
        gpf.kernels.SquaredExponential(
            lengthscales=[5]*(n_inputs-1), active_dims = list(range(n_inputs-1)))
        * gpf.kernels.Exponential(active_dims=[n_inputs-1], lengthscales=[1])
        )
    kernel_var = (gpf.kernels.SquaredExponential(lengthscales=[5]*(n_inputs-1), active_dims = list(range(n_inputs-1)))
                  * gpf.kernels.SquaredExponential(active_dims=[n_inputs-1], lengthscales=[10]))
    kernel = gpf.kernels.SeparateIndependent(
        [
            kernel_mean,# * gpf.kernels.Matern12(lengthscales=[5], active_dims=[8]),  # This is k1, the kernel of f1
            kernel_var
            # gpf.kernels.RationalQuadratic(
            #     lengthscales=[1 for i in range(8)], 
            #     active_dims=[0,1,2,3,4,5,6,7]
            #     ),  # this is k2, the kernel of f2
        ]
    )
    pass
    # model = gpf.models.GPR((X_train_mean, y_train_mean), kernel, likelihood=likelihood)
    # model = gpf.models.

    # Initial inducing points position Z
    # Z = np.linspace(X_train.min(), X_train.max(), M)[:, None]  # Z must be of shape [M, 1]



    inducing_variable = gpf.inducing_variables.SeparateIndependentInducingVariables(
        [
            gpf.inducing_variables.InducingPoints(Z1),  # This is U1 = f1(Z1)
            gpf.inducing_variables.InducingPoints(Z2),  # This is U2 = f2(Z2)
        ]
    )

    model = gpf.models.SVGP(
        kernel=kernel,
        likelihood=likelihood,
        inducing_variable=inducing_variable,
        num_latent_gps=likelihood.latent_dim,
    )
    
    data = (X_train, y_train)
    loss_fn = model.training_loss_closure(data)

    gpf.utilities.set_trainable(model.q_mu, False)
    gpf.utilities.set_trainable(model.q_sqrt, False)

    variational_vars = [(model.q_mu, model.q_sqrt)]
    natgrad_opt = gpf.optimizers.NaturalGradient(gamma=0.1)

    adam_vars = model.trainable_variables
    adam_opt = tf.optimizers.Adam(0.1)
    # de_opt = tfp.optimizer.differential_evolution_minimize(loss_fn, adam_vars)


    @tf.function
    def optimisation_step():
        natgrad_opt.minimize(loss_fn, variational_vars)
        adam_opt.minimize(loss_fn, adam_vars)
   
    epochs = 400
    log_freq = 20

    for epoch in range(1, epochs + 1):
        optimisation_step()

        # For every 'log_freq' epochs, print the epoch and plot the predictions against the data
        if epoch % log_freq == 0:
            print(f"Epoch {epoch} - Loss: {loss_fn().numpy() : .4f}")
            
            # plot_distribution(X, Y, Ymean, Ystd)

    t = datetime.datetime(2022,2,1,5,10)
    steps_ahead = 60
    gp_pred_mean = np.zeros(steps_ahead)
    gp_pred_ub = np.zeros(steps_ahead)
    gp_pred_lb = np.zeros(steps_ahead)
    nwp_pred = np.zeros(steps_ahead)
    wind_meas = np.zeros(steps_ahead)
    times = []
    for i in range(steps_ahead):
        prediction_NWP = utils.get_NWP(weather_data, time=t, steps=i+1)
        nwp_pred[i] = prediction_NWP
        x_i = generate_features(weather_data, t, n_last, input_feature, i+1).reshape((1,-1))
        gp_mean, gp_var = model.predict_y(x_i)
        if label == 'error':
            prediction_GP = prediction_NWP + gp_mean
        elif label == 'measurement':
            prediction_GP = gp_mean
        elif label == 'change of wind speed':
            current_wind_speed = utils.get_wind_value(weather_data, t, steps=0)
            prediction_GP = current_wind_speed + gp_mean
        prediction_GP_lower = prediction_GP - np.sqrt(gp_var)
        prediction_GP_upper = prediction_GP + np.sqrt(gp_var)
        gp_pred_mean[i] = prediction_GP
        gp_pred_lb[i] = prediction_GP_lower
        gp_pred_ub[i] = prediction_GP_upper
        wind_meas[i] = utils.get_wind_value(weather_data, t, i)
        times.append(t + (i+1)*datetime.timedelta(minutes=10))
    plt.figure()
    plt.plot(times, nwp_pred)
    plt.plot(times, gp_pred_lb, color='k')
    plt.plot(times, gp_pred_mean)
    plt.plot(times, gp_pred_ub, color='k')
    plt.plot(times, wind_meas)
    plt.show()
    pass