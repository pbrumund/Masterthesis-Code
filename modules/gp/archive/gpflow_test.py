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

    if feature == 'error':
        return measurements - wind_predictions
    elif feature == 'measurement':
        return measurements
    elif feature == 'measurement & error':
        return np.concatenate([measurements, measurements-wind_predictions])
    elif feature == 'prediction & error':
        return np.concatenate([wind_predictions, measurements-wind_predictions])
    elif feature == 'prediction & measurement':
        return np.concatenate([wind_predictions, measurements])
    elif feature == 'error & nwp':
        return np.concatenate([measurements-wind_predictions, [measurements[-1], wind_prediction_at_step, np.sqrt(cape), temperature]])
    elif feature == 'error & nwp & time':
        return np.concatenate([measurements-wind_predictions, [measurements[-1], wind_prediction_at_step, np.sqrt(cape), temperature, steps_ahead]])
    elif feature == 'measurement & nwp':
        return np.concatenate([measurements, wind_predictions, [wind_prediction_at_step, np.sqrt(cape), temperature]])
    elif feature == 'measurement & nwp & time':
        return np.concatenate([measurements, wind_predictions, [wind_prediction_at_step, np.sqrt(cape), temperature, steps_ahead]])
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
        



if __name__ == "__main__":

    # Data loading
    start_time = datetime.datetime(2020,1,1)
    end_time = datetime.datetime(2020,1,31)

    n_samples = 100
    n_last = 3
    input_feature = 'error & nwp '
    label = 'error'
    steps_ahead = 1
    print(input_feature)
    print(label)
    print(f'n_last = {n_last}, steps_ahead = {steps_ahead}')

    weather_data = load_weather_data(start_time, end_time)
    
    # X_train, y_train = sample_random_inputs(weather_data, n_samples, n_features, input_feature, label, max_steps_ahead=60)
    X_train = np.array([])
    y_train = np.array([])

    i_
    for time in weather_data['times_meas'][n_last-1:-steps_ahead]:
        x_i = generate_features(weather_data, time, n_last=n_last, feature=input_feature, steps_ahead=steps_ahead)
        n_features = len(x_i)
        X_train = np.append(X_train, x_i)
        y_train = np.append(y_train, generate_labels(weather_data, time, label, steps_ahead))
    X_train = np.reshape(X_train, (-1,n_features))

    M = 20
    X_train_mean, y_train_mean = sparsify_training_data(X_train, y_train, M)
    X_train_var, y_train_var = estimate_variance(X_train, y_train, M)
    X_train, y_train = sparsify_training_data(X_train, y_train, 1000)

    likelihood = gpf.likelihoods.HeteroskedasticTFPConditional(
        distribution_class=tfp.distributions.Normal,  # Gaussian Likelihood
        scale_transform=tfp.bijectors.Exp(),  # Exponential Transform
    )

    kernel = gpf.kernels.SeparateIndependent(
        [
            gpf.kernels.SquaredExponential(),  # This is k1, the kernel of f1
            gpf.kernels.SquaredExponential(),  # this is k2, the kernel of f2
        ]
    )
    pass
    # model = gpf.models.GPR((X_train_mean, y_train_mean), kernel, likelihood=likelihood)
    # model = gpf.models.
    M = 20  # Number of inducing variables for each f_i

    # Initial inducing points position Z
    Z = np.linspace(X_train.min(), X_train.max(), M)[:, None]  # Z must be of shape [M, 1]



    inducing_variable = gpf.inducing_variables.SeparateIndependentInducingVariables(
        [
            gpf.inducing_variables.InducingPoints(X_train_mean),  # This is U1 = f1(Z1)
            gpf.inducing_variables.InducingPoints(X_train_var),  # This is U2 = f2(Z2)
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
    adam_opt = tf.optimizers.Adam(0.01)


    @tf.function
    def optimisation_step():
        natgrad_opt.minimize(loss_fn, variational_vars)
        adam_opt.minimize(loss_fn, adam_vars)
   
    epochs = 10000
    log_freq = 20

    for epoch in range(1, epochs + 1):
        optimisation_step()

        # For every 'log_freq' epochs, print the epoch and plot the predictions against the data
        if epoch % log_freq == 0 and epoch > 0:
            print(f"Epoch {epoch} - Loss: {loss_fn().numpy() : .4f}")
            Ymean, Yvar = model.predict_y(X_train)
            Ymean = Ymean.numpy().squeeze()
            Ystd = tf.sqrt(Yvar).numpy().squeeze()
            # plot_distribution(X, Y, Ymean, Ystd)


    pass