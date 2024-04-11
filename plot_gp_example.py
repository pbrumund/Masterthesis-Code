import numpy as np
import gpflow as gpf
import tensorflow as tf
import scipy
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('pgf')
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})
plt.ion()
cm = 1/2.54

rng = np.random.default_rng(seed=3)
x = np.linspace(0,1,200)
# function
f_fun = lambda xi: 10*xi**3-np.cos(xi)
f = f_fun(x)
sigma_fun = lambda xi: np.exp(1.5*np.sin(5*xi))+0.1
sigma = sigma_fun(x)
x_train = rng.random(100)
y_train = np.zeros(100)
for i in range(x_train.shape[0]):
    xi = x_train[i]
    sigma_i = sigma_fun(xi)
    fi = f_fun(xi)
    yi = fi+rng.normal(scale=sigma_i)
    y_train[i] = yi

x_train = x_train.reshape((-1,1))
y_train = y_train.reshape((-1,1))
Z = np.linspace(0,1,10).reshape((-1,1))
inducing_variables = gpf.inducing_variables.SeparateIndependentInducingVariables([
    gpf.inducing_variables.InducingPoints(Z),
    gpf.inducing_variables.InducingPoints(Z)
])

likelihood = gpf.likelihoods.HeteroskedasticTFPConditional()
kernel = gpf.kernels.SeparateIndependent([
    gpf.kernels.SquaredExponential(),gpf.kernels.SquaredExponential()
])
gp = gpf.models.SVGP(kernel=kernel, likelihood=likelihood, inducing_variable=inducing_variables, num_latent_gps=likelihood.latent_dim)

data = (x_train, y_train)
loss = gp.training_loss_closure(data)
gpf.utilities.set_trainable(gp.q_mu, False)
gpf.utilities.set_trainable(gp.q_sqrt, False)
variational_vars = [(gp.q_mu, gp.q_sqrt)]
natgrad_opt = gpf.optimizers.NaturalGradient(gamma=0.1)
adam = tf.optimizers.Adam(0.01)
adam_vars = gp.trainable_variables
@tf.function
def optimisation_step():
    natgrad_opt.minimize(loss, variational_vars)
    adam.minimize(loss, adam_vars)

for epoch in range(1000):
    optimisation_step()

mean_pred, var_pred = gp.predict_y(x.reshape((-1,1)))
sigma_pred = np.sqrt(var_pred).reshape(-1)
mean_pred = mean_pred.numpy().reshape(-1)
# Homoscedastic GP for comparison
inducing_variables = gpf.inducing_variables.InducingPoints(Z)
kernel = gpf.kernels.SquaredExponential()
likelihood = gpf.likelihoods.Gaussian()
gp = gpf.models.SVGP(kernel=kernel, likelihood=likelihood, inducing_variable=inducing_variables)
data = (x_train, y_train)
loss = gp.training_loss_closure(data)
gpf.utilities.set_trainable(gp.q_mu, False)
gpf.utilities.set_trainable(gp.q_sqrt, False)
variational_vars = [(gp.q_mu, gp.q_sqrt)]
natgrad_opt = gpf.optimizers.NaturalGradient(gamma=0.1)
adam = tf.optimizers.Adam(0.01)
adam_vars = gp.trainable_variables
@tf.function
def optimisation_step():
    natgrad_opt.minimize(loss, variational_vars)
    adam.minimize(loss, adam_vars)

for epoch in range(1000):
    optimisation_step()

mean_pred_hom, var_pred_hom = gp.predict_y(x.reshape((-1,1)))
sigma_pred_hom = np.sqrt(var_pred_hom).reshape(-1)
mean_pred_hom = mean_pred_hom.numpy().reshape(-1)

fig, (ax1, ax2) = plt.subplots(2, layout='constrained')
plt_f, = ax2.plot(x, f, color='tab:gray')
ax2.fill_between(x, f-2*sigma, f+2*sigma, color='tab:gray', alpha=0.15)
ax2.fill_between(x, f+sigma, f-sigma, color='tab:gray', alpha=0.35)
plt_y_train = ax2.scatter(x_train, y_train, color='black')
plt_y_pred, = ax2.plot(x, mean_pred, color='tab:blue')
ax2.fill_between(x, mean_pred-2*sigma_pred, mean_pred+2*sigma_pred, color='tab:blue', alpha=0.25)
ax2.fill_between(x, mean_pred+sigma_pred, mean_pred-sigma_pred, color='tab:blue', alpha=0.5)
ax1.plot(x, f, color='tab:gray')
ax1.fill_between(x, f-2*sigma, f+2*sigma, color='tab:gray', alpha=0.15)
ax1.fill_between(x, f+sigma, f-sigma, color='tab:gray', alpha=0.35)
ax1.scatter(x_train, y_train, color='black')
ax1.plot(x, mean_pred, color='tab:blue')
plt_y_pred_hom, = ax1.plot(x, mean_pred_hom, color='tab:orange')
ax1.fill_between(x, mean_pred_hom-2*sigma_pred_hom, mean_pred_hom+2*sigma_pred_hom, color='tab:orange', alpha=0.25)
ax1.fill_between(x, mean_pred_hom+sigma_pred_hom, mean_pred_hom-sigma_pred_hom, color='tab:orange', alpha=0.5)
ax2.set_ylabel('$y$')
ax1.set_ylabel('$y$')
ax2.set_xlabel('$x$')
ax1.margins(x=0)
ax2.margins(x=0)
fig.legend(handles=[plt_f, plt_y_train, plt_y_pred_hom, plt_y_pred], labels=['Ground truth', 'Training data', 'Homoscedastic GP', 'Heteroscedastic GP'],
           ncol=4, loc='upper center', bbox_to_anchor=(0.5,1.15))
fig.set_size_inches(12*cm, 8*cm)
plt.show()
pass
plt.savefig('../Abbildungen/gp_heteroscedastic_example_illustration.pgf', bbox_inches='tight')

# standard gp
rng = np.random.default_rng(seed=1)
f_fun = lambda xi: 10*xi**3-np.cos(xi)+np.exp(1.5*np.sin(5*xi))
x_test = np.linspace(-0.25, 1.25, 300)
f = f_fun(x_test)
sigma = 0.5
x_train = rng.random(6)
y_train = np.zeros(6)
for i in range(x_train.shape[0]):
    xi = x_train[i]
    fi = f_fun(xi)
    yi = fi+rng.normal(scale=sigma)
    y_train[i] = yi
x_train = x_train.reshape((-1,1))
y_train = y_train.reshape((-1,1))
likelihood = gpf.likelihoods.Gaussian(0.25)
gp = gpf.models.GPR((x_train, y_train), kernel=gpf.kernels.SquaredExponential(), likelihood=likelihood)
opt = gpf.optimizers.Scipy()
opt.minimize(gp.training_loss, gp.trainable_variables)

mean_pred, var_pred = gp.predict_y(x_test.reshape((-1,1)))
f_pred, var_f_pred = gp.predict_f(x_test.reshape((-1,1)))
sigma_pred = np.sqrt(var_pred).reshape(-1)
mean_pred = mean_pred.numpy().reshape(-1)

sigma_n_sq = gp.likelihood.variance
sigma_f_sq = gp.kernel.variance
sigma_gp = np.sqrt(sigma_n_sq+sigma_f_sq)
fig, ax = plt.subplots()
plt_f, = ax.plot(x_test, f, color='tab:orange')
# ax.fill_between(x_test, f-2*sigma, f+2*sigma, color='tab:gray', alpha=0.15)
# ax.fill_between(x_test, f+sigma, f-sigma, color='tab:gray', alpha=0.35)
plt_y_train = ax.scatter(x_train, y_train, color='black')
plt_y_pred, = ax.plot(x_test, mean_pred, color='tab:blue')
ax.fill_between(x_test, mean_pred-2*sigma_pred, mean_pred+2*sigma_pred, color='tab:blue', alpha=0.25)
ax.fill_between(x_test, mean_pred+sigma_pred, mean_pred-sigma_pred, color='tab:blue', alpha=0.5)
plt_prior, = ax.plot(x_test, np.zeros(len(x_test)), '--', color='tab:gray')
ax.fill_between(x_test, -2*sigma_gp, 2*sigma_gp, color='tab:gray', alpha=0.1)
ax.fill_between(x_test, -sigma_gp, sigma_gp, color='tab:gray', alpha=0.2)
ax.set_ylabel('$y$')
ax.set_xlabel('$x$')
ax.margins(x=0)
fig.legend(handles=[plt_f, plt_y_train, plt_prior, plt_y_pred], labels=['Ground truth', 'Training data', 'GP prior', 'Prediction'],
           ncol=4, loc='upper center', bbox_to_anchor=(0.5,1.05))
fig.set_size_inches(12*cm, 6*cm)
plt.show()
plt.savefig('../Abbildungen/gp_example_illustration.pgf', bbox_inches='tight')
pass

