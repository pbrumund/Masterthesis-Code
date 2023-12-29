import gpflow as gpf
from gpflow.base import TensorType
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from check_shapes import inherit_check_shapes
from utils import generate_features

class PosteriorKernel(gpf.kernels.Kernel):
    def __init__(self, kernel, gp_prior, get_x_fun, cashing = True):
        super().__init__()
        self.kernel = kernel
        self.gp = gp_prior
        self.get_x_fun = get_x_fun
        self.gp_predictions = {}
        self.cashing = cashing
        print('__init__ for kernel')

    def get_gp_pred(self, X):
        if not self.cashing:
            return self.gp.compiled_predict_y(self.get_x_fun(X))
        # Speed up GP by saving output of prior GP and saving it for future computations
        x_hash = hash(X.numpy().tostring())
        out = self.gp_predictions.get(x_hash)
        if out is not None:
            return out
        X_in = self.get_x_fun(X)
        mean, var = self.gp.compiled_predict_y(X_in)
        self.gp_predictions[x_hash] = (mean, var)
        return mean, var
        # var_out = np.zeros(X.shape)
        # mean_out = np.zeros(X.shape)
        # for i in range(X.shape[0]):
        #     x_hash = hash(X_in[i].numpy().tostring())
        #     out = self.gp_predictions.get(x_hash)
        #     if out is not None:
        #         mean_out[i], var_out[i] = out
        #     else:
        #         mean, var = self.gp.compiled_predict_y(tf.reshape(X_in[i], (1,-1)))
        #         self.gp_predictions[x_hash] = (mean, var)
        #         mean_out[i], var_out[i] = mean, var
        
        # return mean_out, var_out
    
    def reset_gp_prediction(self):
        self.gp_predictions = {}

    def K(self, X, X2=None):
        # TODO: show positive definiteness to prove this is a valid kernel fuction
        if X2 is None:
            k = self.kernel.K(X)
            if self.cashing:
                _, var = self.get_gp_pred(X)
            else:
                _, var = self.gp.compiled_predict_y(self.get_x_fun(X))
            var_mat = tf.linalg.matmul(var, tf.transpose(var))
            return tf.math.multiply(k, tf.math.sqrt(var_mat))
        if self.cashing:
            _, var1 = self.get_gp_pred(X)
            _, var2 = self.get_gp_pred(X2)
        else:
            _, var1 = self.gp.compiled_predict_y(self.get_x_fun(X))
            _, var2 = self.gp.compiled_predict_y(self.get_x_fun(X2))
        var_mat = tf.linalg.matmul(var1, tf.transpose(var2))
    
        k = self.kernel.K(X, X2)
        return tf.math.multiply(tf.math.sqrt(var_mat),k)

    def K_diag(self, X: TensorType):
        if self.cashing:
            _, var = self.get_gp_pred(X)
        else:
            _, var = self.gp.compiled_predict_y(self.get_x_fun(X))
        k = self.kernel.K_diag(X)
        return tf.math.multiply(tf.reshape(var,(-1)), k)
    
class PosteriorMean(gpf.functions.MeanFunction):
    def __init__(self, gp_prior, get_x_fun, cashing=True):
        self.gp = gp_prior
        self.get_x_fun = get_x_fun
        self.cashing = cashing
        self.gp_predictions = {}

    def get_gp_pred(self, X):
        # Speed up GP by saving output of prior GP and saving it for future computations
        x_hash = hash(X.numpy().tostring())
        out = self.gp_predictions.get(x_hash)
        if out is not None:
            return out
        X_in = self.get_x_fun(X)
        mean, var = self.gp.compiled_predict_y(X_in)
        self.gp_predictions[x_hash] = (mean, var)
        return mean, var

    @inherit_check_shapes
    def __call__(self, X):
        if self.cashing:
            mean, _ = self.get_gp_pred(X)
        else:
            x_in = self.get_in_fun(X)
            mean, _ = self.gp.compiled_predict_y(x_in)
        return mean

