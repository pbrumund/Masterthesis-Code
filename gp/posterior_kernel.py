import gpflow as gpf
from gpflow.base import TensorType
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from check_shapes import inherit_check_shapes
from utils import generate_features

class PosteriorKernel(gpf.kernels.Kernel):
    def __init__(self, kernel, gp_prior, get_x_fun, a_init=0.99):
        super().__init__()
        self.kernel = kernel
        self.gp = gp_prior
        self.get_x_fun = get_x_fun
        self.a = gpf.Parameter(a_init, transform=tfp.bijectors.Sigmoid())
    def K(self, X, X2=None):
        if X2 is None:
            k = self.kernel.K(X)
            _, var = self.gp.compiled_predict_y(self.get_x_fun(X))
            var_mat = self.a*tf.linalg.matmul(var, tf.transpose(var))
            var_mat += (1-self.a)*tf.linalg.diag(var[:,0])
            return tf.math.multiply(k, tf.math.sqrt(var_mat))
        
        _, var1 = self.gp.compiled_predict_y(self.get_x_fun(X))
        _, var2 = self.gp.compiled_predict_y(self.get_x_fun(X2))
        var_mat = self.a*tf.linalg.matmul(var1, tf.transpose(var2))
    
        k = self.kernel.K(X, X2)
        return tf.math.multiply(tf.math.sqrt(var_mat),k)

    def K_diag(self, X: TensorType):
        _, var = self.gp.compiled_predict_y(self.get_x_fun(X))
        k = self.kernel.K_diag(X)
        return tf.math.multiply(tf.reshape(var,(-1)), k)
    
class PosteriorMean(gpf.functions.MeanFunction):
    def __init__(self, gp_prior, get_in_fun):
        self.gp = gp_prior
        self.get_in_fun = get_in_fun

    @inherit_check_shapes
    def __call__(self, X):
        x_in = self.get_in_fun(X)
        mean, _ = self.gp.compiled_predict_y(x_in)
        return mean

