from DynamicModel import DynamicModel
import casadi as ca
import numpy as np

class WindTurbine(DynamicModel):
    pass

    

def get_simple_power_curve_model():
    x = ca.SX.sym('x', 0)
    u = ca.SX.sym('u', 0)
    w = ca.SX.sym('wind_speed')
    p = {}
    ode = x
    y = -0.0089 + 0.0111*w - 0.0076*w**2 + 0.0028*w**3 - 0.0002*w**4 + 0.000005*w**5
    wind_turbine = WindTurbine(x, u, p, y, ode, w)
    return wind_turbine

def get_power_curve_model():
    x = ca.SX.sym('x', 0)
    u = ca.SX.sym('u', 0)
    w = ca.SX.sym('wind_speed')
    p = {}
    ode = x
    power_curve_data = np.loadtxt('models\power_curve.csv', delimiter=',')
    wind_speeds = power_curve_data[:,0]
    power_outputs = power_curve_data[:,1]
    power_curve = ca.interpolant('power_curve', 'bspline', [wind_speeds], power_outputs)
    # y = ca.Function('power_curve', [w], [power_curve(w)])
    wind_turbine = WindTurbine(x, u, p, power_curve, ode, w)
    return wind_turbine


if __name__ == "__main__":
    wind_turbine = get_power_curve_model()
    pass