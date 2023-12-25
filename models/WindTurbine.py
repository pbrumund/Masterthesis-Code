from DynamicModel import DynamicModel
import casadi as ca
import numpy as np

class WindTurbine(DynamicModel):
    def __init__(self, x, u, p, power_curve_data, ode, w=None, hub_height = 100):
        wind_speeds = power_curve_data[:,0]
        power_outputs = power_curve_data[:,1]
        power_curve = ca.interpolant('power_curve', 'bspline', [wind_speeds], power_outputs)
        super().__init__(x, u, p, power_curve, ode, w)
        # Inverse power curve: interpolation with inputs/outputs reversed to avoid solving equation
        P_min = np.min(power_outputs)
        P_max = np.max(power_outputs)
        for i, p in enumerate(power_outputs):
            if p==P_min and power_outputs[i+1] > P_min:
                i_start = i
            if p == P_max and power_outputs[i-1] < P_max:
                i_end = i 
        wind_speeds_unique = wind_speeds[i_start:i_end]
        power_outputs_unique = power_outputs[i_start:i_end]
        self.inverse_power_curve = ca.interpolant(
            'inverse_power_curve', 'bspline', [power_outputs_unique], wind_speeds_unique
        )
        self.hub_height = hub_height

    def scale_wind_speed(self, wind_speed):
        """Scale the wind speed from 10 meters to hub height assuming neutral conditions"""
        prediction_height = 10
        alpha = 0.11
        return wind_speed*(self.hub_height/prediction_height)**alpha

    

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
    
    # y = ca.Function('power_curve', [w], [power_curve(w)])
    wind_turbine = WindTurbine(x, u, p, power_curve_data, ode, w, hub_height=110)
    return wind_turbine


if __name__ == "__main__":
    wind_turbine = get_power_curve_model()
    pass