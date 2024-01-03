from .dynamic_model import DynamicModel
import casadi as ca
import numpy as np

class WindTurbine(DynamicModel):
    def __init__(self, x, u, p, y, ode, w=None, hub_height=100):
        super().__init__(x, u, p, y, ode, w)
        self.hub_height = hub_height

    def scale_wind_speed(self, wind_speed):
        """Scale the wind speed from 10 meters to hub height assuming neutral conditions"""
        prediction_height = 10
        # compare https://doi.org/10.1175/1520-0450(1994)033<0757:DTPLWP>2.0.CO;2
        alpha = 0.11
        return wind_speed*(self.hub_height/prediction_height)**alpha

class StaticWindTurbine(WindTurbine):
    def __init__(self, power_curve_fun, inverse_power_curve_fun = None, hub_height = 100):
        x = ca.SX.sym('x', 0)
        u = ca.SX.sym('u', 0)
        p = {}
        y = x
        ode = x
        super().__init__(x, u, p, y, ode, hub_height=hub_height)
        self.set_bounds(lbx=ca.DM.zeros(0), ubx=ca.DM.ones(0), lbu=ca.DM.zeros(0), ubu=ca.DM.ones(0))
        self.x0 = ca.DM.zeros(0)    # initial condition for simulation
        self.power_curve_fun = power_curve_fun
        if inverse_power_curve_fun is None:
            raise NotImplementedError('please provide function to invert power curve')
        self.inverse_power_curve = inverse_power_curve_fun      
        self.hub_height = hub_height    # for scaling wind speed
    def get_power_output(self, x, u, w):
        wind_speed_scaled = self.scale_wind_speed(w)
        return self.power_curve_fun(wind_speed_scaled)
    

    

def get_simple_power_curve_model():
    w = ca.MX.sym('wind_speed')
    y = -0.0089 + 0.0111*w - 0.0076*w**2 + 0.0028*w**3 - 0.0002*w**4 + 0.000005*w**5
    power_curve = ca.Function('power_curve', [w], [y], ['wind speed'], ['P_wtg'])
    wind_turbine = StaticWindTurbine(power_curve)
    return wind_turbine

def get_power_curve_model():
    power_curve_data = np.loadtxt('modules/models/power_curve.csv', delimiter=',')
    wind_speeds = power_curve_data[:,0]
    power_outputs = power_curve_data[:,1]
    power_curve = ca.interpolant('power_curve', 'bspline', [wind_speeds], power_outputs)
    wind_speed_max = np.max(wind_speeds)
    P_min = np.min(power_outputs)
    P_max = np.max(power_outputs)
    w = ca.MX.sym('wind_speed')
    # extrapolate last value for wind speeds higher than biggest value in table
    # power_curve = ca.if_else(w<wind_speed_max, power_curve(w), P_max)
    # Inverse power curve: interpolation with inputs/outputs reversed to avoid solving equation 
    for i, p in enumerate(power_outputs):
        if p==P_min and power_outputs[i+1] > P_min:
            i_start = i
        if p == P_max and power_outputs[i-1] < P_max:
            i_end = i 
    wind_speeds_unique = wind_speeds[i_start:i_end]
    power_outputs_unique = power_outputs[i_start:i_end]
    inverse_power_curve = ca.interpolant(
        'inverse_power_curve', 'bspline', [power_outputs_unique], wind_speeds_unique
    )
    wind_turbine = StaticWindTurbine(power_curve, inverse_power_curve, hub_height=110)
    return wind_turbine


if __name__ == "__main__":
    wind_turbine = get_power_curve_model()
    pass