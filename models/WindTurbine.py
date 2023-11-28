from DynamicModel import DynamicModel
import casadi as ca

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

if __name__ == "__main__":
    wind_turbine = get_simple_power_curve_model()
    pass