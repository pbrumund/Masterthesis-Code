import casadi as ca
from DynamicModel import DynamicModel

class GasTurbineGenerator(DynamicModel):
    pass

def get_GAST_model():
    # States
    V_gtg = ca.SX.sym('V_gtg')  # Fuel valve output (pu)
    P_gtg = ca.SX.sym('P_gtg')  # Power output (kW)
    x = ca.vertcat(V_gtg, P_gtg)

    # Inputs
    T_gtg = ca.SX.sym('T_gtg')  # GTG throttle (pu)
    u = T_gtg

    # Parameters
    tau_gtg_V = ca.SX.sym('tau_gtg_V')  # Time constant of valve
    tau_gtg_P = ca.SX.sym('tau_gtg_P')  # Time constant of power
    P_gtg_max = ca.SX.sym('P_gtg_max')  # Maximum power output of gas turbine
    # alpha_1 = ca.SX.sym('alpha_1')  # coefficient for quadratic term of efficiency
    # alpha_2 = ca.SX.sym('alpha_2')  # coefficient for linear terme of efficiency
    p = {'tau_gtg_V': tau_gtg_V, 'tau_gtg_P': tau_gtg_P, 'P_gtg_max': P_gtg_max, 
         #'alpha_1': alpha_1, 'alpha_2': alpha_2
         }

    # Outputs
    # P_gtg
    # eta_gtg = alpha_1*P_gtg**2+alpha_2*P_gtg
    # y = ca.vertcat(P_gtg, eta_gtg)
    y = P_gtg
    
    # ODE
    d_V_gtg = (T_gtg - V_gtg)/tau_gtg_V
    d_P_gtg = (V_gtg*P_gtg_max-P_gtg)/tau_gtg_P
    dxdt = ca.vertcat(d_V_gtg, d_P_gtg)

    gtg_GAST = GasTurbineGenerator(x, u, p, y, dxdt)
    gtg_GAST.set_parameter('tau_gtg_V', 0.5)
    gtg_GAST.set_parameter('tau_gtg_P', 0.5)
    gtg_GAST.set_parameter('P_gtg_max', 4500)
    return gtg_GAST

def get_static_GTG():
    P_gtg = ca.SX.sym('P_gtg')
    x = ca.SX.sym('x', 0)
    gtg = GasTurbineGenerator(x, P_gtg, {}, P_gtg, x)
    return gtg

if __name__ == "__main__":
    gtg = get_GAST_model()
    pass