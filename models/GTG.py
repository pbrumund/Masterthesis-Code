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
    p = {'tau_gtg_V': tau_gtg_V, 'tau_gtg_P': tau_gtg_P, 'P_gtg_max': P_gtg_max}

    # Outputs
    y = P_gtg
    
    # ODE
    d_V_gtg = (T_gtg - V_gtg)/tau_gtg_V
    d_P_gtg = (V_gtg*P_gtg_max-P_gtg)/tau_gtg_P
    dxdt = ca.vertcat(d_V_gtg, d_P_gtg)

    gtg_GAST = GasTurbineGenerator(x, u, p, y, dxdt)
    return gtg_GAST

if __name__ == "__main__":
    gtg = get_GAST_model()
    pass