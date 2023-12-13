import casadi as ca
from DynamicModel import DynamicModel
#from typing import Dict

class Battery(DynamicModel):
    def set_capacity(self, Q: float):
        self.set_parameter('Q', Q)   
    
    


def get_shepherd_model() -> Battery:
    # State
    int_idt = ca.SX.sym('SOC')  # actual battery charge (Ah)
    x = int_idt

    # Inputs
    i   = ca.SX.sym('i')
    u = i

    # Parameters
    E0  = ca.SX.sym('E0')   # no-load voltage (V)
    K   = ca.SX.sym('K')    # polarisation voltage (V)
    Q   = ca.SX.sym('Q')    # battery capacity (Ah)
    A   = ca.SX.sym('A')    # exponential zone amplitude (V)
    B   = ca.SX.sym('B')    # exponential zone time constant inverse (Ah)^-1
    R   = ca.SX.sym('R')    # internal resistance (Ohm)
    #p   = ca.vertcat(E0, K, Q, A, B, R)
    p = {
        'E0': E0,
        'K': K,
        'Q': Q,
        'A': A,
        'B': B,
        'R': R
    }

    # no-load voltage
    E = E0 - K*Q/(Q-int_idt) + A*ca.exp(-B*int_idt)

    # Outputs
    V_batt = E - R*i
    P_batt = i*V_batt
    y = ca.vertcat(V_batt, P_batt)

    # ODE
    dxdt = i

    # get battery object
    battery_shepherd_model = Battery(x, u, p, y, dxdt)
    return battery_shepherd_model

def get_shepherd_model_LiIon():
    battery = get_shepherd_model()
    battery.set_parameter('E0', 3.7348)
    battery.set_parameter('R', 0.09)
    battery.set_parameter('K', 0.00876)
    battery.set_parameter('A', 0.468)
    battery.set_parameter('B', 3.5294/3600)
    return battery


if __name__ == "__main__":
    battery = get_shepherd_model()
    pass