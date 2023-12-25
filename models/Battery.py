import casadi as ca
from DynamicModel import DynamicModel
#from typing import Dict

class Battery(DynamicModel):
    def set_capacity(self, Q: float):
        self.set_parameter('Q', Q)   
    
    


def get_shepherd_model(N_p=1, N_s=1) -> Battery:
    # State
    int_idt = ca.SX.sym('SOC')  # actual battery charge (As)
    x = int_idt

    # Inputs
    i   = ca.SX.sym('i')
    u = i

    # Parameters
    E0  = ca.SX.sym('E0')   # no-load voltage (V)
    K   = ca.SX.sym('K')    # polarisation voltage (V)
    Q   = ca.SX.sym('Q')    # battery capacity (As)
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
    V_batt = N_s*(E - R*i/N_p)
    P_batt = i*V_batt/1000 # Power in kW
    y = ca.vertcat(V_batt, P_batt)

    # ODE
    dxdt = i/N_p

    # get battery object
    battery_shepherd_model = Battery(x, u, p, y, dxdt)
    return battery_shepherd_model

def get_shepherd_model_LiIon(N_p=1, N_s=1):
    battery = get_shepherd_model(N_p, N_s)
    battery.set_parameter('E0', 3.7348)
    battery.set_parameter('R', 0.09)
    battery.set_parameter('K', 0.00876)
    battery.set_parameter('A', 0.468)
    battery.set_parameter('B', 3.5294/3600)
    battery.set_parameter('Q', 1*3600)
    return battery


if __name__ == "__main__":
    battery = get_shepherd_model_LiIon(N_p=8000, N_s = 170)
    battery.setup_integrator(dt=600)
    p_const = 1000
    n = 30
    P_target = ca.vertcat(p_const*ca.DM.ones(n), -p_const*ca.DM.ones(n))
    u = ca.MX.sym('U', 2*n)
    x_0 = 0# 0.99*3600
    J = 0
    x_k = x_0
    for k in range(2*n):
        x_k = battery.get_next_state(x_k, u[k])
        P_k = battery.outfun(x=x_k, u=u[k])[1]
        J += (P_k - P_target[k])**2
    nlp = {'f': J, 'x': u}
    solver = ca.nlpsol('solver', 'ipopt', nlp)
    sol = solver(x0=ca.DM.zeros(2*n))
    u_opt = sol['x']
    x_traj = ca.DM.zeros(2*n)
    v_traj = ca.DM.zeros(2*n)
    p_traj = ca.DM.zeros(2*n)
    x_k = x_0
    for k in range(2*n):
        x_k = battery.get_next_state(x_k, u_opt[k])
        x_traj[k] = x_k
        y = battery.outfun(x_k, u_opt[k])
        v_traj[k] = y[0]
        p_traj[k] = y[1]
    import matplotlib.pyplot as plt
    import numpy as np
    t = np.arange(0,10*2*n, 10)
    plt.figure()
    plt.plot(t,1-x_traj/3600)
    plt.ylabel('SOC')
    plt.xlabel('t in min')
    plt.figure()
    plt.plot(t,u_opt)
    plt.ylabel('I in A')
    plt.xlabel('t in min')
    plt.figure()
    plt.plot(t,v_traj)
    plt.ylabel('U in V')
    plt.xlabel('t in min')
    plt.figure()
    plt.plot(t,p_traj)
    plt.ylabel('P in kW')
    plt.xlabel('t in min')
    plt.show()
    pass