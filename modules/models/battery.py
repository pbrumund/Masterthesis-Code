from .dynamic_model import DynamicModel
import casadi as ca
#from typing import Dict

class Battery(DynamicModel):
    def get_power_output(self, x, u, w):
        return self.outfun(x, u, self.parameter_values)[0]    
    
def get_shepherd_model() -> Battery:
    # State
    int_idt = ca.SX.sym('int_idt')  # integrated current (Ah)
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
    N_p = ca.SX.sym('N_p')
    N_s = ca.SX.sym('N_s')
    #p   = ca.vertcat(E0, K, Q, A, B, R)
    p = {
        'E0': E0,
        'K': K,
        'Q': Q,
        'A': A,
        'B': B,
        'R': R,
        'N_p': N_p,
        'N_s': N_s
    }

    # no-load voltage
    E = E0 - K*Q/(Q-int_idt)*i/N_p + A*ca.exp(-B*int_idt) # Factor i missing in Tremblay paper

    # Outputs
    V_batt = N_s*(E - R*i/N_p)
    P_batt = i*V_batt/1000 # Power in kW
    SOC = Q-int_idt
    y = ca.vertcat(P_batt, V_batt, SOC)

    get_SOC_fun = ca.Function('get_SOC', [x, u, Q], [SOC], ['x', 'u', 'Q'], ['SOC'])
    # ODE
    dxdt = i/N_p/3600

    # get battery object
    battery_shepherd_model = Battery(x, u, p, y, dxdt)
    battery_shepherd_model.get_SOC_fun = get_SOC_fun
    return battery_shepherd_model


def get_shepherd_model_LiIon(N_p=1, N_s=1):
    battery = get_shepherd_model()
    battery.set_parameter('E0', 3.7348)
    battery.set_parameter('R', 0.09)
    battery.set_parameter('K', 0.00876)
    battery.set_parameter('A', 0.468)
    battery.set_parameter('B', 3.5294)
    battery.set_parameter('Q', 1)
    battery.set_parameter('N_p', N_p)
    battery.set_parameter('N_s', N_s)
    battery.set_bounds(lbx=ca.DM(0.1), ubx=ca.DM(0.9), lbu=ca.DM(-1*N_p), ubu=ca.DM(1*N_p))
    battery.get_SOC_fun = ca.Function('get_SOC', [battery._x, battery._u], 
        [battery.get_SOC_fun(battery._x, battery._u, battery.parameter_values[2])], 
        ['x', 'u'], ['SOC'])
    battery.x0 = ca.DM(0.5) # initial condition for simulation
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
        P_k = battery.get_power_output(x=x_k, u=u[k], w=None)
        # P_k = battery.outfun(x=x_k, u=u[k])[0]
        J += (P_k - P_target[k])**2
    nlp = {'f': J, 'x': u}
    solver = ca.nlpsol('solver', 'ipopt', nlp)
    sol = solver(x0=ca.DM.zeros(2*n))
    u_opt = sol['x']
    x_traj = ca.DM.zeros(2*n)
    v_traj = ca.DM.zeros(2*n)
    p_traj = ca.DM.zeros(2*n)
    soc_traj = ca.DM.zeros(2*n)
    x_k = x_0
    for k in range(2*n):
        x_k = battery.get_next_state(x_k, u_opt[k])
        x_traj[k] = x_k
        y = battery.outfun(x_k, u_opt[k])
        v_traj[k] = y[1]
        p_traj[k] = y[0]
        soc_traj[k] = y[2]
    import matplotlib.pyplot as plt
    import numpy as np
    t = np.arange(0,10*2*n, 10)
    plt.figure()
    plt.plot(t,soc_traj)
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