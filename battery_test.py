import casadi as ca
import matplotlib.pyplot as plt
from modules.models.battery import get_shepherd_model_LiIon

if __name__ == "__main__":
    battery = get_shepherd_model_LiIon(N_p=35000, N_s = 310)
    battery.setup_integrator(dt=600)
    p_const = 32000
    n = 6    
    P_target = ca.vertcat(p_const*ca.DM.ones(n), -p_const*ca.DM.ones(n))
    u = ca.MX.sym('U', 2*n)
    x_0 = 0.1# 0.99*3600
    J = 0
    x_k = x_0
    g = ca.MX.sym('g', 0)
    for k in range(2*n):
        x_k = battery.get_next_state(x_k, u[k])
        P_k = battery.get_power_output(x=x_k, u=u[k], w=None)
        # P_k = battery.outfun(x=x_k, u=u[k])[0]
        J += (P_k - P_target[k])**2
        g = ca.vertcat(g, 0.1-x_k, x_k-0.9)
    lbg = -ca.inf*ca.DM.ones(g.shape[0])
    ubg = ca.DM.zeros(g.shape[0])
    nlp = {'f': J, 'x': u, 'g': g}
    solver = ca.nlpsol('solver', 'ipopt', nlp)
    sol = solver(x0=ca.DM.zeros(2*n), lbg=lbg, ubg=ubg)
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