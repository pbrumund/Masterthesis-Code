import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from modules.models.battery import get_shepherd_model_LiIon

matplotlib.use('pgf')
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})
plt.ion()
cm = 1/2.54

battery = get_shepherd_model_LiIon()
battery.setup_integrator(60)

x0 = 0
i = .1

x_k = x0
V_list = []
x_list = []
SOC_list = []
while x_k < 1:
    x_list.append(x_k)
    y_k = battery.outfun(x_k, 0)
    V_k = y_k[1]
    SOC_k = y_k[2]
    V_list.append(V_k)
    SOC_list.append(SOC_k)
    x_k = battery.get_next_state(x_k, i)

x = np.array(x_list).reshape(-1)[:-1]
V = np.array(V_list).reshape(-1)[:-1]
SOC = np.array(SOC_list).reshape(-1)[:-1]

x = x[V>0]
SOC = SOC[V>0]
V = V[V>0]

# fig, axs = plt.subplots(1,2,layout='constrained')
fig = plt.figure()
axs = plt.axes()
axs.plot(100*SOC, V)

axs.set_ylabel('Battery voltage (V)')
axs.set_xlabel('Battery SOC (\%)')
axs.grid()
fig.set_size_inches(10*cm, 7*cm)
plt.savefig('../Abbildungen/battery_voltage_curve.pgf', bbox_inches='tight')

plt.show()
pass