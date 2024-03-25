import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import norm

from modules.models.ohps import OHPS

matplotlib.use('pgf')
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})
plt.ion()
cm = 1/2.54

ohps = OHPS()
wind_speeds = np.arange(0,30,0.1)
power_outputs = np.array([ohps.wind_turbine.power_curve_fun(w) for w in wind_speeds]).reshape(-1)

fig, axs = plt.subplots(2, sharex=True, layout='constrained')
ax = axs[0]
ax.plot(wind_speeds, power_outputs)
ax.set_ylabel(r'$P_{\mathrm{gtg},k}=g(v_{\mathrm{wind},k})$')
P_g = np.array(ohps.wind_turbine.power_curve_fun(10))[0,0]
ax.set_yticks([P_g])
ax.set_yticklabels([r'$P_{\mathrm{g},k}$'])

# ax.vlines(10, 0, P_g, linestyles='dashed', color='black')
# ax.hlines(P_g, 0, 30, linestyles='dashed', color='black')
ax.set_ylim([0,10000])
ax.set_xticks([10,25])
ax.set_xticklabels([r'$g^{-1}(P_{\mathrm{g},k})$', r'$v_\mathrm{cut-out}$'])
ax.grid()
ax = axs[1]
v_dist = norm(16,4)
pdf = v_dist.pdf(wind_speeds)
ax.plot(wind_speeds, pdf)
ax.fill_between(wind_speeds, 0, pdf, np.logical_or(wind_speeds<=10, wind_speeds>=25), alpha=0.5)
# ax.vlines(10, 0, v_dist.pdf(10), linestyles='dashed', color='black')
# ax.vlines(25, 0, v_dist.pdf(25), linestyles='dashed', color='black')
ax.set_xlabel(r'$v_{\mathrm{wind},k}$')
ax.set_ylabel(r'$f_{k}(v_{\mathrm{wind}})$')
ax.set_xticks([10,25])
ax.set_yticks([])
ax.set_xticklabels([r'$g^{-1}(P_{\mathrm{g},k})$', r'$v_\mathrm{cut-out}$'])
ax.set_xlim([0,30])
ax.grid()
fig.set_size_inches(12*cm,10*cm)
plt.show()
plt.savefig('../Abbildungen/chance_constraints_illustration.pgf', bbox_inches='tight')

fig, axs = plt.subplots(2, sharex=True, layout='constrained')
ax = axs[0]
ax.plot(wind_speeds, power_outputs)
ax.set_ylabel(r'$P_{\mathrm{gtg},k}=g(v_{\mathrm{wind},k})$')
P_g = np.array(ohps.wind_turbine.power_curve_fun(10))[0,0]
ax.set_yticks([8000])
ax.set_yticklabels([r'$P_{\mathrm{wtg,nom}}$'])

# ax.vlines(10, 0, P_g, linestyles='dashed', color='black')
# ax.hlines(P_g, 0, 30, linestyles='dashed', color='black')
ax.set_ylim([0,10000])

ax.grid()
ax = axs[1]
v1 = 11
s1 = 1.5
v_dist = norm(v1,s1)
vmin1 = v1-s1*norm.ppf(0.9)
vmax1 = v1+s1*norm.ppf(0.9)
pdf = v_dist.pdf(wind_speeds)
ax.plot(wind_speeds, pdf)
ax.fill_between(wind_speeds, 0, pdf, np.logical_or(wind_speeds<=min(vmin1,12.5), wind_speeds>25), alpha=0.5)
ax.vlines(vmin1, 0, v_dist.pdf(vmin1), linestyles='dashed', color='tab:blue')
ax.vlines(vmax1, 0, v_dist.pdf(vmax1), linestyles='dashed', color='tab:blue')

v2 = 24
s2 = 2
v_dist = norm(v2,s2)
vmin2 = v2-s2*norm.ppf(0.9)
vmax2 = v2+s2*norm.ppf(0.9)
pdf = v_dist.pdf(wind_speeds)
ax.plot(wind_speeds, pdf)
ax.fill_between(wind_speeds, 0, pdf, np.logical_or(wind_speeds<=min(vmin2,12.5), wind_speeds>25), alpha=0.5)
ax.vlines(vmin2, 0, v_dist.pdf(vmin2), linestyles='dashed', color='tab:orange')
ax.vlines(vmax2, 0, v_dist.pdf(vmax2), linestyles='dashed', color='tab:orange')

v3 = 17
s3 = 2.4
v_dist = norm(v3,s3)
vmin3 = v3-s3*norm.ppf(0.9)
vmax3 = v3+s3*norm.ppf(0.9)
pdf = v_dist.pdf(wind_speeds)
ax.plot(wind_speeds, pdf)
ax.fill_between(wind_speeds, 0, pdf, np.logical_or(wind_speeds<=min(vmin3,12.5), wind_speeds>25), alpha=0.5)
ax.vlines(vmin3, 0, v_dist.pdf(vmin3), linestyles='dashed', color='tab:green')
ax.vlines(vmax3, 0, v_dist.pdf(vmax3), linestyles='dashed', color='tab:green')
# ax.fill_between(wind_speeds, 0, pdf, np.logical_or(wind_speeds<=10, wind_speeds>=25), alpha=0.5)
# ax.vlines(10, 0, v_dist.pdf(10), linestyles='dashed', color='black')
# ax.vlines(25, 0, v_dist.pdf(25), linestyles='dashed', color='black')
ax.set_xlabel(r'$v_{\mathrm{wind},k}$')
ax.set_ylabel(r'$f_{k}(v_{\mathrm{wind}})$')
# ax.set_xticks([vmin1, vmax1, vmin2, vmax2, vmin3, vmax3])
ax.set_yticks([])

ax.grid()
fig.set_size_inches(12*cm,9*cm)
plt.savefig('../Abbildungen/chance_constraints_simplification_illustration.pgf', bbox_inches='tight')