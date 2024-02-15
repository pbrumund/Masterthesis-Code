import datetime

from modules.mpc import get_mpc_opt
from modules.gp import get_gp_opt
from modules.mpc_scoring import get_power_error, get_gtg_power, get_gtg_emissions

# Nominal MPC, Perfect forecast
# mpc_opt = get_mpc_opt(N=30)
# gp_opt = get_gp_opt(dt_pred = mpc_opt['dt'])
# del mpc_opt['use_soft_constraints_state']
# del mpc_opt['param']['r_s_x']
# mpc_opt['param']['r_s_P'] = 1000
# t_start = datetime.datetime(2022,1,1)
# t_end = datetime.datetime(2022,12,31)
# mpc_opt['t_start'] = t_start
# mpc_opt['t_end'] = t_end

# error_abs_pf, error_rel_pf = get_power_error('nominal_mpc_perfect_forecast', mpc_opt, gp_opt)
# P_gtg_abs, P_gtg_rel = get_gtg_power('nominal_mpc_perfect_forecast', mpc_opt, gp_opt)
# P_in_gtg, eta_gtg = get_gtg_emissions('nominal_mpc_perfect_forecast', mpc_opt, gp_opt)
# print(f'Perfect forecast: {error_abs_pf} MWh total error, {100*error_rel_pf}% of demand')
# print(f'GTG power: {P_gtg_abs}, {P_gtg_rel*100}% of total generated power')
# print(f'Mean power demand of GTG: {P_in_gtg}, average efficiency: {eta_gtg}')

# # Nominal MPC, NWP forecast
# mpc_opt = get_mpc_opt(N=30)
# gp_opt = get_gp_opt(dt_pred = mpc_opt['dt'])
# t_start = datetime.datetime(2022, 1, 1)
# t_end = datetime.datetime(2022,12,31)
# mpc_opt['t_start'] = t_start
# mpc_opt['t_end'] = t_end
# error_abs_nwpf, error_rel_nwpf = get_power_error('nominal_mpc_nwp_forecast', mpc_opt, gp_opt)
# P_gtg_abs, P_gtg_rel = get_gtg_power('nominal_mpc_nwp_forecast', mpc_opt, gp_opt)
# P_in_gtg, eta_gtg = get_gtg_emissions('nominal_mpc_nwp_forecast', mpc_opt, gp_opt)
# print(f'NWP forecast: {error_abs_nwpf} MWh total error, {100*error_rel_nwpf}% of demand')
# print(f'GTG power: {P_gtg_abs}, {P_gtg_rel*100}% of total generated power')
# print(f'Mean power demand of GTG: {P_in_gtg}, average efficiency: {eta_gtg}')

# # Nominal MPC, GP forecast
# gp_opt = get_gp_opt(dt_pred = mpc_opt['dt'], steps_forward = mpc_opt['N'])
# error_abs_gpf, error_rel_gpf = get_power_error('nominal_mpc_gp_forecast', mpc_opt, gp_opt)
# P_gtg_abs, P_gtg_rel = get_gtg_power('nominal_mpc_gp_forecast', mpc_opt, gp_opt)
# P_in_gtg, eta_gtg = get_gtg_emissions('nominal_mpc_gp_forecast', mpc_opt, gp_opt)
# print(f'Gaussian process mean: {error_abs_gpf} MWh total error, {100*error_rel_gpf}% of demand')
# print(f'GTG power: {P_gtg_abs}, {P_gtg_rel*100}% of total generated power')
# print(f'Mean power demand of GTG: {P_in_gtg}, average efficiency: {eta_gtg}')

# Chance constrained MPC
mpc_opt = get_mpc_opt(N=30, use_soft_constraints_state=True)
gp_opt = get_gp_opt(dt_pred = mpc_opt['dt'], steps_forward = mpc_opt['N'], verbose = False, n_z=200)
t_start = datetime.datetime(2022, 1, 1)
t_end = datetime.datetime(2022,12,31,23,50)
mpc_opt['t_start'] = t_start
mpc_opt['t_end'] = t_end
error_abs_cc, error_rel_cc = get_power_error('chance_constrained_mpc', mpc_opt, gp_opt)
P_gtg_abs, P_gtg_rel = get_gtg_power('chance_constrained_mpc', mpc_opt, gp_opt)
P_in_gtg, eta_gtg = get_gtg_emissions('chance_constrained_mpc', mpc_opt, gp_opt)
print(f'Chance constrained: {error_abs_cc} MWh total error, {100*error_rel_cc}% of demand')
print(f'GTG power: {P_gtg_abs}, {P_gtg_rel*100}% of total generated power')
print(f'Mean power demand of GTG: {P_in_gtg}, average efficiency: {eta_gtg}')

# Simple Multistage
from scipy.stats import norm
epsilon = 0.1
std_factor = norm.ppf(1-epsilon)
std_list = (-std_factor, 0, std_factor)
mpc_opt = get_mpc_opt(N=30, std_list_multistage=std_list, use_simple_scenarios=True, dE_min=5000)
gp_opt = get_gp_opt(dt_pred = mpc_opt['dt'], steps_forward = mpc_opt['N'], verbose=False)
t_start = datetime.datetime(2022,1,1)
t_end = datetime.datetime(2022,12,31)
mpc_opt['t_start'] = t_start
mpc_opt['t_end'] = t_end
error_abs_ms_simple, error_rel_ms_simple = get_power_error('multi-stage_mpc', mpc_opt, gp_opt)
P_gtg_abs, P_gtg_rel = get_gtg_power('multi-stage_mpc', mpc_opt, gp_opt)
P_in_gtg, eta_gtg = get_gtg_emissions('multi-stage_mpc', mpc_opt, gp_opt)
print(f'Multi-stage with simple scenarios: {error_abs_ms_simple} MWh total error, {100*error_rel_ms_simple}% of demand')
print(f'GTG power: {P_gtg_abs}, {P_gtg_rel*100}% of total generated power')
print(f'Mean power demand of GTG: {P_in_gtg}, average efficiency: {eta_gtg}')

mpc_opt['use_simple_scenarios'] = False
error_abs_ms, error_rel_ms = get_power_error('multi-stage_mpc', mpc_opt, gp_opt)
P_gtg_abs, P_gtg_rel = get_gtg_power('multi-stage_mpc', mpc_opt, gp_opt)
P_in_gtg, eta_gtg = get_gtg_emissions('multi-stage_mpc', mpc_opt, gp_opt)
print(f'Multi-stage: {error_abs_ms} MWh total error, {100*error_rel_ms}% of demand')
print(f'GTG power: {P_gtg_abs}, {P_gtg_rel*100}% of total generated power')
print(f'Mean power demand of GTG: {P_in_gtg}, average efficiency: {eta_gtg}')

