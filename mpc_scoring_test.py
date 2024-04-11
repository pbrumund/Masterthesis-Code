import datetime

from modules.mpc import get_mpc_opt
from modules.gp import get_gp_opt
from modules.mpc_scoring import get_power_error, get_gtg_power, get_gtg_emissions, get_energy_constraint_violation

# Nominal MPC, Perfect forecast
mpc_opt = get_mpc_opt(N=36, use_soft_constraints_state=False)
gp_opt = get_gp_opt(dt_pred = mpc_opt['dt'])

error_abs_pf, error_rel_pf = get_power_error('nominal_mpc_perfect_forecast', mpc_opt, gp_opt)
P_gtg_abs, P_gtg_rel = get_gtg_power('nominal_mpc_perfect_forecast', mpc_opt, gp_opt)
P_in_gtg, eta_gtg = get_gtg_emissions('nominal_mpc_perfect_forecast', mpc_opt, gp_opt)
print(f'Perfect forecast, N=36: {error_abs_pf} MWh total error, {100*error_rel_pf}% of demand')
print(f'GTG power: {P_gtg_abs} GWh, {P_gtg_rel*100}% of total generated power')
print(f'Mean power demand of GTG: {P_in_gtg} MW, average efficiency: {eta_gtg}')

# Nominal MPC, NWP forecast
mpc_opt = get_mpc_opt(N=36, use_soft_constraints_state=False)
gp_opt = get_gp_opt(dt_pred = mpc_opt['dt'])
error_abs_nwpf, error_rel_nwpf = get_power_error('nominal_mpc_nwp_forecast_without_llc', mpc_opt, gp_opt)
P_gtg_abs, P_gtg_rel = get_gtg_power('nominal_mpc_nwp_forecast_without_llc', mpc_opt, gp_opt)
P_in_gtg, eta_gtg = get_gtg_emissions('nominal_mpc_nwp_forecast_without_llc', mpc_opt, gp_opt)
print(f'NWP forecast, N=36: {error_abs_nwpf} MWh total error, {100*error_rel_nwpf}% of demand')
print(f'GTG power: {P_gtg_abs} GWh, {P_gtg_rel*100}% of total generated power')
print(f'Mean power demand of GTG: {P_in_gtg} MW, average efficiency: {eta_gtg}')

# Nominal MPC, GP forecast
mpc_opt = get_mpc_opt(N=36, use_soft_constraints_state=False)
gp_opt = get_gp_opt(dt_pred = mpc_opt['dt'], steps_forward = mpc_opt['N'], verbose=False)

error_abs_gpf, error_rel_gpf = get_power_error('nominal_mpc_gp_forecast_without_llc', mpc_opt, gp_opt)
P_gtg_abs, P_gtg_rel = get_gtg_power('nominal_mpc_gp_forecast_without_llc', mpc_opt, gp_opt)
P_in_gtg, eta_gtg = get_gtg_emissions('nominal_mpc_gp_forecast_without_llc', mpc_opt, gp_opt)
print(f'Gaussian process mean, N=36: {error_abs_gpf} MWh total error, {100*error_rel_gpf}% of demand')
print(f'GTG power: {P_gtg_abs} GWh, {P_gtg_rel*100}% of total generated power')
print(f'Mean power demand of GTG: {P_in_gtg} MW, average efficiency: {eta_gtg}')

# Chance constrained MPC
mpc_opt = get_mpc_opt(N=36, use_soft_constraints_state=False)
gp_opt = get_gp_opt(dt_pred = mpc_opt['dt'], steps_forward = mpc_opt['N'], verbose=False)

error_abs_cc, error_rel_cc = get_power_error('chance_constrained_mpc_without_llc_rerun', mpc_opt, gp_opt)
P_gtg_abs, P_gtg_rel = get_gtg_power('chance_constrained_mpc_without_llc_rerun', mpc_opt, gp_opt)
P_in_gtg, eta_gtg = get_gtg_emissions('chance_constrained_mpc_without_llc_rerun', mpc_opt, gp_opt)
print(f'Chance constrained, N=36: {error_abs_cc} MWh total error, {100*error_rel_cc}% of demand')
print(f'GTG power: {P_gtg_abs} GWh, {P_gtg_rel*100}% of total generated power')
print(f'Mean power demand of GTG: {P_in_gtg} MW, average efficiency: {eta_gtg}')

# Chance constrained MPC with direct GP
mpc_opt['N'] = 36
gp_opt['steps_forward'] = 36
error_abs_cc, error_rel_cc = get_power_error('chance_constrained_mpc_without_llc_direct_rerun', mpc_opt, gp_opt)
P_gtg_abs, P_gtg_rel = get_gtg_power('chance_constrained_mpc_without_llc_direct_rerun', mpc_opt, gp_opt)
P_in_gtg, eta_gtg = get_gtg_emissions('chance_constrained_mpc_without_llc_direct_rerun', mpc_opt, gp_opt)
print(f'Chance constrained with direct GP: {error_abs_cc} MWh total error, {100*error_rel_cc}% of demand')
print(f'GTG power: {P_gtg_abs} GWh, {P_gtg_rel*100}% of total generated power')
print(f'Mean power demand of GTG: {P_in_gtg} MW, average efficiency: {eta_gtg}')

mpc_opt['N'] = 36
mpc_opt['epsilon_chance_constraint'] = 0.05
gp_opt['steps_forward'] = 36
error_abs_cc, error_rel_cc = get_power_error('chance_constrained_mpc_without_llc_rerun', mpc_opt, gp_opt)
P_gtg_abs, P_gtg_rel = get_gtg_power('chance_constrained_mpc_without_llc_rerun', mpc_opt, gp_opt)
P_in_gtg, eta_gtg = get_gtg_emissions('chance_constrained_mpc_without_llc_rerun', mpc_opt, gp_opt)
print(f'Chance constrained with epsilon = 0.05: {error_abs_cc} MWh total error, {100*error_rel_cc}% of demand')
print(f'GTG power: {P_gtg_abs} GWh, {P_gtg_rel*100}% of total generated power')
print(f'Mean power demand of GTG: {P_in_gtg} MW, average efficiency: {eta_gtg}')
# Simple Multistage
from scipy.stats import norm
epsilon = 0.1
std_factor = norm.ppf(1-epsilon)
std_list = (-std_factor, 0, std_factor)
mpc_opt = get_mpc_opt(N=36, std_list_multistage=std_list, use_simple_scenarios=True, dE_min=5000, 
                      use_soft_constraints_state=False, include_last_measurement=True)
gp_opt = get_gp_opt(dt_pred = mpc_opt['dt'], steps_forward = mpc_opt['N'], verbose=False)
error_abs_ms_simple, error_rel_ms_simple = get_power_error('multi-stage_mpc_without_llc', mpc_opt, gp_opt)
P_gtg_abs, P_gtg_rel = get_gtg_power('multi-stage_mpc_without_llc', mpc_opt, gp_opt)
P_in_gtg, eta_gtg = get_gtg_emissions('multi-stage_mpc_without_llc', mpc_opt, gp_opt)
print(f'Multi-stage with simple scenarios, dE = 5 MWh: {error_abs_ms_simple} MWh total error, {100*error_rel_ms_simple}% of demand')
print(f'GTG power: {P_gtg_abs} GWh, {P_gtg_rel*100}% of total generated power')
print(f'Mean power demand of GTG: {P_in_gtg} MW, average efficiency: {eta_gtg}')

mpc_opt = get_mpc_opt(N=36, std_list_multistage=std_list, use_simple_scenarios=True, dE_min=0, 
                      use_soft_constraints_state=False, include_last_measurement=True)
gp_opt = get_gp_opt(dt_pred = mpc_opt['dt'], steps_forward = mpc_opt['N'], verbose=False)
error_abs_ms_simple, error_rel_ms_simple = get_power_error('multi-stage_mpc_without_llc_branch_immediately', mpc_opt, gp_opt)
P_gtg_abs, P_gtg_rel = get_gtg_power('multi-stage_mpc_without_llc_branch_immediately', mpc_opt, gp_opt)
P_in_gtg, eta_gtg = get_gtg_emissions('multi-stage_mpc_without_llc_branch_immediately', mpc_opt, gp_opt)
print(f'Multi-stage with simple scenarios, dE = 0 MWh: {error_abs_ms_simple} MWh total error, {100*error_rel_ms_simple}% of demand')
print(f'GTG power: {P_gtg_abs} GWh, {P_gtg_rel*100}% of total generated power')
print(f'Mean power demand of GTG: {P_in_gtg} MW, average efficiency: {eta_gtg}')

# Multistage with 9 branches
mpc_opt = get_mpc_opt(N=36, std_list_multistage=std_list, use_simple_scenarios=False, dE_min=5000, 
                      use_soft_constraints_state=False, include_last_measurement=True)
gp_opt = get_gp_opt(dt_pred = mpc_opt['dt'], steps_forward = mpc_opt['N'], verbose=False)
error_abs_ms_simple, error_rel_ms_simple = get_power_error('multi-stage_mpc_without_llc', mpc_opt, gp_opt)
P_gtg_abs, P_gtg_rel = get_gtg_power('multi-stage_mpc_without_llc', mpc_opt, gp_opt)
P_in_gtg, eta_gtg = get_gtg_emissions('multi-stage_mpc_without_llc', mpc_opt, gp_opt)
print(f'Multi-stage with more scenarios, dE = 0 MWh: {error_abs_ms_simple} MWh total error, {100*error_rel_ms_simple}% of demand')
print(f'GTG power: {P_gtg_abs} GWh, {P_gtg_rel*100}% of total generated power')
print(f'Mean power demand of GTG: {P_in_gtg} MW, average efficiency: {eta_gtg}')

# Shifting
# Perfect forecast
mpc_opt = get_mpc_opt(N=36, t_end_sim=datetime.datetime(2022,12,31))
mpc_opt['param']['k_dP'] = 10
mpc_opt['param']['r_s_E'] = 100
mpc_opt['param']['k_bat'] = 0
mpc_opt['use_path_constraints_energy'] = True
mpc_opt['use_soft_constraints_state'] = False
gp_opt = get_gp_opt(dt_pred = mpc_opt['dt'])
cv_mean, cv_max, cv_percent, cv_rms = get_energy_constraint_violation('nominal_mpc_perfect_forecast_shifting', mpc_opt, gp_opt)
P_gtg_abs, P_gtg_rel = get_gtg_power('nominal_mpc_perfect_forecast_shifting', mpc_opt, gp_opt)
P_in_gtg, eta_gtg = get_gtg_emissions('nominal_mpc_perfect_forecast_shifting', mpc_opt, gp_opt)
print('Nominal MPC with perfect forecast, shifting')
print(f'Mean constraint violation: {cv_mean} MWh, max constraint violation: {cv_max} MWh, RMS {cv_rms}, share of contraint violation: {cv_percent*100}%')
print(f'GTG power: {P_gtg_abs} GWh, {P_gtg_rel*100}% of total generated power')
print(f'Mean power demand of GTG: {P_in_gtg} MW, average efficiency: {eta_gtg}')

# NWP forecast
mpc_opt = get_mpc_opt(N=36, t_start_sim=datetime.datetime(2022,1,1), use_soft_constraints_state=False)
mpc_opt['param']['k_dP'] = 10
mpc_opt['param']['r_s_E'] = 100
mpc_opt['param']['k_bat'] = 0
mpc_opt['use_path_constraints_energy'] = True
cv_mean, cv_max, cv_percent, cv_rms = get_energy_constraint_violation('nominal_mpc_nwp_forecast_shifting_fixed_demand', mpc_opt, gp_opt)
P_gtg_abs, P_gtg_rel = get_gtg_power('nominal_mpc_nwp_forecast_shifting_fixed_demand', mpc_opt, gp_opt)
P_in_gtg, eta_gtg = get_gtg_emissions('nominal_mpc_nwp_forecast_shifting_fixed_demand', mpc_opt, gp_opt)
print('Nominal MPC with NWP forecast, shifting')
print(f'Mean constraint violation: {cv_mean} MWh, max constraint violation: {cv_max} MWh, RMS {cv_rms}, share of contraint violation: {cv_percent*100}%')
print(f'GTG power: {P_gtg_abs} GWh, {P_gtg_rel*100}% of total generated power')
print(f'Mean power demand of GTG: {P_in_gtg} MW, average efficiency: {eta_gtg}')

# GP forecast
mpc_opt = get_mpc_opt(N=36, t_start_sim=datetime.datetime(2022,1,1), use_soft_constraints_state=False)
mpc_opt['param']['k_dP'] = 10
mpc_opt['param']['r_s_E'] = 100
mpc_opt['param']['k_bat'] = 0
mpc_opt['use_path_constraints_energy'] = True
mpc_opt['N_p'] = 8000
gp_opt = get_gp_opt(dt_pred = mpc_opt['dt'], steps_forward = mpc_opt['N'], verbose=False)
cv_mean, cv_max, cv_percent, cv_rms = get_energy_constraint_violation('nominal_mpc_gp_forecast_shifting_fixed_demand', mpc_opt, gp_opt)
P_gtg_abs, P_gtg_rel = get_gtg_power('nominal_mpc_gp_forecast_shifting_fixed_demand', mpc_opt, gp_opt)
P_in_gtg, eta_gtg = get_gtg_emissions('nominal_mpc_gp_forecast_shifting_fixed_demand', mpc_opt, gp_opt)
print('Nominal MPC with GP forecast, shifting')
print(f'Mean constraint violation: {cv_mean} MWh, max constraint violation: {cv_max} MWh, RMS {cv_rms}, share of contraint violation: {cv_percent*100}%')
print(f'GTG power: {P_gtg_abs} GWh, {P_gtg_rel*100}% of total generated power')
print(f'Mean power demand of GTG: {P_in_gtg} MW, average efficiency: {eta_gtg}')

# Multi-stage
mpc_opt = get_mpc_opt(N=36, std_list_multistage=std_list, use_simple_scenarios=True, dE_min=0, t_start_sim=datetime.datetime(2022,1,1), use_soft_constraints_state=False, include_last_measurement=True)#,  t_start=datetime.datetime(2022,12,6), t_end=datetime.datetime(2022,12,8))
mpc_opt['param']['k_dP'] = 10
mpc_opt['param']['r_s_E'] = 100
mpc_opt['param']['k_bat'] = 0
mpc_opt['use_path_constraints_energy'] = True
gp_opt = get_gp_opt(dt_pred = mpc_opt['dt'], steps_forward = mpc_opt['N'], verbose=False)
cv_mean, cv_max, cv_percent, cv_rms = get_energy_constraint_violation('multi-stage_mpc_shifting_fixed_demand_keep_tree', mpc_opt, gp_opt)
P_gtg_abs, P_gtg_rel = get_gtg_power('multi-stage_mpc_shifting_fixed_demand_keep_tree', mpc_opt, gp_opt)
P_in_gtg, eta_gtg = get_gtg_emissions('multi-stage_mpc_shifting_fixed_demand_keep_tree', mpc_opt, gp_opt)
print('Multi-stage MPC, shifting')
print(f'Mean constraint violation: {cv_mean} MWh, max constraint violation: {cv_max} MWh, RMS {cv_rms}, share of contraint violation: {cv_percent*100}%')
print(f'GTG power: {P_gtg_abs} GWh, {P_gtg_rel*100}% of total generated power')
print(f'Mean power demand of GTG: {P_in_gtg} MW, average efficiency: {eta_gtg}')

# 50 MWh
# Perfect forecast
mpc_opt = get_mpc_opt(N=36, t_end_sim=datetime.datetime(2022,12,31))
mpc_opt['param']['k_dP'] = 10
mpc_opt['param']['r_s_E'] = 100
mpc_opt['param']['k_bat'] = 0
mpc_opt['use_path_constraints_energy'] = True
mpc_opt['use_soft_constraints_state'] = False
gp_opt = get_gp_opt(dt_pred = mpc_opt['dt'])
cv_mean, cv_max, cv_percent, cv_rms = get_energy_constraint_violation('nominal_mpc_perfect_forecast_shifting_50MWh', mpc_opt, gp_opt, E_backoff=50)
P_gtg_abs, P_gtg_rel = get_gtg_power('nominal_mpc_perfect_forecast_shifting_50MWh', mpc_opt, gp_opt)
P_in_gtg, eta_gtg = get_gtg_emissions('nominal_mpc_perfect_forecast_shifting_50MWh', mpc_opt, gp_opt)
print('Nominal MPC with perfect forecast, shifting')
print(f'Mean constraint violation: {cv_mean} MWh, max constraint violation: {cv_max} MWh, RMS {cv_rms}, share of contraint violation: {cv_percent*100}%')
print(f'GTG power: {P_gtg_abs} GWh, {P_gtg_rel*100}% of total generated power')
print(f'Mean power demand of GTG: {P_in_gtg} MW, average efficiency: {eta_gtg}')

# NWP forecast
mpc_opt = get_mpc_opt(N=36, t_start_sim=datetime.datetime(2022,1,1), use_soft_constraints_state=False)
mpc_opt['param']['k_dP'] = 10
mpc_opt['param']['r_s_E'] = 100
mpc_opt['param']['k_bat'] = 0
mpc_opt['use_path_constraints_energy'] = True
cv_mean, cv_max, cv_percent, cv_rms = get_energy_constraint_violation('nominal_mpc_nwp_forecast_shifting_50MWh', mpc_opt, gp_opt, E_backoff=50)
P_gtg_abs, P_gtg_rel = get_gtg_power('nominal_mpc_nwp_forecast_shifting_50MWh', mpc_opt, gp_opt)
P_in_gtg, eta_gtg = get_gtg_emissions('nominal_mpc_nwp_forecast_shifting_50MWh', mpc_opt, gp_opt)
print('Nominal MPC with NWP forecast, shifting')
print(f'Mean constraint violation: {cv_mean} MWh, max constraint violation: {cv_max} MWh, RMS {cv_rms}, share of contraint violation: {cv_percent*100}%')
print(f'GTG power: {P_gtg_abs} GWh, {P_gtg_rel*100}% of total generated power')
print(f'Mean power demand of GTG: {P_in_gtg} MW, average efficiency: {eta_gtg}')

# GP forecast
mpc_opt = get_mpc_opt(N=36, t_start_sim=datetime.datetime(2022,1,1), use_soft_constraints_state=False)
mpc_opt['param']['k_dP'] = 10
mpc_opt['param']['r_s_E'] = 100
mpc_opt['param']['k_bat'] = 0
mpc_opt['use_path_constraints_energy'] = True
mpc_opt['N_p'] = 8000
gp_opt = get_gp_opt(dt_pred = mpc_opt['dt'], steps_forward = mpc_opt['N'], verbose=False)
cv_mean, cv_max, cv_percent, cv_rms = get_energy_constraint_violation('nominal_mpc_gp_forecast_shifting_50MWh', mpc_opt, gp_opt, E_backoff=50)
P_gtg_abs, P_gtg_rel = get_gtg_power('nominal_mpc_gp_forecast_shifting_50MWh', mpc_opt, gp_opt)
P_in_gtg, eta_gtg = get_gtg_emissions('nominal_mpc_gp_forecast_shifting_50MWh', mpc_opt, gp_opt)
print('Nominal MPC with GP forecast, shifting')
print(f'Mean constraint violation: {cv_mean} MWh, max constraint violation: {cv_max} MWh, RMS {cv_rms}, share of contraint violation: {cv_percent*100}%')
print(f'GTG power: {P_gtg_abs} GWh, {P_gtg_rel*100}% of total generated power')
print(f'Mean power demand of GTG: {P_in_gtg} MW, average efficiency: {eta_gtg}')

# Multi-stage
mpc_opt = get_mpc_opt(N=36, std_list_multistage=std_list, use_simple_scenarios=True, dE_min=0, t_start_sim=datetime.datetime(2022,1,1), use_soft_constraints_state=False, include_last_measurement=True)#,  t_start=datetime.datetime(2022,12,6), t_end=datetime.datetime(2022,12,8))
mpc_opt['param']['k_dP'] = 10
mpc_opt['param']['r_s_E'] = 100
mpc_opt['param']['k_bat'] = 0
mpc_opt['use_path_constraints_energy'] = True
gp_opt = get_gp_opt(dt_pred = mpc_opt['dt'], steps_forward = mpc_opt['N'], verbose=False)
cv_mean, cv_max, cv_percent, cv_rms = get_energy_constraint_violation('multi-stage_mpc_shifting_50MWh', mpc_opt, gp_opt, E_backoff=50)
P_gtg_abs, P_gtg_rel = get_gtg_power('multi-stage_mpc_shifting_50MWh', mpc_opt, gp_opt)
P_in_gtg, eta_gtg = get_gtg_emissions('multi-stage_mpc_shifting_50MWh', mpc_opt, gp_opt)
print('Multi-stage MPC, shifting')
print(f'Mean constraint violation: {cv_mean} MWh, max constraint violation: {cv_max} MWh, RMS {cv_rms}, share of contraint violation: {cv_percent*100}%')
print(f'GTG power: {P_gtg_abs} GWh, {P_gtg_rel*100}% of total generated power')
print(f'Mean power demand of GTG: {P_in_gtg} MW, average efficiency: {eta_gtg}')


# half load
mpc_opt = get_mpc_opt(N=36, use_soft_constraints_state=False)
gp_opt = get_gp_opt(dt_pred = mpc_opt['dt'])

error_abs_pf, error_rel_pf = get_power_error('nominal_mpc_perfect_forecast_half_load_rerun', mpc_opt, gp_opt)
P_gtg_abs, P_gtg_rel = get_gtg_power('nominal_mpc_perfect_forecast_half_load_rerun', mpc_opt, gp_opt)
P_in_gtg, eta_gtg = get_gtg_emissions('nominal_mpc_perfect_forecast_half_load_rerun', mpc_opt, gp_opt)
print(f'Perfect forecast, N=36: {error_abs_pf} MWh total error, {100*error_rel_pf}% of demand')
print(f'GTG power: {P_gtg_abs} GWh, {P_gtg_rel*100}% of total generated power')
print(f'Mean power demand of GTG: {P_in_gtg} MW, average efficiency: {eta_gtg}')

mpc_opt = get_mpc_opt(N=36)#, t_end_sim=datetime.datetime(2022,12,31))
mpc_opt['param']['k_dP'] = 10
mpc_opt['param']['r_s_E'] = 100
mpc_opt['param']['k_bat'] = 0
mpc_opt['use_path_constraints_energy'] = True
mpc_opt['use_soft_constraints_state'] = False
gp_opt = get_gp_opt(dt_pred = mpc_opt['dt'])
cv_mean, cv_max, cv_percent, cv_rms = get_energy_constraint_violation('nominal_mpc_perfect_forecast_shifting_half_load_rerun', mpc_opt, gp_opt,E_backoff=50)
P_gtg_abs, P_gtg_rel = get_gtg_power('nominal_mpc_perfect_forecast_shifting_half_load_rerun', mpc_opt, gp_opt)
P_in_gtg, eta_gtg = get_gtg_emissions('nominal_mpc_perfect_forecast_shifting_half_load_rerun', mpc_opt, gp_opt)
print('Nominal MPC with perfect forecast, shifting')
print(f'Mean constraint violation: {cv_mean} MWh, max constraint violation: {cv_max} MWh, RMS {cv_rms}, share of contraint violation: {cv_percent*100}%')
print(f'GTG power: {P_gtg_abs} GWh, {P_gtg_rel*100}% of total generated power')
print(f'Mean power demand of GTG: {P_in_gtg} MW, average efficiency: {eta_gtg}')
pass

# Perfect forecast
mpc_opt = get_mpc_opt(N=36)
mpc_opt['param']['k_dP'] = 10
mpc_opt['param']['r_s_E'] = 100
mpc_opt['param']['k_bat'] = 0
mpc_opt['param']['k_E_shifted'] = 0.01
mpc_opt['use_path_constraints_energy'] = True
mpc_opt['use_soft_constraints_state'] = False
gp_opt = get_gp_opt(dt_pred = mpc_opt['dt'])
cv_mean, cv_max, cv_percent, cv_rms = get_energy_constraint_violation('nominal_mpc_perfect_forecast_shifting', mpc_opt, gp_opt, E_backoff=50)
P_gtg_abs, P_gtg_rel = get_gtg_power('nominal_mpc_perfect_forecast_shifting', mpc_opt, gp_opt)
P_in_gtg, eta_gtg = get_gtg_emissions('nominal_mpc_perfect_forecast_shifting', mpc_opt, gp_opt)
print('Nominal MPC with perfect forecast, shifting')
print(f'Mean constraint violation: {cv_mean} MWh, max constraint violation: {cv_max} MWh, RMS {cv_rms}, share of contraint violation: {cv_percent*100}%')
print(f'GTG power: {P_gtg_abs} GWh, {P_gtg_rel*100}% of total generated power')
print(f'Mean power demand of GTG: {P_in_gtg} MW, average efficiency: {eta_gtg}')

# NWP forecast
mpc_opt = get_mpc_opt(N=36, t_start_sim=datetime.datetime(2022,1,1), use_soft_constraints_state=False)
mpc_opt['param']['k_dP'] = 10
mpc_opt['param']['r_s_E'] = 100
mpc_opt['param']['k_bat'] = 0
mpc_opt['param']['k_E_shifted'] = 0.01
mpc_opt['use_path_constraints_energy'] = True
cv_mean, cv_max, cv_percent, cv_rms = get_energy_constraint_violation('nominal_mpc_nwp_forecast_shifting', mpc_opt, gp_opt, E_backoff=50)
P_gtg_abs, P_gtg_rel = get_gtg_power('nominal_mpc_nwp_forecast_shifting', mpc_opt, gp_opt)
P_in_gtg, eta_gtg = get_gtg_emissions('nominal_mpc_nwp_forecast_shifting', mpc_opt, gp_opt)
print('Nominal MPC with NWP forecast, shifting')
print(f'Mean constraint violation: {cv_mean} MWh, max constraint violation: {cv_max} MWh, RMS {cv_rms}, share of contraint violation: {cv_percent*100}%')
print(f'GTG power: {P_gtg_abs} GWh, {P_gtg_rel*100}% of total generated power')
print(f'Mean power demand of GTG: {P_in_gtg} MW, average efficiency: {eta_gtg}')

# GP forecast
mpc_opt = get_mpc_opt(N=36, t_start_sim=datetime.datetime(2022,1,1), use_soft_constraints_state=False)
mpc_opt['param']['k_dP'] = 10
mpc_opt['param']['r_s_E'] = 100
mpc_opt['param']['k_bat'] = 0
mpc_opt['param']['k_E_shifted'] = 0.01
mpc_opt['use_path_constraints_energy'] = True
mpc_opt['N_p'] = 8000
gp_opt = get_gp_opt(dt_pred = mpc_opt['dt'], steps_forward = mpc_opt['N'], verbose=False)
cv_mean, cv_max, cv_percent, cv_rms = get_energy_constraint_violation('nominal_mpc_gp_forecast_shifting', mpc_opt, gp_opt, E_backoff=50)
P_gtg_abs, P_gtg_rel = get_gtg_power('nominal_mpc_gp_forecast_shifting', mpc_opt, gp_opt)
P_in_gtg, eta_gtg = get_gtg_emissions('nominal_mpc_gp_forecast_shifting', mpc_opt, gp_opt)
print('Nominal MPC with GP forecast, shifting')
print(f'Mean constraint violation: {cv_mean} MWh, max constraint violation: {cv_max} MWh, RMS {cv_rms}, share of contraint violation: {cv_percent*100}%')
print(f'GTG power: {P_gtg_abs} GWh, {P_gtg_rel*100}% of total generated power')
print(f'Mean power demand of GTG: {P_in_gtg} MW, average efficiency: {eta_gtg}')

# Multi-stage
mpc_opt = get_mpc_opt(N=36, std_list_multistage=std_list, use_simple_scenarios=True, dE_min=0, t_start_sim=datetime.datetime(2022,1,1), use_soft_constraints_state=False, include_last_measurement=True)#,  t_start=datetime.datetime(2022,12,6), t_end=datetime.datetime(2022,12,8))
mpc_opt['param']['k_dP'] = 10
mpc_opt['param']['r_s_E'] = 100
mpc_opt['param']['k_bat'] = 0
mpc_opt['param']['k_E_shifted'] = 0.01
mpc_opt['use_path_constraints_energy'] = True
gp_opt = get_gp_opt(dt_pred = mpc_opt['dt'], steps_forward = mpc_opt['N'], verbose=False)
cv_mean, cv_max, cv_percent, cv_rms = get_energy_constraint_violation('multi-stage_mpc_shifting', mpc_opt, gp_opt, E_backoff=50)
P_gtg_abs, P_gtg_rel = get_gtg_power('multi-stage_mpc_shifting', mpc_opt, gp_opt)
P_in_gtg, eta_gtg = get_gtg_emissions('multi-stage_mpc_shifting', mpc_opt, gp_opt)
print('Multi-stage MPC, shifting')
print(f'Mean constraint violation: {cv_mean} MWh, max constraint violation: {cv_max} MWh, RMS {cv_rms}, share of contraint violation: {cv_percent*100}%')
print(f'GTG power: {P_gtg_abs} GWh, {P_gtg_rel*100}% of total generated power')
print(f'Mean power demand of GTG: {P_in_gtg} MW, average efficiency: {eta_gtg}')

mpc_opt = get_mpc_opt(N=36)
mpc_opt['param']['k_dP'] = 10
mpc_opt['param']['r_s_E'] = 100
mpc_opt['param']['k_bat'] = 0
mpc_opt['param']['k_E_shifted'] = 1
mpc_opt['use_path_constraints_energy'] = True
mpc_opt['use_soft_constraints_state'] = False
gp_opt = get_gp_opt(dt_pred = mpc_opt['dt'])
cv_mean, cv_max, cv_percent, cv_rms = get_energy_constraint_violation('nominal_mpc_perfect_forecast_shifting', mpc_opt, gp_opt, E_backoff=50)
P_gtg_abs, P_gtg_rel = get_gtg_power('nominal_mpc_perfect_forecast_shifting', mpc_opt, gp_opt)
P_in_gtg, eta_gtg = get_gtg_emissions('nominal_mpc_perfect_forecast_shifting', mpc_opt, gp_opt)
print('Nominal MPC with perfect forecast, shifting')
print(f'Mean constraint violation: {cv_mean} MWh, max constraint violation: {cv_max} MWh, RMS {cv_rms}, share of contraint violation: {cv_percent*100}%')
print(f'GTG power: {P_gtg_abs} GWh, {P_gtg_rel*100}% of total generated power')
print(f'Mean power demand of GTG: {P_in_gtg} MW, average efficiency: {eta_gtg}')

# NWP forecast
mpc_opt = get_mpc_opt(N=36, t_start_sim=datetime.datetime(2022,1,1), use_soft_constraints_state=False)
mpc_opt['param']['k_dP'] = 10
mpc_opt['param']['r_s_E'] = 100
mpc_opt['param']['k_bat'] = 0
mpc_opt['param']['k_E_shifted'] = 1
mpc_opt['use_path_constraints_energy'] = True
cv_mean, cv_max, cv_percent, cv_rms = get_energy_constraint_violation('nominal_mpc_nwp_forecast_shifting', mpc_opt, gp_opt, E_backoff=50)
P_gtg_abs, P_gtg_rel = get_gtg_power('nominal_mpc_nwp_forecast_shifting', mpc_opt, gp_opt)
P_in_gtg, eta_gtg = get_gtg_emissions('nominal_mpc_nwp_forecast_shifting', mpc_opt, gp_opt)
print('Nominal MPC with NWP forecast, shifting')
print(f'Mean constraint violation: {cv_mean} MWh, max constraint violation: {cv_max} MWh, RMS {cv_rms}, share of contraint violation: {cv_percent*100}%')
print(f'GTG power: {P_gtg_abs} GWh, {P_gtg_rel*100}% of total generated power')
print(f'Mean power demand of GTG: {P_in_gtg} MW, average efficiency: {eta_gtg}')

# GP forecast
mpc_opt = get_mpc_opt(N=36, t_start_sim=datetime.datetime(2022,1,1), use_soft_constraints_state=False)
mpc_opt['param']['k_dP'] = 10
mpc_opt['param']['r_s_E'] = 100
mpc_opt['param']['k_bat'] = 0
mpc_opt['param']['k_E_shifted'] = 1
mpc_opt['use_path_constraints_energy'] = True
mpc_opt['N_p'] = 8000
gp_opt = get_gp_opt(dt_pred = mpc_opt['dt'], steps_forward = mpc_opt['N'], verbose=False)
cv_mean, cv_max, cv_percent, cv_rms = get_energy_constraint_violation('nominal_mpc_gp_forecast_shifting', mpc_opt, gp_opt, E_backoff=50)
P_gtg_abs, P_gtg_rel = get_gtg_power('nominal_mpc_gp_forecast_shifting', mpc_opt, gp_opt)
P_in_gtg, eta_gtg = get_gtg_emissions('nominal_mpc_gp_forecast_shifting', mpc_opt, gp_opt)
print('Nominal MPC with GP forecast, shifting')
print(f'Mean constraint violation: {cv_mean} MWh, max constraint violation: {cv_max} MWh, RMS {cv_rms}, share of contraint violation: {cv_percent*100}%')
print(f'GTG power: {P_gtg_abs} GWh, {P_gtg_rel*100}% of total generated power')
print(f'Mean power demand of GTG: {P_in_gtg} MW, average efficiency: {eta_gtg}')

# Multi-stage
mpc_opt = get_mpc_opt(N=36, std_list_multistage=std_list, use_simple_scenarios=True, dE_min=0, t_start_sim=datetime.datetime(2022,1,1), use_soft_constraints_state=False, include_last_measurement=True)#,  t_start=datetime.datetime(2022,12,6), t_end=datetime.datetime(2022,12,8))
mpc_opt['param']['k_dP'] = 10
mpc_opt['param']['r_s_E'] = 100
mpc_opt['param']['k_bat'] = 0
mpc_opt['param']['k_E_shifted'] = 1
mpc_opt['use_path_constraints_energy'] = True
gp_opt = get_gp_opt(dt_pred = mpc_opt['dt'], steps_forward = mpc_opt['N'], verbose=False)
cv_mean, cv_max, cv_percent, cv_rms = get_energy_constraint_violation('multi-stage_mpc_shifting', mpc_opt, gp_opt, E_backoff=50)
P_gtg_abs, P_gtg_rel = get_gtg_power('multi-stage_mpc_shifting', mpc_opt, gp_opt)
P_in_gtg, eta_gtg = get_gtg_emissions('multi-stage_mpc_shifting', mpc_opt, gp_opt)
print('Multi-stage MPC, shifting')
print(f'Mean constraint violation: {cv_mean} MWh, max constraint violation: {cv_max} MWh, RMS {cv_rms}, share of contraint violation: {cv_percent*100}%')
print(f'GTG power: {P_gtg_abs} GWh, {P_gtg_rel*100}% of total generated power')
print(f'Mean power demand of GTG: {P_in_gtg} MW, average efficiency: {eta_gtg}')