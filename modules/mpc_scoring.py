import hashlib
import json

import numpy as np

def get_filename_save(mpc_name, mpc_opt, gp_opt):
    # get unique filename for each configuration
    mpc_opt_sorted = {key: mpc_opt[key] for key in sorted(mpc_opt) if key != 'param'}
    cost_param_sorted = {key: mpc_opt['param'][key] for key in sorted(mpc_opt['param'])}
    gp_opt_sorted = {key: gp_opt[key] for key in sorted(gp_opt)}
    opt_dict = {**mpc_opt_sorted, **cost_param_sorted, **gp_opt_sorted}
    run_id = hashlib.md5(str(opt_dict).encode('UTF-8')).hexdigest()[:10]
    filename_data = f'data/simulation_runs/{mpc_name}_data_{run_id}.csv'
    filename_times = f'data/simulation_runs/{mpc_name}_times_{run_id}.csv'
    filename_dict = f'data/simulation_runs/{mpc_name}_param_{run_id}.json'
    return filename_data, filename_times, filename_dict

def get_power_error(mpc_name, mpc_opt, gp_opt, run_id=None):
    """Get absolute energy error in MWh and error relative to total demand"""
    array_dims = {'Power output': 4, 'Power demand': 1, 'SOC': 1, 'Inputs': 2}
    data_loader = DataSaving(mpc_name, mpc_opt, gp_opt, array_dims, run_id)
    data, times = data_loader.load_trajectories()
    P_demand = data['Power demand'].reshape(-1)[:52272] # leave out last two days to have same length of the trajectory
    P_total = data['Power output'][:,-1][:52272]
    E_error = np.sum(np.abs((P_demand-P_total))/6000) # MWh
    E_demand = np.sum((P_demand)/6000)
    return E_error, E_error/E_demand

def get_gtg_power(mpc_name, mpc_opt, gp_opt, run_id=None):
    """Get absolute GTG power and power relative to total produced power"""
    array_dims = {'Power output': 4, 'Power demand': 1, 'SOC': 1, 'Inputs': 2}
    data_loader = DataSaving(mpc_name, mpc_opt, gp_opt, array_dims, run_id)
    data, times = data_loader.load_trajectories()
    P_prod = (data['Power output'][:,0] + data['Power output'][:,2])[:52272]
    P_gtg = data['Power output'][:,0][:52272]
    P_gtg_total = np.sum(P_gtg)/6000000 # GWh
    P_prod_total = np.sum(P_prod)/6000000
    return P_gtg_total, P_gtg_total/P_prod_total

def get_gtg_emissions(mpc_name, mpc_opt, gp_opt, run_id=None):
    """Get absolute GTG power and power relative to total produced power"""
    from modules.models import OHPS
    ohps = OHPS()
    array_dims = {'Power output': 4, 'Power demand': 1, 'SOC': 1, 'Inputs': 2}
    data_loader = DataSaving(mpc_name, mpc_opt, gp_opt, array_dims, run_id)
    data, times = data_loader.load_trajectories()
    P_gtg = data['Power output'][:,0].reshape(-1)
    eta_gtg = np.array([ohps.gtg.eta_fun(P_gtg_i/ohps.P_gtg_max) for P_gtg_i in P_gtg]).reshape(-1)
    fuel = np.where(P_gtg<1, 0, P_gtg/eta_gtg)/1000 # MW
    return np.mean(fuel[:52272]), np.mean(eta_gtg[:52272])
    
def get_energy_constraint_violation(mpc_name, mpc_opt, gp_opt, run_id=None, E_backoff=10):
    array_dims = {'Power output': 4, 'Power demand': 1, 'SOC': 1, 'Inputs': 2}
    data_loader = DataSaving(mpc_name, mpc_opt, gp_opt, array_dims, run_id)
    data, times = data_loader.load_trajectories()
    P_demand = data['Power demand'].reshape(-1)[:52272] # leave out last two days to have same length of the trajectory
    P_total = data['Power output'][:,-1][:52272]
    E_shifted = np.cumsum((P_total-P_demand)/6000)
    constraint_violation = np.where(
        np.abs(E_shifted)>E_backoff,
        np.abs(E_shifted)-E_backoff,
        0
    )
    return np.mean(constraint_violation), np.max(constraint_violation), np.nonzero(constraint_violation)[0].shape[0]/E_shifted.shape[0],np.sqrt( np.mean(np.square(constraint_violation)))

class DataSaving:
    def __init__(self, mpc_name, mpc_opt, gp_opt, array_dims, run_id=None) -> None:
        self.mpc_name = mpc_name
        self.mpc_opt = mpc_opt
        self.gp_opt = gp_opt
        if run_id is None:
            self.filename_data, self.filename_times, filename_opt = get_filename_save(
                mpc_name, mpc_opt, gp_opt)
            with open(filename_opt, 'w') as fp:
                mpc_opt_sorted = {key: str(mpc_opt[key]) for key in sorted(mpc_opt) if key != 'param'}
                cost_param_sorted = {key: str(mpc_opt['param'][key]) for key in sorted(mpc_opt['param'])}
                gp_opt_sorted = {key: str(gp_opt[key]) for key in sorted(gp_opt)}
                opt_dict = {**mpc_opt_sorted, **cost_param_sorted, **gp_opt_sorted}
                json.dump(opt_dict, fp, indent=2)
        else:
            self.filename_data = f'data/simulation_runs/{mpc_name}_data_{run_id}.csv'
            self.filename_times = f'data/simulation_runs/{mpc_name}_times_{run_id}.csv'
        self.array_dims = array_dims

    def save_trajectories(self, time, data, mode='a'):
        data_arrays = []
        for key in self.array_dims:
            data_arrays.append(data[key])
        data_concat = np.concatenate(data_arrays, axis=1)
        with open(self.filename_data, mode) as file:
            np.savetxt(file, data_concat)
        if not isinstance(time, np.ndarray):
            time = np.array(time).reshape(-1)
        with open(self.filename_times, mode) as file:
            np.savetxt(file, time, fmt='%s')

    def load_trajectories(self):
        try:
            data_concat = np.loadtxt(self.filename_data)
            i = 0
            data = {}
            for key, dim in self.array_dims.items():
                data[key] = data_concat[:,i:i+dim]
                i += dim
            times = np.loadtxt(self.filename_times, dtype=str)
            return data, times
        except:
            return None, None



