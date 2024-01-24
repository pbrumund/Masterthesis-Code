import numpy as np

def get_filename_save(mpc_name, mpc_opt, gp_opt):
    # get unique filename for each configuration
    mpc_opt_sorted = {key: mpc_opt[key] for key in sorted(mpc_opt) if key != 'param'}
    cost_param_sorted = {key: mpc_opt['param'][key] for key in sorted(mpc_opt['param'])}
    gp_opt_sorted = {key: gp_opt[key] for key in sorted(gp_opt)}
    opt_dict = {**mpc_opt_sorted, **cost_param_sorted, **gp_opt_sorted}
    filename_data = f'data/simulation_runs/{mpc_name}_data_{hash(str(opt_dict))}.csv'
    filename_times = f'data/simulation_runs/{mpc_name}_times_{hash(str(opt_dict))}.csv'
    return filename_data, filename_times

def get_power_error(mpc_name, mpc_opt, gp_opt):
    array_dims = {'Power output': 4, 'Power demand': 1, 'SOC': 1, 'Inputs': 2}
    data_loader = DataSaving(mpc_name, mpc_opt, gp_opt, array_dims)
    data, times = data_loader.load_trajectories()
    P_demand = data['Power demand']
    P_total = data['Power output'][:,-1]
    E_error = np.sum((P_demand-P_total)/6000)
    E_demand = np.sum((P_demand)/6000)
    return E_error, E_error/E_demand

class DataSaving:
    def __init__(self, mpc_name, mpc_opt, gp_opt, array_dims) -> None:
        self.mpc_name = mpc_name
        self.mpc_opt = mpc_opt
        self.gp_opt = gp_opt
        self.filename_data, self.filename_times = get_filename_save(mpc_name, mpc_opt, gp_opt)
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
            for key, dim in self.array_dims.items:
                data[key] = data_concat[:,i:i+dim]
                i += dim
            times = np.loadtxt(self.filename_times)
            return data, times
        except:
            return None



