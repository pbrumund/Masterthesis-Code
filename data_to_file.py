import datetime

import numpy as np

from modules.gp import DataHandler, get_gp_opt

if __name__ == '__main__':
    t_start = datetime.datetime(2020,1,1)
    t_end = datetime.datetime(2021,12,31)
    t_start_data = datetime.datetime(2020,1,1)
    t_end_data = datetime.datetime(2022,12,31)
    opt = get_gp_opt()
    data_handler = DataHandler(t_start_data, t_end_data, opt)
    times = data_handler.weather_data['times_meas']
    times = times[times>t_start]
    times = times[times<t_end]
    wind_speeds_nwp = [data_handler.get_NWP(t) for t in times]
    wind_speeds_meas = [data_handler.get_measurement(t) for t in times]
    np.savetxt(f'data/wind_speed_nwp_{t_start.year}_{t_end.year}.csv', wind_speeds_nwp)
    np.savetxt(f'data/wind_speed_actual_{t_start.year}_{t_end.year}.csv', wind_speeds_meas)