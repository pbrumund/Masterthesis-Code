import datetime
import time
import random

import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    from modules.gp import DirectGPEnsemble, get_gp_opt, TimeseriesModel
    gp_opt = get_gp_opt(dt_pred=10, verbose=True)
    direct_gp = DirectGPEnsemble(gp_opt)
    gp_opt = get_gp_opt(dt_pred=10, verbose=False)
    timeseries_gp = TimeseriesModel(gp_opt)

    # t_list = [
    #     datetime.datetime(2022,2,15),
    #     datetime.datetime(2022,5,1,12),
    #     datetime.datetime(2022,9,10,5),
    #     datetime.datetime(2022,7,30,20),
    #     datetime.datetime(2022,8,6,12,50),
    #     datetime.datetime(2022,3,5,7),
    #     datetime.datetime(2022,11,30,2)
    # ]
    steps = 60

    plt.ion()
    for i in range(10):
        year = 2022
        month = random.randint(1,12)
        day = random.randint(1,28)
        hour = random.randint(0,23)
        minute = 10*random.randint(0,5)
        t = datetime.datetime(year, month, day, hour, minute)
        times = [t + i*datetime.timedelta(minutes=10) for i in range(0, steps)]
        start = time.perf_counter()
        means_direct, vars_direct = direct_gp.predict_trajectory(t, steps)
        stop = time.perf_counter()
        print(f'Time for direct model: {(stop-start): .3f} seconds')
        start = time.perf_counter()
        means_timeseries, vars_timeseries = timeseries_gp.predict_trajectory(start_time=t, steps=steps, train=True)
        stop = time.perf_counter()
        print(f'Time for timeseries model with training: {(stop-start): .3f} seconds')
        start = time.perf_counter()
        means_timeseries_, vars_timeseries_ = timeseries_gp.predict_trajectory(start_time=t, steps=steps, train=False)
        stop = time.perf_counter()
        print(f'Time for timeseries model without training: {(stop-start): .3f} seconds')
        timeseries_gp.gp_predictions = None
        start = time.perf_counter()
        means_timeseries_, vars_timeseries_ = timeseries_gp.predict_trajectory(start_time=t, steps=steps, train=False)
        stop = time.perf_counter()
        print(f'Time for timeseries model without training and without cashing: {(stop-start): .3f} seconds')
        fig = plt.figure()
        ax = plt.axes()
        plot_direct, = ax.plot(times, means_direct, color='tab:blue')
        ax.fill_between(times, means_direct - 2*np.sqrt(vars_direct), 
                        means_direct + 2*np.sqrt(vars_direct), alpha=0.2, color='tab:blue')
        ax.fill_between(times, means_direct - 1*np.sqrt(vars_direct), 
                        means_direct + 1*np.sqrt(vars_direct), alpha=0.4, color='tab:blue')
        plot_timeseries, = ax.plot(times, means_timeseries, color='tab:orange')
        ax.fill_between(times, means_timeseries - 2*np.sqrt(vars_timeseries), 
                        means_timeseries + 2*np.sqrt(vars_timeseries), alpha=0.2, color='tab:orange')
        ax.fill_between(times, means_timeseries - 1*np.sqrt(vars_timeseries), 
                        means_timeseries + 1*np.sqrt(vars_timeseries), alpha=0.4, color='tab:orange')
        plot_measurement, = ax.plot(times, [direct_gp.data_handler.get_measurement(t_i, 0)
                                           for t_i in times], color='tab:green')
        plot_nwp, = ax.plot(times, [direct_gp.data_handler.get_NWP(t, steps)
                                           for steps in range(steps)], color='tab:red')
        ax.set_xlabel('Time')
        ax.set_ylabel('Wind speed in m/s')
        ax.legend([plot_direct, plot_timeseries, plot_measurement, plot_nwp], 
                   ['Direct GP', 'Timeseries GP', 'Measurement', 'NWP wind prediction'])
        plt.show()
        plt.pause(0.5)
    pass