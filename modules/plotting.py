import matplotlib.pyplot as plt
import casadi as ca
import numpy as np

plt.ion()

class TimeseriesPlot:
    def __init__(self, xlabel, ylabel, timeseries_names = None, title=None):
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.timeseries_names = timeseries_names
        self.is_setup = False
        self.title = title
    
    def setup_plot(self):
        self.fig = plt.figure(self.title)
        self.ax = plt.axes()
        self.is_setup = True

    def plot(self, x, y, fill_between=None):
        if not self.is_setup: self.setup_plot()
        self.ax.cla()
        if type(y) in (ca.DM, np.array):
            n_plots = y.shape[1]
            is_matrix = True
        elif type(y) is list or type(y) is tuple: 
            n_plots = len(y)
            is_matrix = False
        else: n_plots = 1

        plots = []
        for i in range(n_plots):
            if n_plots == 1:
                y_i = y
            else:
                if is_matrix: y_i = y[:,i]
                else: y_i = y[i]
            if self.timeseries_names is not None:
                label = self.timeseries_names[i]
            else: label = None
            plot = self.ax.plot(x, y_i, label=label)
            plots.append(plot)
        if fill_between is not None:
            for f in fill_between:
                self.ax.fill_between(x, f[0], f[1], alpha=0.5, color='tab:green')
        self.ax.set_xlabel(self.xlabel)
        self.ax.set_ylabel(self.ylabel)
        if self.timeseries_names is not None:
            self.ax.legend()
        # plt.pause(0.01)

