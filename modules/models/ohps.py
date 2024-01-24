from .dynamic_model import DynamicModel
from .battery import get_shepherd_model_LiIon
from .gtg import get_GAST_model, get_static_GTG, get_integrator_GTG
from .wind_turbine import get_power_curve_model

import casadi as ca
import numpy as np

class OHPS(DynamicModel):
    def __init__(self):
        """combine subsystems into one, trying to make code modular to be able to use different models"""
        # self.gtg = get_GAST_model() # does not make sense for discretization interval of multiple minutes
        # self.gtg = get_integrator_GTG(60*10)
        self.gtg = get_static_GTG()
        self.battery = get_shepherd_model_LiIon(N_s=330,N_p = 4*8000) 
        #~4*9.5 MWh, 1200 V, 4*6MW (for 5000A max current), ca. 12 ISO 40 containers
        # N_s=170 for ~5 MWh is not enough to satisfy demand for all scenarios 
        # at current power rating even with perfect forecast
        self.n_wind_turbines = 4
        self.wind_turbine = get_power_curve_model(n_turbines=self.n_wind_turbines)

        x = ca.vertcat(self.gtg.state, self.battery.state, self.wind_turbine.state)
        u = ca.vertcat(self.gtg.inputs, self.battery.inputs, self.wind_turbine.inputs)

        p_gtg = self.gtg.parameters
        p_bat = self.battery.parameters
        p_wtg = self.wind_turbine.parameters
        # p_gtg = {f'p_gtg_{key}': val for key, val in p_gtg.items()}
        # p_bat = {f'p_bat_{key}': val for key, val in p_bat.items()}
        # p_wtg = {f'p_wtg_{key}': val for key, val in p_wtg.items()}
        p = ca.vertcat(self.gtg._p, self.battery._p, self.battery._p)
        parameters = {**p_gtg, **p_bat, **p_wtg}

        y = ca.vertcat(self.gtg.outputs, self.battery.outputs, self.wind_turbine.outputs)
        ode = ca.vertcat(self.gtg.ode, self.battery.ode, self.wind_turbine.ode)

        super().__init__(x, u, parameters, y, ode)

        lbx = ca.vertcat(self.gtg.lbx, self.battery.lbx, self.wind_turbine.lbx)
        ubx = ca.vertcat(self.gtg.ubx, self.battery.ubx, self.wind_turbine.ubx)
        lbu = ca.vertcat(self.gtg.lbu, self.battery.lbu, self.wind_turbine.lbu)
        ubu = ca.vertcat(self.gtg.ubu, self.battery.ubu, self.wind_turbine.ubu)
        self.set_bounds(lbx=lbx, ubx=ubx, lbu=lbu, ubu=ubu)

        self.x0 = ca.vertcat(self.gtg.x0, self.battery.x0, self.wind_turbine.x0)

        # Helper functions to get variables for subsystem
        self.get_x_gtg = ca.Function('get_x_gtg', [self._x], [self.gtg._x], ['x'], ['x_gtg'])
        self.get_x_bat = ca.Function('get_x_bat', [self._x], [self.battery._x], ['x'], ['x_bat'])
        self.get_x_wtg = ca.Function('get_x_wtg', [self._x], [self.wind_turbine._x], ['x'], ['x_wtg'])
        self.get_u_gtg = ca.Function('get_u_gtg', [self._u], [self.gtg._u], ['u'], ['u_gtg'])
        self.get_u_bat = ca.Function('get_u_bat', [self._u], [self.battery._u], ['u'], ['u_bat'])
        self.get_u_wtg = ca.Function('get_u_wtg', [self._u], [self.wind_turbine._u], ['u'], ['u_wtg'])

        self.P_gtg_max = self.gtg.P_max
        self.P_wtg_max = self.wind_turbine.P_max
        
    def get_P_gtg(self, x, u, w):
        x_gtg = self.get_x_gtg(x)
        u_gtg = self.get_u_gtg(u)
        return self.gtg.get_power_output(x_gtg, u_gtg, w)
    
    def get_P_bat(self, x, u, w):
        x_bat = self.get_x_bat(x)
        u_bat = self.get_u_bat(u)
        return self.battery.get_power_output(x_bat, u_bat, w)
    
    def get_P_wtg(self, x, u, w):
        x_wtg = self.get_x_wtg(x)
        u_wtg = self.get_u_wtg(u)
        return self.wind_turbine.get_power_output(x_wtg, u_wtg, w)
    
    def get_P_tot(self, x, u, w):
        P_gtg = self.get_P_gtg(x, u, w)
        P_bat = self.get_P_bat(x, u, w)
        P_wtg = self.get_P_wtg(x, u, w)
        return P_gtg + P_bat + P_wtg
    
    def get_SOC_bat(self, x, u, w):
        x_bat = self.get_x_bat(x)
        u_bat = self.get_u_bat(u)
        return self.battery.get_SOC_fun(x_bat, u_bat)
    
    @property
    def parameter_values(self):
        return ca.vertcat(self.gtg.parameter_values, 
                          self.battery.parameter_values, 
                          self.wind_turbine.parameter_values)
    
    def set_parameter_values(self, p):
        np_gtg = self.gtg.np
        np_bat = self.battery.np
        np_wtg = self.wind_turbine.np
        p_gtg = p[:np_gtg]
        p_bat = p[np_gtg:np_gtg+np_bat]
        if np_wtg > 0:
            p_wtg = p[np_gtg+np_bat:np_gtg+np_bat+np_wtg]
        else:
            p_wtg = ca.DM([])
        self.gtg.set_parameter_values(p_gtg)
        self.battery.set_parameter_values(p_bat)
        self.wind_turbine.set_parameter_values(p_wtg)
        super().set_parameter_values(p)


if __name__ == "__main__":
    ohps = OHPS()
    pass