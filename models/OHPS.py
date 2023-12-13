from DynamicModel import DynamicModel
from Battery import get_shepherd_model_LiIon
from GTG import get_GAST_model
from WindTurbine import get_simple_power_curve_model

import casadi as ca
import numpy as np

class OHPS(DynamicModel):
    def __init__(self):
        #self._w = None
        self.gtg = get_GAST_model()
        self.battery = get_shepherd_model_LiIon()
        self.wind_turbine = get_simple_power_curve_model()

        self.state = ca.vertcat(self.gtg.state, self.battery.state, self.wind_turbine.state)
        self.inputs = ca.vertcat(self.gtg.inputs, self.battery.inputs, self.wind_turbine.inputs)
        self.disturbance = self.wind_turbine.disturbance
    
        # get dict for all parameters, adding prefixes to prevent ambiguity
        p_gtg = self.gtg.parameters
        p_bat = self.battery.parameters
        p_wtg = self.wind_turbine.parameters
        p_gtg = {f'p_gtg_{key}': val for key, val in p_gtg.items()}
        p_bat = {f'p_bat_{key}': val for key, val in p_bat.items()}
        p_wtg = {f'p_wtg_{key}': val for key, val in p_wtg.items()}
        self.parameters = {**p_gtg, **p_bat, **p_wtg}

        self.outputs = ca.vertcat(self.gtg.outputs, self.battery.outputs, self.wind_turbine.outputs)
        self.ode = ca.vertcat(self.gtg.ode, self.battery.ode, self.wind_turbine.ode)

        self.gtg.set_parameter_values(ca.SX([0.5, 0.5, 4500]))
    
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
        
