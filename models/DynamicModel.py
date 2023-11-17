import casadi as ca

class DynamicModel():
    def __init__(self, x, u, p, y, ode):
        self.state = x
        self.inputs = u
        self.parameters = p
        self.outputs = y
        self.ode = ode
        self._odefun = None
        self._odefun_p = None
        self._outfun = None
        self._outfun_p = None
        self._param_values = None

    @property
    def state(self):
        return self._x
    
    @state.setter
    def state(self, x: ca.SX.sym):
        self._x = x
        self._nx = x.shape[0]

    @property
    def nx(self):
        return self._nx
    
    @property
    def inputs(self):
        return self._u
    
    @inputs.setter
    def inputs(self, u: ca.SX.sym):
        self._u = u
        self._nu = u.shape[0]

    @property
    def nu(self):
        return self._nu
        
    @property
    def parameters(self):
        return self._p
    
    @parameters.setter
    def parameters(self, p: dict):
        self._p_dict = p
        self._p = ca.vertcat(*list(p.values()))
        self._np = self._p.shape[0]
        self._parameter_names = list(p.keys())

    @property
    def parameter_values(self):
        return self._param_values
    
    @property
    def np(self):
        return self._np
    
    @property
    def ode(self):
        return self._ode
    
    @ode.setter
    def ode(self, ode: ca.SX.sym):
        if ode.shape != self._x.shape:
            raise ValueError('The shape of the provided ODE does not match the shape of the state')
        self._ode = ode
        self._odefun = ca.Function('ode', [self._x,self._u,self._p], [self._ode],
                                   ['x','u','p'], ['dxdt'])
    
    @property 
    def odefun(self):
        if self._odefun_p is not None:
            return self._odefun_p
        return self._odefun
    
    @property
    def outputs(self):
        return self._y
    
    @outputs.setter
    def outputs(self, y: ca.SX.sym):
        self._y = y
        self._ny = y.shape[0]
        self._outfun = ca.Function('out', [self._x,self._u,self._p], [self._y],
                                   ['x','u','p'], ['y'])
        
    @property
    def outfun(self):
        if self._outfun_p is not None:
            return self._outfun_p
        return self._outfun

    @property
    def ny(self):
        return self._ny
    
    def set_parameter_values(self, p):
        if p.shape != self._p.shape:
            raise ValueError('Shape of provided parameter values does not match expected shape')
        self._param_values = p
        self.outputs = self._y
        self.ode = self._ode
        self._outfun_p = ca.Function('out', [self._x,self._u], [self._outfun(self._x, self._u, p)],
                                   ['x','u'], ['y'])
        self._odefun_p = ca.Function('ode', [self._x,self._u], [self._odefun(self._x, self._u, p)],
                                   ['x','u'], ['dxdt'])
        
    def set_parameter(self, name: str, value):
        if self._param_values is None:
            param_val = self._p.__copy__()
        else:
            param_val = self._param_values.__copy__()
        
        if name in self._parameter_names:
            index = self._parameter_names.index(name)
            param_val[index] = value
        else:
            p_new = ca.SX.sym(name)
            self._p_dict[name] = p_new
            self.parameters = self._p_dict
            param_val = ca.vertcat(param_val, value)
        self.set_parameter_values(param_val)

    def setup(self, dt):
        ode = {'x': self._x, 'p': self._u, 'ode': self._ode}
        self._integrator = ca.integrator('integrator', 'rk', ode, 0, dt)

    def get_next_state(self, x_i, u_i):
        result = self._integrator(x0=x_i, p=u_i)
        return result['xf']