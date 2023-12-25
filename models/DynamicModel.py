import casadi as ca

class DynamicModel():
    def __init__(self, x, u, p, y, ode, w=None):
        self.state = x
        self.inputs = u
        self.disturbance = w
        self.parameters = p
        self._param_values = None
        self.outputs = y
        self.ode = ode
        # self._odefun = None
        # self._outfun = None
        self._bounds = {}

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
    def disturbance(self):
        return self._w
    
    @disturbance.setter
    def disturbance(self, w):
        self._w = w
        if w is not None:
            self._nw = w.shape[0]
        else:
            self._nw = 0

    @property
    def nw(self):
        return self._nw
        
    @property
    def parameters(self):
        return self._p_dict
    
    @parameters.setter
    def parameters(self, p: dict):
        self._p_dict = p
        self._p = ca.vertcat(*list(p.values()))
        self._np = self._p.shape[0]
        self._parameter_names = list(p.keys())

    # @property
    # def parameter_values(self):
    #     return self._param_values
    
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
        if self._w is not None:
            self._odefun = ca.Function('ode', [self._x,self._u,self._p, self._w], [self._ode],
                                        ['x','u','p','w'], ['dxdt'])
        else:
            self._odefun = ca.Function('ode', [self._x,self._u,self._p], [self._ode],
                                        ['x','u','p'], ['dxdt'])
    
    @property 
    def odefun(self):
        return self._odefun
    
    @property
    def outputs(self):
        return self._y
    
    @outputs.setter
    def outputs(self, y: ca.SX.sym):
        if isinstance(y, ca.Function):
            self._ny = y.n_out()
            self._y = ca.SX.sym('y', self._ny)
            self._outfun = y
            return
        self._y = y
        self._ny = y.shape[0]
        if self._w is not None:
            self._outfun = ca.Function('out', [self._x,self._u,self._p, self._w], [self._y],
                                        ['x','u','p', 'w'], ['y'])
        else:
            self._outfun = ca.Function('out', [self._x,self._u,self._p], [self._y],
                                        ['x','u','p'], ['y'])
        

    def outfun(self, x=None, u=None, p=None):
        if x is None:
            return self._outfun
        if self.parameter_values is None:
            return self._outfun(x,u,p)
        else:
            return self._outfun(x,u,ca.DM(self.parameter_values))

    @property
    def ny(self):
        return self._ny
    
    @property
    def bounds(self):
        return self._bounds
    
    def set_bounds(self, **kwargs):
        if 'lbx' in kwargs:
            if kwargs['lbx'].shape[0] != self._nx:
                raise ValueError('Provided shape for lbx does not match state shape')
            self._bounds['lbx'] = kwargs['lbx']
        if 'ubx' in kwargs:
            if kwargs['ubx'].shape[0] != self._nx:
                raise ValueError('Provided shape for ubx does not match state shape')
            self._bounds['ubx'] = kwargs['ubx']
        if 'lbu' in kwargs:
            if kwargs['lbu'].shape[0] != self._nu:
                raise ValueError('Provided shape for lbu does not match state shape')
            self._bounds['lbu'] = kwargs['lbu']
        if 'ubu' in kwargs:
            if kwargs['ubu'].shape[0] != self._nu:
                raise ValueError('Provided shape for ubu does not match state shape')
            self._bounds['ubu'] = kwargs['ubu']

    @property
    def parameter_values(self):
        if self._param_values is None:
            return ca.SX([])
        return self._param_values    
    
    def set_parameter_values(self, p):
        if p.shape != self._p.shape:
            raise ValueError('Shape of provided parameter values does not match expected shape')
        self._param_values = p
        self.outputs = self._y
        self.ode = self._ode
        
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

    def setup_integrator(self, dt):
        if self._w is not None:
            ode = {'x': self._x, 
                   'p': ca.vertcat(self._u, self._w), 
                   'ode': self.odefun(self.state, self.inputs, self.parameter_values, self.disturbance)}
        else:
            ode = {'x': self._x, 
                   'p': self._u, 
                   'ode': self.odefun(self.state, self.inputs, self.parameter_values)}
        self._integrator = ca.integrator('integrator', 'rk', ode, 0, dt)

    def get_next_state(self, x_i, u_i):
        result = self._integrator(x0=x_i, p=u_i)
        return result['xf']