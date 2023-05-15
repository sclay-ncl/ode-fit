
import numpy as np
from scipy.integrate import solve_ivp
from lmfit import Parameters
from Logger import setup_logger

class Model:
    """Stores information parameters and current parameter values, provides solutions to the ODE integration"""

    def __init__(self, p_ode_f, p_time, p_params, p_y0, p_max_val, p_atol, p_rtol, p_integration_method):
        self.logger = setup_logger("model_logger")
        self.ode_f = p_ode_f
        self.time = p_time
        self.params = p_params
        self.y0 = p_y0
        self.max_val = p_max_val
        self.atol = p_atol
        self.rtol = p_rtol
        self.allowed_methods = ['RK23', 'RK45', 'DOP853', 'Radau', 'BDF', 'LSODA']
        self.integration_method = p_integration_method
        
    @property
    def ode_f(self):
        return self._ode_f
    
    @property
    def time(self):
        return self._time

    @property
    def params(self):
        return self._params

    @property
    def y0(self):
        return self._y0
    
    @property
    def max_val(self):
        return self._max_val

    @property
    def rtol(self):
        return self._rtol

    @property
    def atol(self):
        return self._atol

    @property
    def integration_method(self):
        return self._integration_method
    
    @ode_f.setter
    def ode_f(self, f):
        if not callable(f):
            raise ValueError("ode_f must be a function")
        else:
            self._ode_f = f

    @time.setter
    def time(self, value):
        if not isinstance(value, np.ndarray):
            raise ValueError("time must be of type np.ndarray")
        elif value.ndim != 1:
            raise ValueError("time array must be 1 dimensional")
        else:
            self._time = value
    
    @params.setter
    def params(self, value):
        if not isinstance(value, Parameters):
            raise ValueError("params must be of type lmfit.Parameters")
        else:
            self._params = value

    @y0.setter
    def y0(self, value):
        if not isinstance(value, tuple):
            raise ValueError("y0 must be of type tuple")
        else:
            self._y0 = value

    @max_val.setter
    def max_val(self, value):
        if not (isinstance(value, float) or isinstance(value, int)):
            raise ValueError("max_val must be of type float or int")
        else:
            self._max_val = value

    @rtol.setter
    def rtol(self, value):
        if not isinstance(value, float):
            raise ValueError("rtol must be of type float")
        else:
            self._rtol = value

    @atol.setter
    def atol(self, value):
        if not isinstance(value, float):
            raise ValueError("atol must be of type float")
        else:
            self._atol = value

    @integration_method.setter
    def integration_method(self, value):
        if not isinstance(value, str):
            raise ValueError("integration_method must be of type string")
        elif value not in self.allowed_methods:
            link = "https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html"
            raise ValueError("integration_method must one of the following: {}. \n Refer to {} for more details".format(self.allowed_methods, link))
        else:
            self._integration_method = value

    def integrate(self):
        """Solves the intital value problem for the ODE model and current parameter values with Scipy"""
        current_params = {}
        for p in self.params:
            current_params[p] = self.params[p].value
        sol = solve_ivp(fun=self.ode_f, 
                        args=(current_params,),
                        y0 = self.y0,
                        t_span=(self.time[0], self.time[-1]), 
                        t_eval=self.time, 
                        method=self.integration_method,
                        atol=self.atol, 
                        rtol=self.rtol,
                        dense_output=True)
        return sol

    def normalised(self):
        """Returns the model solution normalised against the maximum value for the product"""
        return self.integrate().y[-1]/self.max_val