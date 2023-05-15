import lmfit
import numpy as np
import pandas as pd
from Model import Model
from Logger import setup_logger

class Fitter:
    """Peforms the model fitting against the data using the LMfit library"""

    def __init__(self, p_model, p_time, p_y_data, p_control_data):
        self.logger = setup_logger("fitter_logger")
        self.model = p_model
        self.time = p_time
        self.y_data = p_y_data
        self.control_data = p_control_data
        self.normalised_data = self.y_data / self.control_data
        self.noise_window = 20
        self.noise = self.estimate_noise()
        self.mini = None

        if np.array_equal(self.time, self.y_data):
            raise ValueError("time and y_data must be the same shape")
        if np.array_equal(self.y_data, self.control_data):
            raise ValueError("y_data and control_data must be the same shape")
    
    @property
    def model(self):
        return self._model
    
    @property
    def time(self):
        return self._time
    
    @property
    def y_data(self):
        return self._y_data

    @property
    def control_data(self):
        return self._control_data

    @model.setter
    def model(self, value):
        if not isinstance(value, Model):
            raise ValueError("model must be instance of Model class")
        else:
            self._model = value
    
    @time.setter
    def time(self, value):
        if not isinstance(value, np.ndarray):
            raise ValueError("time must be of type np.ndarray")
        elif value.ndim != 1:
            raise ValueError("time array must be 1 dimensional")
        else:
            self._time = value
    
    @y_data.setter
    def y_data(self, value):
        if not isinstance(value, np.ndarray):
            raise ValueError("y_data must be of type np.ndarray")
        elif value.ndim != 1:
            raise ValueError("y_data array must be 1 dimensional")
        else:
            self._y_data = value

    @control_data.setter
    def control_data(self, value):
        if not isinstance(value, np.ndarray):
            raise ValueError("control_data must be of type np.ndarray")
        elif value.ndim != 1:
            raise ValueError("control_data array must be 1 dimensional")
        else:
            self._control_data = value

    def estimate_noise(self):
        """Estimates the noise of each data points with a rolling window"""
        data = self.normalised_data
        rolling_std = pd.Series(data).rolling(window=self.noise_window, min_periods=1).std().values
        rolling_mean = pd.Series(data).rolling(window=self.noise_window, min_periods=1).mean().values
        noise = np.abs(data - rolling_mean) / rolling_std
        return noise
    
    def residuals(self, params): # calc residuals
        """Integrates and calculates the residual between the IVP solution for the ODE model and the experimental data """
        self.model.params = params
        solution = self.model.integrate()
        normalised_sol = solution.y[-1]/self.model.max_val # normalise solution against the maximum value of the product

        if len(normalised_sol) != len(self.normalised_data):
               print(solution.message)
               exit(0)
        return (normalised_sol[1:] - self.normalised_data[1:]) / self.noise[1:]

    def fit(self):
        ''' Fits the model against the data using LMfit's minimise function'''
        self.logger.info("Fitting Model.")
        self.mini = lmfit.Minimizer(self.residuals, self.model.params)
        result = self.mini.minimize()
        if not result.errorbars: # check if parameter errors were succesfully calculated
            self.logger.info("Fit suceeded, but failed to estimate errors.")
        self.model.params = result.params
        return result