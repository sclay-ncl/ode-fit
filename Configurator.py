import yaml
import numpy as np
import importlib.util
from lmfit import Parameters
from schema import SchemaError
from string import ascii_uppercase

from Assay import Assay
from Model import Model
from Fitter import Fitter
from config_schema import config_schema
from Logger import setup_logger

class Configurator:
    """
    Reads in config file and sets-up model fitting by instantiating an Assay, Model and Fitter class.

    Config file must be in yaml format, following the schema defined in config_schema.py.
    An example with comments describing each field is available at config_example.yaml.
    """
    
    def __init__(self, p_config_path):
        self.logger = setup_logger("config_logger")
        self.schema = config_schema
        self.config_path = p_config_path
        self.config = self.load_config()
        self.assay = self.make_assay()
        self.model = self.make_model()
        self.fitter = self.make_fitter()
        self.logger.info("Configuration complete.")

    @property
    def config_path(self):
        return self._config_path

    @config_path.setter
    def config_path(self, value):
        if not isinstance(value, str):
            raise ValueError("config_path must be of type string")
        else:
            self._config_path = value

    def load_config(self):
        """Loads the and validates the configuration file against the schema"""

        self.logger.info("Loading configuration file.")
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f) # parse yaml file contents into data structure
            try:
                self.schema.validate(config) # validate the configuration file
                return config
            except SchemaError as exc: # raise any SchemaErrors
                raise exc
    
    def make_assay(self):
        """Instantiates the the Assay class with the configuration data"""

        self.logger.info("Loading assay data.")
        assay_config = self.config["assay"]
        file_path = assay_config["file_path"]
        cols = assay_config["cols"]
        rows = assay_config["rows"]
        return Assay(file_path, cols, rows)

    def make_model(self):
        """Instantiates the the Model class with the configuration data"""

        self.logger.info("Loading model.")
        model_config = self.config["model"]
        func_path = model_config["func_path"]
        
        # fetch ode function 
        spec = importlib.util.spec_from_file_location("ODE", func_path) # create module to package function
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        ode_f = getattr(module, "ode_f") # retrieve function from module

        # create lmfit parameter objects
        param_vals = model_config["parameters"]
        params = Parameters()
        for k, v in param_vals.items():
            params.add(name=k, value=v["init_guess"], min=v["min"], max=v["max"])
        
        # load other configs
        y0 = tuple(model_config["y0"])
        max_val = model_config["max_value"]
        time = self.assay.time

        integration_config = self.config["integration"]
        atol = integration_config["atol"]
        rtol = integration_config["rtol"]
        integration_method = integration_config["method"]

        return Model(ode_f, time, params, y0, max_val, atol, rtol, integration_method)

    def make_fitter(self):
        """Instantiates the the Fitter class with the configuration data"""

        self.logger.info("Loading data fitting configuration.")
        letter_to_number = {letter: index for index, letter in enumerate(ascii_uppercase)}
        def parse_wells(well_coords):
            """parses the well coordinate strings (e.g. C3) from the configuration file"""
            wells = np.empty((0, self.assay.cycles))
            for well_str in well_coords:
                well_str = well_str.upper()
                if ":" in well_str: # if selection is a range (C3:G4)
                    well_str = well_str.split(":")
                    start_row = letter_to_number[well_str[0][0]]
                    end_row = letter_to_number[well_str[1][0]]
                    start_col = int(well_str[0][1])-1
                    end_col = int(well_str[1][1])-1
                    selection = self.assay.matrix[start_row:end_row+1, start_col:end_col+1]
                    for w in selection:
                        wells = np.vstack((wells, w))
                else:
                    row = letter_to_number[well_str[0]]
                    col = int(well_str[1:])-1
                    wells = np.vstack((wells, self.assay.matrix[row][col]))

            if np.isnan(wells).any():
                raise ValueError("One or more selected wells contains a nan value")
            return np.sum(wells, axis=0)/len(wells) # average across well range and return values
        
        fitter_config = self.config["fitter"]
        y_data = parse_wells(fitter_config["data_wells"])
        control_data = parse_wells(fitter_config["control_wells"])

        return Fitter(self.model, self.assay.time, y_data, control_data)