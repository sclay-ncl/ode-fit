
# ODE Model Fitting to CLARIOstar Data

This command-line tool reads in data from the output files generated by the CLARIOstar plate-assay reader, and attempts to fit an ODE model defined in Python against that data, producing a PDF report of the results.

## Installation

To install, begin by downloading the repository:

```bash
git clone https://github.com/sclay-ncl/ode-fit
```


Next, navigate to the repository directory and install the neseccary dependencies with pip.

```bash
  cd ode-fit
  pip install -r requirements.txt
```

If you wish, using a virtual environment may stop potential dependency clashes.

That's all!
## Usage

To use the tool, navigate to the tool's directory and run `main.py`.

```bash
python3 main.py --config /path/to/config.yaml --output /path/to/output.pdf
```

The tool takes two arguements, `--config` or `-c`, and `--output` or `-o`. The former points to the location of the configuration file, and the latter points to the location to output the PDF report. 


### The Config File

The configuration file contains the information used to perform the model fitting. The file must be a YAML file, each of the elements of the file must be present: title, model, fitter, integration and assay.

The *model* elements contains the following fields: 
- func_path - a path pointing to the file that contains a Python function representation of the ODE model being fitted.
- parameters - a nested structure containing model parameter definition each parameter must contain:
    - init_guess - an initial guess for the parameter value.
    - max - the maximum value allowed for the parameter.
    - min - the minimum value allowed for the parameter.
- y0 - A list of the initial values, at time zero, for all the dependant variables of the ODE model. 
- max_value - The maximum value possible of the dependant variable that is being fit against the fluorescence data, the last value of y0. 

The *fitter* element contains:

- data_wells 
- control_wells

Each is a list of strings. Each string in the list represents the selection of a well, or range of wells, from the plate-assay data file

The *integration* configuration contains the parameters for performing the integration of the ODE model:
- atol - the absolute tolerance, a fixed value that determines the maximum allowable difference between the exact solution and the numerical solution. 
- rtol - the relative tolerance, a percentage of the current solution value that determines the maximum allowable difference between the exact solution and the numerical solution.
- method - the method to use for ODE integration, from a set of allowed methods for SciPy’s scipy.integrate.solve_ivp function. 

Finally, the *assay* configuration describes where and how the plate-assay data is stored.
- file_path - the file path to the Clariostar output sheet
- cols and rows - the number of columns and rows of the plate-assay

That's quite a long description, to see a commented example of a configuration file see the `examples` directory.


### The ODE Model
The definition of the ODE being used must meet the following requirements:

- The ODE must be a first-order ODE
- The ODE must be a function
- The function name must be ode_f
- The function must take three arguments in this order:
    - t - a list of time values for the integration
    - y0 - a list of the initial values of each of the dependant variables
    - params - a dictionary containing the parameter values
- The function must return a vector representation of the calculations for each of the dependant variables, in the order they appear in y0

- Define the ODE function in an otherwise empty `.py` file.

Once again, to see `examples` for a commented example. 

### Data File

For the data to be read in from the spreadsheet, the spreadsheet must be exported in .xls format.

The sheet containing the kinetic data must be renamed "DATA", and ensure the the "Protocol information" sheet is present and named as so.
