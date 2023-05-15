import re
from schema import Schema, Use, And, Regex

assay_schema = Schema({
    "file_path": And(str, lambda n: n.endswith(".xls"), error="File must be of type .xls"),
    "cols": int,
    "rows": int
})

param_schema = Schema({
    'init_guess': Use(float), # Use call coerces ints to floats on validation
    'max': Use(float),
    'min': Use(float),
})

parameters_schema = Schema({
    str: param_schema
})

model_schema = Schema({
    "func_path": And(str, lambda n: n.endswith(".py"), error="File must be of type .py"),
    "parameters": And(parameters_schema, lambda n: len(n)>=1),
    "y0": And(list, lambda n: all((isinstance(v, float) or isinstance(v, int))  for v in n)),
    "max_value": Use(float)
})

integration_schema = Schema({
    "atol": Use(float),
    "rtol": Use(float),
    "method": str
})


fitter_schema = Schema({
    "data_wells": [Regex(r'^[A-Z]\d{1,2}(:[A-Z]\d{1,2})?$')],
    "control_wells": [Regex(r'^[A-Z]\d{1,2}(:[A-Z]\d{1,2})?$')]
})

config_schema = Schema({
    "title": str,
    "assay": assay_schema,
    "model": model_schema,
    "integration": integration_schema,
    "fitter": fitter_schema,
})


