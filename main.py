import argparse
from os.path import exists, dirname
from Configurator import Configurator
from Report import Report
import numpy as np

def main(args):
    # check that the input and output locations exist
    if not exists(args.config_f):
        print("Configuration file not found.")
    elif not exists(dirname(args.out_f)):
        print("Output directory not found.")
    else:
        config = Configurator(args.config_f) # load configuration, instantiate classes
        assay = config.assay
        model = config.model
        fitter = config.fitter
        title = config.config["title"]
        fit = fitter.fit() # begin the fitting process
        model_sol = model.normalised()
        report = Report(title, fitter.normalised_data, assay.time, model_sol, fit, fitter.mini, args.out_f)
        report.generate_pdf() # create and save the report 

if __name__ == '__main__':
    # set up arguments for command-line interface
    parser = argparse.ArgumentParser(description="Placeholder description")
    parser.add_argument('-c', '--config', dest='config_f', help='Path to yaml configuration file.')
    parser.add_argument('-o', '--output', dest='out_f', help='Path to output PDF report file.')
    args = parser.parse_args()
    main(args)