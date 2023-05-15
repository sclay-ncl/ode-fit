import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import norm
from fpdf import FPDF, HTMLMixin
from lmfit.minimizer import MinimizerResult, Minimizer
from datetime import datetime
from Logger import setup_logger


script_path = os.path.dirname(os.path.abspath(__file__))
mm = 1/25.4


class Report:
    """Generates a PDF report from the data produced by the fitting procedure"""

    def __init__(self, p_title, p_y_data, p_time, p_model_sol, p_fit, p_mini, p_out):
        self.logger = setup_logger('report_logger')
        self.title = p_title
        self.y_data = p_y_data
        self.time = p_time
        self.model_sol = p_model_sol
        self.fit = p_fit
        self.mini = p_mini
        self.out = p_out

    @property
    def title(self):
        return self._title
    
    @property
    def y_data(self):
        return self._y_data
    
    @property
    def time(self):
        return self._time
    
    @property
    def model_sol(self):
        return self._model_sol

    @property
    def fit(self):
        return self._fit

    @property
    def mini(self):
        return self._mini
    
    @property
    def out(self):
        return self._out
    
    @title.setter
    def title(self, value):
        if not isinstance(value, str):
            raise ValueError("title must be of type str")
        else:
            self._title = value
    
    @y_data.setter
    def y_data(self, value):
        if not isinstance(value, np.ndarray):
            raise ValueError("y_data must be of type np.ndarray")
        elif value.ndim != 1:
            raise ValueError("y_data array must be 1 dimensional")
        else:
            self._y_data = value
    
    @time.setter
    def time(self, value):
        if not isinstance(value, np.ndarray):
            raise ValueError("time must be of type np.ndarray")
        elif value.ndim != 1:
            raise ValueError("time array must be 1 dimensional")
        else:
            self._time = value
    
    @model_sol.setter
    def model_sol(self, value):
        if not isinstance(value, np.ndarray):
            raise ValueError("model_sol must be of type np.ndarray")
        elif value.ndim != 1:
            raise ValueError("model_sol array must be 1 dimensional")
        else:
            self._model_sol = value
    
    @fit.setter
    def fit(self, value):
        if not isinstance(value, MinimizerResult):
            raise ValueError("fit must be of type lmfit.minimizer.MinimizerResult")
        else:
            self._fit = value

    @mini.setter
    def mini(self, value):
        if not isinstance(value, Minimizer):
            raise ValueError("fit must be of type lmfit.minimizer.Minimizer")
        else:
            self._mini = value

    @out.setter
    def out(self, value):
        if not isinstance(value, str):
            raise ValueError("out must be of type str")
        else:
            self._out = value

    @staticmethod
    def format_float(num):
        """Formats numbers to a specific character length"""

        abs_num = abs(num)
        if abs_num >= 1e10:
            return f"{num:.5e}"
        elif abs_num == 0:
            return "0"
        else:
            max_digits = 10 - int(abs(math.log10(abs_num))) - 1
            format_str = f"%.{max_digits}f"
            return format_str % num
    
    def fit_plot(self):
        """Creates a plot of the model solution against the experimental data"""

        self.logger.info('Creating model fit plot.')
        plt.clf()  
        fig, ax = plt.subplots(figsize=(130*mm, 80*mm))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#BBBBBB')
        ax.spines['bottom'].set_color('#BBBBBB')
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, color='#DDDDDD')
        ax.xaxis.grid(True, color='#DDDDDD')
        ax.set_ylabel("Reaction Completion Fraction")
        ax.set_xlabel("Time (s)")
        ax.plot(self.time, self.y_data, label="Data", alpha=0.7)
        ax.plot(self.time, self.model_sol, label="Fit")
        ax.set_xticks(ax.get_xticks()[1::2])
        ax.legend()
        image_path = os.path.join(script_path, 'resources', 'fit_plot.png')   
        fig.savefig(image_path, dpi=300, bbox_inches='tight')
        return image_path
        
    def covar_plot(self):
        """Creates a heatmap plot of the parameter covariance matrix"""

        self.logger.info('Creating parameter covariance plot.')
        plt.clf()
        fig, ax = plt.subplots()
        covar = self.fit.covar
        labels = self.fit.var_names
        cmap = plt.get_cmap("RdYlGn").reversed()
        plot = sns.heatmap(covar,
                           cmap=cmap, 
                           xticklabels=labels, yticklabels=labels,
                           linewidths=2,
                           annot=True,
                           ax=ax,
                           cbar_kws={"format": "%.2g"})
        image_path = os.path.join(script_path, 'resources', 'covar_plot.png')   
        fig.savefig(image_path, dpi=300, bbox_inches='tight')
        return image_path
    
    def cdf_plot(self):
        """Creates a CDF and PDF plot of the model errors"""

        self.logger.info('Creating error plots.')
        plt.clf()
        errors = self.model_sol - self.y_data
        sorted_errors = np.sort(errors)
        mean_error = np.mean(errors)
        std_error = np.std(errors)


        cdf = np.cumsum(np.ones_like(sorted_errors)) / len(sorted_errors)   
        fig, ax  = plt.subplots(2, figsize=(130*mm, 80*mm), layout="constrained")
        ax[0].plot(sorted_errors, cdf)
        # ax[0].set_xlabel('Error')
        # ax[0].set_ylabel('Proportion of Errors <= x')
        
        ax[1].hist(errors, bins=30, density=True, alpha=0.5, color='blue')
        x = np.linspace(errors.min(), errors.max(), 1000) # gaussian distribution line 
        ax[1].plot(x, norm.pdf(x, loc=mean_error, scale=std_error), color='orange')
        # ax[1].set_xlabel('Error')
        # ax[1].set_ylabel('Probability Density')

        image_path = os.path.join(script_path, 'resources', 'cdf_plot.png')   
        fig.savefig(image_path, bbox_inches='tight')
        return image_path

    def param_table(self):
        """Creates the parameter value table plot"""

        self.logger.info('Creating parameter table.')
        if self.fit.errorbars: # if errors were estimated, include StdErr
            param_data = [["Parameter", "Value", "StdErr", "Inital Value"]]
            for name, param in self.fit.params.items():
                row = [name, self.format_float(param.value), self.format_float(param.stderr), self.format_float(param.init_value)]
                param_data.append(row)
            param_df = pd.DataFrame(param_data[1:], columns=param_data[0])
        
        else: # if they weren't, dont
            param_data = [["Parameter", "Value", "Inital Value"]]
            for name, param in self.fit.params.items():
                row = [name, self.format_float(param.value), self.format_float(param.init_value)]
                param_data.append(row)
            param_df = pd.DataFrame(param_data[1:], columns=param_data[0])
        

        fig, ax = plt.subplots(figsize=(185*mm,60*mm), layout='constrained')
        ax.axis('off')
        ax.axis('tight')
        ax.autoscale(True)
        ax.table(cellText=param_df.values, colLabels=param_df.columns, loc='top')

        image_path = os.path.join(script_path, 'resources', 'param_table.png')   
        fig.savefig(image_path, bbox_inches="tight")
        return image_path

    def gof_table(self):
        """Creates the goodness-of-fit table plot"""
        
        self.logger.info('Creating GOF table.')
        df = pd.DataFrame({
            "Chi-Square":[self.format_float(self.fit.chisqr)],
            "Reduced Chi-Square": [self.format_float(self.fit.redchi)],
            "Akaike info crit": [self.format_float(self.fit.aic)],
            "Bayesian info crit": [self.format_float(self.fit.bic)]
            })
        data = df.T.values
        fig, ax = plt.subplots(figsize=(185*mm,9*mm))
        ax.axis('off')
        ax.axis('tight')
        ax.table(cellText=data, rowLabels=df.columns, loc='top', fontsize=60, cellLoc='center')
        ax.autoscale(True)
        image_path = os.path.join(script_path, 'resources', 'gof_table.png')   
        fig.savefig(image_path, bbox_inches="tight")
        return image_path
        
    def generate_pdf(self):
        """Generates the PDF using the individual components created by this class
        """
        self.logger.info('Compiling report.')
        pdf = PDF('L', 'mm', 'A4') # instatiate PDF class
        pdf.add_page()
        pdf.set_auto_page_break(auto=False)
        pdf.set_font("helvetica", size=13)
        # add title
        pdf.set_xy(31, 0)
        pdf.cell(162, 8, self.title, align='C')
        pdf.set_xy(223, 0)
        # add date
        date = datetime.now().strftime('%d/%m/%Y')
        pdf.cell(75, 8, date, align="C") 
        
        # add covariance plot
        if self.fit.errorbars:
            pdf.set_xy(187, 23)
            covar_path = self.covar_plot()
            pdf.add_image(covar_path, h=80)
        
        # add fit plot
        fit_path = self.fit_plot()
        pdf.set_xy(1, 121)
        pdf.add_image(fit_path, h=88)
        
        # add parameter table
        pdf.set_xy(1, 16)
        param_path = self.param_table() 
        pdf.add_image(param_path, h=60)
        
        # add GOF table
        pdf.set_xy(1, 85)
        gof_path = self.gof_table()   
        pdf.add_image(gof_path, h=27)
        
        pdf.set_xy(148, 122)
        cdf_path = self.cdf_plot()
        pdf.add_image(cdf_path, h=86)

        pdf.output(self.out)
        self.logger.info('Done, report saved at: {}'.format(os.path.abspath(self.out)))

class PDF(FPDF, HTMLMixin):
    ''' Subclass of the FPDF class, sets default background image'''

    def header(self):
        image_path = os.path.join(script_path, 'resources', 'background.png')
        self.image(image_path, 0, 0, self.w, self.h)

    def add_image(self, image, h):
        '''Adds images to the page at the current cursor location, limited by the height value'''
        self.image(image, x=self.get_x(), y=self.get_y(), h=h, type='png')
