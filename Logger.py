import logging 
import sys

def setup_logger(logger_name):
    """ Sets up a logger"""
    logger =logging.getLogger(logger_name)
    logger.setLevel("INFO")
    format = logging.Formatter('%(message)s')
    handler = logging.StreamHandler()
    handler.setFormatter(format)
    logger.addHandler(handler)
    return logger