import numpy as np 

"""
Module for utils functions in python in tarea1.py
"""


def N(x,mu,std):
	return (np.exp(0.5*(x-mu)**2/std**2)/np.sqrt(2*np.pi*std))
