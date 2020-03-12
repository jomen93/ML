
import numpy as np 


def N(x,mu,std):
	return (np.exp(0.5*(x-mu)**2/std**2)/np.sqrt(2*np.pi*std))


def f(x):
	return x