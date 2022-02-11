#!/usr/bin/env python3
# coding: utf-8
import numpy as np
from pprint import pformat

class prior_funcs:
    def __init__():
        pass
    
    class gaussian:  
        def __init__(self,mu,sigma):
            self.mu=mu
            self.sigma=sigma
        def __repr__(self):
            return self.__class__.__name__+":\n"+pformat(vars(self), indent=4, width=1) 
        def eval(self,x,**kwargs):
            x = (x - self.mu) / self.sigma
            return np.exp(-x*x/2.0) / np.sqrt(2.0*np.pi) / self.sigma
            
    
    class physical_velocity_prior:  
        def __init__(self,rads,vel_zero_index):
            self.rads=rads
            self.zero_index=vel_zero_index
        def __repr__(self):
            return self.__class__.__name__  
        def eval(self,x,allvalues=[],ival=0):
            if ival==self.zero_index:
                return 1
            else:
                if ((allvalues[ival-1]**2*self.rads[ival-1 - self.zero_index]/self.rads[ival - self.zero_index])<x**2):
                    return 1
                else:
                    return 1e-300
