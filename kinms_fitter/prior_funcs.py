#!/usr/bin/env python3
# coding: utf-8
import numpy as np
from pprint import pformat

class prior_funcs:
    """
    Priors for use in KinMS_fitter.
    """
    def __init__():
        pass
    
    class gaussian:  
        def __init__(self,mu,sigma):
            self.mu=mu
            self.sigma=sigma
        def __repr__(self):
            return self.__class__.__name__+":\n"+pformat(vars(self), indent=4, width=1) 
        def eval(self,x,**kwargs):
            xs = (x - self.mu) / self.sigma
            return (-(xs*xs)/2.0) - np.log(2.5066282746310002*self.sigma)
            
    
    class physical_velocity_prior:  
        def __init__(self,rads,vel_zero_index):
            self.rads=rads
            self.zero_index=vel_zero_index
        def __repr__(self):
            return self.__class__.__name__  
        def eval(self,x,allvalues=[],ival=0):
            if ival==self.zero_index:
                return 0
            else:
                if ((allvalues[ival-1]**2*self.rads[ival-1 - self.zero_index]/self.rads[ival - self.zero_index])<x**2):
                    return 0
                else:
                    return -1e50
    class gaussian_rolling:  
        def __init__(self,sigma,index_to_start):
            self.sigma=sigma
            self.zero_index=index_to_start
        def __repr__(self):
            return self.__class__.__name__  
        def eval(self,x,allvalues=[],ival=0):
            if ival==len(allvalues)+self.zero_index:
                return 0
            else:    
                mu=allvalues[ival-1]
                xs = (x - mu) / self.sigma
                return (-(xs*xs)/2.0) - np.log(2.5066282746310002*self.sigma)