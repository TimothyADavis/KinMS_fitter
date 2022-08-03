#!/usr/bin/env python3
# coding: utf-8
import numpy as np
from pprint import pformat

class sb_profs:
    def __init__(self):
        pass    
    
    def eval(modellist,r,params):    
        indices=np.append(np.array([0]),np.cumsum([i.freeparams for i in modellist]))
        operations=[i.operation for i in modellist]
        out=np.zeros(r.size)
        for i,mod in enumerate(modellist):
            # do addative first
            if operations[i]=='add':            
                out= out + mod(r,params[indices[i]:indices[i+1]])
        for i,mod in enumerate(modellist): 
            # then do multiplicative      
            if operations[i]=='mult':
                out= out * mod(r,params[indices[i]:indices[i+1]])    
        return out
            
    
    class expdisk:  
        def __init__(self,guesses,minimums,maximums,priors=None,precisions=None,fixed=None):
            self.guess=np.array(guesses)
            if self.guess.size == 1:
                self.freeparams=1
                self.labels=['Rscale_exp']
                self.units=['arcsec']
            else:
                if self.guess.size == 2:
                    self.freeparams=2
                    self.labels=np.array(['PeakFlux_exp','Rscale_exp'])
                    self.units=['arb','arcsec']
                else:
                    raise('Wrong number of guesses, expected one or two [(optionally PeakFlux_exp), Rscale_exp]')

            self.min=np.array(minimums)
            self.max=np.array(maximums)
            self.guess=np.array(guesses)
            self.operation='add'
            if np.any(fixed) == None:
                self.fixed=np.resize(False,self.freeparams)
            else:
                self.fixed=fixed
            
            if np.any(priors) == None:
                self.priors=np.resize(None,self.freeparams)
            else:
                self.priors=priors
                        
            if np.any(precisions) == None:
                self.precisions=np.resize(((self.max-self.min)/10.),self.freeparams)
            else:
                self.precisions=precisions
    
        def __repr__(self):
            keys=['labels','min','max','fixed']
            return self.__class__.__name__+":\n"+pformat({key: vars(self)[key] for key in keys}, indent=4, width=1)
                    
        def __call__(self,x,args):
            if self.freeparams==1:
                return np.exp(-x/args[0])
            else:
                return args[0]*np.exp(-x/args[1])
            
    class gaussian:  
        def __init__(self,guesses,minimums,maximums,priors=None,precisions=None,fixed=None):
            self.guess=np.array(guesses)
            if self.guess.size == 2:
                self.freeparams=2
                self.labels=['Mean_gauss','sigma_gauss']
                self.units=['arcsec','arcsec']
            else:
                if self.guess.size == 3:
                    self.freeparams=3
                    self.labels=np.array(['PeakFlux_gauss','Mean_gauss','sigma_gauss'])
                    self.units=['arb','arcsec','arcsec']
                else:
                    raise('Wrong number of guesses, expected two or ftree [(optionally PeakFlux_gauss), Mean_gauss, sigma_gauss]')
                    
            self.min=np.array(minimums)
            self.max=np.array(maximums)
            self.operation='add'
        
            if np.any(fixed) == None:
                self.fixed=np.resize(False,self.freeparams)
            else:
                self.fixed=fixed
        
            if np.any(priors) == None:
                self.priors=np.resize(None,self.freeparams)
            else:
                self.priors=priors
                    
            if np.any(precisions) == None:
                self.precisions=np.resize(((self.max-self.min)/10.),self.freeparams)
            else:
                self.precisions=precisions
        def __repr__(self):
            keys=['labels','min','max','fixed']
            return self.__class__.__name__+":\n"+pformat({key: vars(self)[key] for key in keys}, indent=4, width=1)
                 
        def __call__(self,x,args):
            if self.freeparams==2:
                args=np.append(1,args)
            z = (x - args[1]) / args[2]
            return args[0]*np.exp(-z*z/2.0)
            
    class cutoff:  
        def __init__(self,guesses,minimums,maximums,priors=None,precisions=None,fixed=None):
            self.freeparams=2
            self.labels=np.array(['Start_cutoff','End_cutoff'])
            self.units=['arcsec','arcsec']
            self.min=np.array(minimums)
            self.max=np.array(maximums)
            self.guess=np.array(guesses)
            self.operation='mult'
        
            if np.any(fixed) == None:
                self.fixed=np.resize(False,self.freeparams)
            else:
                self.fixed=fixed
        
            if np.any(priors) == None:
                self.priors=np.resize(None,self.freeparams)
            else:
                self.priors=priors
                    
            if np.any(precisions) == None:
                self.precisions=np.resize(((self.max-self.min)/10.),self.freeparams)
            else:
                self.precisions=precisions
        def __repr__(self):
            keys=['labels','min','max','fixed']
            return self.__class__.__name__+":\n"+pformat({key: vars(self)[key] for key in keys}, indent=4, width=1)
                 
        def __call__(self,x,args):
            return ~((x>=args[0])&(x<args[1]))