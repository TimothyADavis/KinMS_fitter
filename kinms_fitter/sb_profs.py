#!/usr/bin/env python3
# coding: utf-8
import numpy as np

class sb_profs:
    def __init__(self):
        pass    
    
    def eval(modellist,r,params):    
        indices=np.append(np.array([0]),np.cumsum([i.freeparams for i in modellist]))
        operations=[i.operation for i in modellist]
        out=np.zeros(r.size)
        for i,mod in enumerate(modellist):
            if operations[i]=='add':            
                out= out + mod(r,params[indices[i]:indices[i+1]])
            if operations[i]=='mult':
                out= out * mod(r,params[indices[i]:indices[i+1]])    
        return out
            
    
    class expdisk:  
        def __init__(self,guesses,minimums,maximums,priors=None,precisions=None,fixed=None):
            self.freeparams=2
            self.labels=np.array(['PeakFlux_exp','Rscale_exp'])
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
                self.priors=fixed
                        
            if np.any(precisions) == None:
                self.precisions=np.resize((self.max-self.min/10.),self.freeparams)
            else:
                self.precisions=precisions
                
        def __call__(self,x,args):
            return args[0]*np.exp(-x/args[1])
            
    class gaussian:  
        def __init__(self,guesses,minimums,maximums,priors=None,precisions=None,fixed=None):
            self.freeparams=3
            self.labels=np.array(['PeakFlux_gauss','Mean_gauss','sigma_gauss'])
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
                self.priors=fixed
                    
            if np.any(precisions) == None:
                self.precisions=np.resize((self.max-self.min/10.),self.freeparams)
            else:
                self.precisions=precisions
                
        def __call__(self,x,args):
            z = (x - args[1]) / args[2]
            return args[0]*np.exp(-z*z/2.0)
            
    class cutoff:  
        def __init__(self,guesses,minimums,maximums,priors=None,precisions=None,fixed=None):
            self.freeparams=2
            self.labels=np.array(['Start_cutoff','End_cutoff'])
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
                self.priors=fixed
                    
            if np.any(precisions) == None:
                self.precisions=np.resize((self.max-self.min/10.),self.freeparams)
            else:
                self.precisions=precisions
                
        def __call__(self,x,args):
            return ~((x>=args[0])&(x<args[1]))