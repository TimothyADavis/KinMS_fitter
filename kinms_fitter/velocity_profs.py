#!/usr/bin/env python3
# coding: utf-8
import numpy as np
from jampy.mge_vcirc import mge_vcirc

class velocity_profs:
    def __init__(self):
        pass    
    
    def eval(modellist,r,params):    
        indices=np.append(np.array([0]),np.cumsum([i.freeparams for i in modellist]))
        out=np.zeros(r.size)
        for i,mod in enumerate(modellist):
            out= np.sqrt(out**2 + mod(r,np.array(params[indices[i]:indices[i+1]], dtype='float64'))**2)
        return out
            
    
    class tilted_rings:  
        def __init__(self,bincentroids,guesses,minimums,maximums,priors=None,precisions=None,fixed=None):
            self.freeparams=bincentroids.size
            self.bincentroids=bincentroids
            self.labels=np.array([])
            for i in range(0,self.freeparams):
                self.labels=np.append(self.labels,"V"+str(i))

            self.min=np.array(minimums)
            self.max=np.array(maximums)
            self.guess=np.array(guesses)

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
            ## return the velocity 
            return np.interp(x,self.bincentroids,args)
            
        
    class keplarian:  
        def __init__(self,distance,guesses,minimums,maximums,priors=None,precisions=None,fixed=None):
            self.freeparams=1
            self.distance=distance
            self.labels=['logCentralMass']

            self.min=np.array(minimums)
            self.max=np.array(maximums)
            self.guess=np.array(guesses)

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
            ## return the velocity
            return np.sqrt(4.301e-3*(10**args[0])/(4.84*self.distance*x))       
            
    class mge_vcirc:  
        def __init__(self,surf,sigma,qobs,distance,guesses,minimums,maximums,priors=None,precisions=None,fixed=None):
            self.guess=np.array(guesses)
            if self.guess.size == 2:
                self.freeparams=2
                self.labels=['M/L','inc']
            else:
                if self.guess.size == 3:
                    self.freeparams=3
                    self.labels=['M/L','inc','logMBH']
                else:
                    raise('Wrong number of guesses, expected two or three [M/L, inc (and optionally logBHmass)]')    
                
            self.distance=distance
            self.surf=surf
            self.sigma=sigma
            self.qobs=qobs
            self.min=np.array(minimums)
            self.max=np.array(maximums)
            

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
            ## return the velocity 
            if self.freeparams==2:
                bhmass=0
            else:
                bhmass=args[2]
            return mge_vcirc(self.surf*args[0], self.sigma, self.qobs, args[1], 10**bhmass, self.distance, x)               
            
    class arctan:  
        def __init__(self,guesses,minimums,maximums,priors=None,precisions=None,fixed=None):
            self.freeparams=2

            self.labels=['Vmax','Rturn']

            self.min=np.array(minimums)
            self.max=np.array(maximums)
            self.guess=np.array(guesses)

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
            ## return the velocity
            return ((2*args[0])/np.pi)*np.arctan(x/args[1])   

            
