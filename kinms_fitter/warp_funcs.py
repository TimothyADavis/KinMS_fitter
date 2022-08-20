#!/usr/bin/env python3
# coding: utf-8
import numpy as np
from pprint import pformat

class warp_funcs:
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

    class linear:  
        def __init__(self,guesses,minimums,maximums,priors=None,precisions=None,fixed=None,labels_prefix='PA'):
            self.guess=np.array(guesses)
            if self.guess.size == 2:
                self.freeparams=2
                self.labels=[labels_prefix+' grad',labels_prefix+' intercept']
                self.units=['deg/arcsec','deg']
            else:
                if self.guess.size == 3:
                    self.freeparams=3
                    self.labels=np.array([labels_prefix+' grad',labels_prefix+' intercept',labels_prefix+' cutoff'])
                    self.units=['deg/arcsec','deg','arcsec']
                else:
                    raise('Wrong number of guesses, expected two or three [gradient, intercept,(optionally cutoff)]')
                    

            self.min=np.array([minimums])
            self.max=np.array([maximums])
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
                out=args[0]*x + args[1]
                if self.freeparams==3:
                    out[x>args[2]]=args[0]*args[2] + args[1]
                return out
            
    
    class flat:  
        def __init__(self,guesses,minimums,maximums,priors=None,precisions=None,fixed=None,labels=['PA'],units=['deg']):
            self.guess=np.array([guesses])
            if self.guess.size == 1:
                self.freeparams=1
                self.labels=np.array([labels])
                self.units=np.array([units])
            else:
                raise('Wrong number of guesses, expected one')

            self.min=np.array([minimums])
            self.max=np.array([maximums])
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
                return args[0]

    class tilted_rings:  
        def __init__(self,bincentroids,guesses,minimums,maximums,priors=None,precisions=None,fixed=None,labels_prefix='PA'):
            self.freeparams=bincentroids.size
            self.bincentroids=bincentroids
            self.operation="add"
            self.labels=np.array([])
            self.units=[]
            for i in range(0,self.freeparams):
                self.labels=np.append(self.labels,labels_prefix+str(i))
                self.units.append('Deg')
            self.min=np.array(minimums)
            self.max=np.array(maximums)
            self.guess=np.array(guesses)
            self.units=np.array(self.units)
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
        def __call__(self,x,args,**kwargs):
            ## return the velocity 
            return np.interp(x,self.bincentroids,args)            
