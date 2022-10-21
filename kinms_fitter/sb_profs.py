#!/usr/bin/env python3
# coding: utf-8
import numpy as np
from pprint import pformat

class sb_profs:
    """
    Creates flexible surface brightness profiles that can be combined together.
    """
    def __init__(self):
        pass    
    
    def eval(modellist,r,params):    
        """
        Evaluates a list of surface brightness profiles.
        
        Inputs
        ------
        modellist : list of objects
            List of sb_prof objects, or objects that have the same methods/inputs.
        r: ndarray of floats
            Radius array, units of arcseconds.
        params: ndarray of floats
            Parameters to use in each model in the list.
        Returns
        -------
        out : ndarray of floats
            Output combined surface brightness profile.
        """
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
        """
        Creates an exponentially declining surface brightness profile.     
        
        Inputs
        ------
        guesses : ndarray of float
            Initial guesses. If a single element, then the disc profile is normalised (peak=1). If two elements then elements should be ['PeakFlux_exp','Rscale_exp']. Rscale units of arcsec.
        minimums  : ndarray of float
            Minimums for the given parameters.
        maximums  : ndarray of float
            Maximums for the given parameters.
        priors : ndarray of objects
            Optional- Priors for the given parameters (see GAStimator priors).
        precisions : ndarray of float
            Optional - Precision you want to reach for the given parameters. [Default = 10 percent of range]
        fixed: ndarray of bool
            Optional - Fix this parameter to the input value in guesses. [Default = All False]
        """
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
            """
            Returns the required profile.
            
            Inputs
            ------
            x : ndarray of float
                Input radial array in arcseconds
            args : ndarray of float
                Input arguments to evalue the profile with
            
            """
            if self.freeparams==1:
                return np.exp(-x/args[0])
            else:
                return args[0]*np.exp(-x/args[1])
            
    class gaussian:  
        """
        Creates an gaussian surface brightness profile.     
        
        Inputs
        ------
        guesses : ndarray of float
            Initial guesses. If a two elements, then the gaussian profile is normalised (peak=1). If three elements then elements should be ['PeakFlux_gauss','Mean_gauss','sigma_gauss']. Units of mean/sigma are arcsec.
        minimums  : ndarray of float
            Minimums for the given parameters.
        maximums  : ndarray of float
            Maximums for the given parameters.
        priors : ndarray of objects
            Optional- Priors for the given parameters (see GAStimator priors).
        precisions : ndarray of float
            Optional - Precision you want to reach for the given parameters. [Default = 10 percent of range]
        fixed: ndarray of bool
            Optional - Fix this parameter to the input value in guesses. [Default = All False]
        """
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
            """
            Returns the required profile.
            
            Inputs
            ------
            x : ndarray of float
                Input radial array in arcseconds
            args : ndarray of float
                Input arguments to evalue the profile with
            
            """
            if self.freeparams==2:
                args=np.append(1,args)
            z = (x - args[1]) / args[2]
            return args[0]*np.exp(-z*z/2.0)
            
    class cutoff:  
        """
        Creates an cutoff in a surface brightness profile between two radii.     
        
        Inputs
        ------
        guesses : ndarray of float
            Initial guesses. Two elements ['Start_cutoff','End_cutoff']. Units of arcsec.
        minimums  : ndarray of float
            Minimums for the given parameters.
        maximums  : ndarray of float
            Maximums for the given parameters.
        priors : ndarray of objects
            Optional- Priors for the given parameters (see GAStimator priors).
        precisions : ndarray of float
            Optional - Precision you want to reach for the given parameters. [Default = 10 percent of range]
        fixed: ndarray of bool
            Optional - Fix this parameter to the input value in guesses. [Default = All False]
        """
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
            """
            Returns the required profile.
            
            Inputs
            ------
            x : ndarray of float
                Input radial array in arcseconds
            args : ndarray of float
                Input arguments to evalue the profile with
            
            """
            return ~((x>=args[0])&(x<args[1]))
            
    class mod_sersic:  
        """
        Creates an "modified sersic" surface brightness profile. This profile seemlessly morphs
        between an exponential disc at n=1, a gaussian at n=2, and can be made to peak at any
        given radius, falling off either side.      
        
        Inputs
        ------
        guesses : ndarray of float
            Initial guesses. If a two elements, then the gaussian profile is normalised (peak=1). If three elements then elements should be ['PeakFlux_gauss','Mean_gauss','sigma_gauss']. Units of mean/sigma are arcsec.
        minimums  : ndarray of float
            Minimums for the given parameters.
        maximums  : ndarray of float
            Maximums for the given parameters.
        priors : ndarray of objects
            Optional- Priors for the given parameters (see GAStimator priors).
        precisions : ndarray of float
            Optional - Precision you want to reach for the given parameters. [Default = 10 percent of range]
        fixed: ndarray of bool
            Optional - Fix this parameter to the input value in guesses. [Default = All False]
        """
        def __init__(self,guesses,minimums,maximums,priors=None,precisions=None,fixed=None):
            self.guess=np.array(guesses)
            if self.guess.size == 3:
                self.freeparams=3
                self.labels=['mean_modsersic','sigma_modsersic','index_modsersic']
                self.units=['arcsec','arcsec','arbitary']
            else:
                if self.guess.size == 4:
                    self.freeparams=4
                    self.labels=['peak_modsersic','mean_modsersic','sigma_modsersic','index_modsersic']
                    self.units=['arbitary','arcsec','arcsec','arbitary']
                else:
                    raise('Wrong number of guesses, expected three or four [(optionally peak_modsersic),mean_modsersic,sigma_modsersic,index_modsersic]')
                    
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
            """
            Returns the required profile.
            
            Inputs
            ------
            x : ndarray of float
                Input radial array in arcseconds
            args : ndarray of float
                Input arguments to evalue the profile with
            
            """
            if self.freeparams==3:
                args=np.append(1,args)
            #z = (x - args[1]) / args[2]
            return args[0]*np.exp(-(np.abs(x-args[1])**args[3]/((args[2])**args[3])))            