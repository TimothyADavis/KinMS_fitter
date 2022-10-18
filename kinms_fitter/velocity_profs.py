#!/usr/bin/env python3
# coding: utf-8
import numpy as np
from jampy.mge_vcirc import mge_vcirc
from kinms.radial_motion import radial_motion
import scipy.integrate as integrate 
from pprint import pformat
from scipy import special
class velocity_profs:
    """
    Circular velocity curves from physical and arbitary models that can be combined together.
    """
    def __init__(self):
        pass    
    
    def eval(modellist,r,params,inc=90):    
        """
        Evaluates a list of velocity_profs objects, returning the total circular velocity profile.
        
        Inputs
        ------
        modellist : list of objects
            List of sb_prof objects, or objects that have the same methods/inputs.
        r: ndarray of floats
            Radius array, units of arcseconds.
        params: ndarray of floats
            Parameters to use in each model in the list.
        inc: float
            Inclination of system, in degrees.
        """
        indices=np.append(np.array([0]),np.cumsum([i.freeparams for i in modellist]))
        operations=[i.operation for i in modellist]
        out=np.zeros(r.size)
        if len(modellist) == 1:
            out= modellist[0](r,params,inc=inc)
        else:
            for i,mod in enumerate(modellist):
                if operations[i]=='quad':   
                    
                    out= out + mod(r,np.array(params[indices[i]:indices[i+1]], dtype='float64'),inc=inc)**2
            out=np.sqrt(out)
            for i,mod in enumerate(modellist):
                if operations[i]=='mult':
                    
                    out= out * mod(r,params[indices[i]:indices[i+1]])  
        return out
            
    
    class tilted_rings:  
        """
        Arbitary velocity profile using the tilted ring formalism. 
        
        Inputs
        ------
        bincentroids: ndarray of float
            Radii at which to constrain the profile. Linearly interpolated between these points.
        guesses : ndarray of float
            Initial guesses. Size of bincentroids, units of km/s.
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
        def __init__(self,bincentroids,guesses,minimums,maximums,priors=None,precisions=None,fixed=None):
            self.freeparams=bincentroids.size
            self.bincentroids=bincentroids
            self.operation="quad"
            self.labels=np.array([])
            self.units=[]
            for i in range(0,self.freeparams):
                self.labels=np.append(self.labels,"V"+str(i))
                self.units.append('km/s')

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
            """
            Returns the required profile.
            
            Inputs
            ------
            x : ndarray of float
                Input radial array in arcseconds
            args : ndarray of float
                Input arguments to evalue the profile with
            
            """
            return np.interp(x,self.bincentroids,args)
            
        
    class keplarian:  
        """
        Keplarian velocity profile for a point mass at r=0. 
        
        Inputs
        ------
        distance: ndarray of float
            Distance to the object in Mpc.
        guesses : ndarray of float
            Initial guesses. One element log10 of the central mass (in Msun).
        minimums  : ndarray of float
            Minimums for the given parameter.
        maximums  : ndarray of float
            Maximums for the given parameter.
        priors : ndarray of objects
            Optional- Priors for the given parameter (see GAStimator priors).
        precisions : ndarray of float
            Optional - Precision you want to reach for the given parameter. [Default = 10 percent of range]
        fixed: ndarray of bool
            Optional - Fix this parameter to the input value in guesses. [Default = False]
        """
        def __init__(self,distance,guesses,minimums,maximums,priors=None,precisions=None,fixed=None):
            self.freeparams=1
            self.distance=distance
            self.labels=['logCentralMass']
            self.units=['log_Msun']
            self.operation="quad"
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
                self.priors=priors
                        
            if np.any(precisions) == None:
                self.precisions=np.resize(((self.max-self.min)/10.),self.freeparams)
            else:
                self.precisions=precisions
        def __repr__(self):
            keys=['labels','min','max','fixed']
            return self.__class__.__name__+":\n"+pformat({key: vars(self)[key] for key in keys}, indent=4, width=1)        
        def __call__(self,x,args,**kwargs):
            """
            Returns the required profile.
            
            Inputs
            ------
            x : ndarray of float
                Input radial array in arcseconds
            args : ndarray of float
                Input arguments to evalue the profile with
            
            """
            return np.sqrt(4.301e-3*(10**args[0])/(4.84*self.distance*x))       
            
    class mge_vcirc:  
        """
        Evaulate an MGE model of the potential, with or without a central point mass.  
        
        Inputs
        ------
        surf : ndarray of float
            Luminosity of each gaussian component, units of Lsun.
        sigma : ndarray of float
            Width of each gaussian, units of arcsec.
        qobs : ndarray of float
            Axial ratio of each gaussian.
        distance: ndarray of float
            Distance to the object in Mpc.
        guesses : ndarray of float
            Initial guesses. One or two elements [M/L and optionally log10_CentralMass in Msun].
        minimums  : ndarray of float
            Minimums for the given parameter.
        maximums  : ndarray of float
            Maximums for the given parameter.
        priors : ndarray of objects
            Optional- Priors for the given parameter (see GAStimator priors).
        precisions : ndarray of float
            Optional - Precision you want to reach for the given parameter. [Default = 10 percent of range]
        fixed: ndarray of bool
            Optional - Fix this parameter to the input value in guesses. [Default = False]
        """
        def __init__(self,surf,sigma,qobs,distance,guesses,minimums,maximums,priors=None,precisions=None,fixed=None):
            self.guess=np.array(guesses)
            self.operation="quad"
            if self.guess.size == 1:
                self.freeparams=1
                self.labels=['M/L']
                self.units=['Msun/Lsun']
            else:
                if self.guess.size == 2:
                    self.freeparams=2
                    self.labels=['M/L','logMBH']
                    self.units=['Msun/Lsun','log_Msun']
                else:
                    raise('Wrong number of guesses, expected one or two [M/L, (and optionally logBHmass)]')    
                
            self.distance=distance
            self.surf=surf
            self.sigma=sigma
            self.qobs=qobs
            self.mininc=np.max(np.rad2deg(np.arccos(qobs-0.05)))
            self.min=np.array(minimums)
            self.max=np.array(maximums)
            

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
                bhmass=0
            else:
                bhmass=args[1]    
            return mge_vcirc(self.surf*args[0], self.sigma, self.qobs, np.clip(kwargs['inc'],self.mininc,90), 10**bhmass, self.distance, x)               

    # class mge_vcirc_innerml:
    #     def __init__(self,surf,sigma,qobs,distance,guesses,minimums,maximums,ninner=1,priors=None,precisions=None,fixed=None):
    #         self.guess=np.array(guesses)
    #         self.operation="quad"
    #         self.freeparams=3
    #         self.labels=['M/Linner','M/Louter','logMBH']
    #         self.units=['Msun/Lsun','Msun/Lsun','log_Msun']
    #         self.distance=distance
    #         self.surf=surf
    #         self.sigma=sigma
    #         self.ninner=ninner
    #         self.qobs=qobs
    #         self.mininc=np.max(np.rad2deg(np.arccos(qobs-0.05)))
    #         self.min=np.array(minimums)
    #         self.max=np.array(maximums)
    #
    #
    #         if np.any(fixed) == None:
    #             self.fixed=np.resize(False,self.freeparams)
    #         else:
    #             self.fixed=fixed
    #
    #         if np.any(priors) == None:
    #             self.priors=np.resize(None,self.freeparams)
    #         else:
    #             self.priors=priors
    #
    #         if np.any(precisions) == None:
    #             self.precisions=np.resize(((self.max-self.min)/10.),self.freeparams)
    #         else:
    #             self.precisions=precisions
    #     def __repr__(self):
    #         keys=['labels','min','max','fixed']
    #         return self.__class__.__name__+":\n"+pformat({key: vars(self)[key] for key in keys}, indent=4, width=1)
    #     def __call__(self,x,args,**kwargs):
    #         """
    #         Returns the required profile.
    #
    #         Inputs
    #         ------
    #         x : ndarray of float
    #             Input radial array in arcseconds
    #         args : ndarray of float
    #             Input arguments to evalue the profile with
    #
    #         """
    #         if self.freeparams==1:
    #             bhmass=0
    #         else:
    #             bhmass=args[2]
    #         return mge_vcirc(self.surf*np.append(np.resize(10**args[0],self.ninner),np.resize(args[1],self.surf.size-self.ninner)), self.sigma, self.qobs, np.clip(kwargs['inc'],self.mininc,90), 10**bhmass, self.distance, x)
    #

            
    class arctan:  
        """
        Arctangent velocity profile. 
        
        Inputs
        ------
        guesses : ndarray of float
            Initial guesses. Vmax and Rturn in units of km/s and arcseconds.
        minimums  : ndarray of float
            Minimums for the given parameter.
        maximums  : ndarray of float
            Maximums for the given parameter.
        priors : ndarray of objects
            Optional- Priors for the given parameter (see GAStimator priors).
        precisions : ndarray of float
            Optional - Precision you want to reach for the given parameter. [Default = 10 percent of range]
        fixed: ndarray of bool
            Optional - Fix this parameter to the input value in guesses. [Default = False]
        """
        def __init__(self,guesses,minimums,maximums,priors=None,precisions=None,fixed=None):
            self.freeparams=2
            self.operation="quad"
            self.labels=['Vmax','Rturn']
            self.units=['km/s','arcsec']
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
                self.priors=priors
                        
            if np.any(precisions) == None:
                self.precisions=np.resize(((self.max-self.min)/10.),self.freeparams)
            else:
                self.precisions=precisions
        def __repr__(self):
            keys=['labels','min','max','fixed']
            return self.__class__.__name__+":\n"+pformat({key: vars(self)[key] for key in keys}, indent=4, width=1)        
        def __call__(self,x,args,**kwargs):
            """
            Returns the required profile.
            
            Inputs
            ------
            x : ndarray of float
                Input radial array in arcseconds
            args : ndarray of float
                Input arguments to evalue the profile with
            
            """
            return ((2*args[0])/np.pi)*np.arctan(x/args[1])   
            
    class radial_barflow:  
        """
        Radial barflow from Spekkens+Sellwood.
        
        Inputs
        ------
        guesses : ndarray of float
            Initial guesses. Vtangential, Vradial, BarRadius and BarPA in units of km/s, km/s, arcsec, degrees.
        minimums  : ndarray of float
            Minimums for the given parameter.
        maximums  : ndarray of float
            Maximums for the given parameter.
        priors : ndarray of objects
            Optional- Priors for the given parameter (see GAStimator priors).
        precisions : ndarray of float
            Optional - Precision you want to reach for the given parameter. [Default = 10 percent of range]
        fixed: ndarray of bool
            Optional - Fix this parameter to the input value in guesses. [Default = False]
        """
        def __init__(self,guesses,minimums,maximums,priors=None,precisions=None,fixed=None):
            self.freeparams=4
            self.operation="quad"
            self.labels=['Vt','Vr','Rbar','phibar']
            self.units=['km/s','km/s','arcsec','deg']
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
                self.priors=priors
                        
            if np.any(precisions) == None:
                self.precisions=np.resize(((self.max-self.min)/10.),self.freeparams)
            else:
                self.precisions=precisions
        def __repr__(self):
            keys=['labels','min','max','fixed']
            return self.__class__.__name__+":\n"+pformat({key: vars(self)[key] for key in keys}, indent=4, width=1)        
        def __call__(self,x,args,**kwargs):
            """
            Returns the required profile.
            
            Inputs
            ------
            x : ndarray of float
                Input radial array in arcseconds
            args : ndarray of float
                Input arguments to evalue the profile with
            
            """
            v2t=np.heaviside(args[2]-x,1.0)*args[0]
            v2r=np.heaviside(args[2]-x,1.0)*args[1]
            phib=args[3]
            return radial_motion.bisymmetric_flow(x,v2t,v2r,phib)     

              
    class sersic:  
        """
        Velocity curve arising from a sersic mass distribution (spherically symmetric). 
        
        Inputs
        ------
        distance: float
            Distance to the object in Mpc.
        guesses : ndarray of float
            Initial guesses. Total mass, effective radius and sersic index. Units of log10(Msun), arcsec and unitless.
        minimums  : ndarray of float
            Minimums for the given parameter.
        maximums  : ndarray of float
            Maximums for the given parameter.
        priors : ndarray of objects
            Optional- Priors for the given parameter (see GAStimator priors).
        precisions : ndarray of float
            Optional - Precision you want to reach for the given parameter. [Default = 10 percent of range]
        fixed: ndarray of bool
            Optional - Fix this parameter to the input value in guesses. [Default = False]
        """
        def __init__(self,distance,guesses,minimums,maximums,priors=None,precisions=None,fixed=None):
            self.guess=np.array(guesses)
            self.freeparams=3
            self.labels=['mtot','re','n']
            self.units=['log_Msun','arcsec','unitless']
            self.operation="quad"
            self.distance=distance
            self.min=np.array(minimums)
            self.max=np.array(maximums)


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
        
        def mass(self,r,theargs,norm=1):
            if not hasattr(r, "__len__") and r == 0:
                return 0
                
            b=1.9992*theargs[2]-0.327 # for n between 0.5 and 10
            p=1.0 - (0.6097/theargs[2]) + (0.05563/(theargs[2]**2))
        
            masses=norm*((4/3.)*np.pi*r**3)*((r/theargs[1])**(-p))*np.exp(-b*(((r/theargs[1])**(1/theargs[2])))-1)
        
            if hasattr(r, "__len__"):
                masses[r==0]=0

                    
                    
            return masses
            
        def __repr__(self):
            keys=['labels','min','max','fixed']
            return self.__class__.__name__+":\n"+pformat({key: vars(self)[key] for key in keys}, indent=4, width=1)    

        def __call__(self,x,theargs,**kwargs):
            """
            Returns the required profile.
            
            Inputs
            ------
            x : ndarray of float
                Input radial array in arcseconds
            args : ndarray of float
                Input arguments to evalue the profile with
            
            """
            
            myargs=theargs.copy()
            
            r=(4.84*self.distance*x)
            myargs[1]=myargs[1]*4.84*self.distance
            
            radfac=10**(0.1763869*myargs[2] + 0.10969795)
            myargs[1]=myargs[1]/radfac
            
            
            totmass,err=integrate.quad(self.mass,0,np.inf,args=(myargs))
            
            
            
            norm=((10**myargs[0])/totmass)
            
                
            smallestmass,errs=integrate.quad(self.mass,0,r[0],args=(myargs,norm))
            
            mass=integrate.cumtrapz(self.mass(r,myargs,norm), r,initial=smallestmass)
            
            vsqr = (4.301e-3*mass)/r
            vsqr[r==0]=0
            return np.sqrt(vsqr)
            
    class bulge_disc:  
        """
        Velocity curve arising from a combination of a n=4 and n=1 sersic mass distributions (spherically symmetric). 
        
        Inputs
        ------
        distance: float
            Distance to the object in Mpc.
        guesses : ndarray of float
            Initial guesses. Total mass, effective radius and sersic index. Units of log10(Msun), arcsec and unitless.
        minimums  : ndarray of float
            Minimums for the given parameter.
        maximums  : ndarray of float
            Maximums for the given parameter.
        priors : ndarray of objects
            Optional- Priors for the given parameter (see GAStimator priors).
        precisions : ndarray of float
            Optional - Precision you want to reach for the given parameter. [Default = 10 percent of range]
        fixed: ndarray of bool
            Optional - Fix this parameter to the input value in guesses. [Default = False]
        """
        def __init__(self,distance,guesses,minimums,maximums,priors=None,precisions=None,fixed=None):
            self.guess=np.array(guesses)
            self.guesses=np.array(guesses)
            self.freeparams=4
            self.labels=['mtot','re_d','re_b','b_to_t']
            self.units=['log_Msun','arcsec','arcsec','unitless']
            self.distance=distance
            self.minimums=np.array(minimums)
            self.maximums=np.array(maximums)
            self.operation="quad"
            self.max=np.array(maximums)
            self.min=np.array(minimums)

            if np.any(fixed) == None:
                self.fixed=np.resize(False,self.freeparams)
            else:
                self.fixed=fixed
            
            if np.all(np.array(priors) == None):
                self.priors=np.resize(None,self.freeparams)
            else:
                self.priors=priors
                        
            if np.any(precisions) == None:
                self.precisions=np.resize(((self.max-self.min)/10.),self.freeparams)
            else:
                self.precisions=precisions
            
            self.mymodel = [velocity_profs.sersic(self.distance,guesses=[self.guesses[0]+np.log10(1-self.guesses[3]),self.guesses[1],1],minimums=[self.minimums[0]+np.log10(1-self.maximums[3]),self.minimums[1],1],maximums=[self.maximums[0]+np.log10(1-self.minimums[3]),self.maximums[1],1],fixed=[self.fixed[0]&self.fixed[3],self.fixed[1],True]),\
                       velocity_profs.sersic(self.distance,guesses=[self.guesses[0]+np.log10(self.guesses[3]),self.guesses[2],4],minimums=[self.minimums[0]+np.log10(self.maximums[3]),self.minimums[2],4],maximums=[self.maximums[0]+np.log10(self.minimums[3]),self.maximums[2],4],fixed=[self.fixed[0]&self.fixed[3],self.fixed[2],True])]
        
        def __repr__(self):
            keys=['labels','min','max','fixed']
            return self.__class__.__name__+":\n"+pformat({key: vars(self)[key] for key in keys}, indent=4, width=1)
            
        def __call__(self,x,theargs,**kwargs):
            """
            Returns the required profile.
            
            Inputs
            ------
            x : ndarray of float
                Input radial array in arcseconds
            args : ndarray of float
                Input arguments to evalue the profile with
            
            """
            
            return velocity_profs.eval(self.mymodel,x,[theargs[0]+np.log10(1-theargs[3]),theargs[1],1,theargs[0]+np.log10(theargs[3]),theargs[2],4])
            

    class nfw:  
        """
        Velocity curve arising from an NFW halo. 
        
        Inputs
        ------
        distance: float
            Distance to the object in Mpc.
        guesses : ndarray of float
            Initial guesses. M200 and concentration. Units of log10(Msun) and unitless.
        minimums  : ndarray of float
            Minimums for the given parameter.
        maximums  : ndarray of float
            Maximums for the given parameter.
        priors : ndarray of objects
            Optional- Priors for the given parameter (see GAStimator priors).
        precisions : ndarray of float
            Optional - Precision you want to reach for the given parameter. [Default = 10 percent of range]
        fixed: ndarray of bool
            Optional - Fix this parameter to the input value in guesses. [Default = False]
        """
        def __init__(self,distance,guesses,minimums,maximums,priors=None,precisions=None,fixed=None,hubbleconst=68.):
            self.guess=np.array(guesses)
            self.guesses=np.array(guesses)
            self.freeparams=2
            self.hubbleparam=hubbleconst
            self.operation="quad"
            self.labels=['M200','log_c']
            self.units=['log_Msun','log unitless']
            self.distance=distance
            self.minimums=np.array(minimums)
            self.maximums=np.array(maximums)
            self.max=np.array(maximums)
            self.min=np.array(minimums)

            if np.any(fixed) == None:
                self.fixed=np.resize(False,self.freeparams)
            else:
                self.fixed=fixed
            
            if np.all(np.array(priors) == None):
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

        def __call__(self,xs,theargs,**kwargs):
            """
            Returns the required profile.
            
            Inputs
            ------
            x : ndarray of float
                Input radial array in arcseconds
            args : ndarray of float
                Input arguments to evalue the profile with
            
            """
            r=self.distance*4.84e-3*xs
            c=10**theargs[1]
            m200=10**theargs[0]
            r200=10*(((4.301e-3*m200)/(100*(self.hubbleparam**2)))**(1/3))
            v200=r200*(self.hubbleparam/100)
            x=r/r200
            top=np.log(1+(c*x)) - ((c*x)/(1+(c*x)))
            bottom=np.log(1+c) - ((c)/(1+(c)))     
            return v200*np.sqrt((1/x)*(top/bottom))
            
    # class mlgrad_linear:
    #     def __init__(self,guesses,minimums,maximums,priors=None,precisions=None,fixed=None):
    #         self.freeparams=3
    #
    #         self.labels=['MLgrad','MLinter','MLendpoint']
    #         self.units=['Msun/Lsun/arcsec','Msun/Lsun','arcsec']
    #         self.operation="mult"
    #         self.min=np.array(minimums)
    #         self.max=np.array(maximums)
    #         self.guess=np.array(guesses)
    #
    #         if np.any(fixed) == None:
    #             self.fixed=np.resize(False,self.freeparams)
    #         else:
    #             self.fixed=fixed
    #
    #         if np.any(priors) == None:
    #             self.priors=np.resize(None,self.freeparams)
    #         else:
    #             self.priors=priors
    #
    #         if np.any(precisions) == None:
    #             self.precisions=np.resize(((self.max-self.min)/10.),self.freeparams)
    #         else:
    #             self.precisions=precisions
    #
    #     def __repr__(self):
    #         keys=['labels','min','max','fixed']
    #         return self.__class__.__name__+":\n"+pformat({key: vars(self)[key] for key in keys}, indent=4, width=1)
    #
    #     def __call__(self,x,args,**kwargs):
    #         """
    #         Returns the required profile.
    #
    #         Inputs
    #         ------
    #         x : ndarray of float
    #             Input radial array in arcseconds
    #         args : ndarray of float
    #             Input arguments to evalue the profile with
    #
    #         """
    #         return np.clip(x,0,args[2])*args[0] + args[1]      
                          

    class exponential_disc:  
        """
        Velocity curve arising from a razor thin exponential disc. Based on eqn 8.74 in "Dynamics and Astrophysics of Galaxies" by Bovy.
        
        Inputs
        ------
        distance: float
            Distance to the object in Mpc.
        guesses : ndarray of float
            Initial guesses. Mdisk and Scaleradius. Units of log10(Msun) and arcsec.
        minimums  : ndarray of float
            Minimums for the given parameter.
        maximums  : ndarray of float
            Maximums for the given parameter.
        priors : ndarray of objects
            Optional- Priors for the given parameter (see GAStimator priors).
        precisions : ndarray of float
            Optional - Precision you want to reach for the given parameter. [Default = 10 percent of range]
        fixed: ndarray of bool
            Optional - Fix this parameter to the input value in guesses. [Default = False]
        """
        def __init__(self,distance,guesses,minimums,maximums,priors=None,precisions=None,fixed=None):
            self.guess=np.array(guesses)
            self.guesses=np.array(guesses)
            if self.guess.size == 2:
                self.freeparams=2
                self.labels=['Mdisk','rscale']
                self.units=['log_Msun','arcsec']
            else:
                if self.guess.size == 3:
                    self.freeparams=3
                    self.labels=['Mdisk','rscale','zscale']
                    self.units=['log_Msun','arcsec','arcsec']
                else:
                    raise('Wrong number of guesses, expected two or three [Mdisk, rscale, (and optionally zscale)]')    

            self.operation="quad"
            self.distance=distance
            self.minimums=np.array(minimums)
            self.maximums=np.array(maximums)
            self.max=np.array(maximums)
            self.min=np.array(minimums)

            if np.any(fixed) == None:
                self.fixed=np.resize(False,self.freeparams)
            else:
                self.fixed=fixed
            
            if np.all(np.array(priors) == None):
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
            
        def __call__(self,xs,theargs,**kwargs):
            """
            Returns the required profile.
            
            Inputs
            ------
            x : ndarray of float
                Input radial array in arcseconds
            args : ndarray of float
                Input arguments to evalue the profile with
            
            """
            r=self.distance*4.84*xs
            rd=self.distance*4.84*theargs[1]
            
            x=r/rd
            
            
            ## based on eqn 8.74 in "Dynamics and Astrophysics of Galaxies" by Bovy 
            prefac=((4.301e-3*(10**theargs[0]))/(2*rd))*(x**2)
            endfac=special.i0(x/2.)*special.k0(x/2.) - special.i1(x/2.)*special.k1(x/2.)
            vcsqr=prefac*endfac
            
            if self.freeparams==3:
                zd=theargs[2]*4.84*self.distance
                vcsqr=vcsqr- ((((4.301e-3*(10**theargs[0]))/(rd**2))*zd)*x*np.exp(-x))

            return np.sqrt(vcsqr)    