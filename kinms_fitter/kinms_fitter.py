#!/usr/bin/env python3
# coding: utf-8
import numpy as np
from gastimator import gastimator
from astropy.io import fits
from astropy.table import Table
import matplotlib.pyplot as plt
from gastimator import corner_plot, priors
from scipy.optimize import Bounds
from scipy.optimize import minimize
import matplotlib
import warnings
from joblib import cpu_count 
import matplotlib.gridspec as gridspec
from matplotlib.offsetbox import AnchoredText
import astropy.units as u
from mpl_toolkits.axes_grid1 import make_axes_locatable
from kinms import KinMS
from kinms.utils.KinMS_figures import KinMS_plotter
from kinms_fitter.sb_profs import sb_profs
from kinms_fitter.warp_funcs import warp_funcs
from kinms_fitter.transformClouds import transformClouds
from kinms_fitter.velocity_profs import velocity_profs      
from kinms_fitter.prior_funcs import prior_funcs
import time
from spectral_cube import SpectralCube
from spectral_cube.utils import SpectralCubeWarning
warnings.filterwarnings(action='ignore', category=SpectralCubeWarning, append=True)
         
class kinms_fitter:
    def __init__(self,filename,spatial_trim=None,spectral_trim=None,linefree_chans=[1,5]):
        """
        Initalise the KinMS_fitter instance, reading in the observed datacube and trimming it as needed.

        :param filename: Path to datacube in FITS format.
        :type filename: string
        :param spatial_trim: Spatial trim in pixels [xstart,xstop,ystart,ystop]
        :type spatial_trim: list[int] or None
        :param spectral_trim: Channels to include spectrally [channel_start, channel_stop]
        :type spectral_trim: list[int] or None
        :param linefree_chans: Channels to use when estimating the RMS [channel_start, channel_stop]
        :type linefree_chans: list[int] or None
        
        """
        self.pa_guess=0
        self.linefree_chans_start = linefree_chans[0]
        self.linefree_chans_end = linefree_chans[1]
        self.chans2do=spectral_trim
        self.spectralcube=None
        self.bunit=None
        self.filename=filename
        self.cube =self.read_primary_cube(filename)
        self.spatial_trim=spatial_trim
        self.clip_cube()
        self._xc_img=np.nanmedian(self.x1)
        self._yc_img=np.nanmedian(self.y1)
        self.xc_guess=np.nanmedian(self.x1)
        self.yc_guess=np.nanmedian(self.y1)
        self.vsys_guess=np.nanmedian(self.v1)
        self.vsys_mid=np.nanmedian(self.v1)
        self.skySampClouds=np.array([])
        self.maxextent=np.max([np.max(np.abs(self.x1-self._xc_img)),np.max(np.abs(self.y1-self._yc_img))])*3600.
        self.nrings=np.floor(self.maxextent/self.bmaj).astype(np.int)
        self.vel_guess= np.nanstd(self.v1)
        self.cellsize=np.abs(self.hdr['CDELT1']*3600)
        self.nprocesses=None
        self.niters=3000
        self.pdf=True
        self.silent=False
        self.show_corner= True
        self.totflux_guess=np.nansum(self.cube)
        self.expscale_guess=self.maxextent/5.
        self.inc_guess=45.
        self.velDisp_guess=8.
        self.velDisp_range=np.array([0,50])
        self.inc_range=np.array([1,89])
        self.expscale_range=np.array([0,self.maxextent])
        self.totflux_range=np.array([0,np.nansum(self.cube)*3.])
        self.xcent_range=np.array([np.min(self.x1),np.max(self.x1)])
        self.ycent_range=np.array([np.min(self.y1),np.max(self.y1)])
        self.pa_range=np.array([0,360])
        self.vsys_range=np.array([np.nanmin(self.v1),np.nanmax(self.v1)])
        self.vel_range=np.array([0,(np.nanmax(self.v1)-np.nanmin(self.v1))/2.])
        self.sbRad=np.arange(0,self.maxextent*2,self.cellsize/3.)
        self.output_initial_model=False
        self.nSamps=np.int(5e5)
        self.sb_profile=None
        self.radial_motion=None
        self.vel_profile=None
        self.inc_profile=None
        self.pa_profile=None
        self.timetaken=0
        self.initial_guesses=None
        self.chi2_var_correct=True
        self.chi2_correct_fac=None
        self.n_pavars=1
        self.n_incvars=1
        self.mask_sum=0
        self.labels=None
        self.text_output=True
        self.pa_prior,self.xc_prior,self.yc_prior,self.vsys_prior,self.inc_prior,self.totflux_prior,self.velDisp_prior = None,None,None,None,None,None,None
        self.save_all_accepted=True
        self.interactive=True
        self.show_plots=True
        self.output_cube_fileroot=False
        self.lnlike_func=None # dont override default    
        self.tolerance=0.1 ## tolerance for simple fit. Smaller numbers are more stringent (longer runtime)
        try:
            self.objname=self.hdr['OBJECT']
        except:
            self.objname="Object"
    
    def colorbar(self,mappable,ticks=None):
        """
        Return a list of random ingredients as strings.

        :param kind: Optional "kind" of ingredients.
        :type kind: list[str] or None
        :raise lumache.InvalidKindError: If the kind is invalid.
        :return: The ingredients list.
        :rtype: list[str]

        """
        ax = mappable.axes
        fig = ax.figure
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("top", size="5%", pad=0.05)
        cb=fig.colorbar(mappable, cax=cax,ticks=ticks,orientation="horizontal")
        cax.xaxis.set_ticks_position("top")
        cax.xaxis.set_label_position('top')
        return cb
            

    def read_in_a_cube(self,path):
        self.spectralcube=SpectralCube.read(path).with_spectral_unit(u.km/u.s, velocity_convention='radio')#, rest_value=self.restfreq)
        self.bunit=self.spectralcube.unit.to_string()
        hdr=self.spectralcube.header
        cube = np.squeeze(self.spectralcube.filled_data[:,:,:].T).value #squeeze to remove singular stokes axis if present
        cube[np.isfinite(cube) == False] = 0.0
        
        try:
            self.pdf_rootname=hdr['OBJECT']+"_KinMS_fitter"
        except:
            self.pdf_rootname="KinMS_fitter"
            
        try:
            beamtab=self.spectralcube.beam
        except:
            try:
                beamtab=self.spectralcube.beams[np.floor(self.spectralcube.beams.size/2).astype(int)]
            except:
            #try flipping them
                try:
                    beamvals=[self.spectralcube.header['bmaj'],self.spectralcube.header['bmin']]
                    beamtab=Beam(major=np.max(beamvals)*u.deg,minor=np.min(beamvals)*u.deg,pa=self.spectralcube.header['bpa']*u.deg)
                except:
                    beamtab=False
 
        return cube, hdr, beamtab
        
    def get_header_coord_arrays(self,hdr):

        y,x=self.spectralcube.spatial_coordinate_map

        x1=np.median(x[0:hdr['NAXIS2'],0:hdr['NAXIS1']],0).value
        y1=np.median(y[0:hdr['NAXIS2'],0:hdr['NAXIS1']],1).value
        v1=self.spectralcube.spectral_axis.value

        cd3= np.median(np.diff(v1))
        cd1= np.median(np.diff(x1))
        
        if np.any(np.diff(x1) > 359):
            # we have RA=0 wrap issue
            x1[x1 > 180]-=360
            
            
        return x1,y1,v1,np.abs(cd1*3600),cd3
            
           
    def rms_estimate(self,cube,chanstart,chanend):
        quarterx=np.array(self.x1.size/4.).astype(np.int)
        quartery=np.array(self.y1.size/4.).astype(np.int)
        return np.nanstd(cube[quarterx*1:3*quarterx,1*quartery:3*quartery,chanstart:chanend])
        
    def from_fits_history(self, hdr):
        """
        Stolen from radio_beam, with thanks!
        """
        # a line looks like
        # HISTORY AIPS   CLEAN BMAJ=  1.7599E-03 BMIN=  1.5740E-03 BPA=   2.61
        if 'HISTORY' not in hdr:
            return None

        aipsline = None
        for line in hdr['HISTORY']:
            if 'BMAJ' in line:
                aipsline = line

        # a line looks like
        # HISTORY Sat May 10 20:53:11 2014
        # HISTORY imager::clean() [] Fitted beam used in
        # HISTORY > restoration: 1.34841 by 0.830715 (arcsec)
        #        at pa 82.8827 (deg)

        casaline = None
        for line in hdr['HISTORY']:
            if ('restoration' in line) and ('arcsec' in line):
                casaline = line
        #assert precedence for CASA style over AIPS
        #        this is a dubious choice

        if casaline is not None:
            bmaj = float(casaline.split()[2]) 
            bmin = float(casaline.split()[4]) 
            bpa = float(casaline.split()[8]) 
            return bmaj, bmin, bpa

        elif aipsline is not None:
            bmaj = float(aipsline.split()[3]) 
            bmin = float(aipsline.split()[5]) 
            bpa = float(aipsline.split()[7])
            return bmaj, bmin, bpa

        else:
            return None,None,None
                        
    def read_primary_cube(self,cube):
        
        ### read in cube ###
        datacube,hdr,beamtab = self.read_in_a_cube(cube)
        
        try:
           self.bmaj= beamtab.major.to(u.arcsec).value
           self.bmin= beamtab.minor.to(u.arcsec).value
           self.bpa=beamtab.pa.to(u.degree).value
        except:     
           try:
               self.bmaj=hdr['BMAJ']*3600.
               self.bmin=hdr['BMIN']*3600.
               self.bpa=hdr['BPA']
           except:
               self.bmaj,self.bmin,self.bpa = self.from_fits_history(hdr)
               if self.bmaj == None:
                   raise Exception('No beam information found')
        
        self.hdr=hdr
                          
        self.x1,self.y1,self.v1,self.cellsize,self.dv = self.get_header_coord_arrays(self.hdr)
        
        if self.chans2do == None:
            self.chans2do=[0,self.v1.size]
            
        if self.dv < 0:
            datacube = np.flip(datacube,axis=2)
            self.dv*=(-1)
            self.v1 = np.flip(self.v1, axis=0)
            self.chans2do=[len(self.v1)-self.chans2do[1],len(self.v1)-self.chans2do[0]]
            self.flipped=True
                
        self.rms= self.rms_estimate(datacube,self.linefree_chans_start,self.linefree_chans_end) 
        return datacube 
    
        
    def setup_params(self):
        nums=np.arange(0,self.nrings)
                
         
        # xcen, ycen, vsys, totflux, veldisp
        initial_guesses=np.array([self.xc_guess,self.yc_guess,self.vsys_guess,self.totflux_guess,self.velDisp_guess])
        minimums=np.array([self.xcent_range[0],self.ycent_range[0],self.vsys_range[0],self.totflux_range[0],self.velDisp_range[0]])
        maximums=np.array([self.xcent_range[1],self.ycent_range[1],self.vsys_range[1],self.totflux_range[1],self.velDisp_range[1]])
        labels=np.array(["Xc","Yc","Vsys","totflux","veldisp"])
        fixed= minimums == maximums
        priors=np.array([self.xc_prior,self.yc_prior,self.vsys_prior,self.totflux_prior,self.velDisp_prior])#np.resize(None,fixed.size)
        precision=(maximums-minimums)/10.    
        
        if 'beam' in self.bunit:
            newunit=(self.spectralcube.unit*u.beam*u.km/u.s).to_string()
        else:
            newunit=(self.spectralcube.unit*u.km/u.s).to_string()
            
        units=np.array(['Deg','Deg','km/s',newunit,'km/s'])
        
            
        
        vars2look=[self.pa_profile,self.inc_profile]
        
    
        if len(self.skySampClouds) == 0:
            vars2look.append(self.sb_profile)
        vars2look.append(self.vel_profile)
        if self.radial_motion != None:
            vars2look.append(self.radial_motion)

            
            
        for list_vars in vars2look:
            initial_guesses= np.append(initial_guesses,np.concatenate([i.guess for i in list_vars])) 
            minimums=np.append(minimums,np.concatenate([i.min for i in list_vars]))
            maximums=np.append(maximums,np.concatenate([i.max for i in list_vars]))
            fixed=np.append(fixed,np.concatenate([i.fixed for i in list_vars]))
            priors=np.append(priors,np.concatenate([i.priors for i in list_vars]))
            precision=np.append(precision,np.concatenate([i.precisions for i in list_vars]))
            labels=np.append(labels,np.concatenate([i.labels for i in list_vars]))
            units=np.append(units,np.concatenate([i.units for i in list_vars]))
            
        if np.any(self.initial_guesses != None):
            initial_guesses= self.initial_guesses
        
        return initial_guesses,labels,minimums,maximums,fixed, priors,precision,units
        
                     
    def clip_cube(self):
        
        if self.spatial_trim == None:
            self.spatial_trim = [0, self.x1.size, 0, self.y1.size]
            
            
        self.cube=self.cube[self.spatial_trim[0]:self.spatial_trim[1],self.spatial_trim[2]:self.spatial_trim[3],self.chans2do[0]:self.chans2do[1]]
        self.x1=self.x1[self.spatial_trim[0]:self.spatial_trim[1]]
        self.y1=self.y1[self.spatial_trim[2]:self.spatial_trim[3]]
        self.v1=self.v1[self.chans2do[0]:self.chans2do[1]]
        

    def model_simple(self,param,fileName=''):
        xc=param[0]
        yc=param[1]
        vsys=param[2]
        totflux=param[3]
        veldisp=param[4]
        phasecen=[xc,yc]
        
        
        pa=warp_funcs.eval(self.pa_profile,self.sbRad,param[5:5+self.n_pavars])
        inc=warp_funcs.eval(self.inc_profile,self.sbRad,param[5+self.n_pavars:5+self.n_pavars+self.n_incvars])

        if len(self.skySampClouds) >0:
            inClouds=transformClouds(self.skySampClouds[:,0:3],posAng = pa,inc = inc,cent = phasecen,sbRad=self.sbRad)
            sbprof=None
            myargs={'vPhaseCent': phasecen,'inClouds': inClouds, 'flux_clouds': self.skySampClouds[:,3]}
        else:
            sbprof=sb_profs.eval(self.sb_profile,self.sbRad,param[5+self.n_pavars+self.n_incvars:5+self.n_pavars+self.n_incvars+self.n_sbvars])
            myargs={'phaseCent': phasecen}
        vrad=velocity_profs.eval(self.vel_profile,self.sbRad,param[5+self.n_pavars+self.n_incvars+self.n_sbvars:],inc=inc[0])
        
        if self.n_radmotionvars >0:
            radmotion=self.radial_motion[0](self.sbRad,param[5+self.n_pavars+self.n_incvars+self.n_velvars+self.n_sbvars:])
        else:
            radmotion=None
        
        
        new=self.kinms_instance.model_cube(inc,sbProf=sbprof,sbRad=self.sbRad,velRad=self.sbRad,velProf=vrad,gasSigma=veldisp,intFlux=totflux,posAng=pa,vOffset=vsys - self.vsys_mid,vSys=vsys,radial_motion_func=radmotion,ra=(phasecen[0]/3600.)+self._xc_img,dec=(phasecen[1]/3600.)+self._yc_img,fileName=fileName,bunit=self.bunit,**myargs)
        #old=KinMSold(self.x1.size*self.cellsize,self.y1.size*self.cellsize,self.v1.size*self.dv,self.cellsize,self.dv,[self.bmaj,self.bmin,self.bpa],inc,nSamps=self.nSamps,sbProf=sbprof,sbRad=self.sbRad,velRad=self.sbRad,velProf=vrad,gasSigma=veldisp,intFlux=totflux,posAng=pa,vOffset=vsys - self.vsys_mid,vSys=vsys,radial_motion_func=radmotion,ra=(phasecen[0]/3600.)+self._xc_img,dec=(phasecen[1]/3600.)+self._yc_img,fileName=fileName,bunit=self.bunit,**myargs,verbose=True).model_cube(toplot=True)

        return new

            
        
    def mcmc_fit(self,initial_guesses,labels,minimums,maximums,fixed,priors,precision):
        mcmc = gastimator(self.model_simple)
        
        mcmc.labels=labels
        mcmc.guesses=initial_guesses
        mcmc.min=minimums
        mcmc.max=maximums
        mcmc.fixed=fixed
        mcmc.prior_func=priors
        mcmc.silent=self.silent
        mcmc.precision= precision
        self.fixed=fixed
        
        if callable(self.lnlike_func): #lnlike func overridden
            mcmc.lnlike_func=self.lnlike_func
        
        if self.nprocesses == None:
            cpucount = np.int(cpu_count())-1
            mcmc.nprocesses = np.clip(np.floor(self.niters/1250),1,cpucount).astype(int)
        else:
            mcmc.nprocesses= int(self.nprocesses)
            
            if self.niters/self.nprocesses < 1250:
                self.errors_warnings.append('WARNING: The chain assigned to each processor was very short (<1250 steps)')
                
        if not self.silent:    
            print("Parameters Fixed:",labels[mcmc.fixed]) 
    
        
        if self.chi2_var_correct & (self.chi2_correct_fac == None):
           self.chi2_correct_fac=(2*(self.mask_sum**0.25))

        if not self.chi2_var_correct:
            self.chi2_correct_fac=1
            
        if not self.silent: 
            if self.chi2_var_correct:
                print("Correction for chi-sqr variance applied:",self.chi2_correct_fac)
            else:
                print("Correction for chi-sqr variance not applied")    
        
        
        outputvalue, outputll= mcmc.run(self.cube,self.error*self.chi2_correct_fac,self.niters,nchains=1,plot=False)

        bestvals=np.median(outputvalue,1)    
        besterrs=np.std(outputvalue,1)
        
        return bestvals, besterrs, outputvalue, outputll    

    def simple_chi2(self,theargs,info):
            model=self.model_simple(theargs)
            chi2=np.nansum((self.cube-model)**2)/(np.nansum((self.error*self.chi2_correct_fac))**2)
            if chi2==0:
                chi2=10000000
            if not self.silent:     
                if info['Nfeval']%50 == 0:
                    print("Steps:",info['Nfeval'],"chi2:",chi2)
            info['Nfeval'] += 1 
            return chi2
    
    def logo(self):
        return """
        ██╗  ██╗██╗███╗   ██╗███╗   ███╗███████╗
        ██║ ██╔╝██║████╗  ██║████╗ ████║██╔════╝
        █████╔╝ ██║██╔██╗ ██║██╔████╔██║███████╗
        ██╔═██╗ ██║██║╚██╗██║██║╚██╔╝██║╚════██║
        ██║  ██╗██║██║ ╚████║██║ ╚═╝ ██║███████║
        ╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝╚═╝     ╚═╝╚══════╝
        """    

            
    def simple_fit(self,initial_guesses,labels,minimums,maximums,fixed):
        if self.chi2_var_correct & (self.chi2_correct_fac == None):
           self.chi2_correct_fac=(2*(self.mask_sum**0.25))
        
        minimums[fixed]=initial_guesses[fixed]
        maximums[fixed]=initial_guesses[fixed]
        
        self.bounds = Bounds(minimums, maximums,keep_feasible=True)

        res = minimize(self.simple_chi2, initial_guesses, args={'Nfeval':0},method ='Powell' ,bounds=self.bounds, options={'disp': True,'adaptive':True,'maxfev':self.niters,'maxiter':self.niters,'ftol':self.tolerance}) 
        
        results=res.x
            
        return results 

    def plot(self,block=True,overcube=None,savepath=None,**kwargs):
        pl=KinMS_plotter(self.cube.copy(), self.x1.size*self.cellsize,self.y1.size*self.cellsize,self.v1.size*self.dv,self.cellsize,self.dv,[self.bmaj,self.bmin,self.bpa], posang=self.pa_guess,overcube=overcube,rms=self.rms,savepath=savepath,savename=self.pdf_rootname,**kwargs)
        pl.makeplots(block=block,plot2screen=self.show_plots)
        self.mask_sum=pl.mask.sum()
        return pl
    
    def write_text(self,bestvals, errup,errdown,units, fixed,runtime,mode,errors_warnings='None',fname="KinMS_fitter_output.txt"):
        
        
        t=Table([self.labels,bestvals,errup,errdown,units,fixed],names=('Quantity', 'Best-fit', '1sig Error-up', '1sig Error-down','Units','Fixed'))

               
        t.meta['comments']=self.logo().split('\n')[:-1]
        t.meta['comments'].append('')
        try:
            t.meta['comments'].append(' Cube Fitted:    '+self.filename)
        except:
            pass
        t.meta['comments'].append(' Fitting method: '+mode)
        if mode =='MCMC':
            t.meta['comments'].append(' MCMC iterations:'+str(self.niters))
        t.meta['comments'].append(' Runtime:        '+"{:.1f}".format(runtime)+"s")    
        t.meta['comments'].append(' SBprofile:')
        for sbcomp in self.sb_profile:
            for line in sbcomp.__repr__().split('\n'):
                 t.meta['comments'].append('                 '+line)
        t.meta['comments'].append(' Velocity profile:')
        for velcomp in self.vel_profile:
            for line in velcomp.__repr__().split('\n'):
                 t.meta['comments'].append('                 '+line)         
        if errors_warnings ==[]:
            errnone="None"
        else:
            errnone=""
        t.meta['comments'].append(' Warnings:       '+errnone)
        for errmsg in errors_warnings:
            t.meta['comments'].append('                 '+errmsg)
        t.meta['comments'].append('')
                
        t.write(fname,format='ascii.fixed_width_two_line',comment='#',overwrite=True)
        
            
    def run(self,method='mcmc',justplot=False,savepath='./',**kwargs):
        method=method.lower()
        if (method!='mcmc')&(method!='both')&(method!='simple'):
            raise Exception('Method must be one of "mcmc", "simple", or "both"')
        self.bincentroids=np.arange(0,self.nrings)*self.bmaj
        self.error=self.rms
        self.errors_warnings=[]
        ### set up offset coordinates
        self.xc_guess=(self.xc_guess-self._xc_img)*3600.
        self.yc_guess=(self.yc_guess-self._yc_img)*3600.
        self.xcent_range=(np.array(self.xcent_range)-self._xc_img)*3600.
        self.ycent_range=(np.array(self.ycent_range)-self._yc_img)*3600.
        
        if np.any(self.inc_profile) == None:
            self.inc_profile=[warp_funcs.flat(self.inc_guess,self.inc_range[0],self.inc_range[1],priors=self.inc_prior,fixed=[self.inc_range[1]==self.inc_range[0]],labels='inc',units='deg')]
        self.n_incvars = np.sum([i.freeparams for i in self.inc_profile])
            
        if np.any(self.pa_profile) == None:
            self.pa_profile=[warp_funcs.flat(self.pa_guess,self.pa_range[0],self.pa_range[1],priors=self.pa_prior,fixed=[self.pa_range[1]==self.pa_range[0]],labels='PA',units='deg')]
        self.n_pavars = np.sum([i.freeparams for i in self.pa_profile])    
        
        if np.any(self.sb_profile) == None:
            # default SB profile is a single exponential disc
            self.sb_profile=[sb_profs.expdisk(guesses=[self.expscale_guess],minimums=[self.expscale_range[0]],maximums=[self.expscale_range[1]],fixed=[False])]
        
        if len(self.skySampClouds) >0:
            self.n_sbvars = 0
        else:
            self.n_sbvars = np.sum([i.freeparams for i in self.sb_profile])
         
            
        if np.any(self.vel_profile) == None:
            # default vel profile is tilted rings
            self.vel_profile=[velocity_profs.tilted_rings(self.bincentroids,guesses=np.resize(self.vel_guess,self.nrings),minimums=np.resize(self.vel_range[0],self.nrings),maximums=np.resize(self.vel_range[1],self.nrings),priors=np.resize(prior_funcs.physical_velocity_prior(self.bincentroids,5+self.n_pavars+self.n_incvars+self.n_sbvars).eval,self.nrings))]
        self.n_velvars = np.sum([i.freeparams for i in self.vel_profile])
        
        
        if np.any(self.radial_motion) == None:
            self.n_radmotionvars=0
        else:
            self.n_radmotionvars = np.sum([i.freeparams for i in self.radial_motion])
        
        
        initial_guesses,labels,minimums,maximums,fixed, priors,precision,units = self.setup_params()
        central_inc_guess=warp_funcs.eval(self.inc_profile,np.array([0]),initial_guesses[5+self.n_pavars:5+self.n_pavars+self.n_incvars])
        if central_inc_guess < 87: 
            inc_multfac= (1/np.cos(np.deg2rad(central_inc_guess)))
        else:
            inc_multfac= 20
        
        if inc_multfac > 2:    
            self.sbRad=np.append(np.arange(0,self.maxextent*2,self.cellsize/3.),np.arange(self.maxextent*2,self.maxextent*inc_multfac,self.cellsize*3.))
        
        self.labels=labels
        if self.output_initial_model != False:
            fileName=savepath+self.output_initial_model
        else:
            fileName=''
            
        ### setup the KinMS instance
        self.kinms_instance=KinMS(self.x1.size*self.cellsize,self.y1.size*self.cellsize,self.v1.size*self.dv,self.cellsize,self.dv,[self.bmaj,self.bmin,self.bpa],nSamps=self.nSamps)
        
        self.pa_guess=warp_funcs.eval(self.pa_profile,np.array([0]),initial_guesses[5:5+self.n_pavars])[0]
        t=time.time()
        init_model=self.model_simple(initial_guesses,fileName=fileName)
        self.timetaken=(time.time()-t)

        
        if not self.silent: 
            print("==============   Welcome to KinMS_fitter!   ==============")
            print(self.logo())
            print("==========================================================")
            print("One model evaluation takes {:.2f} seconds".format(self.timetaken))
        
        self.figout=self.plot(overcube=init_model,savepath=savepath,block=self.interactive,**kwargs)
        
        if justplot:
            return initial_guesses,1,1,1,1
        else:    
        
            t=time.time()
            if (method=='simple') or (method=='both'):
                if not self.silent: 
                    print("============== Begin simple fitting process ==============")
                bestvals=self.simple_fit(initial_guesses,labels,minimums,maximums,fixed)
                besterrs, outputvalue, outputll=0,0,0
                initial_guesses=bestvals
                runtime=(time.time()-t)
                if not self.silent: 
                    print("Simple fitting process took {:.2f} seconds".format(runtime))
                    print("Best fitting parameters:")
                    for name,val in zip(labels,bestvals):
                        print("   "+name+":",val)
                            
            if (method=='mcmc') or (method=='both'):    
                if not self.silent:
                    print("==============  Begin MCMC fitting process  ==============")
                    
                bestvals, besterrs, outputvalue, outputll = self.mcmc_fit(initial_guesses,labels,minimums,maximums,fixed,priors,precision)
                runtime=(time.time()-t)
                if not self.silent: 
                    print("MCMC fitting process took {:.2f} seconds".format(runtime))
                    
                if np.any(np.nanmax(outputvalue[~fixed,:],1)==np.nanmin(outputvalue[~fixed,:],1)):
                    self.errors_warnings.append('WARNING: Some parameters had no accepted guesses. Trying increasing niters.')
                    self.show_corner=False
                    if not self.silent:
                        print('Some parameters had no accepted guesses. Skipping corner plot. Trying increasing niters.')
                    

            self.pa_guess=warp_funcs.eval(self.pa_profile,np.array([0]),bestvals[5:5+self.n_pavars])[0]
            
            if self.output_cube_fileroot != False:
                fileName=self.output_cube_fileroot
            else:
                fileName=''
            best_model=self.model_simple(bestvals,fileName=fileName)
            
            bestvals[0]=(bestvals[0]/3600.)+self._xc_img
            bestvals[1]=(bestvals[1]/3600.)+self._yc_img
            
            if (method=='mcmc'):
                besterrs[0]=besterrs[0]/3600.
                besterrs[1]=besterrs[1]/3600.
            
            if (method=='mcmc')&(self.save_all_accepted):
                np.savez(self.pdf_rootname+".npz",bestvals=bestvals, besterrs=besterrs, outputvalue=outputvalue, outputll=outputll,fixed=fixed,labels=self.labels)

                                        
            if self.text_output:
                if self.pdf_rootname != "KinMS_fitter":
                    fname=self.pdf_rootname+"_KinMS_fitter_output.txt"
                else:
                    fname="KinMS_fitter_output.txt"
                if ((method=='mcmc') or (method=='both')):    
                    perc = np.percentile(outputvalue, [15.86, 50, 84.14], axis=1)
                    sig_bestfit_up = (perc[2][:] - perc[1][:])
                    sig_bestfit_down = (perc[1][:] - perc[0][:])
                    sig_bestfit_up[0:2]=sig_bestfit_up[0:2]/3600. ## into degrees
                    sig_bestfit_down[0:2]=sig_bestfit_down[0:2]/3600. ## into degrees
                else:
                    sig_bestfit_up=sig_bestfit_down=np.zeros(bestvals.size)
                self.write_text(bestvals, sig_bestfit_up, sig_bestfit_down, units, fixed,runtime,method,errors_warnings=self.errors_warnings,fname=fname)
            
            
                
            self.figout=self.plot(overcube=best_model,savepath=savepath,block=self.interactive,**kwargs)
            
            
            if ((method=='mcmc') or (method=='both')) and self.show_corner:
                fig=corner_plot.corner_plot(outputvalue[~fixed,:].T,like=outputll,\
                                    quantiles=[0.16, 0.5, 0.84],labels=self.labels[~fixed],verbose=False)
                if self.pdf:
                    plt.savefig(self.pdf_rootname+"_MCMCcornerplot.pdf")
                if self.show_plots:    
                    if self.interactive==False:
                        plt.draw()
                        plt.pause(1e-6)
                    else:
                        plt.show()
                    
                 
            return bestvals, besterrs, outputvalue, outputll, fixed
        