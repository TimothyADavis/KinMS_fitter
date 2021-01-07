#!/usr/bin/env python3
# coding: utf-8
import numpy as np
from gastimator import gastimator
from astropy.io import fits
import matplotlib.pyplot as plt
from gastimator import corner_plot, priors
from astropy import wcs
import matplotlib
import warnings
warnings.filterwarnings('ignore', category=wcs.FITSFixedWarning, append=True)
import matplotlib.gridspec as gridspec
from matplotlib.offsetbox import AnchoredText
import astropy.units as u
from mpl_toolkits.axes_grid1 import make_axes_locatable
from kinms import KinMS
from kinms.utils.KinMS_figures import KinMS_plotter

def gaussian(values,x):
    mu=values[0]
    sigma=values[1]
    x = (x - mu) / sigma
    return np.exp(-x*x/2.0) / np.sqrt(2.0*np.pi) / sigma
    
def random_walk_prior(val,allvalues=[],ival=0):
    # random walk prior
    if ival==0:
        mean=0
    else:
        mean= allvalues[ival-1]

    return gaussian([mean,20.],val)
    
      
class kinms_fitter:
    def __init__(self,filename,spatial_trim=None,spectral_trim=None,linefree_chans=[1,5]):
        
        self.pa_guess=0
        self.linefree_chans_start = linefree_chans[0]
        self.linefree_chans_end = linefree_chans[1]
        self.cube =self.read_primary_cube(filename)
        self.spatial_trim=spatial_trim
        self.chans2do=spectral_trim
        self.clip_cube()
        
        self.wcs=wcs.WCS(self.hdr)
        self.xc_guess=np.nanmedian(self.x1)
        self.yc_guess=np.nanmedian(self.y1)
        self.vsys_guess=np.nanmedian(self.v1)
        self.maxextent=np.max([np.max(np.abs(self.x1)),np.max(np.abs(self.y1))])
        self.nrings=np.floor(self.maxextent/self.bmaj).astype(np.int)
        self.vel_guess= np.nanstd(self.v1)
        self.cellsize=np.abs(self.hdr['CDELT1']*3600)
        self.nprocesses=None
        self.niters=3000
        self.pdf=False
        self.silent=False
        self.show_corner= False
        self.totflux_guess=np.nansum(self.cube)
        self.expscale_guess=np.max(np.abs(self.y1))/5.
        self.inc_guess=45.
        self.inc_range=[1,89]
        self.expscale_range=[0,np.max(np.abs(self.y1))]
        self.totflux_range=[0,np.nansum(self.cube)*3.]
        self.xcent_range=[np.min(self.x1),np.max(self.x1)]
        self.ycent_range=[np.min(self.y1),np.max(self.y1)]
        self.pa_range=[0,360]
        self.vsys_range=[np.nanmin(self.v1),np.nanmax(self.v1)]
        self.vel_range=[0,(np.nanmax(self.v1)-np.nanmin(self.v1))/2.]
        self.sbRad=np.arange(0,self.maxextent*2,self.cellsize/2.)
        self.nSamps=np.int(5e5)

        
        try:
            self.objname=self.hdr['OBJECT']
        except:
            self.objname="Object"
    
    def colorbar(self,mappable,ticks=None):
        ax = mappable.axes
        fig = ax.figure
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("top", size="5%", pad=0.05)
        cb=fig.colorbar(mappable, cax=cax,ticks=ticks,orientation="horizontal")
        cax.xaxis.set_ticks_position("top")
        cax.xaxis.set_label_position('top')
        return cb
            
    def read_in_a_cube(self,path):
        hdulist=fits.open(path)
        hdr=hdulist[0].header
        cube = np.squeeze(hdulist[0].data.T) #squeeze to remove singular stokes axis if present
        cube[np.isfinite(cube) == False] = 0.0
        
        try:
            if hdr['CASAMBM']:
                beamtab = hdulist[1].data
        except:
            beamtab=None
            
        return cube, hdr, beamtab
        
    def get_header_coord_arrays(self,hdr):
        self.wcs=wcs.WCS(hdr)
        self.wcs=self.wcs.sub(['longitude','latitude','spectral'])
        maxsize=np.max([hdr['NAXIS1'],hdr['NAXIS2'],hdr['NAXIS3']])

        xp,yp=np.meshgrid(np.arange(0,maxsize),np.arange(0,maxsize))
        zp = xp.copy()

        x,y,spectral = self.wcs.all_pix2world(xp,yp,zp, 0)
        


        x1=np.median(x[0:hdr['NAXIS1'],0:hdr['NAXIS2']],0)
        y1=np.median(y[0:hdr['NAXIS1'],0:hdr['NAXIS2']],1)
        spectral1=spectral[0,0:hdr['NAXIS3']]

        if (hdr['CTYPE3'] =='VRAD') or (hdr['CTYPE3'] =='VELO-LSR'):
            v1=spectral1
            try:
                if hdr['CUNIT3']=='m/s':
                     v1/=1e3
                     
            except:
                 if np.max(v1) > 1e5:
                     v1/=1e3


        else:
           f1=spectral1*u.Hz
           try:
               self.restfreq = hdr['RESTFRQ']*u.Hz
           except:
               self.restfreq = hdr['RESTFREQ']*u.Hz
           v1=f1.to(u.km/u.s, equivalencies=u.doppler_radio(self.restfreq))
           v1=v1.value

        cd3= np.median(np.diff(v1))
        cd1= np.median(np.diff(x1))
        return x1,y1,v1,np.abs(cd1*3600),cd3
           
    def rms_estimate(self,cube,chanstart,chanend):
        quarterx=np.array(self.x1.size/4.).astype(np.int)
        quartery=np.array(self.y1.size/4.).astype(np.int)
        return np.nanstd(cube[quarterx*1:3*quarterx,1*quartery:3*quartery,chanstart:chanend])
        
            
    def read_primary_cube(self,cube):
        
        ### read in cube ###
        datacube,hdr,beamtab = self.read_in_a_cube(cube)
        
        try:
           self.bmaj=np.median(beamtab['BMAJ'])
           self.bmin=np.median(beamtab['BMIN'])
           self.bpa=np.median(beamtab['BPA'])
        except:     
           self.bmaj=hdr['BMAJ']*3600.
           self.bmin=hdr['BMIN']*3600.
           self.bpa=hdr['BPA']
        
        self.hdr=hdr
                          
        self.x1,self.y1,self.v1,self.cellsize,self.dv = self.get_header_coord_arrays(self.hdr)
        
        if self.dv < 0:
            datacube = np.flip(datacube,axis=2)
            self.dv*=(-1)
            self.v1 = np.flip(self.v1)
            self.flipped=True
                
        self.rms= self.rms_estimate(datacube,self.linefree_chans_start,self.linefree_chans_end) 
        return datacube 
    
        
    def setup_params(self):
        nums=np.arange(0,self.nrings)
                
         
        # xcen, ycen, vsys
        initial_guesses=np.array([self.pa_guess,self.xc_guess,self.yc_guess,self.vsys_guess,self.inc_guess,self.expscale_guess,self.totflux_guess])
        minimums=np.array([self.pa_range[0],self.xcent_range[0],self.ycent_range[0],self.vsys_range[0],self.inc_range[0],self.expscale_range[0],self.totflux_range[0]])
        maximums=np.array([self.pa_range[1],self.xcent_range[1],self.ycent_range[1],self.vsys_range[1],self.inc_range[1],self.expscale_range[1],self.totflux_range[1]])
        labels=np.array(["PA","Xc","Yc","Vsys","inc","expscale","totflux"])
        fixed= np.array([False,False,False,False,False,False,False])
        priors=np.resize(None,fixed.size)
                

        #V(r) 
        initial_guesses= np.append(initial_guesses,np.resize(self.vel_guess,self.nrings)) 
        minimums=np.append(minimums,np.resize(self.vel_range[0],self.nrings))
        maximums=np.append(maximums,np.resize(self.vel_range[1],self.nrings))
        fixed=np.append(fixed,np.resize(False,self.nrings))
        priors=np.append(priors,np.resize(random_walk_prior,self.nrings))
        
        
        for i in nums:
            labels=np.append(labels,"V"+str(i))

        self.error=self.rms
        
        

        return initial_guesses,labels,minimums,maximums,fixed, priors
        
                     
    def clip_cube(self):
        
        if self.chans2do == None:
            self.chans2do=[0,self.v1.size]

        if self.spatial_trim == None:
            self.spatial_trim = [0, self.x1.size, 0, self.y1.size]
            
            
        self.cube=self.cube[self.spatial_trim[0]:self.spatial_trim[1],self.spatial_trim[2]:self.spatial_trim[3],self.chans2do[0]:self.chans2do[1]]
        self.x1=self.x1[self.spatial_trim[0]:self.spatial_trim[1]]
        self.y1=self.y1[self.spatial_trim[2]:self.spatial_trim[3]]
        self.v1=self.v1[self.chans2do[0]:self.chans2do[1]]
        
        
            
        
    def model(self,param):
        pa=param[0]
        xc=param[1]
        yc=param[2]
        vsys=param[3]
        inc=param[4]
        rscale=param[5]
        totflux=param[6]
        velprof=param[7:]
        vrad=np.interp(self.sbRad,self.bincentroids,velprof)
        sbprof=np.exp(-self.sbRad/rscale)
        
    
        return KinMS(self.x1.size*self.cellsize,self.y1.size*self.cellsize,self.v1.size*self.dv,self.cellsize,self.dv,\
                 [self.bmaj,self.bmin,self.bpa],inc,sbProf=sbprof,sbRad=self.sbRad,velRad=self.sbRad,velProf=vrad,\
                 intFlux=totflux,posAng=pa,fixSeed=True,vOffset=vsys - self.vsys_guess,phaseCent=[(xc-self.xc_guess)*3600.,(yc-self.yc_guess)*3600.],nSamps=self.nSamps,vSys=vsys).model_cube()
                
    
        
            
        
    def mcmc_fit(self,initial_guesses,labels,minimums,maximums,fixed,priors):
        mcmc = gastimator(self.model)
        
        mcmc.labels=labels
        mcmc.guesses=initial_guesses
        mcmc.min=minimums
        mcmc.max=maximums
        mcmc.fixed=fixed
        mcmc.prior=priors
        mcmc.silent=self.silent
        #mcmc.lnlike_func=mylnlike
        mcmc.precision= (mcmc.max-mcmc.min)/10.
        if self.nprocesses != None:
            mcmc.nprocesses= int(self.nprocesses)
        outputvalue, outputll= mcmc.run(self.cube,self.error*((2*self.cube.size)**0.25),self.niters,nchains=1,plot=False)
        bestvals=np.median(outputvalue,1)    
        besterrs=np.std(outputvalue,1)
        
        if self.show_corner:
            figure = corner_plot.corner_plot(outputvalue.T,like=outputll,labels=mcmc.labels,quantiles=[0.16, 0.5, 0.84],verbose=False)

            if self.pdf:
                plt.savefig(self.objname+"_cornerplots.pdf", bbox_inches = 'tight')
            else:
                plt.show()
        
        return bestvals, besterrs    

            
            

    
        
    def run(self):
        self.bincentroids=np.arange(0,self.nrings)*self.bmaj
        initial_guesses,labels,minimums,maximums,fixed, priors = self.setup_params()
        print(initial_guesses)
        import time
        t=time.time()
        for i in range (0,9):
            init_model=self.model(initial_guesses)
        print("One model took this long:",(time.time()-t)/10.)

        KinMS_plotter(self.cube, self.x1.size*self.cellsize,self.y1.size*self.cellsize,self.v1.size*self.dv,self.cellsize,self.dv,[self.bmaj,self.bmin,self.bpa], posang=initial_guesses[0],overcube=init_model,rms=self.error).makeplots()
        
        bestvals, besterrs = self.mcmc_fit(initial_guesses,labels,minimums,maximums,fixed,priors)
        best_model=self.model(bestvals)

        KinMS_plotter(self.cube, self.x1.size*self.cellsize,self.y1.size*self.cellsize,self.v1.size*self.dv,self.cellsize,self.dv,[self.bmaj,self.bmin,self.bpa], posang=bestvals[0],overcube=best_model,rms=self.error).makeplots()
        
        