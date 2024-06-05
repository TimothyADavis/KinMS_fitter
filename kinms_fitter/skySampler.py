# Second build of SkySampler. This program now includes multiple routines - the main, most-costly,
# sampling algorithm and the lightweight transformation program, which rotates the 
# sky-coordinates of clouds to the galaxy plane

# Modifications:
# v2. December 2017, MDS, Oxford
# Testing and correction 25 December 2017, MDS, Horsham
# Updated for Python 3 compatibility 26 December 2017, MDS, Horsham
# Bug fixing for odd dimensioned cubes, added temprorary assertion to require square inputs, 25 June 2019, MDS, Oxford
# Included in KinMS_fitter with small modifications 05/03/24, TAD, Cardiff

import numpy as np
from scipy import interpolate
from astropy.io import fits

class skySampler:
    def __init__(self,filename,cellsize=None):
        hdu=fits.open(filename)
        self.filename=filename
        self.sb=hdu[0].data.T
        if cellsize == None:
            try:
                self.cellsize=np.abs(hdu[0].header['CDELT1']*3600)
            except:
                self.cellsize=np.abs(hdu[0].header['CD1_1']*3600)
        else:
            self.cellsize=cellsize
        
        
            
    def sample(self,nSamps = 0, sampFact = 20, weighting = None, allow_undersample = False, verbose = True,save=True):
        """
        Given a 2d image or 3d cube, will generate coordinates for a point-source model of that image
    
        ================
        Inputs:
    
        sb: np.ndarray of 2 or 3 dimensions containing the astronomical data to be modelled. If 3D, the data will be summed along the third axis.
    
        cellSize: float arcsec/px The scale of each pixel, used to convert cloud positions in pixels to physical unity
    
        nSamps: int The total number of point samples to draw from the distribution, where these are distributed weighted by the intensity.
    
        sampFact: int If nSamps not specified, assume a uniform sampling of clouds, with this many per pixel
                   WARNING: For large images, this can lead to excessive slow-downs
    
        weighting: np.ndarray of 2 dimensions of same size as sb (or with same first dimensions). Used to weight the sampling by distributions other than intensity, eg. by velocity dispersion
    
        allow_undersample: bool Default FALSE. Prevents using a small number of clouds to sample a large matrix. If the matrix is sparse, this can be disabled.
    
        ================
        Outputs:
    
        clouds: An array of [x,y,I] values corresponding to particle positions relative to the cube centre
    
        """
        if save:
            try:
                savedclouds=np.load(self.filename+"_clouds.npz")
                return savedclouds['arr_0']
            except:
                pass
                
        sb=self.sb
        
    
        assert len(sb.shape) == 2 or len(sb.shape) == 3, "The input array must be 2 or 3 dimensions"
    
        # if len(sb.shape) == 3: sb = sb.sum(axis=2)
        # assert sb.shape[0] == sb.shape[1], "Currently, correct gridding is only handled for square matrices"

        if not nSamps == 0 and allow_undersample == False: assert nSamps > sb.size, "There are insufficiently many clouds to sample the distribution. If this is a sparse array, use allow_undersample = True"
    
        inFlux = sb.sum()
    
        #Convert to a list of pixel values
        cent = [sb.shape[0]/2, sb.shape[1]/2]
        #Handle odd and even arrays
        coordsX = (np.arange(0,sb.shape[0])-cent[0]) * self.cellsize
        coordsY = (np.arange(0,sb.shape[1])-cent[1]) * self.cellsize
        delY,delX = np.meshgrid(coordsY,coordsX)     #Order for consistency with index order
    
        #breakpoint()
        sbList = np.zeros((delX.size,3))
        sbList[:,0] = delX.flatten()
        sbList[:,1] = delY.flatten()
        sbList[:,2] = sb.flatten()
        sb = sbList
    
        #Calculate number of clouds to use per pixel. By default, weight uniformly, other modes are intensity-weighted (sampFact set, nSamps not) or custom weighting
        #Priority: Custom, Intensity, Uniform
        if nSamps: 
            if not np.any(weighting):
                scheme = 'Intensity-weighted'
                weighting = sb[:,2]
            else: 
                scheme = 'custom weighting'
                weighting = weighting.flatten()
            weighting[np.isfinite(weighting)==False]=0.0
            weighting=np.abs(weighting)
            intWeight = np.nansum(weighting)
            iCloud = intWeight/nSamps
            nClouds = np.floor(weighting/iCloud)
        else: 
            scheme = 'uniform'
            nClouds = np.full(sb[:,2].shape,sampFact)
        if verbose: print('Using a ',scheme,' scheme to sample with ',np.nansum(nClouds),' clouds.')
    
        # Generate the final list of all clouds
        clouds = np.zeros([int(np.nansum(nClouds)),4])
        k=0
    
        pixSmoothing = 0.5*self.cellsize              #Smooth cloud positions to within a cell, rather than gridded to the centre of that cell
        for i in np.arange(0,sb.shape[0]):
            if not nClouds[i] == 0:
                for j in np.arange(0,nClouds[i]):
                    #print i,j,nClouds[i],k
                    clouds[k,:] = np.array([[sb[i,0]+pixSmoothing*np.random.uniform(low=-1.,high=1.),sb[i,1]+pixSmoothing*np.random.uniform(low=-1.,high=1.),0.,sb[i,2]/nClouds[i]]])
                    k = k + 1

        #Sanity checking:
        if not (clouds[:,3].sum() - inFlux) < 1e-3: print('Flux not conserved: '+str(100*(clouds[:,3].sum()-inFlux)/inFlux)+'%')
        #print(clouds)
        if save:
            np.savez(self.filename+"_clouds.npz",clouds)
            
        return clouds


