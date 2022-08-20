import numpy as np

def transformClouds(inclouds, posAng = 90., inc = 0., cent = [0.,0.],sbRad=None):
    """
    Calculate the galaxy co-ordinates of clouds from the sky plane. This MUST be used if any of the following conditions are true:
    inc != 0
    posAng != 90
    cent != [0,0]
    
    This exists as a stand-alone routine since an MCMC fit to the galaxy will likely need to run this every step, 
    and the sampleClouds routine is computationally expensive
    ============
    Inputs:
    
    clouds: np.ndarray The output of the sampleClouds array [x,y,I]
    
    posAng: 0=<float<360 The position angle of the galaxy major axis, measured from the y-axis of the cube
    
    inc: 0=<float<90 The inclination of the galaxy, with 90 being edge-on
    
    cent: [float,float] The photometric centre of the galaxy relative to the centre of the cube
    
    ============
    Outputs:
    
    clouds: np.ndarray Positions and intensities of particles [x',y',I]
    """
    clouds = inclouds.copy()
    
    if isinstance(posAng, (list, tuple, np.ndarray)):
        # posAng isnt a scalar- potential warp?
        if np.any(posAng!=posAng[0]): #all the same, so not actually warped
            posang_interp=True
        else:
            posang_interp=False
            posang2use=posAng[0]
    if isinstance(inc, (list, tuple, np.ndarray)):
        # posAng isnt a scalar- potential warp?
        if np.any(inc!=inc[0]): 
            inc_interp=True
        else:
            inc_interp=False #all the same, so not actually warped
            inc2use=inc[0]
                    
    if inc_interp or posang_interp:
        rflat_sqr=np.sqrt((clouds[:,0] - cent[0])**2 + (clouds[:,1] - cent[1])**2)
        if posang_interp:
            posang2use=np.interp(rflat,sbRad,posAng)
        if inc_interp:
           inc2use=np.interp(rflat,sbRad,inc)    
            
    
    clouds[:,0:2] = np.array([clouds[:,0] - cent[0],clouds[:,1] - cent[1]]).T
    posang2use = np.radians(90-posang2use)
    xNew = np.cos(posang2use) * clouds[:,0] - np.sin(posang2use) * clouds[:,1]
    yNew = np.sin(posang2use) * clouds[:,0] + np.cos(posang2use) * clouds[:,1]
    yNew = yNew / np.cos(np.radians(inc2use))
    
    clouds[:,0] = xNew
    clouds[:,1] = yNew
    
    return clouds