import numpy as np

def transformClouds(inclouds, posAng = 90., inc = 0., cent = [0.,0.]):
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
    clouds[:,0:2] = np.array([clouds[:,0] - cent[0],clouds[:,1] - cent[1]]).T
    posAng = np.radians(90-posAng)
    xNew = np.cos(posAng) * clouds[:,0] - np.sin(posAng) * clouds[:,1]
    yNew = np.sin(posAng) * clouds[:,0] + np.cos(posAng) * clouds[:,1]
    yNew = yNew / np.cos(np.radians(inc))
    
    clouds[:,0] = xNew
    clouds[:,1] = yNew
    
    return clouds