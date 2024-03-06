<img style="float:top,right" src="https://kinms.space/assets/img/logo_small.png">

# KinMS_fitter

[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-382/) [![PyPI version](https://badge.fury.io/py/kinms-fitter.svg)](https://badge.fury.io/py/kinms-fitter) 


Wrapper for KinMSpy that automates most common galaxy fitting tasks, and has a flexible interface for defining surface brightness/velocity profile functions. Find out more at the KinMS website: [https://www.kinms.space](https://www.kinms.space).

### Install

KinMSfitter can be installed KinMS with `pip install kinms-fitter`. Alternatively you can download the code, navigate to the directory you unpack it too, and run `python setup.py install`.
    
It requires the following modules:

* numpy
* matplotlib
* scipy
* astropy
* KinMS
* jampy
* gastimator


### Documentation

A simple iPython notebook tutorial on KinMS_fitter can be found here: [KinMS_fitter tutorial](https://github.com/TimothyADavis/KinMS_fitter/blob/main/kinms_fitter/docs/KinMS_fitter_tutorial.ipynb)

Full API documentation is avaliable [here](https://timothyadavis.github.io/KinMS_fitter/index.html).

New in KinMS_fitter v>0.5.2, is the ability to use the skySampler tool to deal with non-uniform flux distributions. See an example and some discussion [here](https://github.com/TimothyADavis/KinMS_fitter/blob/main/kinms_fitter/docs/KinMS+skySampler.ipynb)

### Commumication

If you find any bugs, or wish to be kept up to date when new versions of this software are released, please raise an issue here on github, or email me at DavisT -at- cardiff.ac.uk

### License

KinMS_fitter has a GPL-3.0 License, as detailed in the LICENSE file.


Many thanks,

Dr Timothy A. Davis

Cardiff, UK