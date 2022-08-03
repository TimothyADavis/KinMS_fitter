from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

 
setup(name='kinms_fitter',
       version='0.3.0',
       description='Wrapper for KinMSpy that automates most common galaxy fitting tasks',
       url='https://github.com/TimothyADavis/KinMS_fitter',
       author='Timothy A. Davis',
       author_email='DavisT@cardiff.ac.uk',
       long_description=long_description,
       long_description_content_type="text/markdown",
       license='GNU GPLv3',
       packages=['kinms_fitter','kinms_fitter.docs'],
       install_requires=[
           'numpy',
           'gastimator',
           'astropy',
           'matplotlib',
           'spectral-cube',
           'scipy>=1.3.3',
           'jampy',
           'kinms>=3.0.0',
       ],
       classifiers=[
         'Development Status :: 3 - Alpha',
         'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
         'Programming Language :: Python :: 3',
         'Operating System :: OS Independent',
       ],
       zip_safe=True)
       
