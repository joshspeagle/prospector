from copy import deepcopy
import numpy as np
from prospect.models import priors, sedmodel
from prospect.sources import CSPBasis
tophat = priors.tophat
from sedpy.observate import load_filters

## model dust1 as well (2:1 rule) -- need to constrain
## model dust_index as well (since attenuation now matters)
## non-parameteric SFH -- 5 params (6 bins + mass)
## logzneb -- fixed and set to logzsol (might want to leave free?)
## choose filter intrpolation dynamically based on spectral resolution
## S/N=4 reasonable for mocks, fit spectophotometry
## COSMOS?
## L_sol/M_f, M_f=M_star/S (mextra)
## fl * l**2/c = fnu

# --------------
# RUN_PARAMS
# --------------

run_params = {'verbose':True,
              'debug':False,
              'outfile':'output/demo_mock',
              # Fitter parameters
              'nwalkers':128,
              'nburn':[32, 32, 64], 'niter':512,
              'do_powell': True,
              'ftol':0.5e-5, 'maxfev':5000,
              'initial_disp':0.1,
              # Mock data parameters
              'snr': 20.0,
              'add_noise': False,
              # Mock model parameters
              'mass': 1e7,
              'logzsol': -0.5,
              'tage': 0.5,
              'tau': 80.,
              'dust2': 0.3,
              # Data manipulation parameters
              'logify_spectrum':False,
              'normalize_spectrum':False,
              'wlo':3750., 'whi':7200.,
              # SPS parameters
              'zcontinuous': 1,
              }

# --------------
# OBS
# --------------

# Here we are going to put together some filter names
# All these filters are available in sedpy.  If you want to use other filters,
# add their transmission profiles to sedpy/sedpy/data/filters/ with appropriate
# names (and format)
galex = ['galex_FUV', 'galex_NUV']
spitzer = ['spitzer_irac_ch'+n for n in ['1','2','3','4']]
sdss = ['sdss_{0}0'.format(b) for b in ['u','g','r','i','z']]


def load_obs(snr=10.0, add_noise=True, **kwargs):
    """Make a mock dataset.  Feel free to add more complicated kwargs, and put
    other things in the run_params dictionary to control how the mock is
    generated.
    """
    # We'll put the mock data in this dictionary, just as we would for real
    # data.  But we need to know which filters (and wavelengths if doing
    # spectroscopy) with which to generate mock data.
    mock = {}
    mock['wavelength'] = np.arange(3800, 9200, 2.75) # SDSS spectrum
    #mock['wavelength'] = None
    filterset = galex + sdss + spitzer[:2] # only warm spitzer
    mock['filters'] = load_filters(filterset)

    # We need the models to make a mock
    sps = load_sps(**kwargs)
    mod = load_model(**kwargs)

    # Now we get the mock params from the kwargs dict
    params = {}
    for p in mod.params.keys():
        if p in kwargs:
            params[p] = np.atleast_1d(kwargs[p])

    # And build the mock
    mod.params.update(params)
    spec, phot, _ = mod.mean_model(mod.theta, mock, sps=sps)
    # Now store some output
    mock['true_spectrum'] = spec.copy()
    mock['true_maggies'] = phot.copy()
    mock['mock_params'] = deepcopy(mod.params)
    # And add noise
    pnoise_sigma = phot / snr
    pnoise = np.random.normal(0, 1, len(phot)) * pnoise_sigma
    if add_noise:
        mock['maggies'] = phot + pnoise
    else:
        mock['maggies'] = phot.copy()
    mock['maggies_unc'] = pnoise_sigma
    mock['mock_snr'] = snr
    mock['phot_mask'] = np.ones(len(phot), dtype=bool)

    return mock

# --------------
# SPS Object
# --------------

def load_sps(zcontinuous=1, compute_vega_mags=False, **extras):
    sps = CSPBasis(zcontinuous=zcontinuous,
                   compute_vega_mags=compute_vega_mags)
    return sps

# -----------------
# Gaussian Process
# ------------------

def load_gp(**extras):
    return None, None

# --------------
# MODEL_PARAMS
# --------------

# You'll note below that we have 5 free parameters:
# mass, logzsol, tage, tau, dust2
# Each has tophat priors. They are all scalars.
#
# The other parameters are all fixed, but we want to explicitly set their
# values.

model_params = []

# --- Distance ---
model_params.append({'name': 'zred', 'N': 1,
                        'isfree': False,
                        'init': 0.0,
                        'units': '',
                        'prior_function':tophat,
                        'prior_args': {'mini':0.0, 'maxi':4.0}})

# --- SFH --------
model_params.append({'name': 'sfh', 'N': 1,
                        'isfree': False,
                        'init': 4,
                        'units': 'type'
                    })

model_params.append({'name': 'mass', 'N': 1,
                        'isfree': True,
                        'init': 1e7,
                        'init_disp': 1e6,
                        'units': r'M_\odot',
                        'prior_function':tophat,
                        'prior_args': {'mini':1e6, 'maxi':1e9}})

model_params.append({'name': 'logzsol', 'N': 1,
                        'isfree': True,
                        'init': -0.3,
                        'init_disp': 0.3,
                        'units': r'$\log (Z/Z_\odot)$',
                        'prior_function': tophat,
                        'prior_args': {'mini':-1, 'maxi':0.19}})

# If zcontinuous > 1, use 3-pt smoothing
model_params.append({'name': 'pmetals', 'N': 1,
                        'isfree': False,
                        'init': -99,})
                        
model_params.append({'name': 'tau', 'N': 1,
                        'isfree': True,
                        'init': 1.0,
                        'init_disp': 0.5,
                        'units': 'Gyr',
                        'prior_function':tophat,
                        'prior_args': {'mini':0.1, 'maxi':100}})

model_params.append({'name': 'tage', 'N': 1,
                        'isfree': True,
                        'init': 5.0,
                        'init_disp': 3.0,
                        'units': 'Gyr',
                        'prior_function':tophat,
                        'prior_args': {'mini':0.101, 'maxi':14.0}})

model_params.append({'name': 'sfstart', 'N': 1,
                        'isfree': False,
                        'init': 0.0,
                        'units': 'Gyr',
                        'prior_function':tophat,
                        'prior_args': {'mini':0.1, 'maxi':14.0}})

model_params.append({'name': 'tburst', 'N': 1,
                        'isfree': False,
                        'init': 0.0,
                        'units': '',
                        'prior_function':tophat,
                        'prior_args': {'mini':0.0, 'maxi':1.3}})

model_params.append({'name': 'fburst', 'N': 1,
                        'isfree': False,
                        'init': 0.0,
                        'units': '',
                        'prior_function':tophat,
                        'prior_args': {'mini':0.0, 'maxi':0.5}})

# --- Dust ---------
model_params.append({'name': 'dust1', 'N': 1,
                        'isfree': False,
                        'init': 0.0,
                        'units': '',
                        'prior_function':tophat,
                        'prior_args': {'mini':0.1, 'maxi':2.0}})

model_params.append({'name': 'dust2', 'N': 1,
                        'isfree': True,
                        'init': 0.35,
                        'reinit': True,
                        'init_disp': 0.3,
                        'units': '',
                        'prior_function':tophat,
                        'prior_args': {'mini':0.0, 'maxi':2.0}})

model_params.append({'name': 'dust_index', 'N': 1,
                        'isfree': False,
                        'init': -0.7,
                        'units': '',
                        'prior_function':tophat,
                        'prior_args': {'mini':-1.5, 'maxi':-0.5}})

model_params.append({'name': 'dust1_index', 'N': 1,
                        'isfree': False,
                        'init': -1.0,
                        'units': '',
                        'prior_function':tophat,
                        'prior_args': {'mini':-1.5, 'maxi':-0.5}})

model_params.append({'name': 'dust_tesc', 'N': 1,
                        'isfree': False,
                        'init': 7.0,
                        'units': 'log(Gyr)',
                        'prior_function_name': None,
                        'prior_args': None})

model_params.append({'name': 'dust_type', 'N': 1,
                        'isfree': False,
                        'init': 0,
                        'units': 'index'})

model_params.append({'name': 'add_dust_emission', 'N': 1,
                        'isfree': False,
                        'init': True,
                        'units': 'index'})

model_params.append({'name': 'duste_umin', 'N': 1,
                        'isfree': False,
                        'init': 1.0,
                        'units': 'MMP83 local MW intensity'})

# --- Stellar Pops ------------
model_params.append({'name': 'tpagb_norm_type', 'N': 1,
                        'isfree': False,
                        'init': 2,
                        'units': 'index'})

model_params.append({'name': 'add_agb_dust_model', 'N': 1,
                        'isfree': False,
                        'init': True,
                        'units': 'index'})

model_params.append({'name': 'agb_dust', 'N': 1,
                        'isfree': False,
                        'init': 1,
                        'units': 'index'})

# --- Nebular Emission ------

# For speed we turn off nebular emission in the demo
model_params.append({'name': 'add_neb_emission', 'N': 1,
                        'isfree': False,
                        'init': True})

model_params.append({'name': 'add_neb_continuum', 'N': 1,
                        'isfree': False,
                        'init': True})

model_params.append({'name': 'nebemlineinspec', 'N': 1,
                        'isfree': False,
                        'init': False})

# Here is a really simple function that takes a **dict argument, picks out the
# `logzsol` key, and returns the value.  This way, we can have gas_logz find
# the value of logzsol and use it, if we uncomment the 'depends_on' line in the
# `gas_logz` parameter definition.
#
# One can use this kind of thing to transform parameters as well (like making
# them linear instead of log, or divide everything by 10, or whatever.) You can
# have one parameter depend on several others (or vice versa).  Just remember
# that a parameter with `depends_on` must always be fixed.

def stellar_logzsol(logzsol=0.0, **extras):
    return logzsol

model_params.append({'name': 'gas_logz', 'N': 1,
                        'isfree': False,
                        'init': 0.0,
                        'units': r'log Z/Z_\odot',
#                        'depends_on': stellar_logzsol,
                        'prior_function':tophat,
                        'prior_args': {'mini':-2.0, 'maxi':0.5}})

model_params.append({'name': 'gas_logu', 'N': 1,
                        'isfree': False,
                        'init': -2.0,
                        'units': '',
                        'prior_function':tophat,
                        'prior_args': {'mini':-4, 'maxi':-1}})

# --- Calibration ---------
model_params.append({'name': 'phot_jitter', 'N': 1,
                        'isfree': False,
                        'init': 0.0,
                        'units': 'mags',
                        'prior_function':tophat,
                        'prior_args': {'mini':0.0, 'maxi':0.2}})

def load_model(**extras):
    # In principle (and we've done it) you could have the model depend on
    # command line arguments (or anything in run_params) by making changes to
    # `model_params` here before instantiation the SedModel object.  Up to you.
    return sedmodel.SedModel(model_params)

