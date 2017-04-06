import sys, os
import numpy as np
from scipy import interp
from scipy.interpolate import interp1d
from scipy.optimize import fsolve
#from sedpy.observate import getSED
from numpy import random

c = 3e18 # speed of light in AA/s
lsun = 3.846e33  # ergs/s
pc = 3.085677581467192e18  # cm
jansky_mks = 1e-26 # Jy in mks
jy_cgs = 1e-23 # Jy in cgs
AB_zero = 3631 # AB zeropoint in Jy
to_cgs = lsun/(4.0 * np.pi * (pc*10)**2 ) # L_sun to erg/s/cm^2 at 10 pc




###### emission line stuff


# Normal/Gaussian
def gaussian(mu, sigma, x):    
    dif = x - mu
    norm = 1. / np.sqrt(2*np.pi) / sigma
    return norm * np.exp(-0.5 * np.square(dif/sigma)) # normalized Gaussian


# emission line model
def emline(phi, x, sig_trunc=5.):
    mu, sigma, L = phi # grab mean, standard deviation, and scalefactor
    xlow, xhigh = mu - sig_trunc*sigma, mu + sig_trunc*sigma # get bounds
    xsel = (x > xlow) & (x < xhigh) # select elements within bounds
    Fline = np.zeros_like(x)
    Fline[xsel] += L * gaussian(mu, sigma, x[xsel]) # add truncated Gaussian to line model
    return Fline


# combination of emission lines
def emlines(phis, x, sig_trunc=5., return_lines=False):
    phis = np.array(phis).reshape(-1,3) # grab triplets of emission line parameters
    lines = np.array([emline(phi, x, sig_trunc) for phi in phis]) # collect line models
    if return_lines:
        return lines.sum(axis=0), lines
    else:
        return lines.sum(axis=0)





##### model utilities


# get emission line strengths
def get_emline_L(sps):
    Lfrac = sps.csp.emline_luminosity # luminosity [L_sun/mformed]
    mass = sps.params['mass'] # stellar mass [M_sun]
    mfrac = sps.csp.stellar_mass # mass fraction (stellar mass / mass formed)
    mformed = mass / mfrac # mass formed [M_sun]
    return  Lfrac * mformed # emission line strengths [L_sun]


# generate photometry from input spectrum
def gen_phot(wave, flambda, filters):
    # compute filter convolutions normalized by AB spectrum
    phot = np.array([filt.obj_counts(wave, flambda) / filt.ab_zero_counts for i,filt in enumerate(filters)])
    return phot


# generate emission line model
def line_model(phi, wave, obs, return_components=False):
    fl_to_fnu = obs['wavelength']**2 / c / (AB_zero * jy_cgs)
    espec, especs = emlines(phi, wave, return_lines=True)
    ephot = gen_phot(wave, espec, obs['filters'])
    espec = fl_to_fnu * np.interp(obs['wavelength'], wave, espec, left=0., right=0.)
    if return_components:
        xsels = (especs > 0)
        ephots = np.array([gen_phot(wave, especs[i], obs['filters']) for i in xrange(len(especs))])
        especs = fl_to_fnu * np.array([np.interp(obs['wavelength'], wave, es, left=0., right=0.) for es in especs])
        return espec, ephot, especs, ephots
    else:
        return espec, ephot


# generate combined continuum + lines model
def combined_model(theta, model, obs, sps, add_emline=True, phi=None, vdisp=150., return_components=False):
    mspec, mphot, mfrac = model.mean_model(theta, obs, sps=sps) # grab model spectrum [maggies], photometry [maggies], and mass fraction

    if add_emline:
        wave = sps.csp.wavelengths # wavelength basis
        fl_to_fnu = obs['wavelength']**2/ c / (AB_zero * jy_cgs) # conversion from fl [erg/s/cm^2/AA] to fnu [maggies]
        if phi is None: # initialize defaults
            mu = sps.csp.emline_wavelengths # emission line wavelengths
            sigma = vdisp*1e13/c * mu # velocity dispersions
            L = get_emline_L(sps)  # emission line luminosities [L_sun]
            lparams = np.c_[mu, sigma, L].flatten() # emission line parameters
        else:
            lparams = phi.copy()
        lparams[2::3] *= to_cgs # convert from [L_sun] to [ergs/s/cm^2]
        espec, ephot = line_model(lparams, wave, obs) # compute line model spectrum [maggies] and photometry [maggies]
    else:
        espec, ephot = 0., 0.

    if return_components:
        return mspec+espec, mphot+ephot, mfrac, [mspec, mphot], [espec, ephot]
    else:
        return mspec+espec, mphot+ephot, mfrac




##### MH utilities


# transition probability
def mh_tprob(lnp_prop, lnp_old):
    return np.exp(lnp_prop - lnp_old)

# MH update
def mh_update(pos, lnp, prop_cov, lnprobfn):
    pos_new = pos + random.multivariate_normal(np.zeros_like(pos), prop_cov) # propose new position
    lnp_new = lnprobfn(pos_new) # new lnP
    tprob = mh_tprob(lnp_new, lnp) # transition probability 
    if random.rand() <= tprob: # accept/reject step
        return pos_new, lnp_new
    else:
        return pos, lnp

# MH sampler
def mh_sampler(N, pos, lnp, prop_cov, lnprobfn):
    
    pos_chain, lnp_chain = [], [] # intialize chain
    
    for n in xrange(N):
        if n%1000==0: sys.stderr.write(str(n)+' ')
        pos, lnp = mh_update(pos, lnp, prop_cov, lnprobfn) # MH update
        pos_chain.append(pos.copy())
        lnp_chain.append(lnp)
        
    return np.array(pos_chain), np.array(lnp_chain)


######## emcee utilities


# affine-invariant proposal
def aff_inv_proposal(z, a=2):
    norm = np.sqrt(a) / 2 * (a-1) # normalization
    if 1./a <= z <= a: # bounds
        return norm / np.sqrt(z)
    else:
        return 0.


# initialize proposal distribution
a = 2. # stretch scale
zgrid = np.concatenate(([1./a-1e-10], np.logspace(-1*np.log10(a), 1*np.log10(a), 1000), [a+1e-10]), axis=0) # z grid
gzgrid = np.array([aff_inv_proposal(z) for z in zgrid]) # g(z) PDF
gzcdf = np.array([np.trapz(gzgrid[:i+1], zgrid[:i+1]) for i in xrange(len(zgrid-1))]) # g(z) CDF
fzprob = interp1d(gzcdf, zgrid) # g(z) inverse CDF


# select walker for proposal
def pick_walker(idx, Nwalkers):
    return (idx + random.randint(1, Nwalkers)) % Nwalkers # permute index by N-1


# stretch move
def stretch():
    return fzprob(random.rand()) # sample from g(z) using the inverse CDF


# transition probability
def emcee_tprob(lnp_prop, lnp_old, Z, Ndim):
    return np.exp(lnp_prop - lnp_old) * Z**(Ndim-1) # return transition probability


# update a walker
def emcee_walker_update(idx, pos, lnp, lnprobfn, Ndim, Nwalkers, T=1.):
    idx_new = pick_walker(idx, Nwalkers) # pick a random walker
    pos_new, pos_old = pos[idx_new], pos[idx] # grab positions
    Z = stretch() # sample stretch factor
    pos_prop = pos_new + Z * (pos_old - pos_new) # propose a new position
    lnp_prop, lnp_old = lnprobfn(pos_prop), lnp[idx] # compute/grab posteriors
    tprob = emcee_tprob(lnp_prop, lnp_old, Z, Ndim) # compute transition probability
    if T != 1.: tprob = tprob**(1./T) # simulated annealing modification
    if random.rand() <= tprob: # accept/reject step
        return pos_prop, lnp_prop
    else:
        return pos_old, lnp_old


# emcee sampler w/ simulated annealing burn-in assistance
def emcee_sampler(Niter, pos_init, lnp_init, lnprobfn, Nburnin, T_max, burn_schedule=10):
    
    Nwalkers, Ndim = pos_init.shape # initialize dimensions
    pos_burn = np.empty((Nburnin, Nwalkers, Ndim), dtype='float32') # positions
    lnp_burn = np.empty((Nburnin, Nwalkers), dtype='float32') # lnP
    pos_burn[0], lnp_burn[0] = pos_init, lnp_init # starting positions

    # set temperature schedule
    if T_max > 1.: 
        T_schedule = np.linspace(T_max, 1, Nburnin-1) 
    else:
        T_schedule = np.ones(Nburnin-1)
    
    # burn in phase
    for i in xrange(1, Nburnin):
        if i%10==0: 
            sys.stderr.write(str(i)+' ')
        if i%burn_schedule==0:
            good_walkers = lnp_burn[i-1] > np.median(lnp_burn[i-1]) # select better half of walkers
            mu = np.mean(pos_burn[i-1,good_walkers], axis=0) # get mean positions
            sig = np.cov(pos_burn[i-1,good_walkers], rowvar=False) # get covariance around mean positions
            pos_burn[i-1,~good_walkers] = random.multivariate_normal(mu, sig, size=np.sum(~good_walkers)) # reassign positions of worse half
        for j in xrange(Nwalkers): # serially iterate over each walker
            pos_burn[i,j], lnp_burn[i,j] = emcee_walker_update(j, pos_burn[i-1], lnp_burn[i-1], 
                                                               lnprobfn, Ndim, Nwalkers, T=T_schedule[i-1])
    
    # initialize chain for sampling
    pos_chain = np.empty((Niter, Nwalkers, Ndim), dtype='float32') # positions
    lnp_chain = np.empty((Niter, Nwalkers), dtype='float32') # lnP
    for j in xrange(Nwalkers):
        pos_chain[0,j], lnp_chain[0,j] = emcee_walker_update(j, pos_burn[-1], lnp_burn[-1], lnprobfn, Ndim, Nwalkers)
    
    # sampling phase
    for i in xrange(1, Niter):
        if i%10==0: sys.stderr.write(str(i)+' ')
        for j in xrange(Nwalkers):
            pos_chain[i,j], lnp_chain[i,j] = emcee_walker_update(j, pos_chain[i-1], lnp_chain[i-1], lnprobfn, Ndim, Nwalkers)
    return pos_chain, lnp_chain, pos_burn, lnp_burn




##### HMC utilities 


# HMC leapfrog integrator
def leapfrog(L, epsilon, Minv, q, p, gradfn, return_run=False):
    q_chain, p_chain = [], []
    q_t, p_t = q.copy(), p.copy() # initialize (position, momentum)
    p_t = p_t + 0.5 * epsilon * gradfn(q_t) # initial momentum half-step
    p_chain.append(p_t)
    q_t = q_t + epsilon * np.dot(Minv, p_t) # initial position step
    q_chain.append(q_t)
    for t in xrange(L):
        p_t = p_t + epsilon * gradfn(q_t) # momentum step
        p_chain.append(p_t)
        q_t = q_t + epsilon * np.dot(Minv, p_t) # position step
        q_chain.append(q_t)
    p_t = p_t + 0.5 * epsilon * gradfn(q_t) # final momentum half-step
    p_chain.append(p_t)
    
    if return_run:
        return q_t, -p_t, np.array(q_chain), np.array(p_chain)
    else:
        return q_t, -p_t


# HMC transition probability
def hmc_tprob(lnp, p, lnp_new, p_new, Minv):
    H = -lnp + 0.5 * np.dot(p.T, np.dot(Minv, p))
    H_new = -lnp_new + 0.5 * np.dot(p_new.T, np.dot(Minv, p_new))
    prob = np.exp(-H_new + H)
    return prob


# HMC update
def hmc_update(theta, lnp, L, epsilon, M, Minv, lnprobfn, gradfn):
    p = random.multivariate_normal(np.zeros_like(theta), M) # sample momentum 
    q, lnp = theta.copy(), lnp # initial (position, momentum, lnP)
    q_out, p_out, q_chain, p_chain = leapfrog(L, epsilon, Minv, q, p, gradfn, return_run=True) # final (position, momentum)
    lnp_out = lnprobfn(q_out) # final lnP
    tprob = hmc_tprob(lnp, p, lnp_out, p_out, Minv) # transition probability
    if random.rand() < tprob: # accept
        return q_out, lnp_out # return new position/lnP
    else: # reject
        return q, lnp # return old position


# HMC sampler
def hmc_sampler(theta, N, L, epsilon, M, Minv, lnprobfn, gradfn, stochastic=True, fvar=0.2):
    
    q = theta.copy() # position 
    lnp = lnprobfn(q) # lnP
    q_chain, lnp_chain = [], [] # intialize chain
    L_init, epsilon_init = np.copy(L), epsilon # initialize integration parameters
    
    for n in xrange(N):
        if n%100==0: sys.stderr.write(str(n)+' ')
        if stochastic: # let L and epsilon slightly vary
            L = random.randint(L_init*(1-fvar), L_init*(1+fvar)+1e-10)
            epsilon = random.uniform(epsilon_init*(1-fvar), epsilon_init*(1+fvar))
        q, lnp = hmc_update(q, lnp, L, epsilon, M, Minv, lnprobfn, gradfn) # HMC update
        q_chain.append(q.copy())
        lnp_chain.append(lnp)
        
    return np.array(q_chain), np.array(lnp_chain)



##### eHMC utilities

# eHMC leapfrog integrator
def ehmc_leapfrog(L, epsilon, Minv, q, p, gradfn_cond, Q0, dQ, return_run=False):
    q_chain, p_chain = [], []
    q_t, p_t, Q_t = q.copy(), p.copy(), Q0 # initialize (position, momentum)
    p_t = p_t + 0.5 * epsilon * gradfn_cond(q_t, Q_t) # initial momentum half-step (conditioned on Q)
    p_chain.append(p_t)
    q_t = q_t + epsilon * np.dot(Minv, p_t) # initial position step
    q_chain.append(q_t)
    Q_t += dQ
    for t in xrange(L):
        p_t = p_t + epsilon * gradfn_cond(q_t, Q_t) # momentum step (conditioned on Q)
        p_chain.append(p_t)
        q_t = q_t + epsilon * np.dot(Minv, p_t) # position step
        q_chain.append(q_t)
        Q_t += dQ
    p_t = p_t + 0.5 * epsilon * gradfn_cond(q_t, Q_t) # final momentum half-step (conditioned on Q)
    p_chain.append(p_t)
    
    if return_run:
        return q_t, -p_t, np.array(q_chain), np.array(p_chain)
    else:
        return q_t, -p_t


# eHMC transition probability
def ehmc_tprob(lnp, p, lnp_new, p_new, Minv, Z, Ndim):
    H = -lnp + 0.5 * np.dot(p.T, np.dot(Minv, p))
    H_new = -lnp_new + 0.5 * np.dot(p_new.T, np.dot(Minv, p_new))
    prob = np.exp(-H_new + H) * Z**(Ndim-1)
    return prob

    
# eHMC update
def ehmc_walker_update(idx, pos, lnp, Ndim_emcee, Nwalkers, L, epsilon, M, Minv, lnprobfn, gradfn_cond, gradsel, T=1.):

    pos_old, lnp_old = pos[idx], lnp[idx] # starting position
    idx_pivot = pick_walker(idx, Nwalkers) # pick a random walker
    pos_pivot = pos[idx_pivot] # grab walker positions
    
    pos_new = np.empty_like(pos_old) # proposed position
    
    pos_old_emcee, pos_pivot_emcee = pos_old[~gradsel], pos_pivot[~gradsel] # non-gradient terms
    Z = stretch() # sample stretch factor
    pos_new_emcee = pos_pivot_emcee + Z * (pos_old_emcee - pos_pivot_emcee) # propose stretch move for non-gradient terms
    delta_pos_emcee = (pos_new_emcee - pos_old_emcee) / (L+1.)

    p_old = random.multivariate_normal(np.zeros(sum(gradsel)), M) # sample momentum for gradient variables
    q_old = pos_old[gradsel] # original position for gradient variables

    # update HMC variables
    q_new, p_new, q_chain, p_chain = ehmc_leapfrog(L, epsilon, Minv, q_old, p_old, gradfn_cond, pos_old_emcee, delta_pos_emcee, return_run=True) # leapfrog integrate conditional posterior

    # propose state
    pos_new[~gradsel] = pos_new_emcee # emcee 
    pos_new[gradsel] = q_new # HMC
    lnp_new = lnprobfn(pos_new) # compute new posterior
    tprob = ehmc_tprob(lnp_old, p_old, lnp_new, p_new, Minv, Z, sum(~gradsel)) # transition probability
    if T != 1.: tprob = tprob**(1./T) # simulated annealing modification
    if random.rand() < tprob: # accept
        return pos_new, lnp_new # return new position/lnP
    else: # reject
        return pos_old, lnp_old # return old position/lnP


# eHMC sampler
def ehmc_sampler(Niter, pos_init, lnp_init, Nburnin, T_max, L, epsilon, M, Minv, lnprobfn, gradfn_cond, gradsel, burn_schedule=10, stochastic=True, fvar=0.2):

    Nwalkers, Ndim = pos_init.shape # initialize dimensions
    Ndim_hmc, Ndim_emcee = sum(gradsel), sum(~gradsel) # subdivide into gradient/non-gradient variables
    L_init, epsilon_init = np.copy(L), epsilon # initialize integration parameters

    pos_burn = np.empty((Nburnin, Nwalkers, Ndim), dtype='float32') # positions
    lnp_burn = np.empty((Nburnin, Nwalkers), dtype='float32') # lnP
    pos_burn[0], lnp_burn[0] = pos_init, lnp_init # starting positions

    # set temperature schedule
    if T_max > 1.: 
        T_schedule = np.linspace(T_max, 1, Nburnin-1) 
    else:
        T_schedule = np.ones(Nburnin-1)

    # burn in phase
    for i in xrange(1, Nburnin):
        if i%10==0: 
            sys.stderr.write(str(i)+' ')
        if i%burn_schedule==0:
            good_walkers = lnp_burn[i-1] > np.median(lnp_burn[i-1]) # select better half of walkers
            mu = np.mean(pos_burn[i-1,good_walkers], axis=0) # get mean positions
            sig = np.cov(pos_burn[i-1,good_walkers], rowvar=False) # get covariance around mean positions
            pos_burn[i-1,~good_walkers] = random.multivariate_normal(mu, sig, size=np.sum(~good_walkers)) # reassign positions of worse half
        for j in xrange(Nwalkers): # serially iterate over each walker
            if stochastic: # let L and epsilon slightly vary
                L = random.randint(L_init*(1-fvar), L_init*(1+fvar)+1e-10)
                epsilon = random.uniform(epsilon_init*(1-fvar), epsilon_init*(1+fvar))
            pos_burn[i,j], lnp_burn[i,j] = ehmc_walker_update(j, pos_burn[i-1], lnp_burn[i-1], Ndim_emcee, Nwalkers,
                                                              L, epsilon, M, Minv, lnprobfn, gradfn_cond, gradsel,
                                                              T=T_schedule[i-1]) # update position
               
    # initialize chain for sampling
    pos_chain = np.empty((Niter, Nwalkers, Ndim), dtype='float32') # positions
    lnp_chain = np.empty((Niter, Nwalkers), dtype='float32') # lnP
    for j in xrange(Nwalkers):
        if stochastic: # let L and epsilon slightly vary
            L = random.randint(L_init*(1-fvar), L_init*(1+fvar)+1e-10)
            epsilon = random.uniform(epsilon_init*(1-fvar), epsilon_init*(1+fvar))
        pos_chain[0,j], lnp_chain[0,j] = ehmc_walker_update(j, pos_burn[-1], lnp_burn[-1], Ndim_emcee, Nwalkers,
                                                            L, epsilon, M, Minv, lnprobfn, gradfn_cond, gradsel)
    
    # sampling phase
    for i in xrange(1, Niter):
        if i%10==0: sys.stderr.write(str(i)+' ')
        for j in xrange(Nwalkers):
            if stochastic: # let L and epsilon slightly vary
                L = random.randint(L_init*(1-fvar), L_init*(1+fvar)+1e-10)
                epsilon = random.uniform(epsilon_init*(1-fvar), epsilon_init*(1+fvar))
            pos_chain[i,j], lnp_chain[i,j] = ehmc_walker_update(j, pos_chain[i-1], lnp_chain[i-1], Ndim_emcee, Nwalkers,
                                                                L, epsilon, M, Minv, lnprobfn, gradfn_cond, gradsel)
    return pos_chain, lnp_chain, pos_burn, lnp_burn
