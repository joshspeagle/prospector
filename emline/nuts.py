## NUTS utilities (based on code by Morgan Fouesneau)

# leapfrog step
def leapfrog(q, p, lnpgrad, epsilon, fn):
    p_half = q + 0.5 * epsilon * lnpgrad # momentum half-step (initial)
    q_new = q + epsilon * p_half # position step
    lnp_new, lnpgrad_new = fn(q_new) # evaluate lnP and gradient
    p_new = p_half + 0.5 * epsilon * lnpgrid_new # momentum half-step (final)
    return q_new, p_new, lnp_new, lnpgrad_new


# simple heuristic for finding a reasonable epsilon
def find_reasonable_epsilon(q, lnp, lnpgrad, fn):
    epsilon = 1. # set epsilon randomly
    p = random.normal(size=len(q)) # sample momentum
    q_new, p_new, lnp_new, lnpgrad_new = leapfrog(q, p, lnpgrad, epsilon, fn) # step forward

    k = 1. # initial scale
    while ~np.isfinite(lnp_new) or (~np.isfinite(lnpgrad_new)).any(): # if one of our gradients has blown up
        k *= 0.5 # choose a smaller step size
        q_new, p_new, lnp_new, lnpgrad_new = leapfrog(q, p, lnpgrad, epsilon * k, fn) # try again
    epsilon = 0.5 * k * epsilon # set trial epsilon

    tprob = np.exp(lnp_new - lnp - 0.5 * (np.dot(p_new, p_new.T) - np.dot(p, p.T))) # transition probability

    a = 2. * float((tprob > 0.5)) - 1. # a={-1.,1.}
    while tprob**a > 2.**-a: # keep moving epsilon in the same direction until tprob>0.5
        epsilon = epsilon * (2.**a)
        q_new, p_new, lnp_new, lnpgrad_new = leapfrog(q, p, lnpgrad, epsilon, fn)
        tprob = np.exp(lnp_new - lnp - 0.5 * (np.dot(p_new, p_new.T) - np.dot(p, p.T)))

    sys.stderr.write('epsilon_guess = '+str(epsilon))
    return epsilon


# stopping criterion for proposing new positions
def stop_criterion(q_minus, q_plus, p_minus, p_plus):
    dq = q_plus - q_minus # differences in furthest trajectories
    stop_flag = (np.dot(dq, p_minus.T) >= 0) & (np.dot(dq, p_plus.T) >= 0) # check if we've doubled back on ourselves
    return stop_flag


# building the tree used for NUTS
def build_tree(q, p, lnpgrad, logu, v, j, epsilon, fn, H, logu_bound=1000.):

    if j == 0: # base case
        q_new, p_new, lnp_new, lnpgrad_new = leapfrog(q, p, lnpgrad, v*epsilon, fn) # leapfrog in v (+/-) direction
        H_new = lnp_new - 0.5 * np.dot(p_new, p_new.T) # new Hamiltonian
        n_new = int(logu < joint) # check if new point is in slice
        s_new = int((logu - logu_bound) < H) # check if simulation is wildly inaccurate

        # set return values (minus=plus for all things here since the "tree" is of depth 0)
        q_minus, q_plus = q_new[:], q_new[:]
        p_minus, p_plus = p_new[:], p_new[:]
        lnpgrad_minus, lnpgrad_plus = lnpgrad_new[:], lnpgrad_new[:]

        a_new = min(1., np.exp(H_new - H)) # acceptance probability
        na_new = 1 # number of points

    else: # recursively build height j-1 left and right subtrees
        minus_var, plus_var, new_var, anc_var = build_tree(q, p, lnpgrad, logu, v, j-1, epsilon, fn, H) # build tree for j-1
        q_minus, p_minus, lnpgrad_minus = minus_var # leftmost entry
        q_plus, p_plus, lnpgrad_plus = plus_var # rightmost entry
        q_new, p_new, lnp_new, lnpgrad_new = new_var # new proposal
        n_new, s_new, a_new, na_new = anc_var # anciliary variables

        # if the stopping criteria were met in the first subtree, we're done
        if s_new == 1: # if simulated trajectory is good

            if v == -1: # if we went backwards in time
                minus_var2, plus_var2, new_var2, anc_var2 = build_tree(q_minus, p_minus, lnpgrad_minus, logu, v, j-1, epsilon, fn, H) # build from leftmost entry
                q_minus, p_minus, lnpgrad_minus = minus_var2[:] # leftmost entry
                q_new2, p_new2, lnp_new2, lnpgrad_new2 = new_var2[:] # new proposal
                n_new2, s_new2, a_new2, na_new2 = anc_var2[:] # new anciliary variables
            else: # if we went forwards in time
                minus_var2, plus_var2, new_var2, anc_var2 = build_tree(q_plus, p_plus, lnpgrad_plus, logu, v, j-1, epsilon, fn, H) # build from rightmost entry
                q_plus, p_plus, lnpgrad_plus = plus_var2[:] # rightmost entry
                q_new2, p_new2, lnp_new2, lnpgrad_new2 = new_var2[:] # new proposal
                n_new2, s_new2, a_new2, na_new2 = anc_var2[:] # new anciliary variables

            # choose which subtree to propagate a sample up from.
            tprob = float(n_new2) / max(float(int(n_new) + int(n_new2)), 1.)) # transition probability
            if random.rand() < tprob: # if we accept, reassign values
                q_new = q_new2[:]
                p_new = p_new2[:]
                lnp_new = lnp_new2
                lnpgrad_new = lnpgrad_new2[:]
            n_new = int(n_new) + int(n_new2) # update number of valid points
            stop_flag = stop_criterion(q_minus, q_plus, p_minus, p_plus) # stopping criterion
            s_new = int(s_new and s_new2 and stop_flag) # update stopping criterion

            # update acceptance statistics
            a_new = a_new + a_new2
            na_new = na_new + na_new2

    return [q_minus, p_minus, lnpgrad_minus], [q_plus, p_plus, lnpgrad_plus], [q_new, p_new, lnp_new, lnpgrad_new], [n_new, s_new, a_new, na_new]


# NUTS step
def nuts_step(q, p, lnp, lnpgrad, Ndim, epsilon, n, fn, adaptive=False, adapt_params=None):

    H = lnp - 0.5 * np.dot(p, p.T) # compute Hamiltonian
    logu = float(H - random.exponential()) # sample slice variable u
    pos_prop, mom_prop, lnp_prop, lnpgrad_prop = q, p, lnp, lnp_grad # assign current values as proposed values
    
    # initialize tree
    q_minus, q_plus = q[:], q[:]
    p_minus, p_plus = p[:], p[:]
    lnpgrad_minus, lnpgrad_plus = lnpgrad[:], lnpgrad[:]
    j, n, s = 0, 1, 1  # initial height (j=0), valid points (n=1), stopping criterion (good)

    while s == 1: # if we haven't stopped
        v = int(2 * (random.rand() < 0.5) - 1) # pick a direction (-1/+1 = backwards/forwards)

        # double the tree.
        if v == -1:
            minus_var, plus_var, new_var, anc_var = build_tree(q_minus, p_minus, lnpgrad_minus, logu, v, j, epsilon, fn, H) # build from the left
            q_minus, p_minus, lnpgrad_minus = minus_var[:] # leftmost vals
            q_new, p_new, lnp_new, lnpgrad_new = new_var[:] # new position
            n_new, s_new, alpha, nalpha = anc_var[:] # new anciliary variables
        else:
            minus_var, plus_var, new_var, anc_var = build_tree(q_plus, p_plus, lnpgrad_plus, logu, v, j, epsilon, fn, H) # build from the right
            q_plus, p_plus, lnpgrad_plus = minus_var[:] # rightmost vals
            q_new, lnp_new, lnpgrad_new = new_var[:] # new position
            n_new, s_new, alpha, nalpha = anc_var[:] # new anciliary variables

        tprob = min(1, float(n_new) / float(n)) # transition probability
        if (s_new == 1) and (random.rand() < tprob): # Metropolis update
            pos_prop = q_new[:] # new position
            mom_prop = p_new[:] # new momentum
            lnp_prop = lnp_new # new lnP
            lnpgrad_prop = lnpgrad_new[:] # new gradient
        n += n_new # update number of valid points seen so far
        s = s_new and stop_criterion(q_minus, q_plus, p_minus, p_plus) # update stopping criterion
        j += 1 # increment depth

    # adapt epsilon if we're still burning in using dual-averaging algorithm
    if adaptive:
        if adapt_params is not None:
            gamma, t0, kappa, mu, epsilonbar, Hbar = adapt_params
        else:
            gamma, t0, kappa, mu = 0.05, 10., 0.75, np.log(10. * epsilon)
            epsilonbar, Hbar = 1., 0.
        eta = 1. / float(n + t0)
        Hbar = (1. - eta) * Hbar + eta * (delta - alpha / float(nalpha))
        epsilon = np.exp(mu - sqrt(n) / gamma * Hbar)
        eta = n ** -kappa
        epsilonbar = np.exp((1. - eta) * np.log(epsilonbar) + eta * np.log(epsilon))
        adapt_params = [gamma, t0, kappa, mu, epsilonbar, Hbar]

    return pos_prop, -mom_prop, lnp_prop, lnpgrad_prop, adapt_params

# NUTS sampler
def sample_nuts(fn, Niter, Nburnin, q_init, epsilon=None, targ_afrac=0.6, adapt_params=None):

    # initialize quantities
    Ndim = len(q_init) # number of parameters to sample
    pos_chain = np.empty((Nburnin + Niter, Ndim), dtype=float) # position samples
    lnp_chain = np.empty(Nburnin + Niter, dtype=float) # lnP
    lnp, lnpgrad = fn(q_init) # grab lnP and gradient
    pos_chain[0, :], lnprob[0] = q_init, lnp # initial values
    adapt_schedule = arange(Nburnin + Niter) < Nburnin # burn-in (adapt) flag

    # initialize epsilon
    if epsilon is None: 
        epsilon = find_reasonable_epsilon(q_init, lnp, lnpgrad, fn) 

    # run sampler
    for i in xrange(1, Nburnin + Niter):
        p = random.normal(size=Ndim) # sample momentum
        pos_chain[i,:], p_new, lnp_chain[i], lnpgrad, adapt_params = nuts_step(pos_chain[i,:], p, lnp_chain[i-1], lnpgrad, Ndim, epsilon, i, fn, 
                                                                                        adaptive=adapt_schedule[i], adapt_params=adapt_params)
 
    # split up samples
    pos_burn, pos_chain = pos_chain[:Nburnin], pos_chain[Nburnin:]
    lnp_burn, lnp_chain = lnp_chain[:Nburnin], lnp_chain[Nburnin:]
    return pos_chain, lnp_chain, pos_burn, lnp_burn, epsilon
