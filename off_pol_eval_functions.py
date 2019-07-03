import scipy.integrate as integrate
from math import exp
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.optimize import minimize
from sklearn.grid_search import GridSearchCV
from sklearn.neighbors.kde import KernelDensity
from sklearn.metrics.pairwise import rbf_kernel
import datetime
import pickle
import sys
# For bandwidth estimation
from scipy.stats import norm 
from sklearn import linear_model
# import numdifftools as nd
from scipy.misc import derivative
from sklearn.neighbors.kde import KernelDensity
from scipy.stats import truncnorm



# !FIXME Global offset value.

# !FIXME
# Currently when changing data generation distributions, need also to change sampling method in evaluate_subsample
# to generate from the appropriate treatment distribution.
'''
Choices for output function.
'''
def oracle_evaluation(**params):
    X = params['x_samp']; tau = params['tau']
    return 2*pow(np.abs(X - tau),1.5)

'''
Different options for kernel function.
'''
def db_exp_kernel(x1, x2, variance = 1):
    return exp(-1 * (np.linalg.norm(x1-x2)) / (2*variance))

def gram_matrix(xs):
    return rbf_kernel(xs, gamma=0.5)
def gaussian_kernel(u):
    return np.exp(-0.5 * u**2 )/(np.sqrt(2*np.pi))
def gaussian_kernel_h(u,h_2):
    return (1/(np.sqrt(h_2)*np.sqrt(2*np.pi)))*np.exp((-0.5)/h_2 * (1.0*u)**2 )
def gaussian_k_bar(u):
    return (1/(np.sqrt(4*np.pi)))*np.exp(.25* np.linalg.norm(1.0*u)**2)
def epanechnikov_kernel(u):
    return 0.75*(1-u**2)*(1 if abs(u) <= 1 else 0)
def epanechnikov_int(lo,hi):
    '''
    :return: Definite integral of the kernel from between lo and hi. Assumes that they are within bounds.
    '''
    return 0.75*(hi-hi**3/3.0) - 0.75*(lo-lo**3/3.0)

'''
Different option for discrete policy functions
Policy functions take in an x vector and return
'''
def discrete_optimal_central_policy(**params):
    '''
    :param params:
    :return: optimal treatment vector
    '''
    x = params['x_samp']
    T = params['T_samp']
    t_lo = min(T)
    t_hi = max(T)
    n_bins = params['n_bins']
    bins = np.linspace(t_lo, t_hi, n_bins)
    T_binned = np.digitize(T, bins).flatten()
    x_binned = np.digitize(x/2.0, bins).flatten()
    bin_means = [T[T_binned == i].mean() for i in range(1, n_bins)]
    # return np.asarray([bin_means[T_bin - 1] for T_bin in x_binned]).flatten()
    return x_binned

def discretize_tau_policy(**params):
    '''
    Discretize the treatment vector 'tau' according to uniform binning.
    '''
    x = params['x_samp']
    T = params['T_samp']
    n_bins = params['n_bins']
    t_lo = min(T)
    t_hi = max(T)
    bins = np.linspace(t_lo, t_hi, n_bins)
    T_binned = np.digitize(T, bins).flatten()
    bin_means = [T[T_binned == i].mean() for i in range(1, n_bins)]
    tau_binned = np.digitize(params['tau'], bins).flatten()
    return tau_binned

'''
Different options for generating data
'''
def generate_data_uniform(m,n, d, t_lo, t_hi, x_scheme = 'unif'): 
    """
    # Generate random features
    # n: number of instances 
    # m: grid length of treatment
    # d: feature dimension
    # x_scheme: switch to determine dependency structure of x 
    """
    xs = np.array(np.random.uniform(0,2,(n,d)))
    t_fullgrid = np.linspace(t_lo, t_hi, m )
    Z_list = [ np.concatenate([xs, np.ones([n,1])*(t_lo + 1.0*i*(t_hi-t_lo)/(m-1))] , axis=1) for i in np.arange(m) ]
    Z = np.concatenate(Z_list, axis=0)
    K = np.array(gram_matrix(Z)).reshape([m*n,m*n])
    T = Z[:,d]
    # mean_vec = np.asarray([ np.mean(z) for z in Z])
    mean_vec = np.ones([m*n,1])
    F = np.random.multivariate_normal(mean_vec.flatten(), 7*K)
# Ensure outcomes are positive
    if min(F) < 0:
        F = F + abs(min(F))  
    Y = F + 0.05*np.random.randn(m*n)

    return { 'y': Y, 'z': Z, 'f': F , 'K': K, 'x': xs}


def generate_data(m,n, d, t_lo, t_hi, mean_vec_f, x_scheme = 'unif'): 
    """
    # Generate random features
    # n: number of instances 
    # m: grid length of treatment
    # d: feature dimension
    # x_scheme: switch to determine dependency structure of x 
    """
    xs = np.array(np.random.uniform(0,1,(n,d)))
    t = np.array(np.random.uniform(0, t_hi, size=(n,1)))
    # change mean vector appropriately
    t_fullgrid = np.linspace(t_lo, t_hi, m )
    Z_list = [ np.concatenate((xs, np.ones([n,1])*(t_lo + 1.0*i*(t_hi-t_lo)/(m-1))) , axis=1) for i in np.arange(m) ]
    Z = np.concatenate(Z_list, axis=0)
    K = np.array(gram_matrix(Z)).reshape([m*n,m*n])
    T = Z[:,d]
    # modify to have T have more of an effect
    mean_vec = np.apply_along_axis(mean_vec_f, 1, Z)
    # mean_vec = 3*np.multiply(T,Z[:,0]) + 2*T + np.multiply(Z[:,0], np.exp(np.multiply(-Z[:,0],T)))
    F = np.random.multivariate_normal(mean_vec, 2*K)
# Ensure outcomes are positive
    if min(F) < 0:
        F = F + abs(min(F))  
    Y = F + 0.05*np.random.randn(m*n)

    return { 'y': Y, 'z': Z, 'f': F , 'K': K, 'x': xs}

def off_pol_estimator(**params): 

    THRESH = params['threshold']
    y_out = params['y']; x = params['x']; h = params['h'];Q = params['Q']; n = params['n']; t_lo = params['t_lo'];  t_hi = params['t_hi']
    kernel = params['kernel_func'];kernel_int =  params['kernel_int_func']
    if ('y_samp' in params.keys()):
        y_out = params['y_samp']
    if ('T_samp' in params.keys()): 
        T = params['T_samp']
    else: 
        T = params['T']
    if ('x_samp' in params.keys()):
        x = params['x_samp']

    BMI_IND = params.get('BMI_IND') # propensity score for warfarin data evaluations 
    if (params.get('DATA_TYPE') == 'warfarin'): 
        x = params['x'][:,BMI_IND]

    loss = 0
    tau = params['tau']
    clip_tau = np.clip(tau, t_lo, t_hi)
    Qs = np.zeros(n)
    for i in np.arange(n): 
        Q_i = Q(x[i], T[i], t_lo, t_hi)
        if (abs(clip_tau[i] - t_lo) <= h):
            alpha = kernel_int((t_lo-clip_tau[i])/h, 1)
        elif (abs(clip_tau[i] - t_hi) <= h):
            alpha = kernel_int(-1,  (t_hi - clip_tau[i])/h )
        else:
            alpha = 1
        Qs[i] = (1.0/h)*kernel( (clip_tau[i] - T[i])/h )/max(Q_i,THRESH)
        loss += kernel( (clip_tau[i] - T[i])/h )*1.0 * y_out[i]/max(Q_i,THRESH) * 1.0/alpha
    norm_sum = np.mean(np.maximum(Qs,THRESH*np.ones(n)))
    return [loss, norm_sum]

def off_policy_variance(**params):
    """
    Takes in a choice of kernel and dictionary of parameters and data required for evaluation
    tau is a vector of treatment values (assumed given)
    If y_samp, T_samp is present, use that instead. 
    """
    [loss, norm_sum] = off_pol_estimator(**params)
    h = params['h']; n = params['n']
    loss = loss / (norm_sum*1.0*n*h)
    loss_mean = np.mean(loss)
    return np.square(loss - loss_mean)

def off_policy_evaluation(**params):
    """
    Takes in a choice of kernel and dictionary of parameters and data required for evaluation
    tau is a vector of treatment values (assumed given)
    If y_samp, T_samp is present, use that instead. 
    """
    [loss, norm_sum] = off_pol_estimator(**params)
    h = params['h']
    n = params['n']
    return loss/(norm_sum*1.0*h*n)

def off_pol_disc_evaluation(policy, **params):
    THRESH = params['threshold']
    y_out = params['y']; x = params['x_samp']; h = params['h']; Q = params['Q']; n = params['n']; t_lo = params['t_lo']; t_hi = params['t_hi']
    n_bins = params['n_bins']
    if ('y_samp' in params.keys()):
        y_out = params['y_samp'].flatten()
    if ('T_samp' in params.keys()):
        T = params['T_samp'].flatten()
    else:
        T = params['T'].flatten()

    BMI_IND = params.get('BMI_IND') # propensity score for warfarin data evaluations 
    if (params.get('DATA_TYPE') == 'warfarin'): 
        x = params['x'][:,BMI_IND]

    t_lo = min(T)
    t_hi = max(T)
    bin_width = t_hi-t_lo
    bins = np.linspace(t_lo, t_hi, n_bins)
    T_binned = np.digitize(T, bins, right = True).flatten()
    bin_means = [T[T_binned == i].mean() for i in range(1, len(bins))]

    loss = 0
    tau_vec = policy(**params).flatten()
    #! FIXME need to establish whether policy returns discrete bins or means
    treatment_overlap = np.where(np.equal(tau_vec.flatten(), T_binned))[0]

    for ind in treatment_overlap:
        Q_i = Q(x[ind], bin_means[T_binned[ind]-1], t_lo, t_hi) * bin_width*1.0/n_bins # BUG FIX: this is going to have to be integrated against 
        loss += y_out[ind]/max(Q_i,THRESH)
    n_overlap = len(treatment_overlap)
    if n_overlap == 0:
        print "no overlap"
        return 0
    return loss/(1.0*n)

# Self normalize disc. off pol evaluation 
# doesn't work well 
# def off_pol_disc_evaluation(policy, **params):
#     THRESH = params['threshold']
#     y_out = params['y']; x = params['x_samp']; h = params['h']; Q = params['Q']; n = params['n']; t_lo = params['t_lo']; t_hi = params['t_hi']
#     n_bins = params['n_bins']
#     if ('y_samp' in params.keys()):
#         y_out = params['y_samp'].flatten()
#     if ('T_samp' in params.keys()):
#         T = params['T_samp'].flatten()
#     else:
#         T = params['T'].flatten()

#     BMI_IND = params.get('BMI_IND') # propensity score for warfarin data evaluations 
#     if (params.get('DATA_TYPE') == 'warfarin'): 
#         x = params['x'][:,BMI_IND]

#     t_lo = min(T)
#     t_hi = max(T)
#     bin_width = t_hi-t_lo
#     bins = np.linspace(t_lo, t_hi, n_bins)
#     T_binned = np.digitize(T, bins, right = True).flatten()
#     bin_means = [T[T_binned == i].mean() for i in range(1, len(bins))]

#     loss = 0
#     tau_vec = policy(**params).flatten()
#     #! FIXME need to establish whether policy returns discrete bins or means
#     treatment_overlap = np.where(np.equal(tau_vec.flatten(), T_binned))[0]
#     n_overlap = len(treatment_overlap)
#     Qs = np.zeros(n_overlap)
#     i=0
#     for ind in treatment_overlap:
#         Q_i = Q(x[ind], bin_means[T_binned[ind]-1], t_lo, t_hi) * bin_width*1.0/n_bins # BUG FIX: this is going to have to be integrated against 
#         Qs[i] = 1.0/max(Q_i,THRESH)
#         loss += y_out[ind]/max(Q_i,THRESH)
#         i+=1 
    
#     norm_sum = np.mean(Qs)
#     if n_overlap == 0:
#         print "no overlap"
#         return 0
#     return loss/(1.0*n*norm_sum)



def off_pol_gaus_lin_grad(beta, *args):
    """
    Compute a gradient for special case of gaussian kernel and linear policy tau
    """
    params = dict(args[0])
    y_out = params['y'];x = params['x'];  T = params['T']; h = params['h']; Q = params['Q']
    n = params['n']; t_lo = params['t_lo']; t_hi = params['t_hi']
    tau = np.dot(x,beta)
    clip_tau = np.clip(tau, t_lo, t_hi)
    d = len(beta)
    grad = np.zeros([d,1])
    for i in np.arange(n):
        Q_i = Q(x[i], T[i],t_lo, t_hi)
        beta_x_i = np.dot(x[i], beta)
        grad += (gaussian_kernel((beta_x_i - T[i])/h) * y_out[i]/Q_i) * (-1.0*x[i]/h**2) * (beta_x_i - T[i])
    return grad/(1.0*h*len(y_out))


def partial_g_n_hat_i(**params): 
    '''
    Compute normalization term 
    '''

def f_g(**params): 
    THRESH = params['threshold']
    y_out = params['y']; x = params['x']; h = params['h'];Q = params['Q']; n = params['n']; t_lo = params['t_lo'];  t_hi = params['t_hi']
    kernel = params['kernel_func'];kernel_int =  params['kernel_int_func']
    if ('y_samp' in params.keys()):
        y_out = params['y_samp']
    if ('T_samp' in params.keys()): 
        T = params['T_samp']
    else: 
        T = params['T']
    if ('x_samp' in params.keys()):
        x = params['x_samp']
    BMI_IND = params.get('BMI_IND') # propensity score for warfarin data evaluations 
        
    loss = 0
    g = 0 # also keep track of normalized probability ratio quantity 
    partial_f = 0 
    partial_g = 0 
    tau = params['tau']
    clip_tau = np.clip(tau, t_lo, t_hi)
    Qs = np.zeros(n)
    for i in np.arange(n): 
        if (params.get('DATA_TYPE') == 'warfarin'): 
            Q_i = Q(x[i,BMI_IND], T[i], t_lo, t_hi)
        else: 
            Q_i = Q(x[i], T[i], t_lo, t_hi)
        if (abs(clip_tau[i] - t_lo) <= h):
            alpha = kernel_int((t_lo-clip_tau[i])/h, 1)
        elif (abs(clip_tau[i] - t_hi) <= h):
            alpha = kernel_int(-1,  (t_hi - clip_tau[i])/h )
        else:
            alpha = 1
        Qs[i] = kernel( (clip_tau[i] - T[i])/h )/max(Q_i,THRESH)
        loss += kernel( (clip_tau[i] - T[i])/h )*1.0 * y_out[i]/max(Q_i,THRESH) * 1.0/alpha
        if abs((clip_tau[i] - T[i])/h) >= 1:
            partial_f += 0 # don't add anything to partial derivatives 
        else:
            partial_g += -1.5 * ((clip_tau[i] - T[i])/h ) * 1.0/max(Q_i,THRESH) * x[i,:]
            partial_f += -1.5 * ((clip_tau[i] - T[i])/h ) * y_out[i]/max(Q_i,THRESH) * x[i,:]
    norm_sum = np.mean(Qs)
    return [loss/(1.0*h*n), 1.0*norm_sum/h, partial_f/(1.0*n*h**2) , partial_g/(1.0*n*h**2) ]


def off_pol_epan_lin_grad(beta, *args):
    """
    Compute a gradient for special case of Epanechnikov kernel and linear policy tau
    """
    # THRESH = 0.001
    d = len(beta) 
    params = dict(args[0])
    #! FIXME x vs xsamp
    tau = np.dot(beta, params['x'].T)
    params['tau'] = tau
    params['beta'] = beta

    THRESH = params['threshold']

    [f, g, nabla_f, nabla_g] = f_g(**params)
    # compute gradient vector via quotient rule
    if g < THRESH: 
        g = THRESH  
    return np.asarray((g*nabla_f - f*nabla_g) / g**2 )

def off_pol_var_lin_grad(beta, *args):
    """
    Compute a gradient for special case of Epanechnikov kernel and linear policy tau
    """
    # THRESH = 0.001
    d = len(beta) 
    params = dict(args[0])
    #! FIXME x vs xsamp
    tau = np.dot(beta, params['x'].T)
    params['tau'] = tau
    params['beta'] = beta

    THRESH = params['threshold']

    [f, g, nabla_f, nabla_g] = f_g(**params)
    # compute gradient vector via quotient rule
    if g < THRESH: 
        g = THRESH  
    return np.asarray((g*nabla_f - f*nabla_g) / g**2 )

def off_pol_gaus_lin_grad_for_max(beta, *args):
    """Wrapper function which multiplies gradient by -1
    """
    return off_pol_gaus_lin_grad(beta, *args)

"""
Options for treatment policies
"""
def tau_test(tau_test_value, x): 
    return tau_test_value 
def linear_tau(x, beta): 
    return np.dot(beta,x)
def unif_Q(x, t, t_lo, t_hi):
    return 1.0/(t_hi-t_lo)
def trunc_norm_Q(x, t, t_lo, t_hi):
    # Get pdf from  truncated normally distributed propensity score (standard normal centered around (x-t)
    sc = 0.5
    mu = x
    a, b = (t_lo - mu) / sc, (t_hi - mu) / sc
    return truncnorm.pdf(t, a,b, loc = mu, scale = sc)
def norm_Q(x, t, t_lo, t_hi):
    OFFSET = 0.1
    std = 0.5
    return 1.0/std *norm.pdf( (t-x - OFFSET)/ std)

def exp_Q(x, t, t_lo, t_hi):
    # Sample from an exponential conditional distribution of T on X using Inverse CDF transform
    return x*np.exp(-t*x)
def sample_exp_T(x): 
    u = np.random.uniform()
    return -np.log(1-u)/x

def sample_norm_T(x):
   # ' Sample randomly from uniform normal distribution'
    sc = 0.5
    OFFSET = 0.1
    return np.random.normal(loc=x + OFFSET, scale = sc)

def evaluate_oracle_outcomes(m,n,f,t_lo,t_hi,tau,X):
    """ 
    Evaluate 'true' outcomes at closest grid point to given tau vector 
    """
    j_taus = np.array( [int(np.round(1.0*t*(m-1)/t_hi)) for t in tau] )
    j_taus = np.clip(j_taus, 0, m-1)
    return np.array( [ f[j_taus[ind]*n+ind] for ind in np.arange(n)]  )

def evaluate_oracle_interpolated_outcomes(**params):
    """ 
    Function is given a spline curve with which to interpolate values at 'tau'
    """
    spline_tck = params['spline']; tau = params['tau']; X = params['x_samp']
    outcomes = [ interpolate.bisplev( X[i], tau[i], spline_tck ) for i in np.arange(len(X)) ] 
    return np.array(outcomes)

def sample_T_given_x(x, t_lo, t_hi, sampling = "uniform"):
        # Sample from propensity score
    # e.g. exponential distribution
    sc = 0.5
    if (sampling == "exp"): 
        sample_exp_T_vec = np.vectorize(sample_exp_T)
        T_sub = sample_exp_T_vec(x / std)
        T_sub = np.clip(T_sub, t_lo, t_hi)
    elif (sampling == "normal"):
        # Unbounded normal sampling
        sample_norm_T_vec = np.vectorize(sample_norm_T)
        T_sub = sample_norm_T_vec(x )
    elif (sampling == "truncated_normal"):
        # Unbounded normal sampling
        # sample_norm_T_vec = np.vectorize(sample_norm_T)
        # T_sub = sample_norm_T_vec(x )
        T_sub = np.zeros([len(x), 1])
        for i in np.arange(len(x)):
            a =(t_lo - x[i]) / sc
            b = (t_hi - x[i]) / sc
            T_sub[i] = truncnorm.rvs(a, b, loc = x[i], scale = sc, size=1)[0]
    else: 
        T_sub = np.array( [ np.random.uniform(low = t_lo, high= t_hi) for x_samp in x ]  )
    return T_sub

def evaluate_subsample( n_sub, verbose = False, evaluation=False, cross_val = True, **param_dict):
    """
    Evaluate off policy evaluation given a subsample of data from full 
    Or just subsample data and return subsampled_dictionary
    """
    Z = param_dict['z']; X = param_dict['x']; t_lo = param_dict['t_lo']; t_hi = param_dict['t_hi']; m = param_dict['m']
    n = param_dict['n']; Y = param_dict['y']; d = param_dict['d']; f = param_dict['f']; data_gen = param_dict['data_gen']
    sampling = param_dict['sampling']; sub_params = param_dict.copy()
    # Subsample data
    if (data_gen == "grid"):
        X_sub = np.random.choice(n-1, n_sub)
        T_sub = sample_T_given_x(X[X_sub], t_lo, t_hi, sampling)
        # Round T to grid values
        j_s = np.array( [int(np.round(1.0*t*(m-1)/t_hi)) for t in T_sub] ).flatten()
        T_grid = np.array([  t_lo + 1.0*np.round(1.0*t*(m-1)/t_hi)*(t_hi-t_lo)/(m-1) for t in T_sub  ])
        Y_sub = np.array( [ Y[j_s[ind]*n+x]  for (ind,x) in enumerate(X_sub)] )
        sub_params['n'] = n_sub
        sub_params['y_samp'] = Y_sub.flatten()
        #! FIXME flattening possibly multidimensional data
        sub_params['x_samp'] = X[X_sub,:]
        sub_params['T_samp'] = T_grid.flatten()

    else: 
        # Uniform sampling
        X_sub = np.random.choice(m*n-1, n_sub)
        sub_params['n'] = n_sub
        sub_params['x_samp'] = X[X_sub,:].reshape([n_sub,1])
        # Toggle how sampling is drawn
        if sampling != "uniform":
            sub_params['T_samp'] = sample_T_given_x( X[X_sub,:], t_lo, t_hi, sampling ).reshape([n_sub,1])
        else: # assume uniform otherwise
            sub_params['T_samp'] = Z[:,d][X_sub].reshape([n_sub,1])
        # Toggle how oracle values are drawn
        if (sub_params['oracle_func']):
            # temporary setting of tau to
            sub_params['tau'] = sub_params['T_samp']
            # adding noise to 'y' values
            sub_params['y_samp'] = oracle_evaluation(**sub_params) #+ np.random.randn(n_sub,1)*0.05
            sub_params['f_samp'] = oracle_evaluation(**sub_params)
            del sub_params['tau']
        else: #Oracle fnc parameter not set
            sub_params['y_samp'] = Y[X_sub].reshape([n_sub,1])
            sub_params['f_samp'] = f[X_sub].reshape([n_sub,1])

    if 'tau' in param_dict.keys():
        sub_params['tau'] = param_dict['tau'][X_sub]
    else:
        if verbose:
                print "No taus given"
    if cross_val:
        h_opt = find_best_h(cv_func, res, **sub_params)
        sub_params['h'] = h_opt

    return sub_params


def plot_surface(plot_sample = False, **params): 
    fig = plt.figure(figsize=plt.figaspect(.2))
    ax = fig.add_subplot(1,3,1, projection='3d')

    if not plot_sample: 
        x = params['z'][:,0]
        t = params['z'][:,1]
        y = params['y']
    else: 
        x = params['x_samp']
        t = params['T_samp']
        y = params['y_samp']

    ax.scatter(x, t, y, s = 0.06)
    ax.set_xlabel('x Label')
    ax.set_ylabel('t Label')
    ax.set_zlabel('y Label')
    ax = fig.add_subplot(1, 3, 2, projection='3d')
    ax.scatter(x, t, y, s = 0.06)
    # Add best beta vector 
    # ax1.scatter(x[40:],y[40:], s=10, c='r', marker="o", label='second') 
    ax.azim = 240
    ax.elev = 20
    ax.set_xlabel('x ')
    ax.set_ylabel('t ')
    ax.set_zlabel('y ')
    plt.show()

def lin_off_policy_loss_evaluation(beta, *args): 
    arg_dict = dict(args[0])
    t_lo = arg_dict['t_lo']
    t_hi = arg_dict['t_hi']
    x = arg_dict['x_samp']
    arg_dict['tau'] = np.clip(np.dot(x,beta), t_lo, t_hi)
    return off_policy_evaluation(**arg_dict)

def constant_off_policy_loss_evaluation(const, *args): 
    arg_dict = dict(args[0])
    x = arg_dict['x_samp']
    arg_dict['tau'] = const * np.ones(arg_dict['n'])
    return off_policy_evaluation(**arg_dict)

def eval_interpolated_oracle_tau(beta, *args):
    params = dict(args[0])
    t_lo = params['t_lo']
    t_hi = params['t_hi']
    spline_tck = params['spline']
    tau_candidate = np.clip(np.dot(beta, params['x_samp'].T), t_lo, t_hi)
    params['tau'] = tau_candidate
    return np.mean(evaluate_oracle_interpolated_outcomes(**params))


def eval_const_interpolated_oracle_tau(const, *args):
    params = dict(args[0])
    t_lo = params['t_lo']
    t_hi = params['t_hi']
    spline_tck = params['spline']
    tau_candidate = const * np.ones(params['n'])
    params['tau'] = tau_candidate
    return np.mean(evaluate_oracle_interpolated_outcomes(**params))

def eval_oracle_tau(beta, *args):
    params = dict(args[0])
    t_lo = params['t_lo']
    t_hi = params['t_hi']
    tau_candidate = np.clip(np.dot(beta, params['x'].T), t_lo, t_hi)
    #!FIXME graceful handling of loss function of y_i
    params['tau'] = tau_candidate
    return np.mean(evaluate_oracle_interpolated_outcomes(**params))
def eval_oracle_tau_evaluation(beta, *args):
    params = dict(args[0])
    t_lo = params['t_lo']
    t_hi = params['t_hi']
    tau_candidate = np.clip(np.dot(beta, params['x'].T), t_lo, t_hi)
    #!FIXME graceful handling of loss function of y_i
    params['tau'] = tau_candidate
    return np.mean(oracle_evaluation(**params))


def pol_opt(verbose = True, samp_func = lin_off_policy_loss_evaluation, oracle_eval = eval_interpolated_oracle_tau, **samp_params):
    """
    Run a policy optimization test, comparing performance of empirical minimizer against the true counterfactual outcomes. 
    """
    d = samp_params['d']
    n = samp_params['n']
    t_lo = samp_params['t_lo']
    t_hi = samp_params['t_hi']
    beta_d = [np.random.uniform() for i in np.arange(d)]
    if samp_params['kernel_func'] == gaussian_kernel:
        res = minimize(samp_func, x0 = beta_d, jac = off_pol_gaus_lin_grad_for_max, bounds = ((0, t_hi/max(samp_params['x']) ),) , args=samp_params.items() )
    else:
        res = minimize(samp_func, x0 = beta_d, jac = off_pol_epan_lin_grad, bounds = ((t_lo/max(samp_params['x_samp']), t_hi/max(samp_params['x_samp']) ),) , args=samp_params.items() )
    emp_best_tau = np.clip(np.dot(res.x, samp_params['x'].T), t_lo, t_hi)
    if verbose:
        print "Optimization results"
        print res
        print "Policy treatments:"
        print emp_best_tau
        print "Observed treatments: "
        print samp_params['T_samp']
    # print "Deviation in treatment vector: "
    # print np.linalg.norm(emp_best_tau - samp_params['T_samp'])
    print 'x: ' + str( res.x ) 
    print 'off pol evaluation value '
    print res.fun
    """
    Optimize a treatment policy over oracle outcomes f 
    """
    # spl_x = samp_params['z'][:,0]
    # spl_t = samp_params['z'][:,1]
    # # f is positive
    # splined_f_tck = interpolate.bisplrep(spl_x,spl_t, samp_params['f'])
    # samp_params['spline'] = splined_f_tck
    samp_params['tau'] = emp_best_tau
    oracle_outcomes = samp_params['oracle_func'](**samp_params)
    ## Evaluate the 'true' performance of this treatment vector
    print 'oracle mean of empirically best feature vector  \n'
    print np.mean(oracle_outcomes)

    # print 'Computing oracle best-in-class linear policy via interpolation of true response surface: \n'
    beta_d = [np.random.uniform() for i in np.arange(d)]
    # print "initial condition: " + str(beta_d)
    # print 'val of initial condition: '
    # print oracle_func(beta_d, samp_params.items())

    oracle_res = minimize(oracle_eval, x0 = beta_d, bounds = ((0, 1.0/np.mean(samp_params['x']) ),) , args=samp_params.items() )
    if verbose:
        print oracle_res
        print 'beta'
        print oracle_res.x
        print 'oracle best linear treatment policy value '
        print  oracle_res.fun

    return [res, oracle_res, splined_f_tck]

def off_pol_opt_test(n_max, n_trials, n_spacing, n_0, t_lo_sub,t_hi_sub, **sub_params):
    n = sub_params['n']; m = sub_params['m']; t_lo = t_lo_sub; t_hi = t_hi_sub
    d = sub_params['d']
    n_space = np.linspace(n_0, n_max, n_spacing)
    best_beta = np.zeros([len(n_space),n_trials])
    best_oracle_beta = np.zeros([len(n_space),n_trials])
    OOS_OPE = np.zeros([len(n_space),n_trials])
    OOS_oracle = np.zeros([len(n_space),n_trials])
    # discrete_off_pol_evals = np.zeros([n_treatments, n_spacing, n_trials])
    oracle_func = sub_params['oracle_func']
    h_orig = sub_params['h']
    TEST_N = 250
    TEST_SET = evaluate_subsample( 250, evaluation = False, cross_val = False, **sub_params )

    for i, n_sub in enumerate(np.linspace(n_0, n_max, n_spacing)): 
        # sub_params['h'] = h_orig * (np.power(n_sub,0.2))/np.power(n_0,0.2)
        n_rnd = int(np.floor(n_sub))
        print "testing with n = " + str(n_rnd)
        for k in np.arange(n_trials):
            subsamples_pm = evaluate_subsample( n_rnd, evaluation = False, cross_val = False, **sub_params )
            # oracle_evals[t_ind, i, k] = np.mean(evaluate_oracle_interpolated_outcomes(splined_f_tck, m,n_rnd, subsamples_pm['f'], t_lo, t_hi, subsamples_pm['tau'], subsamples_pm['x_samp']))
            ### Compute best betas with random restarts 
            oracle_betas = np.zeros([n_restarts, d]);eval_vals = np.zeros([n_restarts, d]);emp_betas = np.zeros([n_restarts, d]);emp_eval_vals = np.zeros([n_restarts, d])
            for i_restart in np.arange(n_restarts):
                beta_d = [np.random.uniform() for i in np.arange(d)]
                res = minimize(lin_off_policy_loss_evaluation, x0 = beta_d, jac = off_pol_epan_lin_grad, bounds = ((t_lo/max(samp_params['x_samp']), t_hi/max(samp_params['x_samp']) ),) , args=samp_params.items() )
                emp_betas[i_restart] = res.x; emp_eval_vals[i_restart] = res.fun

                oracle_res = minimize(oracle_func, x0 = beta_d, bounds = ((0, 1.0/np.mean(samp_params['x']) ),) , args=samp_params.items() )
                oracle_betas[i_restart] = oracle_res.x; eval_vals[i_restart] = oracle_res.fun  

            emp_best_tau = np.clip(np.dot(res.x, samp_params['x_samp'].T), t_lo, t_hi)
            # get best beta value from random restarts
            best_ind = np.argmin(emp_eval_vals)
            best_beta[i,k] =  emp_betas[best_ind,:]
            
            best_oracle_ind = np.argmin(eval_vals)
            best_oracle_beta[i,k] =  oracle_betas[oracle_betas,:]
            TEST_SET['tau'] = best_beta[i,k] * TEST_SET['x_samp']
            OOS_OPE[i,k] = off_policy_evaluation(**TEST_SET)
            OOS_oracle[i,k] = np.mean(oracle_func(**TEST_SET))

    return [best_beta, best_oracle_beta, OOS_OPE, OOS_oracle]


def off_pol_eval_cons_test(n_max, n_trials, n_treatments, n_spacing, n_0,t_lo_sub,t_hi_sub, **sub_params):
    n = sub_params['n']; m = sub_params['m']; t_lo = t_lo_sub; t_hi = t_hi_sub
    treatment_space = np.linspace(t_lo, t_hi, n_treatments)
    off_pol_evals = np.zeros([n_treatments, n_spacing, n_trials])
    oracle_evals = np.zeros([n_treatments, n_spacing, n_trials])
    discrete_off_pol_evals = np.zeros([n_treatments, n_spacing, n_trials])
    oracle_func = sub_params['oracle_func']
    splined_f_tck = sub_params['spline']
    h_orig = sub_params['h']
    for i, n_sub in enumerate(np.linspace(n_0, n_max, n_spacing)): 
        # sub_params['h'] = h_orig * (np.power(n_sub,0.2))/np.power(n_0,0.2)
        n_rnd = int(np.floor(n_sub))
        print "testing with n = " + str(n_rnd)
        for k in np.arange(n_trials):
            for t_ind, t in enumerate(treatment_space):
                subsamples_pm = evaluate_subsample( n_rnd, evaluation = False, cross_val = False, **sub_params )
                subsamples_pm['tau'] = t * np.ones(n_sub)
                oracle_evals[t_ind, i, k] = np.mean(oracle_func(**subsamples_pm))
                # oracle_evals[t_ind, i, k] = np.mean(evaluate_oracle_interpolated_outcomes(splined_f_tck, m,n_rnd, subsamples_pm['f'], t_lo, t_hi, subsamples_pm['tau'], subsamples_pm['x_samp']))
                off_pol_evals[t_ind, i, k] = off_policy_evaluation(**subsamples_pm)
                discrete_off_pol_evals[t_ind, i, k] = off_pol_disc_evaluation(discretize_tau_policy , **subsamples_pm)

    off_pol_evals.dump( str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")) + 'off_pol_vals.np')
    oracle_evals.dump(str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")) + 'off_pol_oracles.np')
    return [oracle_evals, off_pol_evals, discrete_off_pol_evals]


def off_pol_eval_linear_test( n_max, beta_0, beta_hi, n_trials, n_treatments, n_spacing, n_0, **sub_params):
    '''
    Systematically evaluate over a treatment space defined by a linear treatment policy
    '''
    treatment_space = np.linspace(beta_0, beta_hi, n_treatments)
    off_pol_evals = np.zeros([n_treatments, n_spacing, n_trials])
    oracle_evals = np.zeros([n_treatments, n_spacing, n_trials])
    discrete_off_pol_evals = np.zeros([n_treatments, n_spacing, n_trials])
    t_lo = sub_params['t_lo']; t_hi = sub_params['t_hi']; spl_x = sub_params['z'][:,0]; spl_t = sub_params['z'][:,1]
    # f is positive
    splined_f_tck = interpolate.bisplrep(spl_x,spl_t, sub_params['f'])
    sub_params['spline'] = splined_f_tck
    oracle_func = sub_params['oracle_func']
    n = sub_params['n']; m = sub_params['m']

    for i, n_sub in enumerate(np.linspace(n_0, n_max, n_spacing)): 
        n_rnd = int(np.floor(n_sub))
        print "testing n = " + str(n_rnd)
        for k in np.arange(n_trials):
            for beta_ind, beta in enumerate(treatment_space):
                subsamples_pm = evaluate_subsample( n_rnd, evaluation = False, cross_val = False, **sub_params )
                tau = np.clip(np.dot( subsamples_pm['x_samp'], beta ) , t_lo, t_hi)
                subsamples_pm['tau'] = tau
                oracle_evals[beta_ind, i, k] = np.mean(oracle_func(**subsamples_pm))
                # oracle_evals[beta_ind, i, k] = np.mean(evaluate_oracle_interpolated_outcomes(splined_f_tck,m,n_rnd, subsamples_pm['f'], beta_0, beta_hi, tau, subsamples_pm['x_samp']))
                # off_pol_evals[beta_ind, i, k] = off_policy_evaluation(**subsamples_pm)
                off_pol_evals[beta_ind, i, k] = off_policy_evaluation(**subsamples_pm)
                discrete_off_pol_evals[beta_ind, i, k] = off_pol_disc_evaluation(discretize_tau_policy , **subsamples_pm)

    off_pol_evals.dump( str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")) + 'off_pol_linear_vals.np')
    oracle_evals.dump(str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")) + 'off_pol_linear_oracles.np')
    return [oracle_evals, off_pol_evals, discrete_off_pol_evals]
'''
    Systematically evaluate over a treatment space defined by a linear treatment policy

With DM 
'''
def off_pol_eval_linear_test( n_max, beta_0, beta_hi, n_trials, n_treatments, n_spacing, n_0, **sub_params):
    '''
    '''
    treatment_space = np.linspace(beta_0, beta_hi, n_treatments)
    off_pol_evals = np.zeros([n_treatments, n_spacing, n_trials])
    oracle_evals = np.zeros([n_treatments, n_spacing, n_trials])
    discrete_off_pol_evals = np.zeros([n_treatments, n_spacing, n_trials])
    t_lo = sub_params['t_lo']; t_hi = sub_params['t_hi']; spl_x = sub_params['z'][:,0]; spl_t = sub_params['z'][:,1]
    # f is positive
    splined_f_tck = interpolate.bisplrep(spl_x,spl_t, sub_params['f'])
    sub_params['spline'] = splined_f_tck
    oracle_func = sub_params['oracle_func']
    n = sub_params['n']; m = sub_params['m']

    for i, n_sub in enumerate(np.linspace(n_0, n_max, n_spacing)): 
        n_rnd = int(np.floor(n_sub))
        print "testing n = " + str(n_rnd)
        for k in np.arange(n_trials):
            for beta_ind, beta in enumerate(treatment_space):
                subsamples_pm = evaluate_subsample( n_rnd, evaluation = False, cross_val = False, **sub_params )
                tau = np.clip(np.dot( subsamples_pm['x_samp'], beta ) , t_lo, t_hi)
                subsamples_pm['tau'] = tau
                oracle_evals[beta_ind, i, k] = np.mean(oracle_func(**subsamples_pm))
                # oracle_evals[beta_ind, i, k] = np.mean(evaluate_oracle_interpolated_outcomes(splined_f_tck,m,n_rnd, subsamples_pm['f'], beta_0, beta_hi, tau, subsamples_pm['x_samp']))
                # off_pol_evals[beta_ind, i, k] = off_policy_evaluation(**subsamples_pm)
                off_pol_evals[beta_ind, i, k] = off_policy_evaluation(**subsamples_pm)
                discrete_off_pol_evals[beta_ind, i, k] = off_pol_disc_evaluation(discretize_tau_policy , **subsamples_pm)

    off_pol_evals.dump( str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")) + 'off_pol_linear_vals.np')
    oracle_evals.dump(str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")) + 'off_pol_linear_oracles.np')
    return [oracle_evals, off_pol_evals, discrete_off_pol_evals]

def plot_off_pol_evals(off_pol_evals, oracle_evals, off_pol_disc_evals, n_0, n, n_trials, n_treatments, n_spacing, t_lo, t_hi, x_label, title_stem, truncate_y = False):
    mean_off_pol_vals = np.mean(off_pol_evals, axis = 2)
    mean_oracle_vals = np.mean(oracle_evals,axis=2)
    sds_off_pol = np.std(off_pol_evals, axis = 2)
    sds_oracle = np.std(oracle_evals, axis = 2)
    mean_off_pol_disc_evals = np.mean(off_pol_disc_evals,axis=2)
    sds_off_pol_disc = np.std(off_pol_disc_evals, axis = 2)

    ts = np.linspace(t_lo, t_hi, n_treatments)

    ns = np.linspace(n_0, n, n_spacing)
    for i in np.arange(n_spacing):
        plt.figure(i+1)
        error_1 = 1.96*sds_off_pol[:,i]/np.sqrt(n_trials)
        error_2 = 1.96*sds_oracle[:,i]/np.sqrt(n_trials)
        error_3 = 1.96*sds_off_pol_disc[:,i]/np.sqrt(n_trials)

        plt.plot(ts, mean_oracle_vals[:,i], c = "blue")
        plt.fill_between(ts, mean_oracle_vals[:,i]-error_2, mean_oracle_vals[:,i]+error_2, alpha=0.5, edgecolor='blue', facecolor='blue')

        plt.scatter(ts, mean_off_pol_disc_evals[:,i], c = "green")
        plt.fill_between(ts, mean_off_pol_disc_evals[:,i]-error_3, mean_off_pol_disc_evals[:,i]+error_3, alpha=0.4, edgecolor='g', facecolor='g')
        plt.scatter(ts, mean_off_pol_vals[:,i], c = "red")
        plt.fill_between(ts, mean_off_pol_vals[:,i]-error_1, mean_off_pol_vals[:,i]+error_1, alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')

    #     plt.ylim( (0, 10) )
        plt.title(title_stem+ " with n = " + str(ns[i]))
        plt.ylabel("outcome Y")
        plt.xlabel(x_label)
        if truncate_y: 
            plt.ylim((0,truncate_y))
        plt.show()

'''
Helper functions for (noisy) bandwidth estimation: 
'''

def build_linear_model( **samp_params): 
    '''
    Fit a linear response model for use in estimation of bandwidth
    Test code for testing linear model of response
    # test_val = np.random.uniform()
    # samp_params['tau'] = test_val * np.ones([n,1])
    # test_data = np.concatenate( [samp_params['x'], samp_params['tau']], axis = 1 )
    # pred = regr.predict(test_data)
    pred_params = {'z' : test_data, 'y' : pred }
    plot_surface(**pred_params)
    plot_surface(**sub_params) 
    '''
    n = samp_params['n']
    regr = linear_model.LinearRegression()
    samp_params['z_samp'] = np.concatenate( [samp_params['x_samp'], samp_params['T_samp']],axis = 1 )
    regr.fit(samp_params['z_samp'], samp_params['y_samp'])
    return regr

def scores_cond_f_y_given_tau_x(joint_f_t_x, joint_f_y_t_x, test_point):
    """
    Use the estimates of joint density of F_{T,X} and F_{Y,T,X} to estimate
    the conditional density F_{Y|T,X} at the given test point
    Test point: [y, t, x]
    """
    tp = test_point[1:]
    joint_f_tau_x = joint_f_t_x.score_samples( tp.reshape([1,2]) )
    joint_f_y_tau_x = joint_f_y_t_x.score_samples( test_point.reshape([1,3]) )
    return np.exp(joint_f_y_tau_x - joint_f_tau_x)

# def scores_cond_f_y_given_tau_x_caller(test_point): 
#     #FIXME: will look in global scope
#     return scores_cond_f_y_given_tau_x(joint_f_t_x, joint_f_y_t_x, test_point)

def bias_integrand(y, tau, x, hessian): 
    x0 = np.asarray([y, tau, x])
    return y**2 * hessian([y, tau, x])[1][1] * 0.5

def empirical_exp_second_moment(regr, **params): 
    x = params['x']
    tau = params['tau']
    y = params['y_samp']
    T = params['T']
    
    y_pred = regr.predict(np.concatenate([params['x_samp'], params['tau']], axis = 1))
    Q = params['Q']
    Q_vec = np.asarray([Q(x[i], T[i], params['t_lo'], params['t_hi']) for i in range(params['n'])])
    return np.square(y_pred) / Q_vec

def est_h(h_sub, regr, hess, **samp_params):
    R_K = 1.0/(2*np.sqrt(np.pi))
    kappa_two = 1.0 
    C = R_K /(4.0 * samp_params['n'] * kappa_two**2)
    exp_second_moment = np.mean(empirical_exp_second_moment(regr, **samp_params))
    # Assume that tau doesn't change for x_i for now
    bias = 0 
    ymin = min(samp_params['y_samp'])
    ymax = max(samp_params['y_samp'])

    for i in range(h_sub):
        print i
        bias += integrate.quad(lambda u: bias_integrand(u, samp_params['tau'][i], samp_params['x_samp'][i], hess), ymin, ymax)[0]
    mean_bias_sqd = (bias/h_sub)**2
    h = np.power(C*exp_second_moment/(mean_bias_sqd*samp_params['n']), 0.2)

    print "opt h for this treatment vector: " + str(h) 
    return h

    ''' variant of OPE with known propensities
    '''
    ## given Known propensities
def off_policy_evaluation_known_Q(**params):
    """
    Takes in a choice of kernel and dictionary of parameters and data required for evaluation
    tau is a vector of treatment values (assumed given)
    If y_samp, T_samp is present, use that instead. 
    """
    [loss, norm_sum] = off_pol_estimator_known_Q(**params)
    h = params['h']
    n = params['n']
    return loss/(norm_sum*1.0*h*n)

def off_pol_estimator_known_Q(**params): 
    THRESH = params['threshold']
    y_out = params['y']; x = params['x']; h = params['h'];n = params['n']; t_lo = params['t_lo'];  t_hi = params['t_hi']
    kernel = params['kernel_func'];kernel_int =  params['kernel_int_func']
    Q = params['Q_known']; 
    if ('y_samp' in params.keys()):
        y_out = params['y_samp']
    if ('T_samp' in params.keys()): 
        T = params['T_samp']
    else: 
        T = params['T']
    if ('x_samp' in params.keys()):
        x = params['x_samp']

    BMI_IND = params.get('BMI_IND') # propensity score for warfarin data evaluations 
    if (params.get('DATA_TYPE') == 'warfarin'): 
        x = params['x'][:,BMI_IND]

    loss = 0
    tau = params['tau']
    clip_tau = np.clip(tau, t_lo, t_hi)
    Qs = np.zeros(n)
    for i in np.arange(n): 
        Q_i = Q[i]
        if (abs(clip_tau[i] - t_lo) <= h):
            alpha = kernel_int((t_lo-clip_tau[i])/h, 1)
        elif (abs(clip_tau[i] - t_hi) <= h):
            alpha = kernel_int(-1,  (t_hi - clip_tau[i])/h )
        else:
            alpha = 1
        Qs[i] = (1.0/h)*kernel( (clip_tau[i] - T[i])/h )/max(Q_i,THRESH)
        loss += kernel( (clip_tau[i] - T[i])/h )*1.0 * y_out[i]/max(Q_i,THRESH) * 1.0/alpha
    
#         if kernel( (clip_tau[i] - T[i])/h )>0.5: 
#             print y_out[i] 
#             print 'propensity: ' + str(Q_i)
    norm_sum = np.mean(np.maximum(Qs,THRESH*np.ones(n)))
    return [loss, norm_sum]

def bandwidth_selection(n_samp,h_sub, **params):
    '''
    Top-level function for estimating bandwidth. Note that this scales incredibly poorly with the size of the sampled dataset. 
    '''
    def scores_cond_f_y_given_tau_x_caller(test_point): 
        return scores_cond_f_y_given_tau_x(joint_f_t_x, joint_f_y_t_x, test_point)

    n = params['n']

    samp_params = evaluate_subsample(n_samp, cross_val = False, evaluation = False, **params)
    regr = build_linear_model(**samp_params)

    samp_params['tau'] = 0.5 * np.ones([samp_params['n'], 1])

    samp_params['z_samp'] = np.concatenate([samp_params['x_samp'], samp_params['T_samp']], axis = 1)
    bandwidths = {'bandwidth': np.logspace(-1,1,20)}
    grid = GridSearchCV(KernelDensity(), bandwidths)
    grid.fit(samp_params['z_samp'])

    bandwidth_est = grid.best_estimator_.bandwidth
    joint_f_t_x = KernelDensity(kernel='gaussian', bandwidth = bandwidth_est).fit(samp_params['z_samp'] )
    joint_f_y_t_x = KernelDensity(kernel='gaussian', bandwidth = bandwidth_est).fit(
        np.concatenate([samp_params['y_samp'],samp_params['z_samp']],axis=1) )

    cond_dens_hess = nd.Hessian(scores_cond_f_y_given_tau_x_caller)
    h = est_h(h_sub, regr, cond_dens_hess, **samp_params)
    return h