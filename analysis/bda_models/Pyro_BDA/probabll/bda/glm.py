import pyro
assert pyro.__version__.startswith('1.6.0')

import torch
import torch.distributions as td

import numpy as np
from tqdm.auto import tqdm
from tabulate import tabulate
import matplotlib.pyplot as plt
import arviz

from collections import OrderedDict, namedtuple
from pyro import poutine

import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F
from torch.distributions import constraints

import torch.distributions  as td

import pyro.distributions as pd
from pyro.optim import ClippedAdam
from pyro.infer import Predictive, SVI, Trace_ELBO
from pyro.infer import config_enumerate, infer_discrete
from pyro import poutine

from tabulate import tabulate
from collections import namedtuple, OrderedDict


class Family:
    """
    This is a container for classes that specify likelihood functions for generalised linear models.  
    
    A choice of family is used in a model
    
        mu, s ~ N(0, I)
        w[g], b[g] ~ N(0,I)
        
        others ~ prior()
        Y[g]|x[g] ~ F(link(eta), others)
            eta = mu + (s + w[g]) * x + b[g]

    Here you will find instances of F, they involve specifying the link function and priors/approximate posteriors
    on other parameters of the sampling distribution.
    
    Every class should 
    * use the constructor to specify fixed prior parameters and hyperparameters of variational approximations
    * implement link(eta) -> tensor
        this applies a non-linearity to eta in order to parameterise the sampling distribution
    * implement likelihood(link, params) -> a pyro Distribution to model the observation    
    * implement variational_params(device) -> dict of pyro.param tensors, properly initialised for SVI
    * implement posterior(variational_params) -> dict of pyro.sample objects (the approximate posterior distributions on additional parameters of the sampling distribution, if any)
    
    """
    
    class Poisson:
        """
        Use this family to model count data.
        
        Y|eta ~ Poisson(softplus(eta) + eps)
        
        """

        def __init__(self, eps=1e-9):
            """
            Parameters:
                eps: added to the Poisson rate
            """
            self.eps = eps
            
        def prior(self, device):
            """The Poisson does not have additional parameters"""
            return dict()
        
        def link(self, eta):
            """Maps eta to the Poisson rate using softplus"""
            return F.softplus(eta) + self.eps
        
        def likelihood(self, link, params=None):
            """A Poisson(link) object"""
            return pd.Poisson(link)

        def variational_params(self, device):  
            """The Poisson does not have additional parameters"""
            return dict()

        def posterior(self, params):   
            """The Poisson does not have additional parameters"""
            return dict()
    
    class GammaPoisson:
        """
        Use this family to model over-dispersed count data.
        
        beta ~ Gamma(a, b)
        Y|eta ~ GammaPoisson(softplus(eta) + eps, beta)
        """

        def __init__(self, a, b, eps=1e-9):
            """
            Parameters:
                a and b: strictly positive, determine the shape and rate of a Gamma prior on the GammaPoisson rate.
            """
            self.a = a
            self.b = b
            self.eps = eps
            
        def prior(self, device):
            """Gamma prior on rate in GammaPoisson(shape, rate)"""
            ones = torch.ones(1, device=device)
            return {'beta': pd.Gamma(ones * self.a, ones * self.b)}
                
        def link(self, eta):
            """Maps eta to the GammaPoisson shape via softplus"""
            return F.softplus(eta) + self.eps
        
        def likelihood(self, link, params):
            """GammaPoisson(shape=link(eta), rate=beta)"""
            return pd.GammaPoisson(link, params['beta'])

        def variational_params(self, device):
            """The location and scale for the LogNormal approximation to the posterior of the GammaPoisson rate"""
            zeros = torch.zeros(1, device=device)            
            params = {
                'beta_loc': pyro.param('beta_loc', td.Normal(zeros, zeros + 1).sample()),
                'beta_scale': F.softplus(pyro.param('beta_scale', td.Normal(zeros, zeros + 1).sample()))
            }
            return params

        def posterior(self, params):
            """LogNormal approximation to the posterior of the GammaPoisson rate"""
            return {'beta': pd.LogNormal(params['beta_loc'], params['beta_scale'])}

    class Normal:
        """
        Use this family to model continuous univariate data.
        

        sigma ~ Gamma(a, b)
        Y|eta ~ N(eta, sigma^2)
        
        """

        def __init__(self, sigma_a, sigma_b):
            """
            Parameters:
            
                sigma_a, sigma_b: parameters of a Gamma prior on the Normal scale.
            """
            self.sigma_a = sigma_a
            self.sigma_b = sigma_b

        def prior(self, device):
            zero = torch.zeros(1, device=device)
            params = {
                'sigma': pd.Gamma(zero + self.sigma_a, zero + self.sigma_b)
            }
            return params
        
        def link(self, eta):
            return eta
        
        def likelihood(self, link, params):
            return pd.Normal(link, params['sigma'])    

        def variational_params(self, device):
            zeros = torch.zeros(1, device=device)
            params = {
                'sigma_loc': pyro.param('sigma_loc', td.Normal(zeros, zeros + 1).sample()),
                'sigma_scale': F.softplus(pyro.param('sigma_scale', td.Normal(zeros, zeros + 1).sample())),
            }                
            return params

        def posterior(self, params):
            params = {
                'sigma': pd.LogNormal(params['sigma_loc'], params['sigma_scale'])
            }            
            return params


class GeneralisedLinearModelR1:
    """

    Model:
        
        mu ~ N(0, mu_scale^2)
        
        For each input predictor i = 1...I
            
            s[i] ~ N(0, s_scale^2)

            For each group g = 1...G 
                w[g,i] ~ N(0,w_scale^2)
                b[g,i] ~ N(0,b_scale^2)

        For each group g = 1...G              
            phi[g] ~ F.prior()

            For each observation n = 1...N
                eta = mu + (s[i] + w[g,i]) * x[n, g] + b[g,i]                
                y[n, g] ~ F.likelihood(F.link(eta), phi[g])
                
    """
           
    
    def __init__(self, 
        family,
        mu_scale=10., 
        s_scale=1.,
        w_scale=1.,
        b_scale=1.,
        device=torch.device('cuda:1')
    ):
        """
        Parameters:
                    
            device: torch device            
        """
        self.family = family
        self.mu_scale = mu_scale
        self.s_scale = s_scale
        self.w_scale = w_scale
        self.b_scale = b_scale
        
        self.device = device
        self.elbo_values = []
        self.optim = None
        
    def prepare(self, x, y):   
        """Map numpy arrays to tensors in the correct device"""
        return torch.tensor(x, device=self.device, dtype=torch.float32), torch.tensor(y, device=self.device, dtype=torch.float32)

    def model(self, x, y=None):
        N, G, I = x.shape
        device = x.device

        # []
        mu = pyro.sample('mu', pd.Normal(torch.zeros(1, device=device), torch.ones(1, device=device) * self.mu_scale))

        # [I=num_input_features]  
        s = pyro.sample('s', pd.Normal(torch.zeros(I, device=device), torch.ones(I, device=device) * self.s_scale).to_event(1))

        with pyro.plate("groups", G, device=device):                    

            # [G, I=num_input_features]  
            w = pyro.sample('w', pd.Normal(torch.zeros(I, device=device), torch.ones(I, device=device) * self.w_scale).to_event(1))    

            # [G=num_groups]
            b = pyro.sample('b', pd.Normal(torch.zeros(1, device=device), torch.ones(1, device=device) * self.b_scale))
            
            # Some likelihood functions may have additional parameters for which we need a prior
            params = dict()
            for rv_name, rv_dist in self.family.prior(device).items():
                params[rv_name] = pyro.sample(rv_name, rv_dist)

        with pyro.plate("observations", N, device=device):
            # [N, G]
            eta = pyro.deterministic('eta', mu + (x * s).sum(-1) + (x * w).sum(-1) + b)
            link = pyro.deterministic('link', self.family.link(eta))            
            obs = pyro.sample('obs', self.family.likelihood(link, params).to_event(1), obs=y)

        return obs

    def guide(self, x, y=None):

        N, G, I = x.shape
        device = x.device

        # []
        mu_loc = pyro.param('mu_loc', td.Normal(torch.zeros(1, device=device), torch.ones(1, device=device)).sample())
        mu_scale = F.softplus(pyro.param('mu_scale', td.Normal(torch.zeros(1, device=device), torch.ones(1, device=device)).sample()))

        # [I]
        s_loc = pyro.param('s_loc', td.Normal(torch.zeros(I, device=device), torch.ones(I, device=device)).sample())
        s_scale = F.softplus(pyro.param('s_scale', td.Normal(torch.zeros(I, device=device), torch.ones(I, device=device)).sample()))

        # [G, I]
        w_loc = pyro.param('w_loc', td.Normal(torch.zeros((G, I), device=device), torch.ones((G, I), device=device)).sample())
        w_scale = F.softplus(pyro.param('w_scale', td.Normal(torch.zeros((G, I), device=device), torch.ones((G, I), device=device)).sample()))

        # [G]
        b_loc = pyro.param('b_loc', td.Normal(torch.zeros(G, device=device), torch.ones(G, device=device)).sample())
        b_scale = F.softplus(pyro.param('b_scale', td.Normal(torch.zeros(G, device=device), torch.ones(G, device=device)).sample()))

        # In case the likelihood function requires priors
        params = self.family.variational_params(device)

        # Global rvs
        pyro.sample('mu', pd.Normal(mu_loc, mu_scale))
        pyro.sample('s', pd.Normal(s_loc, s_scale).to_event(1))

        # Group rvs
        with pyro.plate("groups", G, device=device):
            # [G, I]
            pyro.sample('w', pd.Normal(w_loc, w_scale).to_event(1))

            # [G]
            pyro.sample('b', pd.Normal(b_loc, b_scale))        
            
            # [G] in case the likelihood function requires posterior inference for some parameter
            for rv_name, rv in self.family.posterior(params).items():                
                pyro.sample(rv_name, rv)               
            
    def print_model_shapes(self, x):
        trace = poutine.trace(self.model).get_trace(x)
        trace.compute_log_prob()
        print(trace.format_shapes())

    def print_guide_shapes(self, x):        
        trace = poutine.trace(self.guide).get_trace(x)
        trace.compute_log_prob()
        print(trace.format_shapes())
    
    def prior_checks(self, x, y):      
        pvalue = (self.model(x) > y).float().mean(0)
        
        return OrderedDict(pvalue=pvalue)     

    def fit(self, x, y, num_iterations=2000, reset=True, progressbar=True, lr=0.005, clip_norm=10.):                
        """Use this to optimise the variational factors, make sure to obtain (x,y) from self.prepare"""
        if reset or self.optim is None:
            self.optim = ClippedAdam({"lr": lr, "clip_norm": clip_norm})
            pyro.clear_param_store()
            self.elbo_values = []
            
        elbo = Trace_ELBO(max_plate_nesting=2)
        svi = SVI(self.model, self.guide, self.optim, loss=elbo)        
        iterator = tqdm(range(num_iterations)) if progressbar else iter(range(num_iterations))
        
        for j in iterator:
            loss = svi.step(x, y)
            self.elbo_values.append(loss)
            iterator.set_postfix(OrderedDict(ELBO=self.elbo_values[-1]))          
    
    def prior_predict(self, num_samples, x, y=None):
        """
        Use this to sample from the prior predictive.
        
        If you fix y, we will not resample it.
        """
        pred = Predictive(self.model, num_samples=num_samples)(x, y)
        return pred
    
    def posterior_predict(self, num_samples, x, y=None):
        """
        Use this to sample from the posterior predictive.
        
        If you fix y, we will not resample it.
        """
        pred = Predictive(self.model, guide=self.guide, num_samples=num_samples)(x, y)
        return pred
            
class Plotting:
    """
    Some plot ideas.
    """

    @staticmethod
    def elbo(model):

        def moving_average(data_set, periods=3):
            weights = np.ones(periods) / periods
            return np.convolve(data_set, weights, mode='valid')
        _ = plt.plot(moving_average(-np.array(model.elbo_values), 20))
        _ = plt.xlabel("Steps")
        _ = plt.ylabel("ELBO")


    @staticmethod
    def obs(x, bins='auto', density=True, figsize=(6, 1), titles=None, xlabels=None, ylabels=None):
        if type(x) is torch.Tensor:
            x = x.detach().cpu().numpy()
        G = x.shape[-1]
        shared_bins = np.histogram_bin_edges(x.flatten(), bins=bins)
        fig, ax = plt.subplots(ncols=1, nrows=G, sharex=True, figsize=(figsize[0], figsize[1] * G))    
        for g in range(G):
            _ = ax[g].hist(x[...,g].flatten(), bins=shared_bins, density=density)
            if xlabels is not None and type(xlabels) in [list, tuple]:
                _ = ax[g].set_xlabel(xlabels[g])
            if titles is not None and type(titles) in [list, tuple]:    
                _ = ax[g].set_title(titles[g])
            if ylabels is not None and type(ylabels) in [list, tuple]:
                _ = ax[g].set_ylabel(ylabels[g])
                _ = ax[g].yaxis.set_label_position("right")

        if titles is not None and type(titles) is str:
            _ = ax[0].set_title(title)
        if xlabels is not None and type(xlabels) is str:
            _ = ax[-1].set_xlabel(xlabels)
        if ylabels is not None and type(ylabels) is str:
            _ = ax[-1].set_ylabel(ylabels)            
        return fig, ax

    @staticmethod
    def pred(posterior, obs=None, bins='auto', density=True, figsize=(6, 1), xlabels=None, ylabels=None, titles=None):
        if type(obs) is torch.Tensor:
            obs = obs.detach().cpu().numpy()
        pred = posterior['obs'].detach().cpu().numpy()

        if obs is not None:
            shared_bins = np.histogram_bin_edges(np.concatenate([pred.flatten(), obs.flatten()]), bins=bins)
        else:
            shared_bins = np.histogram_bin_edges(pred.flatten(), bins=bins)

        G = pred.shape[-1]
        fig, ax = plt.subplots(ncols=1, nrows=G, sharex=True, figsize=[figsize[0], figsize[1] * G])
        for g in range(G):
            if obs is not None:
                _ = ax[g].hist(obs[...,g].flatten(), bins=shared_bins, density=density, color='gray', histtype='step')
            _ = ax[g].hist(pred[...,g].flatten(), color='red', alpha=0.7, density=density, bins=shared_bins)

            if xlabels is not None and type(xlabels) in [list, tuple]:
                _ = ax[g].set_xlabel(xlabels[g])
            if titles is not None and type(titles) in [list, tuple]:    
                _ = ax[g].set_title(titles[g])
            if ylabels is not None and type(ylabels) in [list, tuple]:
                _ = ax[g].set_ylabel(ylabels[g])
                _ = ax[g].yaxis.set_label_position("right")

        if titles is not None and type(titles) is str:
            _ = ax[0].set_title(title)
        if xlabels is not None and type(xlabels) is str:
            _ = ax[-1].set_xlabel(xlabels)
        if ylabels is not None and type(ylabels) is str:
            _ = ax[-1].set_ylabel(ylabels) 

        return fig, ax    

    @staticmethod
    def mu(pred, figsize=(6,4)):        
        _ = arviz.plot_posterior({r'$\mu$': pred['mu'].unsqueeze(0).detach().cpu().numpy()}, figsize=figsize) 

    @staticmethod
    def slope(pred, figsize=(14,4)):
        # Here without the shared contribution of s
        fig, ax = plt.subplots(ncols=1, nrows=3, figsize=figsize)  
        _ = arviz.plot_forest({'s': pred['s'].unsqueeze(0).detach().cpu().numpy()}, ax=ax[0])
        _ = arviz.plot_forest({'w': pred['w'].unsqueeze(0).cpu().numpy()}, ax=ax[1])
        _ = arviz.plot_forest({'s+w': (pred['s']+pred['w']).unsqueeze(0).cpu().numpy()}, ax=ax[2])
     
    @staticmethod
    def bias(pred, figsize=(6,4)):
        # without the shared contribution of mu
        fig, ax = plt.subplots(ncols=2, nrows=1, figsize=figsize)  
        _ = arviz.plot_forest({'b': pred['b'].unsqueeze(0).cpu().numpy()}, ax=ax[0])
        _ = arviz.plot_forest({r'$\mu + b$': (pred['mu'] + pred['b']).unsqueeze(0).cpu().numpy()}, ax=ax[1])
        