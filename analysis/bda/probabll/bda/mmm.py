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
from pyro.infer import Predictive, SVI, Trace_ELBO, TraceEnum_ELBO
from pyro.infer import config_enumerate, infer_discrete
from pyro import poutine

from tabulate import tabulate
from collections import namedtuple, OrderedDict


class Family:
    """
    This is just a container for classes that can be used to specify the latent components
    and corresponding likelihood functions of a mixed-membership model.
    
    Every class should 
    * use the constructor to specify fixed prior parameters and whether the posterior approximation is a pd.Delta (i.e., MAP inference)
    * implement components(device) -> dict of pyro distributions objects (the components in the mixture)
    * implement likelihood(z, params) -> a pyro Distribution wrapping the components selected by z
    * implement variational_params(num_comps, device) -> dict of pyro.param tensors, properly initialised for SVI
    * implement posterior(variational_params) -> dict of pyro.sample objects (the approximate posterior distributions on parameters of components)
    * implement the property dim (None, or int). If dim is None, the data tensor should have shape (samples, groups), otherwise (samples, groups, dim)
    
    """
    
    class Poisson:

        def __init__(self, alpha, beta, delta=False):
            self.alpha = alpha
            self.beta = beta
            self.delta = delta

        @property
        def dim(self):
            return None
            
        def prior(self, device):
            ones = torch.ones(1, device=device)
            return {'lambda': pd.Gamma(ones * self.alpha, ones * self.beta)}
        
        def components(self, params):
            return pd.Poisson(params['lambda'])

        def likelihood(self, z, params):
            return pd.Poisson(params['lambda'][z])

        def variational_params(self, num_comps, device):
            zeros = torch.zeros(num_comps, device=device)
            if self.delta:
                params = {'lambda_': F.softplus(pyro.param('lambda_', td.Normal(zeros, zeros + 1).sample()))}
            else:
                params = {
                    'lambda_loc': pyro.param('lambda_loc', td.Normal(zeros, zeros + 1).sample()),
                    'lambda_scale': F.softplus(pyro.param('lambda_scale', td.Normal(zeros, zeros + 1).sample()))
                }
            return params

        def posterior(self, params):
            if self.delta:
                return {'lambda': pd.Delta(params['lambda_'])}            
            else:
                return {'lambda': pd.LogNormal(params['lambda_loc'], params['lambda_scale'])}

    class Normal:

        def __init__(self, mu_loc, mu_scale, sigma_alpha, sigma_beta, delta=False):
            self.mu_loc = mu_loc
            self.mu_scale = mu_scale
            self.sigma_alpha = sigma_alpha
            self.sigma_beta = sigma_beta
            self.delta = delta
            
        @property
        def dim(self):
            return None    

        def prior(self, device):
            zero = torch.zeros(1, device=device)
            params = {
                'mu': pd.Normal(zero + self.mu_loc, zero + self.mu_scale),
                'sigma': pd.Gamma(zero + self.sigma_alpha, zero + self.sigma_beta)
            }
            return params
        
        def components(self, params):
            return pd.Normal(params['mu'], params['sigma'])

        def likelihood(self, z, params):
            return pd.Normal(params['mu'][z], params['sigma'][z])    

        def variational_params(self, num_comps, device):
            zeros = torch.zeros(num_comps, device=device)
            if self.delta:
                params = {
                    'mu_': pyro.param('mu_', td.Normal(zeros, zeros + 1).sample()),
                    'sigma_': F.softplus(pyro.param('sigma_', td.Normal(zeros, zeros + 1).sample())),
                } 
            else:
                params = {
                    'mu_loc': pyro.param('mu_loc', td.Normal(zeros, zeros + 1).sample()),
                    'mu_scale': F.softplus(pyro.param('mu_scale', td.Normal(zeros, zeros + 1).sample())),
                    'sigma_loc': pyro.param('sigma_loc', td.Normal(zeros, zeros + 1).sample()),
                    'sigma_scale': F.softplus(pyro.param('sigma_scale', td.Normal(zeros, zeros + 1).sample())),
                }
                
            return params

        def posterior(self, params):
            if self.delta:
                params = {
                    'mu': pd.Delta(params['mu_']),
                    'sigma': pd.Delta(params['sigma_'])
                } 
            else:
                params = {
                    'mu': pd.Normal(params['mu_loc'], params['mu_scale']),
                    'sigma': pd.LogNormal(params['sigma_loc'], params['sigma_scale'])
                }            
            return params
        
    class LowRankMVN:

        def __init__(self, dim, rank, mu_loc, mu_scale, cov_diag_alpha, cov_diag_beta, cov_factor_loc, cov_factor_scale, delta=False):
            self._dim = dim
            self.rank = rank            
            self.mu_loc = mu_loc
            self.mu_scale = mu_scale
            self.cov_diag_alpha = cov_diag_alpha
            self.cov_diag_beta = cov_diag_beta
            self.cov_factor_loc = cov_factor_loc
            self.cov_factor_scale = cov_factor_scale
            self.delta = delta
            
        @property
        def dim(self):
            return self._dim

        def prior(self, device):
            zero = torch.zeros(1, device=device)
            params = {
                'mu': pd.Independent(
                    pd.Normal(
                        torch.zeros(self.dim, device=device) + self.mu_loc, 
                        torch.zeros(self.dim, device=device) + self.mu_scale
                    ), 1),
                'cov_factor': pd.Independent(
                    pd.Normal(
                        torch.zeros([self.dim, self.rank], device=device) + self.cov_factor_loc, 
                        torch.ones([self.dim, self.rank], device=device) + self.cov_factor_scale
                    ), 2),
                'cov_diag': pd.Independent(
                    pd.Gamma(
                        torch.zeros(self.dim, device=device) + self.cov_diag_alpha, 
                        torch.zeros(self.dim, device=device) + self.cov_diag_beta
                    ), 1)
            }
            return params
        
        def components(self, params):
            return pd.LowRankMultivariateNormal(params['mu'], params['cov_factor'], params['cov_diag'])

        def likelihood(self, z, params):
            loc = params['mu'][z]
            cov_factor = params['cov_factor'][z]
            cov_diag = params['cov_diag'][z]
            return pd.LowRankMultivariateNormal(loc, cov_factor, cov_diag)

        def variational_params(self, num_comps, device):
            zeros = torch.zeros(num_comps, device=device)
            if self.delta:
                params = {
                    'mu_': pyro.param(
                        'mu_', td.Normal(
                            torch.zeros(self.dim, device=device), 
                            torch.ones(self.dim, device=device)
                        ).sample([num_comps])),
                    'cov_factor_': pyro.param(
                        'cov_factor_', td.Normal(
                            torch.zeros([self.dim, self.rank], device=device), 
                            torch.ones([self.dim, self.rank], device=device)
                        ).sample([num_comps])),
                    'cov_diag_': F.softplus(pyro.param(
                        'cov_diag_', td.Normal(
                            torch.zeros(self.dim, device=device), 
                            torch.ones(self.dim, device=device)
                        ).sample([num_comps]))),
                } 
            else:
                params = {
                    'mu_loc': pyro.param(
                        'mu_loc', td.Normal(
                            torch.zeros(self.dim, device=device), 
                            torch.ones(self.dim, device=device)
                        ).sample([num_comps])),
                    'mu_scale': F.softplus(pyro.param(
                        'mu_scale', td.Normal(
                            torch.zeros(self.dim, device=device), 
                            torch.ones(self.dim, device=device)
                        ).sample([num_comps]))),
                    'cov_diag_loc': pyro.param(
                        'cov_diag_loc', td.Normal(
                            torch.zeros(self.dim, device=device), 
                            torch.ones(self.dim, device=device)
                        ).sample([num_comps])),
                    'cov_diag_scale': F.softplus(pyro.param(
                        'cov_diag_scale', td.Normal(
                            torch.zeros(self.dim, device=device), 
                            torch.ones(self.dim, device=device)
                        ).sample([num_comps]))),
                    'cov_factor_loc': pyro.param(
                        'cov_factor_loc', td.Normal(
                            torch.zeros([self.dim, self.rank], device=device), 
                            torch.ones([self.dim, self.rank], device=device)
                        ).sample([num_comps])),
                    'cov_factor_scale': F.softplus(pyro.param(
                        'cov_factor_scale', td.Normal(
                            torch.zeros([self.dim, self.rank], device=device), 
                            torch.ones([self.dim, self.rank], device=device)
                        ).sample([num_comps]))),
                }
                
            return params

        def posterior(self, params):
            if self.delta:
                params = {
                    'mu': pd.Independent(pd.Delta(params['mu_']), 1),
                    'cov_factor': pd.Independent(pd.Delta(params['cov_factor_']), 2),
                    'cov_diag': pd.Independent(pd.Delta(params['cov_diag_']), 1)
                } 
            else:
                params = {
                    'mu': pd.Independent(pd.Normal(params['mu_loc'], params['mu_scale']), 1),
                    'cov_diag': pd.Independent(pd.LogNormal(params['cov_diag_loc'], params['cov_diag_scale']), 1),
                    'cov_factor': pd.Independent(pd.Normal(params['cov_factor_loc'], params['cov_factor_scale']), 2),
                }            
            return params        

        
    class Multinomial:

        def __init__(self, dim, total_count, alpha=1.0, delta=False):
            """
            
            A likelihood function for count vectors.
            
                theta[t] ~ Dirichlet(alpha * 1) for t=1,...,T
                X|z ~ Multinomial(theta[z], N)
            
            Parameters:
            
            dim: vocabulary size
            total_count: number of draws from the Multinomial component (this has to be the same for every group).
            alpha: symmetric Dirichlet prior parameter          
            delta: Delta posterior instead of a Dirichlet posterior
            """
            self._dim = dim
            self.total_count = total_count
            self.alpha = alpha
            self.delta = delta

        @property
        def dim(self):
            return self._dim    

        def prior(self, device):
            zero = torch.zeros(self.dim, device=device)
            params = {
                'theta': pd.Dirichlet(zero + self.alpha),
            }
            return params

        def components(self, params):
            return pd.Multinomial(probs=params['theta'], total_count=self.total_count)

        def likelihood(self, z, params):
            return pd.Multinomial(probs=params['theta'][z], total_count=self.total_count)

        def variational_params(self, num_comps, device):        
            if self.delta:
                zeros = torch.zeros([num_comps, self.dim], device=device)
                params = {
                    'theta_': F.softmax(pyro.param('theta_', td.Normal(zeros, zeros + 1).sample()), -1),
                } 
            else:
                zeros = torch.zeros([num_comps, self.dim], device=device)
                params = {
                    'theta_alpha': F.softplus(pyro.param('theta_alpha', td.Normal(zeros, zeros + 1).sample())),
                }

            return params

        def posterior(self, params):
            if self.delta:
                params = {
                    'theta': pd.Independent(pd.Delta(params['theta_']), 1),
                } 
            else:
                params = {
                    'theta': pd.Dirichlet(params['theta_alpha']) 
                }            
            return params

    class DirichletMultinomial:

        def __init__(self, dim, total_count, two_params=False, counts_a=1.0, counts_b=1.0, alpha_a=1.0, alpha_b=1.0, delta=False):
            """
            A likelihood function for count vectors.
            
                counts[t] ~ Gamma(counts_a, counts_b) for t=1,...,T

                if two_params:
                    alpha[t] ~ Gamma(alpha_a, alpha_b) for t=1,...,T
                else:
                    alpha[t] = 1
                
                X|z ~ DirichletMultinomial(alpha[z] * counts[z], N)
            
            Parameters:
            
            dim: vocabulary size
            total_count: number of draws from the DirichletMultinomial component (this has to be the same for every group).
            two_params: whether we model a scaling factor separately from the pseudo counts            
            counts_a, counts_b: parameters of the prior on counts
            alpha_a, alpha_b: parameters of the prior on alpha            
            delta: Delta posteriors instead of LogNormal posteriors
            """
            self._dim = dim
            self.total_count = total_count
            self.two_params = two_params
            self.alpha_a, self.alpha_b = alpha_a, alpha_b
            self.counts_a, self.counts_b = counts_a, counts_b
            self.delta = delta

        @property
        def dim(self):
            return self._dim 

        def prior(self, device):
            zero = torch.zeros([1], device=device)
            zeros = torch.zeros([self.dim], device=device)
            params = {
                'counts': pd.Independent(pd.Gamma(zeros + self.counts_a, zeros + self.counts_b), 1),
            }
            if self.two_params:
                params['alpha'] = pd.Independent(pd.Gamma(zero + self.alpha_a, zero + self.alpha_b), 1)
            return params

        def components(self, params):
            if self.two_params:
                return pd.DirichletMultinomial(params['alpha'] * params['counts'], total_count=self.total_count)
            else:
                return pd.DirichletMultinomial(params['counts'], total_count=self.total_count)

        def likelihood(self, z, params):
            if self.two_params:
                return pd.DirichletMultinomial(params['alpha'][z] * params['counts'][z], total_count=self.total_count)
            else:
                return pd.DirichletMultinomial(params['counts'][z], total_count=self.total_count)

        def variational_params(self, num_comps, device):        
            if self.delta:
                zero = torch.zeros([num_comps, 1], device=device)
                zeros = torch.zeros([num_comps, self.dim], device=device)
                params = {
                    'counts_': F.softmax(pyro.param('counts_', td.Normal(zeros, zeros + 1).sample()), -1),
                } 
                if self.two_params:
                    params['alpha_'] = F.softmax(pyro.param('alpha_', td.Normal(zero, zero + 1).sample()), -1)                
            else:
                zero = torch.zeros([num_comps, 1], device=device)
                zeros = torch.zeros([num_comps, self.dim], device=device)
                params = {
                    'counts_loc': pyro.param('counts_loc', td.Normal(zeros, zeros + 1).sample()),
                    'counts_scale': F.softplus(pyro.param('counts_scale', td.Normal(zeros, zeros + 1).sample())),
                }
                if self.two_params:
                    params['alpha_loc'] = pyro.param('alpha_loc', td.Normal(zero, zero + 1).sample())
                    params['alpha_scale'] = F.softplus(pyro.param('alpha_scale', td.Normal(zero, zero + 1).sample()))

            return params

        def posterior(self, params):
            if self.delta:                
                params = {
                    'counts': pd.Independent(pd.Delta(params['counts_']), 1),
                } 
                if self.two_params:
                    params['alpha'] = pd.Independent(pd.Delta(params['alpha_']), 1)
            else:
                params = {
                    'counts': pd.Independent(pd.LogNormal(params['counts_loc'], params['counts_scale']), 1),
                }         
                if self.two_params:
                    params['alpha'] = pd.Independent(pd.LogNormal(params['alpha_loc'], params['alpha_scale']), 1)
            return params            
        

class MixedMembershipRD:
           
    
    def __init__(self, 
        family,
        T=10, 
        DP_alpha=0.1, 
        device=torch.device('cuda:1')
    ):
        """
        A mixed-membership model draws observations from a group-specific mixture of shared latent components 
        with a truncated Dirichlet process prior on the mixing coefficients.
        
        Parameters:
        
            family: the family of the components used to explain the data
            
            T: we mix T components whose parameters are sampled from a shared prior
                e.g., for T Normal distributions
                    mu[t] ~ prior on location (see Family.Normal)
                    sigma[t] ~ prior on scale (see Family.Normal)
                    
            DP_alpha: concentation parameter of the DP (the smaller, the less components)
            
            device: torch device            
        """
        self.family = family
        self.T = T
        self.DP_alpha = DP_alpha
        self.device = device
        self.N = self.G = None
        self.D = family.dim
        self.elbo_values = []
        self.optim = None
    
    @classmethod
    def mix_weights(cls, beta):
        beta1m_cumprod = (1 - beta).cumprod(-1)
        return F.pad(beta, (0, 1), value=1) * F.pad(beta1m_cumprod, (1, 0), value=1)
    
    def prepare(self, groups):
        """
        Return a torch tensor of observations.
        
        Parameters:
        
        groups: nparray with shape [num_observations, num_groups] or [num_observations, num_groups, dim] for multivariate data
        """
        if len(groups.shape) == 2:
            self.N, self.G = groups.shape
            D = None
        elif len(groups.shape) == 3:
            self.N, self.G, D = groups.shape            
        else:
            raise ValueError(f"groups should have shape [N, G] or [N, G, D] got {groups.shape}")
            
        if self.family.dim != D:
            raise ValueError(f"You have a {self.family.dim}-dimensional family and {D}-dimensional data.")
            
        if self.G > self.N:
            print("You have more groups than samples per group, are you sure your array has shape [N, G] or [N, G, D]?")
            
        return torch.tensor(groups, device=self.device, dtype=torch.float32)

    def model(self, x=None, batch_size=None):
        """
        Pyro joint distribution.
        
        Parameter:
        
        x: observations as returned by self.prepare or None. 
            If None, the 'obs' node of the graphical model will be resampled.
        """

        N, G = self.N, self.G
        T = self.T
        if N is None or G is None:        
            raise ValueError("Did you run prepare?")
        device = self.device

        # Sample Gaussian Components
        with pyro.plate("components", T, device=device):  
            # construct the components
            params = dict()
            for rv_name, rv in self.family.prior(device).items():
                params[rv_name] = pyro.sample(rv_name, rv)

        # Sample mixing weights    
        with pyro.plate("DPs", G, device=device):
            # [G, T-1]
            beta = pyro.sample(
                "beta", 
                pd.Beta(
                    torch.ones(1, device=device), 
                    torch.ones(1, device=device) * self.DP_alpha
                ).expand((T-1,)).to_event(1)
            )
        # [G, T]
        omega = pyro.deterministic("omega", self.mix_weights(beta))

        with pyro.plate("groups", G, device=device):
            with pyro.plate("observations", N, device=device, subsample_size=batch_size) as ind:
                # [N, G]
                z = pyro.sample("z", pd.Categorical(probs=omega), infer={"enumerate": "parallel"})
                
                # [N, G]
                # Construct the likelihood function
                obs = None if x is None else x.index_select(0, ind)
                return pyro.sample("obs", self.family.likelihood(z, params), obs=obs)
            
    def guide(self, x=None, batch_size=None):
        """
        Pyro approximate posterior.
        
        Parameter:
        
        x: observations as returned by self.prepare or None.         
        """
        N, G = self.N, self.G
        T = self.T
        if N is None or G is None:        
            raise ValueError("Did you run prepare?")
        device = self.device 
        
        # [G, T-1]
        kappa = pyro.param('kappa', td.Uniform(0, 2).sample([G, T-1]).to(device), constraint=constraints.positive)

        # Construct the variational parameters for T components
        params = self.family.variational_params(T, device)
        
        with pyro.plate("components", T):
            # [T] construct the posterior approximations
            for rv_name, rv in self.family.posterior(params).items():                
                pyro.sample(rv_name, rv)

        with pyro.plate("DPs", G, device=device):
            # [G, T-1]
            beta = pyro.sample("beta", pd.Beta(torch.ones(T-1, device=device), kappa).to_event(1))       
            
    def print_model_shapes(self, batch_size=None):
        trace = poutine.trace(poutine.enum(self.model, first_available_dim=-3)).get_trace(batch_size=batch_size)
        trace.compute_log_prob()
        print(trace.format_shapes())

    def print_guide_shapes(self):        
        trace = poutine.trace(poutine.enum(self.guide, first_available_dim=-3)).get_trace()
        trace.compute_log_prob()
        print(trace.format_shapes())
    
    def prior_checks(self, x, num_checks=100):
        mean_checks = torch.stack([self.model(None).mean(0) > x.mean(0) for _ in range(num_checks)]).float()
        mean_checks = mean_checks.mean(0)
        
        std_checks = torch.stack([self.model(None).std(0) > x.std(0) for _ in range(num_checks)]).float()
        std_checks = std_checks.mean(0)
        
        return OrderedDict(mean=mean_checks, std=std_checks)     

    def fit(self, x, num_iterations=2000, reset=True, batch_size=None, lr=0.005, clip_norm=10., progressbar=True):        
        """
        """
        if reset or self.optim is None:
            self.optim = ClippedAdam({"lr": lr, "clip_norm": clip_norm})
            pyro.clear_param_store()
            self.elbo_values = []
            
        elbo = TraceEnum_ELBO(max_plate_nesting=2)
        svi = SVI(self.model, self.guide, self.optim, loss=elbo)        
        iterator = tqdm(range(num_iterations)) if progressbar else iter(range(num_iterations))
        
        for j in iterator:
            loss = svi.step(x, batch_size=batch_size)
            self.elbo_values.append(loss)
            iterator.set_postfix(OrderedDict(ELBO=self.elbo_values[-1]))          
    
    def prior_predict(self, num_samples, x=None, batch_size=None):
        pred = Predictive(self.model, num_samples=num_samples)(x, batch_size=batch_size)
        return pred
    
    def posterior_predict(self, num_samples, x=None, batch_size=None):
        pred = Predictive(self.model, guide=self.guide, num_samples=num_samples)(x, batch_size=batch_size)
        return pred
            
class Plotting:

    @staticmethod
    def elbo(model):

        def moving_average(data_set, periods=3):
            weights = np.ones(periods) / periods
            return np.convolve(data_set, weights, mode='valid')
        _ = plt.plot(moving_average(-np.array(model.elbo_values), 20))
        _ = plt.xlabel("Steps")
        _ = plt.ylabel("ELBO")

    @staticmethod
    def obs(x, pred=None, group_names=None, bins=100, density=True, figsize=(6, 1), sharex=True, sharey=False, colors=['gray', 'red']):
        """
        Histogram plots for each of the groups, multivariate data are plotted marginally (i.e., flattened).

        Parameters:

        x: [N, G] or [N, G, D]
            rendered in gray
        pred: [S, N, G] or [S, N, G, D]    
            rendered in red
        """
        
        # Adjust shapes
        if pred is not None and len(pred.shape) < len(x.shape):
            raise ValueError(f'Predictions should have the same shape as observations or additional sample dimensions')
        if len(x.shape) == 2:  # Introduce an event dimension
            x = x[...,None]
            if pred is not None:
                pred = pred[...,None]
        elif len(x.shape) != 3:
            raise ValueError(f"x should have shape [N, G] or [N, G, D] got {x.shape}")
            
        # To numpy
        if type(x) is torch.Tensor:
            x = x.detach().cpu().numpy()
        if pred is not None and type(pred) is torch.Tensor:
            pred = pred.detach().cpu().numpy()

        N, G, D = x.shape
        
        # Adjust names
        if group_names is None:
            group_names = np.arange(1, G + 1)
        
        fig, ax = plt.subplots(ncols=1, nrows=G, sharex=True, sharey=sharey, figsize=[figsize[0], figsize[1] * G])        
        fig.tight_layout() 
        
        if pred is not None:
            shared_bins = np.histogram_bin_edges(np.concatenate([pred.flatten(), x.flatten()]), bins=bins)
        else:
            shared_bins = np.histogram_bin_edges(x.flatten(), bins=bins)

        for g in range(G):
            _ = ax[g].hist(x[...,g,:].flatten(), bins=shared_bins, density=density, color=colors[0], histtype='step')
            if pred is not None:
                _ = ax[g].hist(pred[...,g,:].flatten(), color=colors[1], alpha=0.7, density=density, bins=shared_bins)
            _ = ax[g].set_ylabel(group_names[g])
            _ = ax[g].yaxis.set_label_position("right")       

        return fig, ax

    @staticmethod
    def obs_dims(x, pred=None, group_names=None, figsize=[6, 1], sharex=True, sharey=False):
        """
        Violin plots for each of the groups and each of the dimensions of the data samples. 

        Parameters:

        x: [N, G] or [N, G, D]
            rendered in blue
        pred: [S, N, G] or [S, N, G, D]    
            rendered in orange
        """
        # Adjust shapes
        if pred is not None:
            if len(pred.shape) < len(x.shape):
                raise ValueError(f'Predictions should have the same shape as observations or additional sample dimensions')
        if len(x.shape) == 2:  # Introduce an event dimension
            x = x[...,None]
            if pred is not None:
                pred = pred[...,None]
        elif len(x.shape) != 3:
            raise ValueError(f"x should have shape [N, G] or [N, G, D] got {x.shape}")

        # To numpy
        if type(x) is torch.Tensor:
            x = x.detach().cpu().numpy()
        if pred is not None and type(pred) is torch.Tensor:
            pred = pred.detach().cpu().numpy()

        N, G, D = x.shape

        # Adjust names
        if group_names is None:
            group_names = np.arange(1, G + 1)

        # Plot groups
        fig, ax = plt.subplots(ncols=G, nrows=D, sharex=sharex, sharey=sharey, figsize=[figsize[0] * G, figsize[1] * D])
        if len(ax.shape) == 1:
            ax = ax[None, :]
        fig.tight_layout()
        
        for g in range(G):
            for d in range(D):    
                _ = arviz.plot_violin(
                    {
                        'obs': x[...,g,d],
                    },
                    var_names=['obs'],
                    ax=ax[d,g],
                )
                if pred is not None:
                    _ = arviz.plot_violin(
                        {
                            'pred': pred[...,g,d],
                        },
                        var_names=['pred'],
                        ax=ax[d,g],
                    )
                ax[d,g].set_title(f"{g}:{group_names[g]} ({d})")
        return fig, ax

    @staticmethod
    def components(posterior, model, figsize=(4,2), sharex=True, sharey=True, marginal=True):
        """
        Plot samples from the components. 
        
        If marginal=True, plot histograms of (flattened) samples.
        If marginal=False, plot violins for every group and dimension. 
        
        Parameters:
        
        posterior: dictionary with samples from the posterior distribution
        model: an instance of MixedMembershipRD
        marginal: whether or not multivariate data should be flattened
        
        This uses Plotting.obs and Plotting.obs_dims.
        """
        T = model.T
        D = model.D if model.D else 1
                
        samples = model.family.components(posterior).sample().view(-1, T, D).detach().cpu().numpy()
        if marginal:
            return Plotting.obs(samples, group_names=None, figsize=figsize, sharex=sharex, sharey=sharey, colors=['blue'])
        else:
            return Plotting.obs_dims(samples, group_names=None, figsize=figsize, sharex=sharex, sharey=sharey)

    @staticmethod
    def mixture_weights(posterior, model, figsize=(16, 8), cols=4):
        """
        Boxplots of the mixing coefficients for each group.
        """
        G, T = model.G, model.T
        omega = posterior['omega'].reshape([-1, G, T]) #

        fig, ax = plt.subplots(ncols=min(cols, G), nrows=G//cols+bool(G%cols), sharey=True, figsize=figsize)

        if len(ax.shape) == 1:
            ax = ax.reshape(1, -1)

        for g in range(G):
            _ = ax[g//cols, g%cols].boxplot(omega[:,g,:].detach().cpu().numpy())
            _ = ax[g//cols, g%cols].set_xlabel(f"Group {g}")
        return fig, ax


    @staticmethod
    def mean_mixture_weights(posterior, model):        
        """
        Heatmap of posterior mean mixing coefficient per component per group.
        """
        G, T = model.G, model.T
        omega = posterior['omega'].reshape([-1, G, T]) #
        
        _ = plt.imshow(omega.mean(0).detach().cpu().numpy())
        _ = plt.ylabel("groups")
        _ = plt.yticks(np.arange(model.G))
        _ = plt.xticks(np.arange(model.T))
        _ = plt.xlabel("components")


    @staticmethod
    def compare_mixture_weights(posterior, model, ref_group=0, figsize=(8, 16), eps=1e-9, bins=100):
        """
        Compare mixture weights across groups taking one group as reference.
        """

        def tvd(p, q):
            return 0.5*(torch.sum(torch.abs(p.probs - q.probs), -1))

        def js(p, q):
            return (td.kl_divergence(p, q) + td.kl_divergence(q, p)) / 2.

        G, T = model.G, model.T
        omega = posterior['omega'].reshape([-1, G, T]) #

        fig, ax = plt.subplots(ncols=1, nrows=model.G, sharex='col', figsize=figsize)
        s_ref = td.Categorical(logits=(omega[:,ref_group]+eps).log())
        
        for g in range(model.G):    
            s_other = td.Categorical(logits=(omega[:,g]+eps).log())    
            _ = ax[g].hist(js(s_ref, s_other).cpu().numpy(), bins=bins)  

        ax[0].set_title('Jensen-Shannon')

        return fig, ax
        
    @staticmethod
    def expected_divergence(p_from, posterior, model, D=td.kl_divergence, figsize=(10, 5)):
        """
        Compute expected divergence from a given distribution to each of the group-specific posterior mixture model.
        """
        with torch.no_grad():
            fig, ax = plt.subplots(ncols=2, nrows=1, figsize=figsize)

            G, T = model.G, model.T

            comps = model.family.components(posterior)
            comps.batch_shape, comps.event_shape

            p_from = p_from.expand(comps.batch_shape)

            comp_div = D(comps, p_from)

            # sum the last dimension (expectation under Categorical(omega))
            # then average the data dimension (expectation under the data distribution)
            div = ((posterior['omega'] * comp_div).sum(-1)).mean(-2)
            # then average the sample dimension (expectation under the posterior samples
            mean_div = div.mean(0).squeeze(0)

            _ = ax[0].set_ylabel('E[ D(comp, ref) ]')
            _ = ax[0].set_xlabel('group')
            _ = ax[1].set_ylabel('E[ D(comp, ref) ]')
            _ = ax[1].set_xlabel('group')
            _ = ax[0].plot(np.arange(1, G + 1), mean_div.cpu().numpy(), 'ro')
            _ = ax[1].boxplot(div.squeeze(1).cpu().numpy())

        return fig, ax       
    
