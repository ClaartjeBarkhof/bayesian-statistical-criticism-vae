import numpy as np

import numpyro
from numpyro.infer import MCMC, NUTS, Predictive
import numpyro.distributions as dist

from jax import random
from jax.nn import logsumexp
import jax.numpy as jnp

class GenPixelModelMNIST:
    def __init__(self, obs_x, obs_y, num_samples=1000, num_chains=1, num_warmup=100):

        assert obs_x.ndim == 2, f"obs_x is assumed to be [B, D], currrent shape: {obs_x.shape}"
        assert obs_y.ndim == 1, f"obs_y is assumed to be 1D, current shape: {obs_y.shape}"
        assert len(obs_x) == len(obs_y), f"obs_x is assumed to be of the same length as obs_y"

        self.obs_x = obs_x
        self.obs_y = obs_y

        # num components = num digits (10)
        self.T = 10
        self.D = obs_x.shape[1]
        self.N = self.obs_x.shape[0]

        self.active = np.zeros((self.T, self.D))
        self.inactive = np.zeros((self.T, self.D))
        for i in range(self.T):
            self.active[i, :] = self.obs_x[obs_y == i].sum(axis=0)
            self.inactive[i, :] = (1.0 - self.obs_x[obs_y == i]).sum(axis=0)

        self.num_samples = num_samples
        self.num_chains = num_chains
        self.num_warmup = num_warmup

        self.rng_key = random.PRNGKey(0)

        nuts_kernel = NUTS(self.conditional_model)
        self.mcmc = MCMC(nuts_kernel, num_samples=num_samples, num_warmup=num_warmup, num_chains=num_chains)

        self.prior_predictive = None
        self.posterior_predictive = None
        self.posterior_samples = None

    def conditional_model(self, obs_x=None):
        # alpha = numpyro.sample("alpha", dist.Exponential(1.))
        alpha = numpyro.deterministic("alpha", 0.2)
        beta = numpyro.deterministic("beta", 1.0)

        concentration_a = numpyro.deterministic("concentration_a", alpha + self.active)
        concentration_b = numpyro.deterministic("concentration_b", beta + self.inactive)

        with numpyro.plate("observations", self.N):
            concentration_a_y = concentration_a[self.obs_y, :]
            concentration_b_y = concentration_b[self.obs_y, :]

            sampled_x = numpyro.sample("sampled_x",
                                       dist.conjugate.BetaBinomial(concentration1=concentration_a_y,
                                                                   concentration0=concentration_b_y,
                                                                   total_count=1).to_event(1),
                                       obs=obs_x)

        return sampled_x

    def posterior(self):
        alpha = numpyro.deterministic("alpha", 0.2)
        beta = numpyro.deterministic("beta", 1.0)

        concentration_a = numpyro.deterministic("concentration_a", alpha + self.active)
        concentration_b = numpyro.deterministic("concentration_b", beta + self.inactive)

        return dist.Beta(concentration_a, concentration_b)

    def conditional_posterior_predictive(self, obs_ys):
        alpha = numpyro.deterministic("alpha", 0.2)
        beta = numpyro.deterministic("beta", 1.0)

        concentration_a = numpyro.deterministic("concentration_a", alpha + self.active)
        concentration_b = numpyro.deterministic("concentration_b", beta + self.inactive)

        concentration_a_y = concentration_a[obs_ys, :]
        concentration_b_y = concentration_b[obs_ys, :]

        return dist.BetaBinomial(concentration1=concentration_a_y,
                                 concentration0=concentration_b_y, total_count=1).to_event(1)

    def assess_unconditional_posterior_predictive(self, obs_xs):
        """ Assess p(x*|x) for given x* (obs_xs)
            p(x*=x*|x) = \sum_y p(x*, y*|x) = \sum_y p(x*|x, y*)p(y*) = \sum_y p(x*|x, y*)(1/T) """

        # [T * Nx] [0, 1, 2, .., 9, 0, 1, 2, ..., 9, ..., 9]
        y_all_classes = np.concatenate([np.arange(self.T) for _ in range(len(obs_xs))])

        # [T * Nx, D] [x1, x1, ..., x2, x2, ..., x9, x9, ...]
        obs_xs_all_class = np.repeat(obs_xs, self.T, axis=0)

        # [T * Nx]
        post_all_classes = self.conditional_posterior_predictive(y_all_classes)

        # [T*N] -> [N, T]
        log_prob_all_class = post_all_classes.log_prob(obs_xs_all_class).reshape(len(obs_xs), self.T)
        log_prob_all_class -= jnp.log(self.T)

        # [N]
        log_pxs_px = logsumexp(log_prob_all_class, axis=1)

        #         pxs_px = jnp.exp(log_pxs_px)

        return log_pxs_px

    def classification_distribution(self, obs_xs):
        """ Determine p(y*|x, x*) for given x* (obs_xs)
            p(y*|x, x*) = p(x*, y*|x) / p(x*|x) = p(x*|x, y*)p(y*) / p(x*|x)
        """

        # [T * Nx] [0, 1, 2, .., 9, 0, 1, 2, ..., 9, ..., 9]
        y_all_classes = np.concatenate([np.arange(self.T) for _ in range(len(obs_xs))])

        # [T * Nx, D] [x1, x1, ..., x2, x2, ..., x9, x9, ...]
        obs_xs_all_class = np.repeat(obs_xs, self.T, axis=0)

        # [T * Nx]
        post_all_classes = self.conditional_posterior_predictive(y_all_classes)

        # [T*N] -> [N, T]
        log_prob_all_class = post_all_classes.log_prob(obs_xs_all_class).reshape(len(obs_xs), self.T)
        log_prob_all_class -= jnp.log(self.T)
        log_prob_all_class -= logsumexp(log_prob_all_class, axis=1)[:,
                              None]  # to this point it is the same as function above

        #         prob_all_class = jnp.exp(log_prob_all_class)

        return log_prob_all_class

    def run(self):
        self.mcmc.run(self.rng_key, obs_x=self.obs_x)
        # self.mcmc.print_summary()
        self.posterior_samples = self.mcmc.get_samples(group_by_chain=False)

        for k, v in self.posterior_samples.items():
            print(k, v.shape)

    def make_prior_predictive(self, num_prior_samples=100):
        if self.prior_predictive is None:
            self.prior_predictive = Predictive(self.model, num_samples=num_prior_samples)

    def draw_prior_predictions(self):
        if self.prior_predictive is None:
            self.make_prior_predictive()
        rng_key, rng_key_ = random.split(self.rng_key)
        return self.prior_predictive(rng_key_, obs_x=None)

    def make_posterior_predictive(self):
        if self.posterior_samples is None:
            raise RuntimeError("You need to run the sampler first")
        if self.posterior_predictive is None:
            self.posterior_predictive = Predictive(self.model, self.posterior_samples)

    def draw_posterior_predictions(self, plot=False):
        if self.posterior_predictive is None:
            self.make_posterior_predictive()
        rng_key, rng_key_ = random.split(self.rng_key)
        return self.posterior_predictive(rng_key_, obs_x=None)