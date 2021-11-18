import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm
import seaborn as sns; sns.set()
import arviz as az

import numpy as np
from tabulate import tabulate
import pandas as pd
import pickle
import os
import torch
import numpy as np
from scipy import stats
from sklearn import preprocessing

import torch.distributions as td

import numpyro
from numpyro.infer import MCMC, NUTS, Predictive
import numpyro.distributions as dist
from jax import random
from jax.nn import logsumexp
import jax.numpy as jnp

class DPMixture:
    def __init__(self, group_names: list, observations: list, obs_dist="normal",
                 DP_alpha=1., num_comps=5,
                 num_samples=1000, num_chains=1, num_warmup=100):

        self.DP_alpha = DP_alpha
        self.T = num_comps

        self.obs_dist = obs_dist

        # [G]
        self.group_names = group_names
        self.G = len(group_names)

        self.group_name_to_id = {n: i for i, n in enumerate(group_names)}
        self.group_id_to_name = {i: n for i, n in enumerate(group_names)}

        self.N = sum(len(y) for y in observations)

        # [N]
        self.obs_g = np.concatenate([[g] * len(y) for g, y in enumerate(observations)])
        self.obs_y = np.concatenate(observations)

        # print("obs g shape", self.obs_g.shape)
        # print("obs y shape", self.obs_y.shape)

        self.num_samples = num_samples
        self.num_chains = num_chains
        self.num_warmup = num_warmup

        self.rng_key = random.PRNGKey(0)

        nuts_kernel = NUTS(self.model)
        self.mcmc = MCMC(nuts_kernel, num_samples=num_samples, num_warmup=num_warmup, num_chains=num_chains)

        self.prior_predictive = None
        self.posterior_predictive = None
        self.posterior_samples = None

    @classmethod
    def mix_weights(cls, beta):
        beta1m_cumprod = (1 - beta).cumprod(-1)
        return jnp.pad(beta, ((0, 0), (0, 1)), constant_values=1) * jnp.pad(beta1m_cumprod, ((0, 0), (1, 0)),
                                                                            constant_values=1)

    @classmethod
    def stick_break_sorting(cls, beta):
        beta1m_cumprod = (1 - beta).cumprod(-1)
        return jnp.pad(beta, (0, 1), constant_values=1) * jnp.pad(beta1m_cumprod, (1, 0), constant_values=1)

    def model(self, y=None):
        """
        Pyro joint distribution.

        Parameter:

        y: observations as returned by self.prepare or None.
            If None, the 'obs' node of the graphical model will be resampled.
        """

        N, G, T = self.N, self.G, self.T

        loc, scale, df = 0.0, 0.0, 0.0

        # Components
        with numpyro.plate("components", T):
            if self.obs_dist == "binomial":
                lamb = numpyro.sample("lamb", dist.Beta(1.0, 1.0))
                probs = numpyro.deterministic("probs", self.stick_break_sorting(lamb))

                # probs = numpyro.deterministic("probs", jnp.cumprod(lamb, axis=-1))
                # probs = numpyro.deterministic("probs", jnp.sort(lamb, axis=-1))
                # probs = jnp.sort(probs, axis=-1)

            # loc & scale for log normal, normal and student T
            elif "normal" in self.obs_dist or self.obs_dist == "student_t":
                if self.obs_dist == "log_normal":
                    mean = 0.0
                else:
                    mean = np.mean(self.obs_y)

                # [T]
                loc = numpyro.sample('loc', dist.Normal(mean, 1.0))
                idx = jnp.argsort(loc, -1)  # , -1
                loc = loc[idx]

                # scale = numpyro.sample('scale', dist.Gamma(1, 10))
                scale = numpyro.sample('scale', dist.Uniform(0.1, 20))
                # scale = scale[idx]

                # degrees of freedom
                if self.obs_dist == "student_t":
                    df = numpyro.sample("student_df", dist.Exponential(rate=1.0 / 10.0))
                    # df = df[idx]

            else:
                raise NotImplementedError

        # Sample mixing weights
        with numpyro.plate("DPs", G):
            # [G, T-1]
            beta = numpyro.sample("beta",
                                  dist.Beta(np.ones(1), np.ones(1) * self.DP_alpha).expand((T - 1,)).to_event(1))

        # [G, T]
        omega = numpyro.deterministic("omega", self.mix_weights(beta))

        loc_z, scale_z, df_z = 0.0, 0.0, 0.0
        with numpyro.plate("observations", N):
            # Choose component
            z = numpyro.sample("z", dist.Categorical(probs=omega[self.obs_g]))

            if "normal" in self.obs_dist or self.obs_dist == "student_t":
                loc_z = numpyro.deterministic("loc_z", loc[z])
                scale_z = numpyro.deterministic("scale_z", scale[z])

                if self.obs_dist == "student_t":
                    df_z = numpyro.deterministic("df_z", df[z])

            elif self.obs_dist == "binomial":
                probs_z = numpyro.deterministic("probs_z", probs[z])

            # Construct the likelihood function
            if self.obs_dist == "normal":
                sampled_y = numpyro.sample("y", dist.Normal(loc=loc_z, scale=scale_z), obs=y)

            elif self.obs_dist == "log_normal":
                sampled_y = numpyro.sample("y", dist.LogNormal(loc=loc_z, scale=scale_z), obs=y)

            elif self.obs_dist == "student_t":
                sampled_y = numpyro.sample("y", dist.StudentT(loc=loc_z, scale=scale_z, df=df_z), obs=y)

            elif self.obs_dist == "binomial":
                sampled_y = numpyro.sample("y", dist.Binomial(probs=probs_z, total_count=28 * 28), obs=y)

            elif self.obs_dist == "truncated_normal":
                sampled_y = numpyro.sample("y", dist.TruncatedNormal(low=0.0, loc=loc_z, scale=scale_z), obs=y)

            else:
                raise NotImplementedError

            # print("sampled y", sampled_y.shape)

            return sampled_y

    def run(self):
        self.mcmc.run(self.rng_key, y=self.obs_y)
        self.mcmc.print_summary()
        self.posterior_samples = self.mcmc.get_samples(group_by_chain=False)
        print("posterior samples shapes:")
        for k, v in self.posterior_samples.items():
            print(k, v.shape)

    def make_prior_predictive(self, num_prior_samples=100):
        if self.prior_predictive is None:
            self.prior_predictive = Predictive(self.model, num_samples=num_prior_samples)

    def draw_prior_predictions(self):
        if self.prior_predictive is None:
            self.make_prior_predictive()
        rng_key, rng_key_ = random.split(self.rng_key)
        return self.prior_predictive(rng_key_, y=None)["y"]

    def make_posterior_predictive(self):
        if self.posterior_samples is None:
            raise RuntimeError("You need to run the sampler first")
        if self.posterior_predictive is None:
            self.posterior_predictive = Predictive(self.model, self.posterior_samples)

    def draw_posterior_predictions(self, plot=False):
        if self.posterior_predictive is None:
            self.make_posterior_predictive()
        rng_key, rng_key_ = random.split(self.rng_key)
        return self.posterior_predictive(rng_key_, y=None)["y"]