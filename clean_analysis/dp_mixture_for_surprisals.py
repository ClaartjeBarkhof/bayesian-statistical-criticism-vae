import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm
import seaborn as sns;

sns.set()
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


def plot_all_groups_preds_obs(self):
    post_preds = self.draw_posterior_predictions()

    N_groups = len(self.group_names)

    ncols = 5
    nrows = int(np.ceil(N_groups / 5))

    fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(3 * ncols, 2 * nrows))

    for g in range(N_groups):
        row, col = g // ncols, g % ncols

        preds_g = post_preds[:, self.obs_g == g]
        obs_g = self.obs_y[self.obs_g == g]

        axs[row, col].hist(np.array(preds_g).flatten(), bins=40, density=True, lw=0, label="preds", alpha=0.7,
                           color="blue")
        axs[row, col].hist(np.array(obs_g).flatten(), bins=40, density=True, lw=0, label="obs", alpha=0.7,
                           color="lightblue")

        axs[row, col].set_title(self.group_names[g], size=8)

        if (col == ncols - 1) and (row == 0):
            axs[row, col].legend(loc=(1.05, 0.8))

    plt.tight_layout()
    plt.show()


def plot_model_data_preds_obs(self):
    post_preds = self.draw_posterior_predictions()

    ncols = 3
    nrows = 1

    c_dict = {
        "data preds": "green",
        "data obs": "lightgreen",
        "model preds": "blue",
        "model obs": "lightblue"
    }

    fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(5 * ncols, 3 * nrows))

    data_group_id = self.group_names.index("data_group")

    preds_model_groups = post_preds[:, self.obs_g != data_group_id]
    obs_model_groups = self.obs_y[self.obs_g != data_group_id]

    preds_data_group = post_preds[:, self.obs_g == data_group_id]
    obs_data_group = self.obs_y[self.obs_g == data_group_id]

    axs[0].hist(np.array(preds_model_groups).flatten(), bins=40, density=True, lw=0, label="model preds",
                color=c_dict["model preds"], alpha=0.7)
    axs[0].hist(np.array(obs_model_groups).flatten(), bins=40, density=True, lw=0, label="model obs",
                color=c_dict["model obs"], alpha=0.7)
    axs[0].hist(np.array(preds_data_group).flatten(), bins=40, density=True, lw=0, label="data preds",
                color=c_dict["data preds"], alpha=0.7)
    axs[0].hist(np.array(obs_data_group).flatten(), bins=40, density=True, lw=0, label="data obs",
                color=c_dict["data obs"], alpha=0.7)

    axs[1].hist(np.array(preds_data_group).flatten(), bins=40, density=True, lw=0, label="data preds",
                color=c_dict["data preds"], alpha=0.7)
    axs[1].hist(np.array(obs_data_group).flatten(), bins=40, density=True, lw=0, label="data obs",
                color=c_dict["data obs"], alpha=0.7)

    axs[2].hist(np.array(preds_model_groups).flatten(), bins=40, density=True, lw=0, label="model preds",
                color=c_dict["model preds"], alpha=0.7)
    axs[2].hist(np.array(obs_model_groups).flatten(), bins=40, density=True, lw=0, label="model obs",
                color=c_dict["model obs"], alpha=0.7)

    axs[0].legend()
    axs[1].legend()
    axs[2].legend()

    axs[0].set_title("Model and data groups obs & preds")
    axs[1].set_title("Data group obs & preds")
    axs[2].set_title("Model group obs & preds")

    plt.tight_layout()
    plt.show()


def kl_component_dist_and_data_group_distance(self):
    # [N_s, N_g, N_c]
    omega = self.posterior_samples["omega"]
    data_group_id = self.group_names.index("data_group")
    omega_data_group = omega[:, data_group_id, :]

    omega_tensor = torch.FloatTensor(np.array(omega))
    omega_data_group_tensor = torch.FloatTensor(np.array(omega_data_group)).unsqueeze(1)

    omega_dists = td.Categorical(probs=omega_tensor)
    omega_data_group_dists = td.Categorical(probs=omega_data_group_tensor)

    # [N_s, N_g, N_c] -> [N_s, N_g] -> [N_g]
    kl = td.kl_divergence(omega_data_group_dists, omega_dists)
    kl_avg = kl.mean(axis=0)  # avg sample dim

    kl_order = np.argsort(kl_avg.numpy().flatten())
    labels_reorder = np.array(self.group_names)[kl_order]

    kl_comps_data_group = dict()
    for i in range(len(labels_reorder)):
        kl_comps_data_group[labels_reorder[i]] = kl_avg[kl_order][i].item()

    return kl_comps_data_group

def estimate_kl_densities_dp_mixture(dp_mixture):
    # beta (1000, 77, 2)
    # loc (1000, 3)
    # loc_z (1000, 3850)
    # omega (1000, 77, 3)
    # scale (1000, 3)
    # scale_z (1000, 3850)

    # --------------------------------------------------------------------------------------------
    # Sample from the posterior predictive of the data group

    # [S, G, T]
    omega = dp_mixture.posterior_samples["omega"]
    S, G, T = omega.shape

    # [S, T]
    loc, scale = dp_mixture.posterior_samples["loc"], dp_mixture.posterior_samples["scale"]

    idx_perm_1 = np.random.permutation(S)
    idx_perm_2 = np.random.permutation(S)

    data_group_idx = dp_mixture.group_names.index("data_group")

    # Use the first indices to sample mixing coefficients of the data group
    sampled_omega = omega[idx_perm_1, data_group_idx, :]
    assert sampled_omega.shape == (S, 3), f"shape should be (S, 3), currently: {sampled_omega.shape}"
    assert sampled_omega.sum(
        axis=-1).mean() == 1.0, f"the sum of the mixing weights should be 1, currently: {sampled_omega.sum(dim=-1).mean()}"
    sampled_components = td.Categorical(probs=torch.Tensor(np.array(sampled_omega))).sample().numpy()

    # Use the second indices to sample component parameters of the sampled components
    sampled_loc, sampled_scale = loc[idx_perm_2, sampled_components], scale[idx_perm_2, sampled_components]
    sample_q_x = td.Normal(loc=torch.Tensor(np.array(sampled_loc)),
                           scale=torch.Tensor(np.array(sampled_scale))).sample().numpy()

    # --------------------------------------------------------------------------------------------
    # Assess the samples under the densities of the data group
    idx_perm_1 = np.random.permutation(S)
    idx_perm_2 = np.random.permutation(S)

    sampled_omega = omega[idx_perm_1, data_group_idx, :]
    sampled_loc, sampled_scale = loc[idx_perm_2, :], scale[idx_perm_2, :]

    q_mix = td.Categorical(probs=torch.Tensor(np.array(sampled_omega)))
    q_comp = td.Normal(loc=torch.Tensor(np.array(sampled_loc)), scale=torch.Tensor(np.array(sampled_scale)))

    # [S, T]
    q_mixtures = td.MixtureSameFamily(q_mix, q_comp)
    # [S]
    log_q_samples = q_mixtures.log_prob(torch.Tensor(np.array(sample_q_x))).numpy()
    log_q_samples_avg = logsumexp(log_q_samples) - np.log(S)

    kl_density_est = dict()

    # --------------------------------------------------------------------------------------------
    # Assess the samples under the densities of the model groups
    for model_group in dp_mixture.group_names:
        if model_group == "data_group": continue
        model_group_idx = dp_mixture.group_names.index(model_group)

        idx_perm_1 = np.random.permutation(S)
        idx_perm_2 = np.random.permutation(S)

        sampled_omega = omega[idx_perm_1, model_group_idx, :]
        sampled_loc, sampled_scale = loc[idx_perm_2, :], scale[idx_perm_2, :]

        p_mix = td.Categorical(probs=torch.Tensor(np.array(sampled_omega)))
        p_comp = td.Normal(loc=torch.Tensor(np.array(sampled_loc)), scale=torch.Tensor(np.array(sampled_scale)))

        # [S, T]
        p_mixtures = td.MixtureSameFamily(p_mix, p_comp)
        # [S]
        log_p_samples = p_mixtures.log_prob(torch.Tensor(np.array(sample_q_x))).numpy()
        log_p_samples_avg = logsumexp(log_p_samples) - np.log(S)

        kl_est = float(log_q_samples_avg - log_p_samples_avg)

        kl_density_est[model_group] = kl_est

    return kl_density_est