import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

import pandas as pd
import torch
import numpy as np
import torch.distributions as td

import numpyro
from numpyro.infer import MCMC, NUTS, Predictive
import numpyro.distributions as dist
from jax import random
from jax.nn import logsumexp
import jax.numpy as jnp

class DPMixture:
    def __init__(self, group_names: list, observations: list, obs_dist="normal",
                 DP_alpha=1., num_comps=5, truncated_normal_low=0.0,
                 num_samples=1000, num_chains=1, num_warmup=100):

        self.DP_alpha = DP_alpha
        self.T = num_comps

        self.obs_dist = obs_dist


        self.truncated_normal_low = truncated_normal_low

        # [G]
        self.group_names = group_names
        self.G = len(group_names)

        assert self.G > 1, "only data with more than one group is supported for now"


        self.group_name_to_id = {n: i for i, n in enumerate(group_names)}
        self.group_id_to_name = {i: n for i, n in enumerate(group_names)}

        self.N = sum(len(y) for y in observations)

        # [N]
        self.obs_g = np.concatenate([[g] * len(y) for g, y in enumerate(observations)])
        self.obs_y = np.concatenate(observations)

        if self.obs_dist == "log_normal":
            assert np.all(self.obs_y > 0.0), \
                "for log normal, all values need to be > 0.0"

        if self.obs_dist == "truncated_normal":
            assert np.all(self.obs_y > truncated_normal_low), \
                f"for truncated_normal, all values need to be > truncated_normal_low ={truncated_normal_low}"

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

            elif self.obs_dist == "log_normal":
                max_y = max(self.obs_y)
                pre_loc = numpyro.sample("pre_loc", dist.Uniform(1, np.log(max_y)))
                scale = numpyro.sample("scale", dist.Uniform(0, 20))
                loc = numpyro.deterministic("loc", jnp.cumsum(pre_loc))

            # loc & scale for log normal, normal and student T
            elif self.obs_dist == "truncated_normal" or self.obs_dist == "normal" or self.obs_dist == "student_t":

                mean = np.mean(self.obs_y)
                std = np.std(self.obs_y)
                max_y = np.max(self.obs_y)
                min_y = np.min(self.obs_y)

                print("mean y:", mean, "std y:", std, "min_y:", min_y, "max y:", max_y)

                # [T]
                pre_loc = numpyro.sample('pre_loc', dist.Uniform(min_y, max_y))  # dist.Normal(mean, std)

                if min_y > 0:  # if False:
                    loc = numpyro.deterministic("loc", jnp.cumsum(pre_loc))
                else:
                    idx = jnp.argsort(pre_loc, -1)  # , -1
                    loc = numpyro.deterministic("loc", pre_loc[idx])

                # scale = numpyro.sample('scale', dist.Gamma(1, 10))
                scale = numpyro.sample('scale', dist.Uniform(0.1, std * 2.0))  # was 20.0
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
                sampled_y = numpyro.sample("y", dist.TruncatedNormal(low=self.truncated_normal_low,
                                                                     loc=loc_z, scale=scale_z), obs=y)

            else:
                raise NotImplementedError

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

    def draw_prior_predictions(self, num_samples=100):
        if self.prior_predictive is None:
            self.make_prior_predictive(num_prior_samples=num_samples)
        rng_key, rng_key_ = random.split(self.rng_key)
        return self.prior_predictive(rng_key_, y=None)

    def make_posterior_predictive(self):
        if self.posterior_samples is None:
            raise RuntimeError("You need to run the sampler first")
        if self.posterior_predictive is None:
            self.posterior_predictive = Predictive(self.model, self.posterior_samples)

    def draw_posterior_predictions(self, plot=False):
        if self.posterior_predictive is None:
            self.make_posterior_predictive()
        rng_key, rng_key_ = random.split(self.rng_key)
        return self.posterior_predictive(rng_key_, y=None)


def surprisal_dp_plot_checks(model, samples, plot_max_groups=5, bins=30, filter_vals_higher=1e6):
    if (samples > filter_vals_higher).any():
        print(f"Warning, values higher than {filter_vals_higher}, filtering those out.")

    for k in range(model.G):
        fig, ax = plt.subplots(ncols=4, figsize=(9, 2))

        # c = next(pal)
        c = "royalblue"

        yk = model.obs_y[model.obs_g == k]
        # [num_samples, num_data_points]
        yk_ = samples[:, model.obs_g == k]

        if (yk_ > filter_vals_higher).any():

            yk_filter = [x for sub_list in yk_.tolist() for x in sub_list if x < filter_vals_higher]

            yk_mean = [np.mean(sub_list) for sub_list in yk_filter]
            yk_std = [np.std(sub_list) for sub_list in yk_filter]
            yk_median = [np.median(sub_list) for sub_list in yk_filter]

        else:
            yk_mean = np.mean(yk_, 1)
            yk_std = np.std(yk_, 1)
            yk_median = np.median(yk_, 1)

        _ = ax[0].hist(yk_mean, bins=bins, color=c, label='pred' if k == 0 else None)
        _ = ax[0].axvline(np.mean(yk), color='black', linestyle='--', label='obs' if k == 0 else None)
        _ = ax[0].set_xlabel(f'E[Y{k}]')

        _ = ax[1].hist(yk_std, color=c, bins=bins)
        _ = ax[1].axvline(np.std(yk), color='black', linestyle='--')
        _ = ax[1].set_xlabel(f'std[Y{k}]')

        _ = ax[2].hist(yk_median, color=c, bins=bins)
        _ = ax[2].axvline(np.median(yk), color='black', linestyle='--')
        _ = ax[2].set_xlabel(f'median[Y{k}]')

        pvalues = np.mean(yk_ > yk, 1)
        _ = ax[3].hist(pvalues, bins=bins, color=c)
        _ = ax[3].set_xlabel(f'Pr(Y{k} > obs{k})')
        _ = ax[3].axvline(np.median(pvalues), color='black', linestyle=':', label='median' if k == 0 else None)

        plt.show()

        if k + 1 == plot_max_groups:
            break


def plot_all_groups_preds_obs(self, prior=False, num_prior_samples=400, filter_vals_higher=1e6, sharex=True, sharey=True):
    if prior:
        preds = self.draw_prior_predictions(num_samples=num_prior_samples)["y"]
    else:
        preds = self.draw_posterior_predictions()["y"]

    if (preds > filter_vals_higher).any():
        print(f"Warning, values higher than {filter_vals_higher}, filtering those out")

    N_groups = len(self.group_names)

    ncols = 5
    nrows = int(np.ceil(N_groups / 5))

    fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(3 * ncols, 2 * nrows), sharex=sharex, sharey=sharey)

    for g in range(N_groups):
        row, col = g // ncols, g % ncols

        preds_g = preds[:, self.obs_g == g]
        obs_g = self.obs_y[self.obs_g == g]

        preds_g = np.array(preds_g).flatten()
        if (preds_g > filter_vals_higher).any():
            preds_g = preds_g[preds_g < filter_vals_higher]

        axs[row, col].hist(preds_g, bins=40, density=True, lw=0, label="preds", alpha=0.7,
                           color="blue")
        axs[row, col].hist(np.array(obs_g).flatten(), bins=40, density=True, lw=0, label="obs", alpha=0.7,
                           color="lightblue")

        axs[row, col].set_title(self.group_names[g], size=8)

        if (col == ncols - 1) and (row == 0):
            axs[row, col].legend(loc=(1.05, 0.8))

    plt.tight_layout()
    plt.show()


def plot_model_data_preds_obs(self, prior=False, num_prior_samples=400, filter_vals_higher=1e6):
    if prior:
        preds = self.draw_prior_predictions(num_samples=num_prior_samples)["y"]
    else:
        preds = self.draw_posterior_predictions()["y"]

    if (preds > filter_vals_higher).any():
        print(f"Warning, values higher than {filter_vals_higher}, filtering those out.")

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

    preds_model_groups = preds[:, self.obs_g != data_group_id]
    obs_model_groups = self.obs_y[self.obs_g != data_group_id]

    preds_data_group = preds[:, self.obs_g == data_group_id]
    obs_data_group = self.obs_y[self.obs_g == data_group_id]

    preds_model_groups = np.array(preds_model_groups).flatten()
    if (preds_model_groups > filter_vals_higher).any():
        preds_model_groups = preds_model_groups[preds_model_groups < filter_vals_higher]

    axs[0].hist(preds_model_groups, bins=40, density=True, lw=0, label="model preds",
                color=c_dict["model preds"], alpha=0.7)
    axs[0].hist(np.array(obs_model_groups).flatten(), bins=40, density=True, lw=0, label="model obs",
                color=c_dict["model obs"], alpha=0.7)

    preds_data_group = np.array(preds_data_group).flatten()
    if (preds_data_group > filter_vals_higher).any():
        preds_data_group = preds_data_group[preds_data_group < filter_vals_higher]

    axs[0].hist(preds_data_group, bins=40, density=True, lw=0, label="data preds",
                color=c_dict["data preds"], alpha=0.7)
    axs[0].hist(np.array(obs_data_group).flatten(), bins=40, density=True, lw=0, label="data obs",
                color=c_dict["data obs"], alpha=0.7)

    axs[1].hist(preds_data_group, bins=40, density=True, lw=0, label="data preds",
                color=c_dict["data preds"], alpha=0.7)
    axs[1].hist(np.array(obs_data_group).flatten(), bins=40, density=True, lw=0, label="data obs",
                color=c_dict["data obs"], alpha=0.7)

    axs[2].hist(preds_model_groups, bins=40, density=True, lw=0, label="model preds",
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


def estimate_kl_densities_dp_mixture(dp_mixture, num_components=3):
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
    assert sampled_omega.shape == (S, num_components), f"shape should be (S, 3), currently: {sampled_omega.shape}"
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

def plot_divergences_data_model_groups(all_df, sort_on="kl_comp sum", plot_only=None, figsize=(20, 20)):
    was_cols = [c for c in all_df.columns if "wasserstein" in c]
    kl_dens_cols = [c for c in all_df.columns if "kl_dens" in c]
    kl_comp_cols = [c for c in all_df.columns if "kl_comp" in c]

    c_dict = {"unconditional_unconditional": "blue",
              "conditional_conditional": "green",
              "unconditional_conditional": "orange"}

    kl_dens_c_dict = {"kl_dens " + k: v for k, v in c_dict.items()}
    kl_comp_c_dict = {"kl_comp " + k: v for k, v in c_dict.items()}
    was_c_dict = {"wasserstein " + k: v for k, v in c_dict.items()}

    all_df = all_df.sort_values(sort_on)

    if plot_only is None:
        fig, axs = plt.subplots(ncols=3, figsize=figsize)

        df_select = all_df[kl_comp_cols]

        df_select.drop("kl_comp sum", axis=1).plot.barh(rot=0, lw=0, color=kl_comp_c_dict, stacked=True, ax=axs[0])
        axs[0].set_title("kl components")

        df_select = all_df[kl_dens_cols]
        df_select.drop("kl_dens sum", axis=1).plot.barh(rot=0, lw=0, color=kl_dens_c_dict, stacked=True, ax=axs[1])
        axs[1].set_title("kl densities")
        axs[1].set_yticks([])

        df_select = all_df[was_cols]
        df_select.drop("wasserstein sum", axis=1).plot.barh(rot=0, color=was_c_dict, lw=0, stacked=True, ax=axs[2])
        axs[2].set_title("wasserstein distances")
        axs[2].set_yticks([])

        plt.suptitle(f"Sorted by {sort_on}")

    elif plot_only == "kl_comp":
        fig, ax = plt.subplots(figsize=figsize)
        df_select = all_df[kl_comp_cols]
        df_select.drop("kl_comp sum", axis=1).plot.barh(rot=0, lw=0, color=kl_comp_c_dict, stacked=True, ax=ax)
        ax.set_title("kl components")

    elif plot_only == "kl_dens":
        fig, ax = plt.subplots(figsize=figsize)
        df_select = all_df[kl_dens_cols]
        df_select.drop("kl_dens sum", axis=1).plot.barh(rot=0, lw=0, color=kl_dens_c_dict, stacked=True, ax=ax)
        ax.set_title("kl density")

    elif plot_only == "wasserstein":
        fig, ax = plt.subplots(figsize=figsize)
        df_select = all_df[was_cols]
        df_select.drop("wasserstein sum", axis=1).plot.barh(rot=0, lw=0, color=was_c_dict, stacked=True, ax=ax)
        ax.set_title("wasserstein")
    else:
        raise ValueError("plot_only must be None, 'wasserstein', 'kl_dens' or 'kl_comp'")

def plot_divergences_data_model_groups_against_other_stat(all_df, global_stats_df, plot_against, plot_against_name=None, stat="kl_comp",
                                                          figsize=(12, 16)):
    assert stat in ["kl_comp", "kl_dens", "wasserstein"], f"stat {stat} not valid"
    assert stat + " sum" in all_df.columns, f"'{stat} sum' must be in all_df.columns to sort on"
    assert plot_against in global_stats_df.columns, f"{plot_against} must be in cols of global_stats_df"

    if plot_against_name is None:
        plot_against_name = plot_against

    was_cols = [c for c in all_df.columns if "wasserstein" in c]
    kl_dens_cols = [c for c in all_df.columns if "kl_dens" in c]
    kl_comp_cols = [c for c in all_df.columns if "kl_comp" in c]

    c_dict = {"unconditional_unconditional": "blue",
              "conditional_conditional": "green",
              "unconditional_conditional": "orange"}

    kl_dens_c_dict = {"kl_dens " + k: v for k, v in c_dict.items()}
    kl_comp_c_dict = {"kl_comp " + k: v for k, v in c_dict.items()}
    was_c_dict = {"wasserstein " + k: v for k, v in c_dict.items()}

    all_df = all_df.sort_values(stat + " sum")
    all_df = all_df[all_df.index != "data_group"]

    fig, axs = plt.subplots(ncols=2, figsize=figsize)

    # kl comp
    if stat == "kl_comp":
        df_select = all_df[kl_comp_cols]
        df_select.drop("kl_comp sum", axis=1).plot.barh(rot=0, lw=0, color=kl_comp_c_dict, stacked=True, ax=axs[0])
        axs[0].set_title("kl components")

    # kl dens
    elif stat == "kl_dens":
        df_select = all_df[kl_dens_cols]
        df_select.drop("kl_dens sum", axis=1).plot.barh(rot=0, lw=0, color=kl_dens_c_dict, stacked=True, ax=axs[0])
        axs[0].set_title("kl density")

    # wasserstein
    else:
        df_select = all_df[was_cols]
        df_select.drop("wasserstein sum", axis=1).plot.barh(rot=0, lw=0, color=was_c_dict, stacked=True, ax=axs[0])
        axs[0].set_title("wasserstein")

    s = global_stats_df.loc[df_select.index][plot_against]
    s.plot.barh(rot=0, lw=0, stacked=True, ax=axs[1])
    _ = axs[1].axes.yaxis.set_ticklabels([])
    axs[1].set_title(plot_against_name)


def plot_surprisal_dists_against_global_stat(global_stats_df, surprisal_values, sort_on, sort_name,
                                             dataset_name, latent_structure,
                                             xlims, ylims, bins=40, cm_name="gnuplot2", title_size=14, title_y=1.02,
                                             cm_shrink=0.6,
                                             subsample_nrows=None, sort_ascend=True, row_height=1.0):
    assert sort_on in global_stats_df.columns, f"{sort_on} must be in global_stats_df.columns"

    if subsample_nrows is not None:
        if len(global_stats_df) < subsample_nrows:
            print(f"Warning, subsample_nrows {subsample_nrows} < len(global_stats_df) {len(global_stats_df)}")

    model_cols = ['unconditional_unconditional', 'unconditional_conditional', 'conditional_conditional']
    model_col_names = ["-log p(x*|x) unconditional samples", "-log p(x*|x) conditional samples",
                       "-log p(x*|x, y*) conditional samples"]

    ncols = len(model_cols)

    group_names = global_stats_df.index.unique()
    for g in group_names:
        assert g in surprisal_values, f"{g} not in surprisal_values dict"

    ngroups = len(group_names)

    if subsample_nrows is None:
        nrows = ngroups
        every = 1
    else:
        nrows = subsample_nrows
        every = int(np.floor(ngroups / subsample_nrows))

    print(f"Rows: {nrows}, cols: {ncols}, every: {every}, n_groups: {ngroups}")

    hist_kwargs = dict(lw=0, alpha=0.7, density=True, bins=bins)

    # Sort
    df_sort_on = global_stats_df.sort_values(sort_on, ascending=sort_ascend)[sort_on]
    sort_on_labels, sort_on_values = df_sort_on.index, df_sort_on.values
    labels = [f"{l} | {sort_name}={v:.2f}" for l, v in zip(sort_on_labels, sort_on_values)]

    # Make colormap based on sort values
    minima, maxima = min(sort_on_values), max(sort_on_values)
    norm = matplotlib.colors.Normalize(vmin=minima, vmax=maxima, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=plt.get_cmap(cm_name))

    # Fig
    fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(ncols * 6, nrows * row_height))  #

    for col, col_name in enumerate(model_cols):
        row = 0
        for idx, group_name in enumerate(sort_on_labels):
            if idx % every != 0 or (row + 1) > nrows:
                continue

            if group_name not in surprisal_values:
                continue

            color = mapper.to_rgba(sort_on_values[idx])

            axs[row, col].hist(surprisal_values["data_group"][model_cols[col]], color="grey", **hist_kwargs,
                               label="data group")
            axs[row, col].hist(surprisal_values[group_name][model_cols[col]], color=color, **hist_kwargs)

            if col == 0:
                axs[row, col].text(-0.05, .5, labels[idx], color='black', fontsize=10, ha="right", va="center",
                                   transform=axs[row, col].transAxes)

            # axs[row, col].set_yticks([])
            # axs[row, col].set_xticks([])
            if xlims[col] is not None:
                axs[row, col].set_xlim(xlims[col])
            if ylims[col] is not None:
                axs[row, col].set_ylim(ylims[col])

            # only show ticks for bottom row
            if row + 1 < nrows:
                axs[row, col].axes.xaxis.set_ticklabels([])

            axs[row, col].axes.yaxis.set_ticklabels([])

            if row == 0:
                axs[row, col].set_title(model_col_names[col])

            row += 1

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    axs[0, 2].legend(loc=(1.05, 0.2))

    fig.colorbar(mapper, ax=axs[:, 2], shrink=cm_shrink, location='right', anchor=(1.0, 0.0), pad=(0.01))

    if subsample_nrows is not None:
        plt.suptitle(
            f"{dataset_name} | - log p(x) under {latent_structure} | coloured by {sort_name}\nsubsampled {subsample_nrows}/{ngroups} spread over {sort_name} order",
            y=title_y, size=title_size)
    else:
        plt.suptitle(f"{dataset_name} | - log p(x) under {latent_structure} | coloured by {sort_name}", y=title_y,
                     size=title_size)

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
    kl_all = kl.permute(1, 0)  # [N_g, N_s]
    kl_avg = kl.mean(axis=0)  # avg sample dim

    kl_order = np.argsort(kl_avg.numpy().flatten())
    labels_reorder = np.array(self.group_names)[kl_order]

    kl_comps_data_group_avg = dict()
    kl_comps_data_group_dists = dict()
    for i in range(len(labels_reorder)):
        group_idx = kl_order[i]

        kl_comps_data_group_avg[labels_reorder[i]] = kl_avg[group_idx].item()
        kl_comps_data_group_dists[labels_reorder[i]] = kl_all[group_idx].numpy()

    return kl_comps_data_group_avg, kl_comps_data_group_dists


def compute_all_divergences_data_model_groups(dp_mixtures, surprisal_values, num_components=3):
    from scipy import stats

    kl_component_assignments_all_dps = dict()

    kl_comp_dists = {}

    for stat, dp_mixture in dp_mixtures.items():
        kl_comps_data_group_avg, kl_comps_data_group_dists = kl_component_dist_and_data_group_distance(dp_mixture)
        kl_component_assignments_all_dps["kl_comp " + stat] = kl_comps_data_group_avg
        kl_comp_dists[stat] = kl_comps_data_group_dists

    kl_component_assignments_all_dps_df = pd.DataFrame(kl_component_assignments_all_dps)
    kl_component_assignments_all_dps_df["kl_comp sum"] = kl_component_assignments_all_dps_df.sum(axis=1)

    kl_densities_all_dps = dict()
    for stat, dp_mixture in dp_mixtures.items():
        kl_density_est = estimate_kl_densities_dp_mixture(dp_mixture, num_components=num_components)
        kl_densities_all_dps["kl_dens " + stat] = kl_density_est

    kl_densities_all_dps_df = pd.DataFrame(kl_densities_all_dps)
    kl_densities_all_dps_df["kl_dens sum"] = kl_densities_all_dps_df.sum(axis=1)

    wass_dists = dict()

    for k, v in surprisal_values.items():
        if k is not "data_group":
            wass_dists[k] = dict()
            for stat_name, stat in v.items():
                w = stats.wasserstein_distance(stat, surprisal_values["data_group"][stat_name])
                wass_dists[k]["wasserstein " + stat_name] = w

    wasserstein_df = pd.DataFrame(wass_dists).transpose()
    wasserstein_df["wasserstein sum"] = wasserstein_df.sum(axis=1)

    all_df = wasserstein_df.join(kl_component_assignments_all_dps_df).join(kl_densities_all_dps_df)

    return all_df, kl_comp_dists