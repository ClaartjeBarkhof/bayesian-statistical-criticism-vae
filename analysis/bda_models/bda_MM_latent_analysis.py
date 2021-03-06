import random
import numpy as np
from matplotlib import pyplot as plt
from torch_two_sample import MMDStatistic

import torch
import torch.distributions as td

import pyro

import sys
sys.path.append("/home/cbarkhof/fall-2021/analysis/bda_models/Pyro_BDA/probabll/bda")

import pandas as pd
from mmm import Family, MixedMembershipRD, Plotting
import scipy.stats as st
import torch_two_sample as tts

class MixedMembershipLatentAnalysis:
    def __init__(self, run_names, all_latents, clean_names=None, device="cuda:0",
                 seed=0, num_components=6, DP_alpha=0.1):

        self.set_random_state(seed=seed)
        self.all_encodings = dict()
        self.device = device

        self.DP_alpha = DP_alpha

        self.Sx, self.G, self.D = all_latents.shape
        self.all_latents = all_latents

        self.obs_g = np.concatenate([[g] * self.Sx for g in range(self.G)])

        assert self.G == len(run_names), "len(run_names) must be equal to the number of groups inferred from all_latents"

        self.clean_names = list(clean_names) if clean_names is not None else list(run_names)

        self.model = self.get_model(num_components=num_components)
        self.posterior = None

    def set_random_state(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        pyro.set_rng_seed(seed)
        torch.manual_seed(seed)
        # rng = np.random.RandomState(0)

    def get_model(self, num_components=10):
        model = MixedMembershipRD(
            Family.LowRankMVN(
                dim=self.D, rank=1,
                mu_loc=0., mu_scale=10.,
                cov_diag_alpha=0.1, cov_diag_beta=1.,
                cov_factor_loc=0., cov_factor_scale=10.,
                delta=True),
            T=num_components,
            DP_alpha=self.DP_alpha,
            device=torch.device(self.device))
        return model

    def fit_mm(self, plot_elbo=True, n_iterations=200):
        print(f"Sx={self.all_latents.shape[0]}, G={self.all_latents.shape[1]}, D={self.all_latents.shape[2]}")
        x = self.model.prepare(self.all_latents)  # .cpu().numpy() <- does it need to be numpy?
        self.model.fit(x, n_iterations, lr=0.01, clip_norm=10.)

        if plot_elbo:
            Plotting.elbo(self.model)

    def plot_component_dist_groups(self, posterior_predict_n_samples=1000, figsize=(8, 8)):
        if self.posterior is None:
            with torch.no_grad():
                self.posterior = self.model.posterior_predict(num_samples=posterior_predict_n_samples)

        plt.rcParams["axes.grid"] = False

        omega = self.posterior["omega"]
        omega_avg = omega.mean(0).squeeze(0).squeeze(0).detach().cpu().numpy()

        fig, axs = plt.subplots(ncols=2, gridspec_kw={'width_ratios': [1, 0.05]}, figsize=figsize)
        im = axs[0].imshow(omega_avg)  # , cmap=cmap

        axs[0].set_yticks(range(len(self.clean_names)))
        axs[0].set_yticklabels(self.clean_names, size=8)
        axs[0].set_xlabel("Component i")

        plt.colorbar(im, cax=axs[1])
        plt.tight_layout()

        plt.axis("off")
        plt.suptitle("Component distribution ?? for different groups\nin mixed membership model in $R^D$", y=1.1)
        plt.show()

    def approximate_log_q_z(self, posterior_predict_n_samples=1000):
        if self.posterior is None:
            self.posterior = self.model.posterior_predict(num_samples=posterior_predict_n_samples)

        # ----------  --------------------------------
        # mu
        # torch.Size([1000, 1, 10, 10])
        # cov_factor
        # torch.Size([1000, 1, 10, 10, 1])
        # cov_diag
        # torch.Size([1000, 1, 10, 10])
        # beta
        # torch.Size([1000, 1, 9, 9])
        # z
        # torch.Size([1000, 1000, 9])
        # obs
        # torch.Size([1000, 1000, 9, 10])
        # omega
        # torch.Size([1000, 1, 1, 9, 10])
        # ----------  --------------------------------

        log_q_zs = dict()
        for g, cn in enumerate(self.clean_names):
            # omega [S, 1, 1, G, T] -> logits [S, T]
            # batch of S mixtures all of T components

            mix = td.Categorical(logits=self.posterior["omega"][:, 0, 0, g, :])  # S, T

            # print("mix.logits.shape", mix.logits.shape)

            # component
            mu, cov_fact, cov_diag = self.posterior["mu"].squeeze(1), self.posterior["cov_factor"].squeeze(1), \
                                     self.posterior["cov_diag"].squeeze(1)
            comp = td.LowRankMultivariateNormal(loc=mu, cov_factor=cov_fact, cov_diag=cov_diag)

            Ns = self.posterior["omega"].shape[0]

            # print("omega", self.posterior["omega"].shape)
            # print("mix", mix)
            # print("mu.shape, cov_diag.shape, cov_fact.shape", mu.shape, cov_diag.shape, cov_fact.shape)

            batched_mixture = td.MixtureSameFamily(mix, comp)

            # print("mu.shape, cov_fact.shape, cov_diag.shape", mu.shape, cov_fact.shape, cov_diag.shape)

            z_post = self.all_latents[:, g, :].unsqueeze(1)
            #print("z_post.shape", z_post.shape) # [Sx, D]

            # [Sx, S_post] -> float
            log_q_z = batched_mixture.log_prob(z_post) # [Sx, num_samples]
            log_q_z = torch.logsumexp(log_q_z, dim=1) - np.log(Ns)
            log_q_z = log_q_z.mean()

            log_q_zs[cn] = log_q_z.item()

        return log_q_zs

    def get_log_p_z(self):
        log_p_zs = dict()
        for g, cn in enumerate(self.clean_names):
            log_p_z = td.Normal(loc=0.0, scale=1.0).log_prob(self.all_latents[:, g, :]).sum(dim=-1).mean()
            log_p_zs[cn] = log_p_z.item()
        return log_p_zs

    def approximate_marginal_kl(self):
        log_p_zs = self.get_log_p_z()
        log_q_zs = self.approximate_log_q_z()

        # assert log_p_zs.shape == log_q_zs, "Shape of log p z and log q z should be the same."

        approx_marginal_kl = dict()
        for cn in log_p_zs.keys():
            approx_marginal_kl[cn] = log_q_zs[cn] - log_p_zs[cn]

        return approx_marginal_kl

    def get_mmd_encodings(self):
        tts_mmd = dict()

        for g, cn in enumerate(self.clean_names):
            z_post = self.all_latents[:, g, :]
            prior_sample = torch.randn_like(z_post)

            alphas = [0.1 * i for i in range(5)]  # TODO: no clue for these...

            n_1, n_2 = len(z_post), len(prior_sample)

            # print("n_1", n_1, "n_2", n_2)
            # print(prior_sample.shape)
            # print(z_post.shape)

            mmd_stat = MMDStatistic(n_1, n_2)
            tts_mmd[cn] = mmd_stat(z_post, prior_sample, alphas, ret_matrix=False)

        return tts_mmd

def kl_divergence(mu, std):
    """
    Calculates the KL-divergence between the posterior and the prior analytically.
    """

    kl_loss = ((mu.pow(2) + std.pow(2)) / 2) - 0.5 - torch.log(std)
    kl_loss = kl_loss.sum(dim=1).mean(dim=0)

    return kl_loss

def gather_alternative_statistics(encodings, N_encodings=2000, latent_dim=10):
    alphas = [0.1 * i for i in range(1, 6)]

    MMD = tts.statistics_diff.MMDStatistic(N_encodings, N_encodings)

    prior_data = torch.randn(size=(N_encodings, latent_dim))

    res = {}

    for i, (name, enc) in enumerate(encodings.items()):
        print(f"{i}/{len(encodings)} - {name}")
        z = enc["z"][:N_encodings]
        loc = enc["mean"][:N_encodings]
        scale = enc["scale"][:N_encodings]

        if len(z) < N_encodings:
            print(f"Warning, N_encodings < len(z), N_encodings={N_encodings}, len(z)={len(z)}")

        rate = kl_divergence(mu=loc, std=scale)

        one_sample_ks = st.kstest(z.flatten().cpu().numpy(), 'norm')

        # Kolmogorov-Smirnov statistic on 2 samples
        two_sample_ks = st.ks_2samp(prior_data.flatten().cpu().numpy(), z.flatten().cpu().numpy())

        mmd = MMD(prior_data, z, alphas=alphas, ret_matrix=False)

        res[name] = {
            "mmd": mmd.cpu().item(),
            "one_sample_ks": one_sample_ks.statistic,
            "two_sample_kl": two_sample_ks.statistic,
            "rate": rate.item(),
        }

    return res

def prepare_data(encoding_dict, N_encodings=2000, latent_dim=10, device="cpu"):
    group_names = []
    data_samples = []

    for k, v in encoding_dict.items():
        enc = v["z"][:N_encodings]
        assert enc.shape[-1] == latent_dim, f"you are passing encodings of wrong dimensionality, latent_dim = " \
                                            f"{latent_dim}, you passed dim = {enc.shape[-1]}"
        data_samples.append(enc)
        group_names.append(k)

    # Prepend standard prior group
    data_samples = [torch.randn((N_encodings, latent_dim))] + data_samples
    group_names = ["prior"] + group_names

    # stack
    data_samples = torch.stack([t.to(device) for t in data_samples])

    G, S, D = data_samples.shape
    print(f"G={G} S={S} D={D}")

    return data_samples, group_names