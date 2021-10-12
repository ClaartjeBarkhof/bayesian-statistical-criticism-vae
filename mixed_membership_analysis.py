import os
import random
import numpy as np
import matplotlib.pyplot as plt
import arviz
from utils import load_checkpoint_model_for_eval
from analysis import collect_encodings
from matplotlib import pyplot as plt
from torch_two_sample import MMDStatistic

import torch
import torch.distributions as td
import torch.nn.functional as F

from tqdm.auto import tqdm
from tabulate import tabulate
from collections import OrderedDict, namedtuple

import pyro
import pyro.distributions as pd
from pyro import poutine

from bda.probabll.bda.mmm import Family, MixedMembershipRD, Plotting

class MixedMembershipLatentAnalysis:
    def __init__(self, run_names, clean_names=None, data_X=None, device="cuda:0", seed=0, num_components=6,
                 save_encodings=True, code_dir="/home/cbarkhof/fall-2021", add_prior_group=True):

        self.set_random_state(seed=seed)
        self.all_encodings = dict()
        self.device = device

        self.Sx = len(data_X)
        self.G = len(run_names)

        self.checkpoints = [f"{code_dir}/run_files/checkpoints/{n}.pt" for n in run_names]
        self.clean_names = list(clean_names) if clean_names is not None else list(run_names)

        for rn, cn, cp in zip(run_names, clean_names, self.checkpoints):
            enc_p = f"{code_dir}/analysis-files/encodings-N={self.Sx}-{rn}.pt"

            # Load encoded data
            if os.path.isfile(enc_p):
                # print(f"Loading encodings from {enc_p}")
                self.all_encodings[cn] = {k: v.squeeze(0).to(device) for k, v in torch.load(enc_p).items()}

            # Encode data
            else:
                # print(f"Predicting encodings and saving to {enc_p}")
                vae_model = load_checkpoint_model_for_eval(cp, map_location=device, return_args=False)

                # collect_encodings returns a dict with keys:
                # z_post, mean_post, scale_post
                self.all_encodings[cn] = collect_encodings(vae_model, data_X=data_X, Sz=1)
                self.all_encodings[cn] = {k: v.squeeze(0).to(device) for k, v in self.all_encodings[cn].items()}

                if save_encodings:
                    torch.save(self.all_encodings[cn], enc_p)

        self.D = self.get_latent_dim()

        if add_prior_group:
            self.all_encodings["prior"] = dict(
                z_post=torch.randn((self.Sx, self.D)).to(device),
                mean_post=torch.zeros((self.Sx, self.D)).to(device),
                scale_post=torch.ones((self.Sx, self.D)).to(device)
            )
            self.clean_names = ["prior"] + self.clean_names

        self.all_latents = torch.stack([e["z_post"].squeeze(0) for e in self.all_encodings.values()], dim=1)

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
            DP_alpha=0.1,
            device=torch.device(self.device))
        return model

    def get_latent_dim(self):
        latent_dims = [e["z_post"].shape[-1] for e in self.all_encodings.values()]
        assert len(set(latent_dims)) == 1, "checkpoints are of different latent dimensionality"
        D = latent_dims[0]
        return D

    def fit_mm(self, plot_elbo=True, n_iterations=200):
        print(f"Sx={self.all_latents.shape[0]}, G={self.all_latents.shape[1]}, D={self.all_latents.shape[2]}")
        x = self.model.prepare(self.all_latents)  # .cpu().numpy() <- does it need to be numpy?
        self.model.fit(x, n_iterations, lr=0.01, clip_norm=10.)

        if plot_elbo:
            Plotting.elbo(self.model)

    def plot_component_dist_groups(self):
        if self.posterior is None:
            self.posterior = self.model.posterior_predict(num_samples=1000)

        plt.rcParams["axes.grid"] = False

        omega = self.posterior["omega"]
        omega_avg = omega.mean(0).squeeze(0).squeeze(0).detach().cpu().numpy()

        fig, axs = plt.subplots(ncols=2, gridspec_kw={'width_ratios': [1, 0.05]}, figsize=(4, 4))
        im = axs[0].imshow(omega_avg)  # , cmap=cmap

        axs[0].set_yticks(range(len(self.clean_names)))
        axs[0].set_yticklabels(list(self.all_encodings.keys()))
        axs[0].set_xlabel("Component i")

        plt.colorbar(im, cax=axs[1])
        plt.tight_layout()

        plt.axis("off")
        plt.suptitle("Component distribution θ for different groups\nin mixed membership model in $R^D$", y=1.1)
        plt.show()

    def approximate_log_q_z(self):
        if self.posterior is None:
            self.posterior = self.model.posterior_predict(num_samples=1000)

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
            mu, cov_fact, cov_diag = self.posterior["mu"].squeeze(1), self.posterior["cov_factor"].squeeze(1), self.posterior["cov_diag"].squeeze(1)
            comp = td.LowRankMultivariateNormal(loc=mu, cov_factor=cov_fact, cov_diag=cov_diag)
            batched_mixture = td.MixtureSameFamily(mix, comp)

            # print("mu.shape, cov_fact.shape, cov_diag.shape", mu.shape, cov_fact.shape, cov_diag.shape)

            z_post = self.all_encodings[cn]["z_post"].unsqueeze(1)
            # print("z_post.shape", z_post.shape)

            # [Sx, S_post] -> float
            log_q_z = batched_mixture.log_prob(z_post).mean()
            log_q_zs[cn] = log_q_z.item()

        return log_q_zs

    def get_log_p_z(self):
        log_p_zs = dict()
        for g, cn in enumerate(self.clean_names):
            log_p_z = td.Normal(loc=0.0, scale=1.0).log_prob(self.all_encodings[cn]["z_post"]).sum(dim=-1).mean()
            log_p_zs[cn] = log_p_z
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

        for k, e in self.all_encodings.items():
            z_post = e["z_post"].squeeze(0)
            prior_sample = torch.randn_like(z_post)

            alphas = [0.1 * i for i in range(5)]  # TODO: no clue for these...

            n_1, n_2 = len(z_post), len(prior_sample)

            # print("n_1", n_1, "n_2", n_2)
            # print(prior_sample.shape)
            # print(z_post.shape)

            mmd_stat = MMDStatistic(n_1, n_2)
            tts_mmd[k] = mmd_stat(z_post, prior_sample, alphas, ret_matrix=False)

        return tts_mmd
