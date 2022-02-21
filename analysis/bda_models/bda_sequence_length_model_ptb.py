import torch
import torch.distributions as td

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from itertools import cycle

import numpyro
import numpyro.distributions as dist
from numpyro.contrib.indexing import Vindex
from numpyro.infer import MCMC, NUTS, Predictive

from jax import random
import jax.numpy as jnp
import jax

NUM_CHAINS = 1
numpyro.set_host_device_count(NUM_CHAINS)
print(f"Running on NumPryo v{numpyro.__version__}")

palette = cycle(sns.color_palette())
color_m = next(palette)
color_f = next(palette)

class GenSeqLenModelPTB:
    def __init__(self, group_names: list, observations: list,
                 N_valid=3370,
                 gamma_shape=[1., 1.], DP_alpha=1., num_comps=5,
                 num_samples=1000, num_chains=1, num_warmup=100):

        self.gamma_shape = gamma_shape
        self.DP_alpha = DP_alpha
        self.T = num_comps

        self.N_valid=N_valid

        self.group_names = group_names
        self.G = len(group_names)

        self.N = sum(len(y) for y in observations)
        # [N]
        self.x = np.concatenate([[g] * len(y) for g, y in enumerate(observations)])
        self.y = np.concatenate(observations)

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
    def mix_weights1d(cls, beta):
        beta1m_cumprod = (1 - beta).cumprod(-1)
        return jnp.pad(beta, ((0, 1),), constant_values=1) * jnp.pad(beta1m_cumprod, ((1, 0),), constant_values=1)

    def model(self, y=None):
        """
        Pyro joint distribution.

        Parameter:

        y: observations as returned by self.prepare or None.
            If None, the 'obs' node of the graphical model will be resampled.
        """

        N, G = self.N, self.G
        T = self.T

        with numpyro.plate("components", T):
            # construct the components
            # [T]
            lamb = numpyro.sample("lambda", dist.Gamma(self.gamma_shape[0], self.gamma_shape[1]))
            rate = numpyro.deterministic("rate", jnp.cumsum(lamb))  # we could use jnp.sort, but cumsum is simpler

        if G > 1:
            # Sample mixing weights
            with numpyro.plate("DPs", G):
                # [G, T-1]
                beta = numpyro.sample(
                    "beta",
                    dist.Beta(
                        np.ones(1),
                        np.ones(1) * self.DP_alpha
                    ).expand((T - 1,)).to_event(1)
                )
            # [G, T]
            omega = numpyro.deterministic("omega", self.mix_weights(beta))
            # [N, T]
            omega_x = numpyro.deterministic("omega_x", omega[self.x])
        elif G == 1:
            # [T-1]
            beta = numpyro.sample(
                "beta",
                dist.Beta(
                    np.ones(1),
                    np.ones(1) * self.DP_alpha
                ).expand((T - 1,)).to_event(1)
            )
            # [T]
            omega = numpyro.deterministic("omega", self.mix_weights1d(beta))
            # [N, T]
            omega_x = Vindex(jnp.expand_dims(omega, -2))[jnp.zeros_like(self.x)]
        else:
            raise ValueError("I need at least 1 group")

        with numpyro.plate("observations", N):
            # [N]
            z = numpyro.sample("z", dist.Categorical(probs=omega_x))

            # [N]
            # To avoid confusion, I'm no longer creating a deterministic size for rate_z
            #  rate_z = numpyro.deterministic("rate_z", rate[z])
            rate_z = rate[z]
            # [N]
            # Construct the likelihood function
            return numpyro.sample("y", dist.Poisson(rate_z), obs=y)

    def run(self):
        self.mcmc.run(self.rng_key, y=self.y)
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
        return self.prior_predictive(rng_key_, y=None)

    def make_posterior_predictive(self):
        if self.posterior_samples is None:
            raise RuntimeError("You need to run the sampler first")
        if self.posterior_predictive is None:
            self.posterior_predictive = Predictive(self.model, self.posterior_samples, infer_discrete=False)

    def draw_posterior_predictions(self, plot=False):
        if self.posterior_predictive is None:
            self.make_posterior_predictive()
        rng_key, rng_key_ = random.split(self.rng_key)
        return self.posterior_predictive(rng_key_, y=None)

    def _select_local_rates(self, rates, z):
        """
        rates [num_samples, num_components]: these are positive scalars
        z [num_samples, data_size]: these are integers in [0, num_components)
        output: [num_samples, data_size] positive scalars
            which are the rates selected by z for each data point

        z should come from Predictive with infer_discrete=True and conditioned on observations
        """
        T = self.T
        return (jax.nn.one_hot(z, T) * jnp.expand_dims(rates, -2)).sum(-1)

    def infer_local_assignments(self, posterior_samples, y):
        """
        Annotate y with z given posterior samples and y.

        Common use: infer_local_assignments(model.posterior_samples, model.y)

        Return: dictionary with local assignments
            * z: [num_samples, data_size]
            * rate_z: [num_samples, data_size]
        """
        if self.posterior_samples is None:
            raise ValueError("self.posterior_samples is None, have you run the model?")
        if y is None:
            raise ValueError("I need y samples (or observations) that are paired with posterior samples")

        rng_key, rng_key_ = random.split(self.rng_key)
        predictive = Predictive(self.model, posterior_samples, infer_discrete=True)
        discrete_samples = predictive(rng_key_, y=y)
        z = discrete_samples['z']
        rates_z = self._select_local_rates(posterior_samples['rate'], z)
        return {'z': z, 'rate_z': rates_z}

    def estimate_log_p_x(self, lengths, posterior_samples=None):
        if posterior_samples is None:
            posterior_samples = self.draw_posterior_predictions()

        # [S, T]
        omega = np.array(posterior_samples["omega"])

        # [S, T]
        omega_x = torch.FloatTensor(omega)

        # [1, S, T]
        mix = td.Categorical(probs=omega_x[None, :, :])

        # [S, T]
        rates = torch.FloatTensor(np.array(posterior_samples["rate"]))

        # [1, S, T]
        comp = td.Poisson(rate=rates[None, :, :])

        # [S, T]
        mix_model = td.MixtureSameFamily(mix, comp)

        # [N, 1]
        lengths = torch.Tensor(lengths)[:, None]

        # [N, S]
        log_prob = mix_model.log_prob(lengths)
        N, S = log_prob.shape

        # [N]
        log_prob = torch.logsumexp(log_prob, dim=1) - np.log(S)

        return log_prob.numpy()

    def estimate_log_p_x_conditional(self, lengths, local_samples=None):
        if local_samples is None:
            local_samples = self.infer_local_assignments(self.posterior_samples, self.y)

        # [S, N_train + N_valid]
        rates = torch.FloatTensor(np.array(local_samples["rate_z"]))

        # [S, N_valid]
        # this is a bit hacky, but we have stored till what index the validation samples run
        rates_group_validation = rates[:, :self.N_valid]

        rates_group_validation = torch.FloatTensor(np.array(rates_group_validation))
        S, N_group_validation = rates_group_validation.shape

        # print(rates_group_validation.shape, S, N_group_validation)

        assert len(lengths) == N_group_validation, \
            "lengths must have the same length as the number of validation posterior rates" \
            f"currently, len(lengths) == {len(lengths)} and N_group_validation={N_group_validation}"

        poisson_validation = td.Poisson(rate=rates_group_validation)

        lengths = torch.Tensor(lengths)[None, :]

        # [S, N]
        log_prob = poisson_validation.log_prob(lengths)

        log_prob = torch.logsumexp(log_prob, dim=0) - np.log(S)

        # [N]
        return log_prob.numpy()


def plot_predictions(model, samples, bins=[20, 100], density=[True, True], sharex=True, sharey=True,
                     c_preds="#55B9F9", c_true="#EE6A2C", save_as=None):
    fig, ax = plt.subplots(model.G, 2, sharex=sharex, sharey=sharey, figsize=(8, model.G * 2.5))
    if model.G == 1:
        ax = ax.reshape(1, -1)
    pal = cycle(sns.color_palette())
    for k in range(model.G):
        yk = model.y[model.x == k]
        yk_ = samples[:, model.x == k]
        c = next(pal)
        _ = ax[k, 0].hist(yk, bins=bins[0], color=c_true, density=density[0])
        #         _ = ax[k, 0].set_xlabel(f'obs: {model.group_names[k]}')
        _ = ax[k, 0].set_title(f'Observations')
        _ = ax[k, 1].hist(yk_.flatten(), bins=bins[1], color=c_preds, density=density[1])
        #         _ = ax[k, 1].set_xlabel(f'predictive: {model.group_names[k]}')
        _ = ax[k, 1].set_title(f'Predictions')

    fig.tight_layout(h_pad=2, w_pad=2)
    if save_as is not None:
        fig.savefig(save_as, dpi=300, bbox="tight_inches")
    fig.show()


def plot_checks(model, samples, bins=30, c_pval="#356FB2", c="#EE6A2C", save_as=None):
    fig, ax = plt.subplots(model.G, 4, figsize=(12, model.G * 2.5))

    if model.G == 1:
        ax = ax.reshape(1, -1)

    # pal = cycle(sns.color_palette())

    for k in range(model.G):
        # c = next(pal)
        yk = model.y[model.x == k]
        yk_ = samples[:, model.x == k]

        _ = ax[k, 0].hist(np.mean(yk_, 1), bins=bins, color=c, label='pred' if k == 0 else None)
        _ = ax[k, 0].axvline(np.mean(yk), color='black', linestyle='--', label='obs' if k == 0 else None)
        _ = ax[k, 0].set_xlabel(f'E[Y{k}]')

        _ = ax[k, 1].hist(np.std(yk_, 1), color=c, bins=bins)
        _ = ax[k, 1].axvline(np.std(yk), color='black', linestyle='--')
        _ = ax[k, 1].set_xlabel(f'std[Y{k}]')

        _ = ax[k, 2].hist(np.median(yk_, 1), color=c, bins=bins)
        _ = ax[k, 2].axvline(np.median(yk), color='black', linestyle='--')
        _ = ax[k, 2].set_xlabel(f'median[Y{k}]')

        pvalues = np.mean(yk_ > yk, 1)
        _ = ax[k, 3].hist(pvalues, bins=bins)
        _ = ax[k, 3].set_xlabel(f'Pr(Y{k} > obs{k})')
        _ = ax[k, 3].axvline(np.median(pvalues), color=c_pval, linestyle=':', label='median' if k == 0 else None)

    _ = fig.legend(loc='upper center', ncol=3)
    fig.tight_layout(h_pad=2, w_pad=2)
    if save_as is not None:
        fig.savefig(save_as, dpi=300, bbox="tight_inches")
    fig.show()


