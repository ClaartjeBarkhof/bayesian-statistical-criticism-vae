import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import arviz as az

from tabulate import tabulate
import numpy as np

import numpyro
from numpyro.infer import MCMC, NUTS, Predictive
import numpyro.distributions as dist
from jax import random

NUM_CHAINS = 3
numpyro.set_host_device_count(NUM_CHAINS)

print(f"Running on NumPryo v{numpyro.__version__}")


class NumpyroGLM:
    def __init__(self, df, group_name_df, obs_x_list, obs_g_name, obs_y_name, correlate_predictors=False,
                 obs_dist="binomial", num_samples=1000, num_warmup=1000, num_chains=3):
        self.df = df

        self.group_name_df = group_name_df
        self.group_names = group_name_df.sort_values("group_id")["group_name"].values.tolist()

        self.num_samples = num_samples
        self.num_chains = num_chains
        self.num_warmup = num_warmup

        self.obs_dist = obs_dist
        self.correlate_predictors = correlate_predictors

        self.obs_x_list = obs_x_list
        self.obs_g_name = obs_g_name
        self.obs_y_name = obs_y_name

        self.make_init_assertions()

        self.obs_x = df[obs_x_list].values
        self.obs_y = df[obs_y_name].values
        self.obs_g = df[obs_g_name].values

        self.G = len(np.unique(self.obs_g))
        self.D = self.obs_x.shape[1]
        self.N = len(self.obs_x)

        self.total_count = 28 * 28
        self.rng_key = random.PRNGKey(0)

        nuts_kernel = NUTS(self.model)
        self.mcmc = MCMC(nuts_kernel, num_samples=num_samples, num_warmup=num_warmup, num_chains=num_chains)
        self.run_bool = False

        self.prior_predictive = None
        self.prior_predictions = None

        self.posterior_predictive = None
        self.posterior_predictions = None

        self.posterior_samples = None

        self.arviz_data = None

        self.print_init()

    def plot_hist_predictors(self):
        fig, axs = plt.subplots(nrows=2, ncols=len(self.obs_x_list), figsize=(len(self.obs_x_list) * 4, 8))

        for w, var_name in enumerate(self.obs_x_list):
            axs[0, w].hist(self.df[var_name].values.tolist(), density=True, bins=40, lw=0)
            axs[0, w].set_title(f"{var_name}")

        for w, var_name in enumerate(self.obs_x_list):
            for g, gn in enumerate(self.group_names):
                axs[1, w].hist(self.df[self.df["group_id"] == g][var_name].values.tolist(),
                               density=True, bins=40, lw=0, label=gn)
                axs[1, w].set_title(f"{var_name}")

        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_correlations(self):
        cols = [self.obs_y_name] + self.obs_x_list + ["group_name"]
        sns.pairplot(self.df[cols], hue="group_name")
        plt.tight_layout()
        plt.show()

    def print_init(self):
        predictors = " & ".join(self.obs_x_list)
        print(f"Predictor X columns: {predictors}")
        print(f"Predicting y, column: {self.obs_y_name}")
        print(f"Groups (G={len(self.group_name_df)}):")
        display(self.group_name_df)

        print(f"Optimisation:")
        print(f"MCMC NUTS algo: N_warmup={self.num_warmup}, N_samples={self.num_samples}, N_chains={self.num_chains}")

        # print("Model:")
        # fig = numpyro.render_model(self.model, model_args=(self.obs_x, self.obs_g, self.obs_y), render_distributions=True)
        # fig.render(view=True)

    def make_init_assertions(self):
        assert type(self.obs_x_list) == list, "you need to pass the predictor variable names in a list"
        assert all(elem in self.df.columns for elem in
                   self.obs_x_list), "the predictor variable names must be columns of the DF"
        assert self.obs_y_name in self.df.columns, "the target variable y must be a column of the DF"
        assert self.obs_g_name in self.df.columns, "the group_id variable must be a column of the DF"
        valid_dists = ["binomial", "student_t"]
        assert self.obs_dist in valid_dists, f"the observation distribution must be in {valid_dists}"
        assert "group_id" in self.group_name_df.columns, "expects a column group_id in self.group_name_df"
        assert "group_name" in self.group_name_df.columns, "expects a column group_id in self.group_name_df"

    def model(self, obs_x=None, obs_g=None, obs_y=None):
        # GLOBAL BIAS
        global_bias = numpyro.sample(
            "global_bias",
            dist.Normal(
                np.zeros(1),
                np.ones(1)
            )
        )

        # GROUP BIAS
        if obs_g is None:
            group_bias_ = 0.0
        else:
            group_bias = numpyro.sample(
                "group_bias",
                dist.Normal(
                    np.zeros(self.G),
                    np.ones(self.G)
                ).to_event(1)
            )
            group_bias_ = group_bias[obs_g]

        # PREDICTOR WEIGHTS
        if obs_x is None:
            Wx = 0.0
        else:
            # improvement: MVN (predictors do correlate)
            weights = numpyro.sample(
                "weights",
                dist.Normal(
                    np.zeros(self.D),
                    np.ones(self.D)
                ).to_event(1)
            )

            Wx = (weights * obs_x).sum(-1)

        # [N]
        eta = numpyro.deterministic("eta", global_bias + group_bias_ + Wx)

        # Section 15.2 of https://jrnold.github.io/bayesian_notes/robust-regression.html:
        # A reasonable prior distribution for the degrees of freedom parameter is a Gamma
        # distribution with shape parameter 2, and an inverse-scale (rate) parameter of 0.1
        if self.obs_dist == "student_t":
            student_scale = numpyro.sample("student_scale", dist.Gamma(concentration=2.0, rate=0.1))
        else:
            student_scale = 0.0
        # student_scale = 1.0

        with numpyro.plate("data", self.N):
            if self.obs_dist == "binomial":
                dist_y = dist.Binomial(self.total_count, logits=eta)
            elif self.obs_dist == "student_t":
                dist_y = dist.StudentT(df=4, loc=eta, scale=student_scale)

            numpyro.sample("obs", dist_y, obs=obs_y)

    def run(self):
        self.mcmc.run(self.rng_key, obs_x=self.obs_x, obs_g=self.obs_g, obs_y=self.obs_y)
        self.run_bool = True
        self.mcmc.print_summary()
        # group_by_chain=False: [K*D, ...]
        self.posterior_samples = self.mcmc.get_samples(group_by_chain=False)
        print("posterior samples shapes:")
        for k, v in self.posterior_samples.items():
            print(k, v.shape)

    def get_prior_predictions(self, num_prior_samples=100, plot=True):
        rng_key, rng_key_ = random.split(self.rng_key)

        if self.prior_predictive is None:
            self.prior_predictive = Predictive(self.model, num_samples=num_prior_samples)

        # [num_prior_samples, N]: for every x take multiple prior samples
        self.prior_predictions = self.prior_predictive(rng_key_, obs_x=self.obs_x, obs_g=self.obs_g, obs_y=None)["obs"]
        print("prior_predictions shape (n_prior_samples, N):", self.prior_predictions.shape)

        if plot:
            self.plot_prior_predictions()

        return self.prior_predictions

    def plot_prior_predictions(self):
        if self.prior_predictions is None:
            self.get_prior_predictions()

        prior_preds_avg_over_sample = self.prior_predictions.mean(axis=0)
        plt.hist(prior_preds_avg_over_sample, bins=40, density=True, label="prior predictive", alpha=0.7, lw=0)
        plt.hist(self.obs_y, bins=40, density=True, label="y obs", alpha=0.7, lw=0)
        plt.title(f"Prior predictions")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def get_posterior_predictions(self, plot=False):
        self.run_check()

        rng_key, rng_key_ = random.split(self.rng_key)
        if self.posterior_predictive is None:
            self.posterior_predictive = Predictive(self.model, self.posterior_samples)

        # [num_chains * num_samples, N] -> [num_chains, num_samples, N]
        self.posterior_predictions = self.posterior_predictive(rng_key_, obs_x=self.obs_x, obs_g=self.obs_g)["obs"]

        print("posterior_predictions shape (chains, n_samples, N):", self.posterior_predictions.shape)

        if plot:
            self.plot_posterior_predictions()

        return self.posterior_predictions

    def plot_posterior_predictions(self):
        if self.posterior_predictions is None:
            self.get_posterior_predictions()

        fig, axs = plt.subplots(ncols=self.num_chains, figsize=(14, 4))

        posterior_predictions = self.posterior_predictions.reshape(self.num_chains, self.num_samples, -1)
        # [num_chains, num_samples, N] -> [num_chains, N]
        posterior_predictions_avg_over_sample = posterior_predictions.mean(axis=1)

        if self.num_chains > 1:
            for i in range(self.num_chains):
                axs[i].hist(self.obs_y, bins=40, density=True, label="y obs", alpha=0.7, lw=0)
                axs[i].hist(posterior_predictions_avg_over_sample[i, :], bins=40, density=True,
                            label="posterior predictive", alpha=0.7, lw=0)
                axs[i].set_title(f"Posterior predictive (Chain {i})")
        else:
            axs.hist(self.obs_y, bins=40, density=True, label="y obs", alpha=0.7, lw=0)
            axs.hist(posterior_predictions_avg_over_sample[0, :], bins=40, density=True,
                     label="posterior predictive", alpha=0.7, lw=0)
            axs.set_title(f"Posterior predictive (Chain 0)")

        plt.legend()
        plt.tight_layout()
        plt.show()

    def predictive_checks(self, prior=False):
        if prior:
            if self.prior_predictions is None:
                self.get_prior_predictions()
            preds = self.prior_predictions
        else:
            if self.posterior_predictions is None:
                self.get_posterior_predictions()
            preds = self.posterior_predictions.reshape(-1, self.posterior_predictions.shape[-1])

        # SHAPES:
        # posterior_predictions [N_chains*N_samples, N_data]
        # prior_predictions [N_samples, N_data]
        # samples (generalising both): [N_s, N_d]

        obs_mean = np.mean(self.obs_y)
        obs_std = np.std(self.obs_y)
        obs_median = np.median(self.obs_y)
        obs_mode = np.max(self.obs_y)
        obs_skew = obs_mean ** (-0.5)
        obs_kurtosis = obs_mean ** (-1)

        obs_stats = [obs_mean, obs_std, obs_median, obs_mode, obs_skew, obs_kurtosis]

        pred_mean = np.mean(preds, axis=1)
        pred_std = np.std(preds, axis=1)
        pred_median = np.median(preds, axis=1)
        pred_mode = np.max(preds, axis=1)
        pred_skew = pred_mean ** (-0.5)
        pred_kurtosis = pred_mean ** (-1)

        preds_stats = [pred_mean, pred_std, pred_median, pred_mode, pred_skew, pred_kurtosis]
        preds_stats_means = [s.mean() for s in preds_stats]
        preds_stats_std = [s.std() for s in preds_stats]

        # predictive p values
        p_vals = [(p > o).mean() for p, o in zip(preds_stats, obs_stats)]

        stats = ["mean", "std", "median", "max", "skew", "kurtosis"]

        headers = ['check', 'p_val', 'obs', 'pred (mean)', 'pred (std)']
        rows = [['S', 1, None, preds.shape[0], None], ['shape', None, self.obs_y.shape, preds.shape, None]]

        for s, p_v, o, p_m, p_std in zip(stats, p_vals, obs_stats, preds_stats_means, preds_stats_std):
            rows.append([s, f"{p_v:.3f}", f"{o:.3f}", f"{p_m:.3f}", f"{p_std:.3f}"])

        for C in [0.25, 0.5, 0.75, 1., 2.]:
            mean_check = (np.abs(pred_mean - obs_mean) < C * pred_std)
            rows.append([f"mean within {C:.2f}*std", None, None, f"{mean_check.mean():.3f}", f"{mean_check.std():.3f}"])

        print(tabulate(rows, headers=headers))  # , floatfmt=(None, ".3f", ".3f", ".3f")

        ncols = 3
        nrows = int(np.ceil(len(stats) / ncols))

        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 4))

        for i, (s, o, p, pm) in enumerate(zip(stats, obs_stats, preds_stats, preds_stats_means)):
            r, c = i // ncols, i % ncols
            if np.any(np.isinf(p)):
                print(f"inf encountered in preds of {s}")
                continue
            axs[r, c].hist(p, bins=40, lw=0, density=True)
            axs[r, c].axvline(o, color='g', linestyle='dashed', label='obs')
            axs[r, c].axvline(pm, color='r', linestyle='dashed', label='pred mean T')
            axs[r, c].set_title(s)

        title = "Prior predictive checks" if prior else "Posterior predictive checkes"
        plt.suptitle(title)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_ppc(self, num_pp_samples=10):
        if self.posterior_predictions is None:
            self.get_posterior_predictions()
        if "posterior_predictive" not in self.arviz_data.groups():
            a = self.posterior_predictions.reshape(-1, self.posterior_predictions.shape[-1])
            self.arviz_data.add_groups(az.from_numpyro(posterior_predictive={"obs": a}, num_chains=self.num_chains))
        az.plot_ppc(self.arviz_data, num_pp_samples=num_pp_samples)

    def run_check(self):
        if not self.run_bool:
            self.run()

    def plot_check(self):
        self.run_check()
        if self.arviz_data is None:
            # coords={"school": np.arange(eight_school_data["J"])},
            # dims={"theta": ["school"]}
            # self.group_names
            # self.obs_x_list
            #             coords = {"group": np.arange(len(self.group_names)), "predictor": np.arange(len(self.obs_x_list))}
            coords = {"group": self.group_names, "predictor": self.obs_x_list}
            dims = {"group_bias": ["group"], "weights": ["predictor"]}
            self.arviz_data = az.from_numpyro(posterior=self.mcmc, num_chains=self.num_chains, dims=dims, coords=coords)

    def plot_trace(self):
        self.plot_check()
        # coords={"group_bias": self.group_names, "weights": self.obs_x_list}
        az.plot_trace(self.arviz_data, var_names=["global_bias", "group_bias", "weights"], compact=False)

    def plot_posterior(self):
        self.plot_check()
        az.plot_posterior(self.arviz_data, var_names=["global_bias", "group_bias", "weights"])

    def plot_group_bias(self):
        self.plot_check()
        az.plot_forest(self.arviz_data, kind='forestplot', var_names=["group_bias"])

    def full_analysis(self, num_pp_samples=400, num_prior_samples=1000):
        self.plot_hist_predictors()
        self.plot_correlations()
        self.run()
        _ = self.get_prior_predictions(num_prior_samples=num_prior_samples, plot=True)
        self.predictive_checks(prior=True)
        _ = self.get_posterior_predictions(plot=True)
        self.predictive_checks(prior=False)
        self.plot_trace()
        self.plot_ppc(num_pp_samples=num_pp_samples)
        self.plot_posterior()
        self.plot_group_bias()