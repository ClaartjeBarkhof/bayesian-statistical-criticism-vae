import matplotlib.pyplot as plt
import seaborn as sns;

sns.set()
import arviz as az

from tabulate import tabulate
import numpy as np
import pandas as pd

import numpyro
from numpyro.infer import MCMC, NUTS, Predictive
import numpyro.distributions as dist
from jax import random
import jax.numpy as jnp

NUM_CHAINS = 3
numpyro.set_host_device_count(NUM_CHAINS)

print(f"Running on NumPryo v{numpyro.__version__}")

class NumpyroGLM:
    def __init__(self, df, group_name_df, obs_x_list, obs_g_name, obs_y_name, correlate_predictors=False,
                 obs_dist="binomial", num_samples=1000, num_warmup=1000, num_chains=3, inverse_data_transform=None):
        self.df = df

        self.group_name_df = group_name_df
        self.group_names = group_name_df.sort_values("group_id")["group_name"].values.tolist()

        self.num_samples = num_samples
        self.num_chains = num_chains
        self.num_warmup = num_warmup

        self.inverse_data_transform = inverse_data_transform

        self.obs_dist = obs_dist

        self.var_names = ["global_bias", "group_bias", "weights"]

        if self.obs_dist == "student_t":
            self.var_names += ["student_scale", "student_df"]
        elif self.obs_dist == "log_normal":
            self.var_names += ["l_normal_scale"]

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

    def model(self, obs_x=None, obs_g=None, obs_y=None):
        # GLOBAL BIAS
        global_bias = numpyro.sample(
            "global_bias",
            dist.Laplace(
                np.zeros(1),
                np.ones(1) / 2.
            )
        )

        # PREDICTOR WEIGHTS
        if obs_x is None:
            Wx_b = 0.0
        else:
            with numpyro.plate("groups", self.G):
                group_bias = numpyro.sample(
                    "group_bias",
                    dist.Laplace(
                        0.0,
                        1. / 2.
                    )
                )
                if self.correlate_predictors:
                    weights = numpyro.sample(
                        "weights",
                        dist.MultivariateNormal(
                            loc=np.zeros(self.D),
                            covariance_matrix=np.diag(np.ones(self.D)) / 2.
                        )
                    )
                else:
                    weights = numpyro.sample(
                        "weights",
                        dist.Laplace(
                            np.zeros(self.D),
                            np.ones(self.D)
                        ).to_event(1)
                    )

                if self.obs_dist == "log_normal" or self.obs_dist == "normal":
                    l_normal_scale = numpyro.sample("l_normal_scale", dist.Uniform(low=0.1, high=10.0))
                elif self.obs_dist == "student_t":
                    # See section 15.2 of https://jrnold.github.io/bayesian_notes/robust-regression.html:
                    student_scale = numpyro.sample("student_scale", dist.Gamma(concentration=2.0, rate=0.1))
                    student_df = numpyro.sample("student_df", dist.Exponential(rate=1.0 / 10.0))
                else:
                    l_normal_scale, student_scale, student_df = 0.0, 0.0, 0.0

            # [G, D]*[N, D] + [N] -> [N, D]*[N, D] + [N] -> [N]
            obs_x = (obs_x - obs_x.mean(axis=0, keepdims=True)) / obs_x.std(axis=0, keepdims=True)
            Wx_b = (weights[self.obs_g] * obs_x).sum(-1) + group_bias[self.obs_g]

        # [N]
        eta = numpyro.deterministic("eta", (global_bias + Wx_b))

        with numpyro.plate("data", self.N):
            if self.obs_dist == "binomial":
                dist_y = dist.Binomial(self.total_count, logits=eta)
            elif self.obs_dist == "student_t":
                dist_y = dist.StudentT(df=student_df[self.obs_g], loc=eta, scale=student_scale[self.obs_g])
            elif self.obs_dist == "log_normal":
                dist_y = dist.LogNormal(loc=eta, scale=l_normal_scale[self.obs_g])
            elif self.obs_dist == "normal":
                dist_y = dist.Normal(loc=eta, scale=l_normal_scale[self.obs_g])

            numpyro.sample("obs", dist_y, obs=obs_y)

    def plot_correlations(self):
        cols = [self.obs_y_name] + self.obs_x_list + ["group_name"]
        sns.pairplot(self.df[cols], hue="group_name")
        plt.tight_layout()
        plt.show()

    def print_init(self):
        predictors = " & ".join(self.obs_x_list)
        print(f"Predictor X columns: {predictors}")
        print(f"Correlate predictors: {self.correlate_predictors}")
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
        valid_dists = ["binomial", "student_t", "log_normal", "normal"]
        assert self.obs_dist in valid_dists, f"the observation distribution must be in {valid_dists}"
        assert "group_id" in self.group_name_df.columns, "expects a column group_id in self.group_name_df"
        assert "group_name" in self.group_name_df.columns, "expects a column group_id in self.group_name_df"

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
        self.prior_predictions = self.prior_predictive(rng_key_, obs_x=self.obs_x, obs_g=self.obs_g, obs_y=None)

        if plot:
            self.plot_trace_predictions(full=True, prior=True)

        return self.prior_predictions

    def plot_trace_predictions(self, full=True, prior=True):
        posterior_prior = "Prior" if prior else "Posterior"
        if prior:
            print(f"Plotting prior prediction, full={full}")
            if self.prior_predictions is None:
                self.get_prior_predictions()
            predictions = self.prior_predictions
        else:
            print(f"Plotting posterior predictions, full={full}")
            if self.posterior_predictions is None:
                self.get_posterior_predictions()
            predictions = {**self.posterior_predictions, **self.posterior_samples}

        print("plot_trace_predictions: keys shapes of predictions")
        for k, v in predictions.items():
            print(k, v.shape)

        for var in ["obs", "eta", "global_bias"]:
            if not full and var != "obs":
                continue

            fig, ax = plt.subplots(figsize=(8, 4))

            label = f"{var}" if var != "obs" else "pred"

            _ = ax.hist(np.array(predictions[var]).flatten(), label=label, density=True, bins=40, lw=0, alpha=0.7)
            if var == "obs":
                _ = ax.hist(np.array(self.obs_y), label="obs y", density=True, bins=40, lw=0, alpha=0.7)

            ax.set_title(f"{posterior_prior} - {var}")

            plt.legend()
            plt.tight_layout()

            plt.show()

        if full:
            # Group variables
            for var in ["group_bias", "l_normal_scale", "student_scale", "student_df"]:
                if ("student" in var) and ("student" not in self.obs_dist):
                    continue
                if ("normal" in var) and ("normal" not in self.obs_dist):
                    continue

                fig, axs = plt.subplots(ncols=self.G, figsize=(int(4*self.G), 4))

                for g in range(self.G):

                    axs[g].hist(np.array(predictions[var][:, g]).flatten(), density=True, bins=40, lw=0, alpha=0.7)
                    axs[g].set_title(f"{self.group_names[g]}")

                plt.suptitle(f"{posterior_prior} - {var}")
                # plt.legend()
                plt.tight_layout()

                plt.show()

            # Group + predictor weights
            fig, axs = plt.subplots(ncols=self.D, nrows=self.G, figsize=(int(4 * self.D), int(self.G*4)))

            for row in range(self.G):
                for col in range(self.D):

                    axs[row, col].hist(np.array(predictions["weights"][:, row, col]).flatten(), density=True, bins=40, lw=0, alpha=0.7)

                    axs[row, col].set_title(f"G={self.group_names[row]}\nD={self.obs_x_list[col]}", y=1.02)

            plt.suptitle(f"{posterior_prior} - weights linear")
            plt.tight_layout()
            plt.show()

    def get_posterior_predictions(self, plot=False):
        self.run_check()

        rng_key, rng_key_ = random.split(self.rng_key)
        if self.posterior_predictive is None:
            self.posterior_predictive = Predictive(self.model, self.posterior_samples)

        self.posterior_predictions = self.posterior_predictive(rng_key_, obs_x=self.obs_x, obs_g=self.obs_g)

        if plot:
            self.plot_trace_predictions(full=True, prior=False)

        return self.posterior_predictions

    def predictive_checks(self, prior=False):
        if prior:
            if self.prior_predictions is None:
                self.get_prior_predictions()
            preds = np.array(self.prior_predictions["obs"])
        else:
            if self.posterior_predictions is None:
                self.get_posterior_predictions()
            preds = np.array(self.posterior_predictions["obs"]) #.reshape(-1, self.posterior_predictions["obs"].shape[-1])
            preds = preds.reshape(self.num_chains, self.num_samples, self.N)
            preds = preds[0, :, :]  # only consider the first chain, to make plotting less heavy

        # print("predictive checks shape preds:", preds.shape)

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
            if np.any(np.isinf(p)) or np.any(np.isnan(p)):
                print(f"encountered NAN in preds of {s} = {np.any(np.isnan(p))}")
                print(f"encountered INF in preds of {s} = {np.any(np.isinf(p))}")
                continue

            axs[r, c].hist(np.array(p), bins=40, lw=0, density=True)
            axs[r, c].axvline(o, color='g', linestyle='dashed', label='obs')
            axs[r, c].axvline(pm, color='r', linestyle='dashed', label='pred mean T')
            axs[r, c].set_title(s)

        title = "Prior predictive checks" if prior else "Posterior predictive checkes"
        plt.suptitle(title)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_ppc(self, num_pp_samples=10):
        self.plot_check()
        if self.posterior_predictions is None:
            self.get_posterior_predictions()
        if "posterior_predictive" not in self.arviz_data.groups():
            a = self.posterior_predictions["obs"].reshape(-1, self.posterior_predictions["obs"].shape[-1])
            self.arviz_data.add_groups(az.from_numpyro(posterior_predictive={"obs": a}, num_chains=self.num_chains))
        az.plot_ppc(self.arviz_data, num_pp_samples=num_pp_samples)

    def run_check(self):
        if not self.run_bool:
            self.run()

    def plot_check(self):
        self.run_check()
        if self.arviz_data is None:
            l = []
            for g in self.group_names:
                l1 = []
                for x in self.obs_x_list:
                    l1.append(f"{g}-{x}")
                l.append(l1)

            #coords = {"group": self.group_names, "group-predictor": l}
            #dims = {"group_bias": ["group"], "weights": ["group-predictor"]}
            #, dims=dims, coords=coords
            self.arviz_data = az.from_numpyro(posterior=self.mcmc, num_chains=self.num_chains)

    def plot_post_preds_per_group(self, N_cols=2):
        if self.posterior_predictions is None:
            self.get_posterior_predictions()

        # [N_chains * N_samples, N_data]
        predictions = self.posterior_predictions["obs"]
        if self.inverse_data_transform is not None:
            predictions = self.inverse_data_transform(np.array(predictions))
            obs_y = self.inverse_data_transform(np.array(self.obs_y))
        else:
            predictions = np.array(predictions)
            obs_y = np.array(self.obs_y)

        predictions = predictions.reshape(self.num_chains, self.num_samples, -1)[0, :, :]

        N_plots = self.G
        N_rows = int(np.ceil(N_plots / N_cols))

        color_dict = {'blue': '#8caadc',
                      'red': '#c51914',
                      'pink': '#fcb1ca',
                      'orange': '#efb116',
                      'dark_blue': '#000563',
                      'green': '#005f32',
                      'sand': '#cec3bc'}
        colors = list(color_dict.values())

        # make grid spec grid
        fig = plt.figure(figsize=(4 * N_cols, 4 * N_rows + 1))
        gs = fig.add_gridspec(N_rows + 1, N_cols)
        all_ax = fig.add_subplot(gs[0, :])
        g_axs = []
        for row in range(N_rows):
            for col in range(N_cols):
                g_axs.append(fig.add_subplot(gs[row + 1, col]))

        all_ax.hist(predictions.flatten(), color=colors[1], density=True, bins=40, label=f"Preds", lw=0,
                    alpha=0.7)
        all_ax.hist(obs_y.flatten(), color=colors[2], density=True, bins=40, label=f"Data", lw=0,
                    alpha=0.7)
        all_ax.legend()

        for g in range(self.G):
            preds_g = predictions[:, self.obs_g == g].flatten()
            obs_y_g = obs_y[self.obs_g == g].flatten()

            #         sns.histplot(x=preds_g, ax=g_axs, label=f"G={g}", kde=True, stat="density", color=colors[g])
            #         sns.histplot(x=, ax=ax,
            #                      label="Data", kde=True, stat="density", color=colors[g+1])

            g_axs[g].hist(preds_g, color=colors[1], density=True, bins=40, label=f"Preds", lw=0, alpha=0.7)
            g_axs[g].hist(obs_y_g, color=colors[2], density=True, bins=40, label=f"Data", lw=0, alpha=0.7)
            g_axs[g].set_title(self.group_names[g])

        plt.suptitle(f"Posterior predictions versus observations (per group)\n"
                     f"y var: {self.obs_y_name}, inv trans: {str(self.inverse_data_transform)}", y=1.03)

        plt.tight_layout()

        plt.show()

    def plot_trace(self):
        self.plot_check()
        # coords={"group_bias": self.group_names, "weights": self.obs_x_list}
        az.plot_trace(self.arviz_data, var_names=self.var_names, compact=False)

    def plot_weights_correlation(self):
        df = pd.DataFrame(np.array(self.posterior_samples['weights']), columns=self.obs_x_list)
        _ = pd.plotting.scatter_matrix(df, alpha=0.2, figsize=(12, 12))
        plt.show()

    def plot_posterior(self):
        self.plot_check()
        az.plot_posterior(self.arviz_data, var_names=self.var_names)

    def plot_group_bias(self):
        self.plot_check()
        az.plot_forest(self.arviz_data, kind='forestplot', var_names=["group_bias"])

    def full_analysis(self, num_pp_samples=400, num_prior_samples=1000, data_plots=True,
                      prior_plots=True, posterior_plots=True, arviz_plots=True):
        if data_plots:
            self.plot_correlations()

        if prior_plots:
            _ = self.get_prior_predictions(num_prior_samples=num_prior_samples, plot=True)
            self.predictive_checks(prior=True)

        self.run()

        if posterior_plots:
            _ = self.get_posterior_predictions(plot=True)
            self.predictive_checks(prior=False)
            self.plot_post_preds_per_group(N_cols=2)

        if arviz_plots:
            # self.plot_trace()
            self.plot_ppc(num_pp_samples=num_pp_samples)
            self.plot_posterior()
            if self.correlate_predictors:
                self.plot_weights_correlation()
            self.plot_group_bias()