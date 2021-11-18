import numpy as np
import pandas as pd
import arviz as az
from scipy import stats

import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from jax import random
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.diagnostics import hpdi
from numpyro.infer import MCMC, NUTS, Predictive

from numpyro_bda_utils import create_group_df, predictive_checks

NUM_CHAINS = 3
numpyro.set_host_device_count(NUM_CHAINS)
print(f"Running on NumPryo v{numpyro.__version__}")


class NumpyroGLM:
    def __init__(self, df, group_by, obs_x_list, obs_y_name, correlate_predictors=False,
                 obs_dist="binomial", num_samples=1000, num_warmup=1000, num_chains=3, inverse_data_transform=None):

        self.df, self.group_name_df = create_group_df(df, group_by=group_by,
                                                      group_id_col="group_id", group_name_col="group_name")
        self.group_names = self.group_name_df.sort_values("group_id")["group_name"].values.tolist()

        self.num_samples = num_samples
        self.num_chains = num_chains
        self.num_warmup = num_warmup

        self.inverse_data_transform = inverse_data_transform

        self.obs_dist = obs_dist

        self.var_names = ["global_bias", "group_bias", "weights"]

        if self.obs_dist == "student_t":
            self.var_names += ["student_scale", "student_df"]
        elif self.obs_dist == "log_normal":
            self.var_names += ["normal_scale"]

        self.correlate_predictors = correlate_predictors

        self.obs_x_list = obs_x_list
        self.obs_y_name = obs_y_name

        self.make_init_assertions()

        self.obs_x = df[obs_x_list].values
        self.obs_y = df[obs_y_name].values
        self.obs_g = df["group_id"].values

        print("obs_g", self.obs_g)

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

    def model(self, obs_x=None, obs_y=None):
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

                if "normal" in self.obs_dist:
                    normal_scale = numpyro.sample("normal_scale", dist.Uniform(low=0.1, high=10.0))
                elif self.obs_dist == "student_t":
                    # See section 15.2 of https://jrnold.github.io/bayesian_notes/robust-regression.html:
                    student_scale = numpyro.sample("student_scale", dist.Gamma(concentration=2.0, rate=0.1))
                    student_df = numpyro.sample("student_df", dist.Exponential(rate=1.0 / 10.0))
                else:
                    normal_scale, student_scale, student_df = 0.0, 0.0, 0.0

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
                dist_y = dist.LogNormal(loc=eta, scale=normal_scale[self.obs_g])
            elif self.obs_dist == "normal":
                dist_y = dist.Normal(loc=eta, scale=normal_scale[self.obs_g])
            elif self.obs_dist == "truncated_normal":
                dist_y = dist.TruncatedNormal(low=0.0, loc=eta, scale=normal_scale[self.obs_g])

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
        # display(self.group_name_df)

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
        valid_dists = ["binomial", "student_t", "log_normal", "normal", "truncated_normal"]
        assert self.obs_dist in valid_dists, f"the observation distribution must be in {valid_dists}"

    def run(self):
        self.mcmc.run(self.rng_key, obs_x=self.obs_x, obs_y=self.obs_y)
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
        self.prior_predictions = self.prior_predictive(rng_key_, obs_x=self.obs_x, obs_y=None)

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

            if not (np.any(np.isinf(np.array(predictions[var]).flatten())) or np.any(
                    np.isnan(np.array(predictions[var]).flatten()))):
                _ = ax.hist(np.array(predictions[var]).flatten(), label=label, density=True, bins=40, lw=0, alpha=0.7)
            if var == "obs":
                if not (np.any(np.isinf(np.array(self.obs_y))) or np.any(np.isnan(np.array(self.obs_y)))):
                    _ = ax.hist(np.array(self.obs_y), label="obs y", density=True, bins=40, lw=0, alpha=0.7)

            ax.set_title(f"{posterior_prior} - {var}")

            plt.legend()
            plt.tight_layout()

            plt.show()

        if full:
            # Group variables
            for var in ["group_bias", "normal_scale", "student_scale", "student_df"]:
                if ("student" in var) and ("student" not in self.obs_dist):
                    continue
                if ("normal" in var) and ("normal" not in self.obs_dist):
                    continue

                fig, axs = plt.subplots(ncols=self.G, figsize=(int(4 * self.G), 4))

                for g in range(self.G):
                    if not (np.any(np.isinf(np.array(predictions[var][:, g]).flatten())) or np.any(
                            np.isnan(np.array(np.array(predictions[var][:, g]).flatten())))):
                        axs[g].hist(np.array(predictions[var][:, g]).flatten(), density=True, bins=40, lw=0, alpha=0.7)
                    axs[g].set_title(f"{self.group_names[g]}")

                plt.suptitle(f"{posterior_prior} - {var}")
                # plt.legend()
                plt.tight_layout()

                plt.show()

            # Group + predictor weights
            fig, axs = plt.subplots(ncols=self.D, nrows=self.G, figsize=(int(4 * self.D), int(self.G * 4)),
                                    sharex="all", sharey="all")

            for row in range(self.G):
                for col in range(self.D):
                    if not (np.any(np.isinf(np.array(predictions["weights"][:, row, col]).flatten())) or np.any(
                            np.isnan(np.array(np.array(predictions["weights"][:, row, col]).flatten())))):
                        axs[row, col].hist(np.array(predictions["weights"][:, row, col]).flatten(), density=True,
                                           bins=40, lw=0, alpha=0.7)
                    axs[row, col].set_title(f"G={self.group_names[row]}\nD={self.obs_x_list[col]}", y=1.02)

            plt.suptitle(f"{posterior_prior} - weights linear")
            plt.tight_layout()
            plt.show()

    def get_posterior_predictions(self, plot=False):
        self.run_check()

        rng_key, rng_key_ = random.split(self.rng_key)
        if self.posterior_predictive is None:
            self.posterior_predictive = Predictive(self.model, self.posterior_samples)

        self.posterior_predictions = self.posterior_predictive(rng_key_, obs_x=self.obs_x)

        if plot:
            self.plot_trace_predictions(full=True, prior=False)

        return self.posterior_predictions

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

            # coords = {"group": self.group_names, "group-predictor": l}
            # dims = {"group_bias": ["group"], "weights": ["group-predictor"]}
            # , dims=dims, coords=coords
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

        if not (np.any(np.isinf(predictions.flatten())) or np.any(np.isnan(predictions.flatten()))):
            all_ax.hist(predictions.flatten(), color=colors[1], density=True, bins=40, label=f"Preds", lw=0,
                        alpha=0.7)

        if not (np.any(np.isinf(obs_y.flatten())) or np.any(np.isnan(obs_y.flatten()))):
            all_ax.hist(obs_y.flatten(), color=colors[2], density=True, bins=40, label=f"Data", lw=0,
                        alpha=0.7)
        all_ax.legend()

        for g in range(self.G):
            preds_g = predictions[:, self.obs_g == g].flatten()
            obs_y_g = obs_y[self.obs_g == g].flatten()

            if not (np.any(np.isinf(preds_g)) or np.any(np.isnan(preds_g))):
                g_axs[g].hist(preds_g, color=colors[1], density=True, bins=40, label=f"Preds", lw=0, alpha=0.7)
            #         sns.histplot(x=preds_g, ax=g_axs, label=f"G={g}", kde=True, stat="density", color=colors[g])
            #         sns.histplot(x=, ax=ax,
            #                      label="Data", kde=True, stat="density", color=colors[g+1])

            if not (np.any(np.isinf(obs_y_g)) or np.any(np.isnan(obs_y_g))):
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

    def plot_group_specific_params(self, predictor_col="mmd mean"):
        assert predictor_col in self.obs_x_list, f"{predictor_col} must be one of the predictors"
        mmd_col = self.obs_x_list.index(predictor_col)

        weights = self.posterior_samples["weights"][:, :, mmd_col]
        global_bias = self.posterior_samples["global_bias"]
        group_bias = self.posterior_samples["group_bias"]

        if self.obs_dist == "student_t":
            group_student_df = self.posterior_samples["student_df"]
            group_student_scale = self.posterior_samples["student_scale"]
            dist_stats = [group_student_df, group_student_scale]
            dist_stat_names = ["group_student_df", "group_student_scale"]
        elif "normal" in self.obs_dist:
            group_scale = self.posterior_samples["normal_scale"]
            dist_stats = [group_scale]
            dist_stat_names = ["group_normal_scale"]

        stat_list = [weights, group_bias] + dist_stats
        stat_list_names = ["MMD weights", "group_bias"] + dist_stat_names

        ncols = len(stat_list)

        fig, axs = plt.subplots(nrows=self.G, ncols=ncols, figsize=(16, 16), sharex="col")

        for g in range(self.G):
            row = g
            for col, stat in enumerate(stat_list):
                weight_mean = weights[:, g].mean()
                global_bias_mean = global_bias.mean()
                group_bias_mean = group_bias[:, g].mean()
                lin_eq = f"{global_bias_mean:.1f} + {group_bias_mean:.1f} +" \
                         f"{weight_mean:.1f} X_MMD"

                data_g = np.array(stat[:, g])
                iqr = stats.iqr(data_g)
                mean = np.mean(data_g)
                upper_iqr = mean + 0.5 * iqr
                lower_iqr = mean - 0.5 * iqr
                axs[row, col].axvline(x=upper_iqr, color="lightblue")
                axs[row, col].axvline(x=lower_iqr, color="lightblue")
                axs[row, col].axvline(x=0, color='lightgrey', linestyle="--")
                sns.violinplot(x=data_g, ax=axs[row, col])
                axs[row, col].set_title(self.group_names[g] + "\n" + stat_list_names[col] + "\n" + lin_eq, y=1.05)

        plt.tight_layout()
        plt.show()

    def plot_regression_mmd_predictor(self, predictor="mmd mean", predictor_name="MMD", y_label="y_label",
                                      plot_n_regression_lines=500, ncols=4):
        assert predictor in self.obs_x_list, "the you must include 'mmd mean' as a predictor in your regression for this plot"
        pred_col = self.obs_x_list.index(predictor)
        # print(f"{predictor} col", pred_col)

        weights = self.posterior_samples["weights"]  # [S, G, X]
        global_bias = self.posterior_samples["global_bias"]  # [S, 1]
        group_bias = self.posterior_samples["group_bias"]  # [S, G]
        # group_student_df = self.posterior_samples["student_df"]  # [S, G]
        # group_student_scale = self.posterior_samples["student_scale"]  # [S, G]
        preds = self.posterior_predictions["obs"]  # [S, X]
        x_col = self.obs_x[:, pred_col]  # [X]

        min_col, max_col = x_col.min(), x_col.max()
        x_lin = np.linspace(start=min_col, stop=max_col, num=1000)

        # We have N_chains * N_samples, which is too much for plotting, subsample N=<plot_n_regression_lines>
        sub_sample_ids = np.random.randint(0, high=weights.shape[0], size=plot_n_regression_lines, dtype=int)

        nrows = int(np.ceil(self.G / ncols))
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 4, nrows * 4),
                                sharex=True, sharey=True)

        for g in range(self.G):
            row = g // ncols
            col = g % ncols

            x_g = x_col[self.obs_g == g]
            y_g = self.obs_y[self.obs_g == g]
            p_g = preds[0, self.obs_g == g]

            # Scatter predictions & data points
            axs[row, col].scatter(x_g, y_g, alpha=0.1, color="green", label='group data')
            axs[row, col].scatter(x_g, p_g, alpha=0.1, color="blue", label='group preds')

            # Subsample group weights and take the col that relates to the predictor we care about
            sub_weights = weights[sub_sample_ids, g, pred_col][None, :]
            sub_global_bias = global_bias[sub_sample_ids].squeeze(1)[None, :]
            sub_group_bias = group_bias[sub_sample_ids, g][None, :]

            # print("sub_weights.shape", sub_weights.shape)
            # print("sub_global_bias.shape", sub_global_bias.shape)
            # print("sub_group_bias.shape", sub_group_bias.shape)

            weight_mean = weights[:, g, pred_col].mean()
            global_bias_mean = global_bias.mean()
            group_bias_mean = group_bias[:, g].mean()

            # [1000, 1], [1, 100] -> [1000, 100]
            y_pred = x_lin[:, None] * sub_weights + sub_global_bias + sub_group_bias

            for i in range(plot_n_regression_lines):
                label = "sampled reg. line" if i == 0 else None
                y = y_pred[:, i]
                if self.obs_dist == "log_normal":
                    y = np.exp(y)
                axs[row, col].plot(x_lin, y, lw=0.3, alpha=0.2, color="pink", label=label)

            y_mean = x_lin * weight_mean + global_bias_mean + group_bias_mean

            if self.obs_dist == "log_normal":
                y_mean = np.exp(y_mean)

            axs[row, col].plot(x_lin, y_mean, lw=0.8, alpha=0.8, color="red", label="avg reg. line")

            title = f"{self.group_names[g]}\n{global_bias_mean:.1f} + {group_bias_mean:.1f} +" \
                    f"{weight_mean:.1f} {predictor_name}"

            axs[row, col].set_title(title, y=1.05)
            # axs[row, col].set_ylim([50, 100])
            axs[row, col].set_xlabel(predictor_name)
            axs[row, col].set_ylabel(y_label, size=8)

        leg = axs[0, 3].legend(loc=(1.05, 0.75))
        for lh in leg.legendHandles:
            lh.set_alpha(1)

        plt.tight_layout()
        plt.show()

    def residuals_plots(self):

        # [Ns, Nx]
        predictions = self.posterior_predictions["obs"]

        ys_per_group = []
        preds_per_group = []
        res_per_group = []

        # record SAMPLE mean (not prediction mean)
        preds_mean_per_group = []
        res_mean_per_group = []

        hpdi_residuals = []
        hpdi_preds = []
        hpdi_ys = []

        for g in range(self.G):
            y = self.obs_y[self.obs_g == g]
            preds = predictions[:, self.obs_g == g]
            res = y - preds

            # [Ns, Ng]
            preds_per_group.append(preds)
            res_per_group.append(res)

            # [Ng]
            ys_per_group.append(y)

            # [Ng]
            preds_mean_per_group.append(preds.mean(axis=0))
            res_mean_per_group.append(res.mean(axis=0))

            hpdi_residuals.append(hpdi(res.mean(axis=0), 0.9))
            hpdi_ys.append(hpdi(y, 0.9))

            # print(preds.mean(axis=0).shape)

            hpdi_preds.append(hpdi(preds.flatten(), 0.9))

        # [N_groups, 2]
        hpdi_residuals = np.stack(hpdi_residuals, axis=0)
        hpdi_ys = np.stack(hpdi_ys, axis=0)
        hpdi_preds = np.stack(hpdi_preds, axis=0)

        fig, axs = plt.subplots(ncols=2, figsize=(14, int(0.75 * self.G)))
        y = np.arange(self.G)

        axs[0].plot(jnp.zeros(self.G), y, "--", label="zero line")
        # Plot the hpdi of the residuals and a dot in the center
        axs[0].errorbar(
            hpdi_residuals.mean(axis=1), y, xerr=hpdi_residuals[:, 1] - hpdi_residuals.mean(axis=1),
            marker="o", ms=5, mew=4, ls="none", alpha=0.8, color="red", label="residuals", capsize=3.0, capthick=0.5)
        axs[0].set_yticks(y)
        axs[0].set_title("Residual 90% HPDI intervals")
        axs[0].set_yticklabels(self.group_names, fontsize=10);
        axs[0].legend(loc=(1.02, 0.9))

        axs[1].plot(np.ones(self.G) * hpdi_ys.mean(), y, "--", label="global true mean")
        axs[1].errorbar(
            hpdi_ys.mean(axis=1), y, xerr=hpdi_ys[:, 1] - hpdi_ys.mean(axis=1), marker="o", ms=5, mew=4,
            ls="none", alpha=0.7, label="true", capsize=3.0, capthick=0.5
        )
        axs[1].errorbar(
            hpdi_preds.mean(axis=1), y, xerr=hpdi_preds[:, 1] - hpdi_preds.mean(axis=1), marker="o", ms=5,
            mew=4, ls="none", alpha=0.7, label="predicted", color="red", capsize=3.0, capthick=0.5
        )
        axs[1].set_yticks(y)
        axs[1].set_title("True versus predicted 90% HPDI intervals")
        axs[1].set_yticklabels(["" for _ in y], fontsize=10);
        axs[1].legend(loc=(1.02, 0.85))
        plt.tight_layout()
        plt.show()

    def full_analysis(self, num_pp_samples=400, num_prior_samples=1000, data_plots=True, analysis_plots=True,
                      residuals=True, prior_plots=True, posterior_plots=True, arviz_plots=True,
                      predictor_col="mmd mean"):
        if data_plots:
            self.plot_correlations()

        if prior_plots:
            _ = self.get_prior_predictions(num_prior_samples=num_prior_samples, plot=True)
            predictive_checks(self, prior=True)

        self.run()

        if posterior_plots:
            _ = self.get_posterior_predictions(plot=True)
            predictive_checks(self, prior=False)
            self.plot_post_preds_per_group(N_cols=2)

        if arviz_plots:
            # self.plot_trace()
            self.plot_ppc(num_pp_samples=num_pp_samples)
            self.plot_posterior()
            if self.correlate_predictors:
                self.plot_weights_correlation()
            self.plot_group_bias()

        if residuals:
            self.residuals_plots()

        if analysis_plots:
            self.plot_group_specific_params(predictor_col=predictor_col)
            self.plot_regression_mmd_predictor(predictor=predictor_col, predictor_name="MMD", y_label=self.obs_y_name,
                                               plot_n_regression_lines=200, ncols=4)

