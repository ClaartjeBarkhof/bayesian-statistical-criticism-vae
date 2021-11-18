import numpy as np
import arviz as az
import itertools

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


class HierarchicalModel:
    def __init__(self, df, obs_y_name, obs_y_data=None, obs_dist="binomial",
                 num_samples=1000, num_warmup=1000, num_chains=3, inverse_data_transform=None):

        self.df = df

        self.decoder_names = list(df["decoder"].values)
        self.experiment_names = list(df["clean_name"].values)
        self.objective_names = list(df["objective"].values)

        # Add indicators for the data group
        self.df, self.decoder_group_df = create_group_df(self.df, ["decoder"], group_id_col="decoder_group_id",
                                                         group_name_col="decoder_group_name")
        self.df, self.objective_group_df = create_group_df(self.df, ["objective"], group_id_col="objective_group_id",
                                                           group_name_col="objective_group_name")
        self.df, self.experiment_group_df = create_group_df(self.df, ["clean_name"], group_id_col="experiment_group_id",
                                                            group_name_col="experiment_group_name")

        self.num_samples = num_samples
        self.num_chains = num_chains
        self.num_warmup = num_warmup

        self.inverse_data_transform = inverse_data_transform

        self.obs_dist = obs_dist

        self.var_names = ["global_bias", "decoder_objective_group_bias", "experiment_group_bias"]

        if self.obs_dist == "student_t":
            self.var_names += ["student_scale", "student_df"]
        elif self.obs_dist == "log_normal":
            self.var_names += ["normal_scale"]

        self.obs_y_name = obs_y_name

        self.make_init_assertions()

        self.obs_y = df[obs_y_name].values

        self.add_data_group = False if obs_y_data is None else True
        self.obs_y_data = obs_y_data
        self.obs_y_all = self.obs_y

        if self.add_data_group:
            self.obs_y_all = np.concatenate([self.obs_y_all, self.obs_y_data])

        # data group indicator, 0 is non-data group, 1 is data group
        self.data_group_id = np.ones_like(self.obs_y_all)
        if self.add_data_group:
            # set the non-data elements to 0
            self.data_group_id[:len(self.obs_y)] = 0

        self.obs_decoder_group_id = df["decoder_group_id"].values
        self.obs_objective_group_id = df["objective_group_id"].values
        self.obs_experiment_group_id = df["experiment_group_id"].values

        self.obs_decoder_group_name = df["decoder_group_name"].values
        self.obs_objective_group_name = df["objective_group_name"].values
        self.obs_experiment_group_name = df["experiment_group_name"].values

        self.G_dec = len(self.decoder_group_df)
        self.G_obj = len(self.objective_group_df)
        self.G_exp = len(self.experiment_group_df)

        self.N = len(self.df)
        if self.add_data_group:
            self.N += len(obs_y_data)

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

    def model(self, obs_y=None):
        if obs_y is not None:
            assert len(obs_y) == len(self.obs_y_all), f"len(obs_y) =!= len(obs_y_all) {len(obs_y)} =!= {len(self.obs_y_all)}"

        # GLOBAL LEVEL
        # BIAS

        global_bias = numpyro.sample("global_bias", dist.Normal(np.mean(self.obs_y), 1.))

        # TODO: inform this with population mean?
        # [2]
        with numpyro.plate("data_model_global_plate", 2):
            data_model_bias = numpyro.sample("data_model_bias", dist.Normal(global_bias, 1.))

        model_global_bias = data_model_bias[0]
        data_global_bias = data_model_bias[1]

        # SCALE and/or DF
        normal_scale, student_scale, student_df = 0.0, 0.0, 0.0

        if "normal" in self.obs_dist:
            normal_scale = numpyro.sample("normal_scale", dist.Uniform(low=0.1, high=10.0))

        elif self.obs_dist == "student_t":
            print("TEST")
            # See section 15.2 of https://jrnold.github.io/bayesian_notes/robust-regression.html:
            student_scale = numpyro.sample("student_scale", dist.Gamma(concentration=2.0, rate=0.1))
            student_df = numpyro.sample("student_df", dist.Exponential(rate=1.0 / 10.0))

        # HIERARCHICAL
        # DECODER GROUP BIAS [G_dec]
        with numpyro.plate("decoder_groups_plate", self.G_dec):
            decoder_group_bias = numpyro.sample("decoder_group_bias", dist.Normal(model_global_bias, 1.))

        # DECODER-OBJECTIVE GROUP BIAS [G_obj, G_dec]
        with numpyro.plate("decoder_groups_plate", self.G_dec):
            with numpyro.plate("objective_groups_plate", self.G_obj):
                decoder_objective_group_bias = numpyro.sample("decoder_objective_group_bias",
                                                              dist.Normal(decoder_group_bias,
                                                                          jnp.ones_like(decoder_group_bias)))

        # EXPERIMENT GROUP BIAS [G_exp]
        with numpyro.plate("experiment_groups_plate", self.G_exp):
            # I need a double indexer of length G_exp that indexes the
            # objective idx, decoder idx of those experiments
            obj_idx_dec_idx = self.df[["decoder_group_id", "objective_group_id", "experiment_group_id"]].drop_duplicates().sort_values("experiment_group_id")[["objective_group_id", "decoder_group_id"]].values

            experiment_means = jnp.array(decoder_objective_group_bias[obj_idx_dec_idx[:, 0], obj_idx_dec_idx[:, 1]])
            experiment_group_bias = numpyro.sample("experiment_group_bias",
                                                   dist.Normal(experiment_means,
                                                               jnp.ones_like(experiment_means)))

        # Add up all the biases
        bias_model_groups = numpyro.deterministic("bias_model_groups", experiment_group_bias[self.obs_experiment_group_id])

        # For all add the global bias
        bias_data_group = numpyro.deterministic("bias_data_group", jnp.ones((self.data_group_id == 1).sum()) * data_global_bias)

        # For only the NON-data group add the group-objective-experiment specific biases
        # (not sure if in place operations would be accepted)
        bias_all = jnp.concatenate([bias_model_groups, bias_data_group])

        assert len(bias_all) == len(self.data_group_id), "length bias all incorrect"

        # Sample y
        with numpyro.plate("data", self.N):
            if self.obs_dist == "binomial":
                dist_y = dist.Binomial(self.total_count, logits=bias_all)
            elif self.obs_dist == "student_t":
                dist_y = dist.StudentT(df=student_df, loc=bias_all, scale=student_scale)
            elif self.obs_dist == "log_normal":
                dist_y = dist.LogNormal(loc=bias_all, scale=normal_scale)
            elif self.obs_dist == "normal":
                dist_y = dist.Normal(loc=bias_all, scale=normal_scale)
            elif self.obs_dist == "truncated_normal":
                dist_y = dist.TruncatedNormal(low=0.0, loc=bias_all, scale=normal_scale)

            numpyro.sample("obs", dist_y, obs=obs_y)

    def print_init(self):
        print(f"Optimisation:")
        print(f"MCMC NUTS algo: N_warmup={self.num_warmup}, N_samples={self.num_samples}, N_chains={self.num_chains}")

    def make_init_assertions(self):
        assert self.obs_y_name in self.df.columns, "the target variable y must be a column of the DF"
        valid_dists = ["binomial", "student_t", "log_normal", "normal", "truncated_normal"]
        assert self.obs_dist in valid_dists, f"the observation distribution must be in {valid_dists}"
        for name in ["decoder", "objective", "clean_name"]:
            assert name in self.df.columns, f"{name} must be present in columns to group by"

    def run(self):
        self.mcmc.run(self.rng_key, obs_y=self.obs_y_all)
        self.run_bool = True
        self.mcmc.print_summary()
        # group_by_chain=False: [K*D, ...]
        self.posterior_samples = self.mcmc.get_samples(group_by_chain=False)
        print("posterior samples shapes:")
        for k, v in self.posterior_samples.items():
            print(k, v.shape)

    def get_prior_predictions(self, num_prior_samples=100):
        rng_key, rng_key_ = random.split(self.rng_key)

        if self.prior_predictive is None:
            self.prior_predictive = Predictive(self.model, num_samples=num_prior_samples)

        # [num_prior_samples, N]: for every x take multiple prior samples
        self.prior_predictions = self.prior_predictive(rng_key_, obs_y=None)

        return self.prior_predictions

    def get_posterior_predictions(self):
        self.run_check()

        rng_key, rng_key_ = random.split(self.rng_key)
        if self.posterior_predictive is None:
            self.posterior_predictive = Predictive(self.model, self.posterior_samples)

        self.posterior_predictions = self.posterior_predictive(rng_key_)

        return self.posterior_predictions

    def run_check(self):
        if not self.run_bool:
            self.run()

    def plot_check(self):
        self.run_check()
        if self.arviz_data is None:
            self.arviz_data = az.from_numpyro(posterior=self.mcmc, num_chains=self.num_chains)

    def plot_trace(self):
        self.plot_check()
        az.plot_trace(self.arviz_data, var_names=self.var_names, compact=False)

    def plot_posterior(self):
        self.plot_check()
        az.plot_posterior(self.arviz_data, var_names=self.var_names)

    def plot_group_bias(self):
        self.plot_check()

        az.plot_forest(self.arviz_data, kind='forestplot', var_names=["decoder_objective_group_bias"])
        az.plot_forest(self.arviz_data, kind='forestplot', var_names=["experiment_group_bias"])

    def full_analysis(self, num_prior_samples=1000, prior_plots=True, posterior_plots=True,
                      arviz_plots=False, analysis_plots=True):

        if analysis_plots:
            self.plot_data_vs_model()

        if prior_plots:
            _ = self.get_prior_predictions(num_prior_samples=num_prior_samples)
            predictive_checks(self, prior=True)

        self.run()

        if posterior_plots:
            _ = self.get_posterior_predictions()
            predictive_checks(self, prior=False)

        if arviz_plots:
            # self.plot_trace()
            # self.plot_posterior()
            self.plot_group_bias()

        if analysis_plots:
            self.plot_hierarchical_biases()
            self.plot_hierarchical_predictions()

    @staticmethod
    def ax_hist(data, ax, c, label):
        if "light" in c:
            alpha = 0.8
        else:
            alpha = 0.5

        ax.hist(data.flatten(), bins=40, alpha=alpha, density=True, color=c, lw=0, label=label)

    def plot_hierarchical_predictions(self):
        c_dict = {
            "data preds": "green",
            "data obs": "lightgreen",
            "model preds": "blue",
            "model obs": "lightblue"
        }

        # [S, N]
        preds_all = np.array(self.posterior_predictions["obs"])
        preds_model = preds_all[:, self.data_group_id == 0]
        preds_data = preds_all[:, self.data_group_id == 1]

        # [N]
        obs_model = np.array(self.obs_y)
        obs_data = np.array(self.obs_y_data)

        # all
        fig, ax = plt.subplots(figsize=(4, 4))
        self.ax_hist(data=preds_model, ax=ax, label="model preds", c=c_dict["model preds"])
        self.ax_hist(data=preds_data, ax=ax, label="data preds", c=c_dict["data preds"])
        self.ax_hist(data=obs_model, ax=ax, label="model obs", c=c_dict["model obs"])
        self.ax_hist(data=obs_data, ax=ax, label="data obs", c=c_dict["data obs"])
        ax.legend()

        # all data group & all model group
        fig, axs = plt.subplots(ncols=2, figsize=(4 * 2, 4), sharex=True, sharey=True)
        self.ax_hist(data=preds_model, ax=axs[0], label="model preds", c=c_dict["model preds"])
        self.ax_hist(data=preds_data, ax=axs[1], label="data preds", c=c_dict["data preds"])
        self.ax_hist(data=obs_model, ax=axs[0], label="model obs", c=c_dict["model obs"])
        self.ax_hist(data=obs_data, ax=axs[1], label="data obs", c=c_dict["data obs"])
        axs[0].legend()
        axs[1].legend()
        axs[0].set_title("Model groups preds vs obs")
        axs[1].set_title("Data group preds vs obs")

        # TODO: all decoder groups
        # TODO: all objective groups

        # all decoder-objective groups
        ncols = len(self.objective_group_df["objective_group_name"].values)
        nrows = len(self.decoder_group_df["decoder_group_name"].values)
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4 * ncols, 4 * nrows))  # , sharex=True, sharey=True

        for row, d in enumerate(sorted(self.decoder_group_df["decoder_group_id"].unique())):
            for col, o in enumerate(sorted(self.objective_group_df["objective_group_id"].unique())):
                dec = self.decoder_group_df[self.decoder_group_df["decoder_group_id"] == d]["decoder_group_name"].values[0]
                obj = self.objective_group_df[self.objective_group_df["objective_group_id"] == d][
                    "objective_group_name"].values[0]

                obs_model_o_d = self.obs_y[((self.obs_objective_group_id == o) & (self.obs_decoder_group_id == d))]
                preds_model_o_d = preds_model[:,
                                  ((self.obs_objective_group_id == o) & (self.obs_decoder_group_id == d))]

                self.ax_hist(data=obs_model_o_d, ax=axs[row, col], label="model obs", c=c_dict["model obs"])
                self.ax_hist(data=preds_model_o_d, ax=axs[row, col], label="model preds", c=c_dict["model preds"])

                self.ax_hist(data=obs_data, ax=axs[row, col], label="data obs", c=c_dict["data obs"])
                self.ax_hist(data=preds_data, ax=axs[row, col], label="data preds", c=c_dict["data preds"])

                #axs[row, col].legend()
                axs[row, col].set_title(f"{dec} - {obj}")

        axs[row, col].legend(loc=(1.0, 0.6))

        plt.suptitle("Preds versus obs | model versus data groups | per objective-decoder")
        plt.tight_layout()
        plt.show()



    def plot_hierarchical_biases(self):
        conf_prop = 0.9

        samples = self.posterior_samples

        # [S,] = 1
        global_bias = samples["global_bias"]
        conf_global_bias = hpdi(global_bias, axis=0, prob=conf_prop)[:, None]

        # [S, 2] = 2
        data_model_bias = samples["data_model_bias"]
        conf_data_global_bias = hpdi(data_model_bias, axis=0, prob=conf_prop)
        data_model_labels = ["model_bias", "data_bias"]

        # [S, 2] = 2
        decoder_group_bias = samples["decoder_group_bias"]
        conf_dec_group = hpdi(decoder_group_bias, axis=0, prob=conf_prop)
        dec_group_labels = self.decoder_group_df.sort_values("decoder_group_id")["decoder_group_name"].values.tolist()

        #print(conf_dec_group.shape)
        #print(len(dec_group_labels))

        # [S, 4, 2] = 8
        decoder_objective_group_bias = samples["decoder_objective_group_bias"]
        conf_dec_obj_group = hpdi(decoder_objective_group_bias, axis=0, prob=conf_prop)
        conf_dec_obj_group_flat = conf_dec_obj_group.reshape(2, -1)
        objective_group_labels = self.objective_group_df.sort_values("objective_group_id")[
            "objective_group_name"].values.tolist()
        dec_obj_group_labels = list(itertools.product(objective_group_labels, dec_group_labels))

        # print(len(dec_obj_group_labels))
        # print(conf_dec_obj_group_flat.shape)

        # [S, 76] = 76
        experiment_group_bias = samples["experiment_group_bias"]
        conf_exp_group = hpdi(experiment_group_bias, axis=0, prob=conf_prop)
        experiment_group_labels = self.experiment_group_df.sort_values("experiment_group_id")[
            "experiment_group_name"].values.tolist()

        # print(len(experiment_group_labels))
        # print(conf_exp_group.shape)

        all_confs = np.concatenate(
            [conf_global_bias.transpose(), conf_data_global_bias.transpose(), conf_dec_group.transpose(),
             conf_dec_obj_group_flat.transpose(), conf_exp_group.transpose()], axis=0)
        all_labels = [
                         "global_bias"] + data_model_labels + dec_group_labels + dec_obj_group_labels + experiment_group_labels
        all_labels = [f"{len(all_labels) - i} {l}" for i, l in enumerate(all_labels)]

        # print(all_confs.shape)
        # print(len(all_labels))

        means = all_confs.mean(axis=1)
        err = all_confs[:, 1] - means
        y = np.arange(len(all_labels))

        fig, ax = plt.subplots(figsize=(8, 20))
        ax.errorbar(means, y, xerr=err, marker="o", ms=5, mew=4, ls="none", alpha=0.8, capsize=3.0, capthick=0.5)
        ax.set_yticks(y)
        ax.set_yticklabels(all_labels, fontsize=10);
        plt.show()

    def plot_data_vs_model(self):
        if self.add_data_group:
            plt.title(f"{self.obs_y_name} statitic data versus model samples")
            _ = plt.hist(self.obs_y_data, bins=40, alpha=0.5, density=True, label="data")
            _ = plt.hist(self.obs_y, bins=40, alpha=0.5, density=True, label="model samples")
            plt.legend()
            plt.show()