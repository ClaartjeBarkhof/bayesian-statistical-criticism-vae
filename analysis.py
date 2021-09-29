import matplotlib.pyplot as plt
import torch
from arguments import prepare_parser
from dataset_dataloader import ImageDataset
from vae_model.distributions import AutoRegressiveDistribution
import numpy as np
import scipy.stats
from scipy.stats import chisquare


def collect_encodings(vae_model, data_X, Sz=1):
    with torch.no_grad():
        q_z_x, z_post = vae_model.inf_model(data_X[:, :, :, :], n_samples=Sz)

        if isinstance(q_z_x, AutoRegressiveDistribution):
            (mean, scale) = q_z_x.params
            mean, scale = mean[0, :, :], scale[0, :, :]
        else:
            mean, scale = q_z_x.base_dist.loc, q_z_x.base_dist.scale

        return dict(z_post=z_post, mean_post=mean, scale_post=scale)


def get_n_data_samples_x_y(image_dataset_name="bmnist", N_samples=500):
    args = prepare_parser(jupyter=True, print_settings=False)
    args.image_dataset_name = image_dataset_name
    dataset = ImageDataset(args=args)

    data_X, data_y = [], []

    # [B, C, W, H] [B]
    for i, (X, y) in enumerate(dataset.valid_loader(num_workers=1, batch_size=100)):
        data_X.append(X)
        data_y.append(y)

        if (i + 1) * 100 >= N_samples:
            break

    data_X, data_y = torch.cat(data_X, dim=0), torch.cat(data_y, dim=0)

    return data_X[:N_samples], data_y[:N_samples]


def plot_posterior_analysis_grid(vae_model, plot_name, plot_dir, data_X, data_y, n_sampled_reconstructions=5):
    with torch.no_grad():

        # data sample, iw ll, ...sampled_reconstructions...
        fig, axs = plt.subplots(nrows=10, ncols=2 + n_sampled_reconstructions,
                                figsize=(1.6 * (n_sampled_reconstructions + 2), 1.6 * 10))

        for digit in range(10):
            # [B=1, C=1, W, H]
            data_sample = data_X[data_y == digit][0].unsqueeze(0)  # create a synthetic batch dim.
            axs[digit, 0].imshow(data_sample[0, 0, :, :].numpy(), cmap="Greys")
            axs[digit, 0].set_title("Data sample", size=7)

            # [B=1, C=1, W, H]
            log_p_x_per_bit = vae_model.estimate_log_likelihood_batch(data_sample, n_samples=10, per_bit=True)
            axs[digit, 1].imshow(log_p_x_per_bit[0, 0, :, :].exp(), cmap="Greys")
            axs[digit, 1].set_title("p(x) per pixel", size=7)

            # [Sx=1, Sz=n_sampled_reconstructions, B=1, C, W, H] -> [Sx=1, C, W, H]
            reconstructions = vae_model.reconstruct(data_sample, Sx=1, Sz=n_sampled_reconstructions)
            reconstructions = reconstructions.squeeze(2).squeeze(0)

            for i in range(2, n_sampled_reconstructions + 2):
                axs[digit, i].imshow(reconstructions[i - 2, 0, :, :], cmap="Greys")
                axs[digit, i].set_title("Reconstruction sample", size=7)

            for row in range(10):
                for col in range(n_sampled_reconstructions + 2):
                    axs[row, col].axis('off')

        plt.suptitle(plot_name, size=14, y=0.94)
        plt.savefig(f"{plot_dir}/{plot_name}.jpg", dpi=300)
        plt.show()


def do_prior_analysis(vae_model, knn_classifier,  data_X, data_y, gen_batch_size=100, gen_n_batches=10):
    vae_model.eval()

    with torch.no_grad():
        all_samples, all_knn_preds = [], []

        for batch_idx in range(gen_n_batches):
            # Draw samples x ~ p(z)p(x|z)

            # [Sx=1, Sz=500, B=1, C, W, H] -> [n_generative_samples, C, W, H]
            s = vae_model.gen_model.sample_generative_model(Sz=gen_batch_size, Sx=1)
            s = s.squeeze(2).squeeze(0)  # squeeze B and Sx

            # Make flat Numpy vectors
            # [n_generative_samples, C * W * H] = [n_generative_samples, 768]
            samples_flat_np = s.reshape(s.shape[0], -1).numpy()

            # [n_generative_samples]
            preds = knn_classifier.predict(samples_flat_np)

            all_samples.append(samples_flat_np)
            all_knn_preds.append(preds)

        all_samples = np.concatenate(all_samples, axis=0)
        all_knn_preds = np.concatenate(all_knn_preds)

        print("all_samples.shape, all_knn_preds.shape", all_samples.shape, all_knn_preds.shape)

        d = dict(samples=None, data=None, metrics=None)

        for digit in range(10):
            select_digit_data = data_X[data_y == digit]
            avg_data_point = select_digit_data.mean(axis=0)
            p_data_digit = len(data_X[data_y == digit]) / len(data_X)

            select_digit_samples = all_samples[all_knn_preds == digit]
            avg_sample = select_digit_samples.mean(axis=0)
            p_gen_digit = len(select_digit_samples) / len(all_samples)

            l2_d = np.linalg.norm(avg_data_point - avg_sample)

            d["samples"][digit] = select_digit_samples
            d["data"][digit] = select_digit_data
            d["digit_metrics"][digit] = {
                "avg_sample": avg_sample,
                "avg_data": avg_data_point,
                "l2_avg_sample_avg_data": l2_d,
                "frac_data_of_digit_class": p_data_digit,
                "frac_samples_digit_class": p_gen_digit,
            }

        unique, counts = np.unique(all_knn_preds, return_counts=True)
        obs_freq = counts / len(all_knn_preds)
        assert len(unique) == 10, "not all classes in generative samples, perhaps try with more samples!"
        ref_unif_fre = np.array([0.1 for _ in range(10)])
        chi = chisquare(obs_freq, ref_unif_fre)
        chi_stat, chi_p = chi.statistic, chi.pvalue

        d["chi_stat"] = chi_stat
        d["obs_freq"] = obs_freq
        d["chi_pvalue"] = chi_p

        return d


def plot_prior_analysis_grid(vae_model, plot_name, plot_dir, knn_classifier, data_X, data_y,
                             n_recon_samples=5, gen_batch_size=100, gen_n_batches=10):
    with torch.no_grad():
        d = do_prior_analysis(vae_model, knn_classifier, data_X, data_y,
                              gen_batch_size=gen_batch_size, gen_n_batches=gen_n_batches)

        cols = n_recon_samples + 1

        fig, axs = plt.subplots(ncols=cols, nrows=10, figsize=(cols * 1.5, 10 * 1.5))

        for digit in range(10):
            # Plot average data sample belonging to this digit class
            axs[digit, 0].imshow(d["digit_metrics"][digit]["avg_data"].reshape(28, 28), cmap="Greys")
            axs[digit, 0].set_title(f"Avg. data digit {digit} p: {p_data:.2f}", y=1.03)

            # Get some stats
            p_data = d["digit_metrics"][digit]["frac_data_of_digit_class"]
            p_gen = d["digit_metrics"][digit]["frac_samples_digit_class"]
            l2 = d["digit_metrics"][digit]["l2_avg_sample_avg_data"]

            # Plot some individual
            samples_class = d["samples"][digit]
            assert len(samples_class) >= n_recon_samples, f"did not find enough samples of class {digit} to plot"

            for c in range(1, n_recon_samples + 1):
                if c == (n_recon_samples // 2) + 1:
                    axs[digit, c].set_title(f"--- Samples {digit} ---")
                axs[digit, c].imshow(samples_class[c, :].reshape(28, 28), cmap="Greys")

            # Plot average generated sample belonging to this digit class
            axs[digit, -1].imshow(d["digit_metrics"][digit]["avg_sample"].reshape(28, 28), cmap="Greys")
            axs[digit, -1].set_title(f"Avg. sampled digit {digit}  p: {p_gen:.2f} l2 d.: {l2:.2f}", y=1.03)

        for row in range(10):
            for col in range(cols):
                axs[row, col].axis('off')

        plt.suptitle(plot_name, size=14, y=0.93)
        plt.savefig(f"{plot_dir}/prior-grid-{plot_name}.jpg", dpi=300)
        plt.show()

    return d


# Estimated generative sample class proportions
def plot_gen_sample_class_proportions(vae_model, knn_classifier, n_gen_samples, plot_name, plot_dir):
    with torch.no_grad():
        # [Sx=1, Sz=n_gen_samples, B=1, C, W, H] -> [n_generative_samples, C, W, H]
        gen_sample = vae_model.gen_model.sample_generative_model(Sz=n_gen_samples, Sx=1).squeeze(2).squeeze(0)
        gen_sample_flat_np = gen_sample.reshape(gen_sample.shape[0], -1).numpy()

        gen_sample_pred_cls = knn_classifier.predict(gen_sample_flat_np)

        fig, axs = plt.subplots(ncols=10, figsize=(2 * 10, 2))
        for digit in range(10):
            gen_samples_digit = gen_sample_flat_np[gen_sample_pred_cls == digit]
            prop_samples_digit = len(gen_samples_digit) / n_gen_samples
            axs[digit].imshow(gen_samples_digit.mean(axis=0).reshape(28, 28), cmap="Greys")
            axs[digit].set_title(f"{digit} frac. {prop_samples_digit:.2f}")
            axs[digit].axis('off')

        plt.suptitle(f"Estimated generative sample class proportions (N={n_gen_samples})\n{plot_name}", size=14, y=1.25)
        plt.savefig(f"{plot_dir}/prior-proportions-{plot_name}.jpg", dpi=300)
        plt.show()


def plot_latents(encodings, clean_name, plot_N_sep_posteriors=10, plot_N_encodings=200, save=None):
    x = np.linspace(-4, 4, 500)
    y = scipy.stats.norm.pdf(x, 0, 1)

    # [S, B, D] -> [B, D]
    latents = encodings["z_post"][0, :plot_N_encodings, :].cpu().numpy()

    mean = encodings["mean_post"][:plot_N_encodings, :].cpu().numpy()
    mean_mean = mean.mean(axis=0)
    std = encodings["scale_post"][:plot_N_encodings, :].cpu().numpy()
    std_mean = std.mean(axis=0)

    plot_N_dims = 10
    fig, axs = plt.subplots(nrows=1, ncols=plot_N_dims, figsize=(plot_N_dims * 2, 8))

    dims = list(np.arange(plot_N_dims))

    for i, d in enumerate(dims):
        # print(f"{i}/9", end='\r')
        axs[i].grid(b=False, which="both")

        if i > 0:
            axs[i].set_yticks([])

        # axs[i].grid = False
        axs[i].set_xticks([])
        axs[i].hist(latents[:, d], orientation="horizontal", bins=50, density=True, alpha=0.3,
                    label=f"Latent sample histogram\n(N={plot_N_encodings})")
        axs[i].set_ylim([-4, 4])

        for n in range(plot_N_sep_posteriors):
            choice_int = np.random.randint(0, len(latents))

            y_sep_post_i = scipy.stats.norm.pdf(x, mean[choice_int, d], std[choice_int, d])

            label = "$q(z|x_i)$ for randomly chosen $i$" if n == 0 else None

            axs[i].plot(y_sep_post_i, x, color='blue', linewidth=1, label=label, alpha=0.1)

        axs[i].axhline(color='b', linewidth=1)  # xmin=0, xmax=50,
        axs[i].plot(y, x, color='r', linewidth=1, label="Standard Gaussian")
        axs[i].set_title(f"$\mu$ = {mean_mean[d]:.2f}\n $\sigma^2$ = {std_mean[d]:.2f}", size=7, y=1.03)

    axs[-1].legend(loc=(0.0, 1.1), prop={'family': 'serif', 'weight': 300, 'size': 14})

    fig.suptitle(clean_name, size=16, y=1.04)

    if save is not None:
        plt.savefig(save, dpi=300, bbox_inches="tight")

    plt.show()