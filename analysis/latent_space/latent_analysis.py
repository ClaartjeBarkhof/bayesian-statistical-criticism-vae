import scipy
import seaborn as sns; sns.set()
import numpy as np
import matplotlib.pyplot as plt


def plot_latents(encodings, clean_name, plot_N_sep_posteriors=10, plot_N_dims = 10, plot_N_encodings=200, save=None):
    x = np.linspace(-4, 4, 500)
    y = scipy.stats.norm.pdf(x, 0, 1)

    # [S, B, D] -> [B, D]
    latents = encodings["z"][:plot_N_encodings, :].cpu().numpy()

    mean = encodings["mean"][:plot_N_encodings, :].cpu().numpy()
    mean_mean = mean.mean(axis=0)
    std = encodings["scale"][:plot_N_encodings, :].cpu().numpy()
    std_mean = std.mean(axis=0)


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
