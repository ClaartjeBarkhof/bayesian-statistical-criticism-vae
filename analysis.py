import matplotlib.pyplot as plt
import torch
from arguments import prepare_parser
from dataset_dataloader import ImageDataset


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


def build_prior_analysis_grid(vae_model, plot_name, plot_dir, knn_classifier, n_generative_samples, data_X, data_y,
                              show_n_samples=5, avg_n_data_samples=30):
    with torch.no_grad():
        # Draw samples x ~ p(z)p(x|z)
        # [1, n_generative_samples, C, W, H] -> [n_generative_samples, C, W, H]
        s = vae_model.gen_model.sample_generative_model(S=n_generative_samples).squeeze(0)

        # Make flat Numpy vectors
        # [n_generative_samples, C * W * H] = [n_generative_samples, 768]
        samples_flat_np = s.reshape(s.shape[0], -1).numpy()

        # [n_generative_samples]
        preds = knn_classifier.predict(samples_flat_np)

        cols = show_n_samples + 2
        fig, axs = plt.subplots(ncols=cols, nrows=10, figsize=(cols * 1.5, 10 * 1.5))

        for digit in range(10):
            avg_data_point = data_X[data_y == digit][:avg_n_data_samples].mean(axis=0).reshape(28, 28)
            axs[digit, 0].imshow(avg_data_point, cmap="Greys")
            axs[digit, 0].set_title(f"Avg. data digit {digit}")

            select_digit_samples = samples_flat_np[preds == digit]
            for c in range(1, show_n_samples + 1):
                axs[digit, c].imshow(select_digit_samples[c, :].reshape(28, 28), cmap="Greys")
                axs[digit, 0].set_title(f"Sampled digit {digit}")

            axs[digit, -1].imshow(select_digit_samples.mean(axis=0).reshape(28, 28), cmap="Greys")
            axs[digit, -1].set_title(f"Avg. sampled digit {digit}")

        for row in range(10):
            for col in range(cols):
                axs[row, col].axis('off')

        plt.suptitle(plot_name, size=14, y=0.93)
        plt.savefig(f"{plot_dir}/{plot_name}.jpg", dpi=300)
        plt.show()


def build_posterior_analysis_grid(vae_model, plot_name, plot_dir, data_X, data_y, n_sampled_reconstructions=5):
    with torch.no_grad():

        # data sample, iw ll, ...sampled_reconstructions...
        fig, axs = plt.subplots(nrows=10, ncols=2 + n_sampled_reconstructions,
                                figsize=(1.6 * (n_sampled_reconstructions + 2), 1.6 * 10))

        for digit in range(10):
            # [1, C, W, H]
            data_sample = data_X[data_y == digit][0].unsqueeze(0)  # create a synthetic batch dim.
            axs[digit, 0].imshow(data_sample[0, 0, :, :].numpy(), cmap="Greys")
            axs[digit, 0].set_title("Data sample", size=7)

            # [1, C, W, H]
            log_p_x_per_bit = vae_model.estimate_log_likelihood_batch(data_sample, n_samples=10, per_bit=True)
            axs[digit, 1].imshow(log_p_x_per_bit[0, 0, :, :].exp(), cmap="Greys")
            axs[digit, 1].set_title("p(x) per pixel", size=7)

            reconstructions = vae_model.reconstruct(data_sample, n_x_samples=n_sampled_reconstructions).squeeze(
                1).squeeze(1).squeeze(1)
            print("XX Reconstructions.shape", reconstructions.shape)

            for i in range(2, n_sampled_reconstructions + 2):
                axs[digit, i].imshow(reconstructions[i - 2, :, :], cmap="Greys")
                axs[digit, i].set_title("Reconstruction sample", size=7)

            for row in range(10):
                for col in range(n_sampled_reconstructions + 2):
                    axs[row, col].axis('off')

        plt.suptitle(plot_name, size=14, y=0.94)
        plt.savefig(f"{plot_dir}/{plot_name}.jpg", dpi=300)
        plt.show()