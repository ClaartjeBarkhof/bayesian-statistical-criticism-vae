import pytorch_lightning as pl
from lagrangian_opt import constraint
from torch.distributions import kl_divergence
import torch


# TODO: NLL / reconstruction loss / distortion

# TODO: avg. KL divergence / rate

# TODO: MMD

# TODO: total correlation

# class LossTermManager:
#     def __init__(self, name):

def free_bits_kl(q_z_x, z_post, free_bits=0.0, per_dimension=True):
    """
    Calculates the KL-divergence between the posterior and the prior analytically.
    """

    # [B, D]
    mus = q_z_x.loc
    # [B, D, D] -> [B, D] (we assume independent dimensions)
    logvar = torch.log(q_z_x.covariance_matrix.diagonal(dim1=-1))  # TODO: check this

    kl_loss = 0.5 * (mus.pow(2) + logvar.exp() - logvar - 1)
    free_bits_kl_loss = None

    if free_bits > 0.0 and per_dimension:

        # Ignore the dimensions of which the KL-div is already under the
        # threshold, avoiding driving it down even further. Those values do
        # not have to be replaced by the threshold because that would not mean
        # anything to the gradient. That's why they are simply removed. This
        # confused me at first.
        kl_mask = (kl_loss > free_bits).float()

        # Sum over the latent dimensions and average over the batch dimension
        free_bits_kl_loss = (kl_mask * kl_loss).sum(-1).mean(0)

    elif free_bits > 0.0 and not per_dimension:
        # Reduce latent dimensions [B, D] -> [B]
        kl_loss_red = kl_loss.sum(-1)

        # Mask
        kl_mask = (kl_loss_red > free_bits).float()
        free_bits_kl_loss = (kl_mask * kl_loss_red).mean(0)

    kl_loss = kl_loss.sum(-1).mean(0)

    return kl_loss, free_bits_kl_loss


class Objective(pl.LightningModule):
    def __init__(self, objective, vae):
        super().__init__()

        # VAE, FB, MDR,
        self.objective = objective
        self.vae = vae
        self.prior = self.vae.prior

    def compute_loss(self, labels, q_z_x, z_post, p_x_z):
        # TODO: implement other types of losses

        # Expected KL from prior to posterior
        rate = self.kl_post_prior(q_z_x)

        # Expected negative log likelihood under q
        distortion = - p_x_z.log_prob(labels).sum(-1).mean(0).item()

        total_loss = distortion + rate

        return total_loss

    def kl_post_prior(self, q_z_x, z_post=None, analytical=True):
        """
        Computes the KL from prior to posterior, either analytically or empirically,
        depending on whether the posterior distribution is given.

        Args:
        Returns:
        """

        if analytical:
            # [B] (summed over latent dimension)
            kl = kl_divergence(q_z_x, self.prior)

        else:
            log_q_z_x = q_z_x.log_prob(z_post)
            log_p_z = self.prior.log_prob(z_post)
            kl = log_q_z_x - log_p_z

        return kl

    @staticmethod
    def gaussian_kernel(x, y):
        """
        Taken from: https://github.com/aktersnurra/information-maximizing-variational-autoencoders/blob/master/model/loss_functions/mmd_loss.py
        """
        x_size = x.size(0)
        y_size = y.size(0)
        dim = x.size(1)
        x = x.unsqueeze(1)
        y = y.unsqueeze(0)
        tiled_x = x.expand(x_size, y_size, dim)
        tiled_y = y.expand(x_size, y_size, dim)
        kernel_input = (tiled_x - tiled_y).pow(2).mean(2) / float(dim)
        return torch.exp(-torch.tensor(kernel_input))

    def maximum_mean_discrepancy(self, z_post):
        """
        Taken from: https://github.com/aktersnurra/information-maximizing-variational-autoencoders/blob/master/model/loss_functions/mmd_loss.py

        """
        x = torch.randn_like(z_post).to(z_post.device)
        y = z_post
        x_kernel = self.gaussian_kernel(x, x)
        y_kernel = self.gaussian_kernel(y, y)
        xy_kernel = self.gaussian_kernel(x, y)
        mmd = x_kernel.mean() + y_kernel.mean() - 2 * xy_kernel.mean()

        return mmd
