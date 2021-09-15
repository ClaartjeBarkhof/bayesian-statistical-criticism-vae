from lagrangian_opt.constraint import Constraint, ConstraintOptimizer
from torch.distributions import kl_divergence
from torch.optim import RMSprop
import torch
import torch.nn as nn
import torch.distributions as td
import numpy as np
from vae_model.inference_model import AutoRegressiveGaussianDistribution

# TODO: total correlation, dimensionwise KL
# TODO: Free bits KL for Non-Gaussian case

class Objective(nn.Module):
    def __init__(self, args):
        super(Objective, self).__init__()

        self.objective = args.objective
        self.data_distribution = args.data_distribution
        self.image_or_language = args.image_or_language
        self.args = args

        # Rate constraint
        if self.objective == "MDR-VAE":
            self.mdr_constraint = Constraint(args.mdr_value, ">", alpha=0.5)
            self.mdr_optimiser = ConstraintOptimizer(RMSprop(), self.mdr_constraint.parameters(), 0.00005)

    def compute_loss(self, x_in, q_z_x, z_post, p_z, p_x_z):
        if self.image_or_language == "image" and self.data_distribution == "multinomial":
            num_classes = 256
            labels = (x_in * (num_classes - 1)).long()
        else:
            # TODO: for language should be something along the lines of x[:, 1:] (cutting of the start token)
            labels = x_in

        # TODO: implement other types of losses
        # Expected KL from prior to posterior
        kl_prior_post = self.kl_prior_post(p_z=p_z, q_z_x=q_z_x, z_post=z_post, analytical=True)

        # Expected negative log likelihood under q
        # TODO: there might be some kind of masking necessary

        # Reduce all dimensions with sum, except for the batch dimension, average that
        nll = - p_x_z.log_prob(labels).reshape(self.args.batch_size, -1).mean()

        # Maximum mean discrepancy
        # z_post at this point is [1, B, D]
        mmd = self.maximum_mean_discrepancy(z_post.squeeze(0))

        total_loss = None
        if self.objective == "AE":
            total_loss = nll
        elif self.objective == "VAE":
            total_loss = nll + kl_prior_post
        elif self.objective == "BETA-VAE":
            total_loss = nll + self.args.beta_beta
        elif self.objective == "MDR-VAE":
            total_loss = nll + kl_prior_post + self.mdr_constraint(kl_prior_post)
        elif self.objective == "FB-VAE":
            total_loss = nll + self.free_bits_kl(q_z_x, z_post, free_bits=self.args.free_bits,
                                                 per_dimension=self.args.free_bits_per_dimension)
        elif self.objective == "INFO-VAE":
            total_loss = - nll + (1 - self.args.info_alpha) * kl_prior_post \
                         + (self.args.info_alpha + self.args.info_lambda - 1) * mmd

        loss_dict = dict(
            total_loss=total_loss,
            mmd=mmd.item(),
            nll=nll.item(),
            kl_prior_post=kl_prior_post.item()
        )

        return loss_dict


    def kl_prior_post(self, p_z, q_z_x, z_post=None, analytical=False):
        """
        Computes the KL from prior to posterior, either analytically or empirically,
        depending on whether the posterior distribution is given.

        Args:
        Returns:
        """

        # A bit of a hack to avoid this kl that raises a NotImplementedError
        if isinstance(p_z, td.MixtureSameFamily):
            analytical = False

        if analytical:

            kl = kl_divergence(q_z_x, p_z).mean()

        else:
            # [B]
            log_q_z_x = q_z_x.log_prob(z_post)
            log_p_z = p_z.log_prob(z_post)

            kl = (log_q_z_x - log_p_z).mean()

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
        kernel_input = (tiled_x - tiled_y).pow(2).mean(2) / torch.Tensor([float(dim)])

        return torch.exp(-kernel_input)

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
