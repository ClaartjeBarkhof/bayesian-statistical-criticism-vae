from torch.distributions import kl_divergence
from torch.optim import RMSprop
import torch
import torch.nn as nn
import torch.distributions as td


class Objective(nn.Module):
    """
    This class handles the objective we choose to optimise our deep generative latent variable model with.

    It handles the following objectives:
      - VAE
      - AE
      - BETA-VAE, with beta argument set (Higgins et al., 2016)
      - FB-VAE (Kingma et al., 2016)
      - MDR-VAE (Pelsmaeker & Aziz, 2019)
      - INFO-VAE, with alpha and lambda argument set  (Zhao et al., 2017)
      - LAG-INFO-VAE (Zhao et al., 2017)
    """
    def __init__(self, args, mdr_constraint=None):
        super(Objective, self).__init__()

        self.objective = args.objective
        self.data_distribution = args.data_distribution
        self.image_or_language = args.image_or_language
        self.args = args
        self.mdr_constraint = mdr_constraint

    def compute_loss(self, x_in, q_z_x, z_post, p_z, p_x_z):
        """
        This function computes statistics and assembles the loss for which we optimise
        based on which objective is used.

        Args:
            x_in: [B, C, W, H] (image) or [B, L] (language)
                the input batch (language sequences or images)
            q_z_x:
                a posterior distribution(-like) object
            z_post: [B, D]
                batch of posterior samples
            p_z:
                a prior distribution object
            p_x_z:
                a likelihood distribution(-like) object
        Returns:
            loss_dict:
                a dictionary containing statistics to log and 'total_loss' which is used for optimisation
        """

        B = x_in.shape[0]

        if self.image_or_language == "image" and self.data_distribution == "multinomial":
            # map 0, 1 range to 0 256 integer range, code taken from:
            # https://github.com/riannevdberg/sylvester-flows/blob/ \
            # 32dde9b7d696fee94f946a338182e542779eecfe/optimization/loss.py#L74
            num_classes = 256
            labels = (x_in * (num_classes - 1)).long()
        else:
            # TODO: for language should be something along the lines of x[:, 1:] (cutting of the start token)
            labels = x_in

        # Expected KL from prior to posterior
        kl_prior_post = self.kl_prior_post(p_z=p_z, q_z_x=q_z_x, z_post=z_post, analytical=True)

        # Expected negative log likelihood under q
        # TODO: there might be some kind of masking necessary

        # Reduce all dimensions with sum, except for the batch dimension, average that
        if self.args.decoder_network_type == "conditional_made_decoder":
            # In case of the MADE, the evaluation is in flattened form.
            labels = labels.reshape(B, -1)

        nll = - p_x_z.log_prob(labels).reshape(self.args.batch_size, -1).mean()

        # Maximum mean discrepancy
        # z_post at this point is [1, B, D]
        mmd = self.maximum_mean_discrepancy(z_post.squeeze(0))

        total_loss, mdr_loss = None, None
        if self.objective == "AE":
            total_loss = nll
        elif self.objective == "VAE":
            total_loss = nll + kl_prior_post
        elif self.objective == "BETA-VAE":
            total_loss = nll + self.args.beta_beta
        elif self.objective == "MDR-VAE":
            # the gradients for the constraints are recorded at some other point
            mdr_loss = self.mdr_constraint(kl_prior_post).squeeze()
            total_loss = nll + kl_prior_post + mdr_loss
        elif self.objective == "FB-VAE":
            raise NotImplementedError
        elif self.objective == "INFO-VAE":
            # gain = ll - (1 - a)*kl_prior_post - (a + l - 1)*marg_kl
            # loss = nll + (1 - a)*kl_prior_post + (a + l - 1)*marg_kl
            total_loss = nll + ((1 - self.args.info_alpha) * kl_prior_post) \
                         + ((self.args.info_alpha + self.args.info_lambda - 1) * mmd)
        elif self.objective == "LAG-INFO-VAE":
            # https://github.com/ermongroup/lagvae/blob/master/methods/lagvae.py
            raise NotImplementedError

        loss_dict = dict(
            total_loss=total_loss,
            mmd=mmd,
            mdr_loss=mdr_loss,
            nll=nll,
            kl_prior_post=kl_prior_post
        )

        return loss_dict

    @staticmethod
    def kl_prior_post(p_z, q_z_x, z_post=None, analytical=False):
        """Computes the KL from prior to posterior, either analytically or empirically."""

        # A bit of a hack to avoid this kl that raises a NotImplementedError (TODO: make this possible)
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
