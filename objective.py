from torch.distributions import kl_divergence
import torch
import torch.nn as nn
import torch.distributions as td
from pytorch_constrained_opt.constraint import Constraint
from torch_two_sample import MMDStatistic
from vae_model.distributions import AutoRegressiveDistribution
import numpy as np


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

    def __init__(self, args, device="cpu"):
        super(Objective, self).__init__()

        self.device = device
        self.data_distribution = args.data_distribution
        self.image_or_language = args.image_or_language
        self.args = args

        self.mdr_constraint = self.get_mdr_constraint()

    def get_mdr_constraint(self):
        return Constraint(self.args.mdr_value, "ge", alpha=0.5).to(
            self.device) if self.args.objective == "MDR-VAE" else None

    def compute_loss(self, x_in, q_z_x, z_post, p_z, p_x_z):
        """
        This function computes statistics and assembles the loss for which we optimise
        based on which objective is used.

        Args:
            x_in: [B, C, W, H] (image) or [B, L] (language)
                the input batch (language sequences or images)
            q_z_x:
                a posterior distribution(-like) object
            z_post: [S, B, D]
                batch of posterior samples
            p_z:
                a prior distribution object
            p_x_z:
                a likelihood distribution(-like) object:
        Returns:
            loss_dict:
                a dictionary containing statistics to log and 'total_loss' which is used for optimisation
        """

        (S, B, D) = z_post.shape

        if self.image_or_language == "image" and self.data_distribution == "multinomial":
            # map 0, 1 range to 0 256 integer range, code taken from:
            # https://github.com/riannevdberg/sylvester-flows/blob/ \
            # 32dde9b7d696fee94f946a338182e542779eecfe/optimization/loss.py#L74
            num_classes = 256
            labels = (x_in * (num_classes - 1)).long()
        else:
            labels = x_in

        # Expected KL from prior to posterior (scalar)
        kl_prior_post = self.kl_prior_post(p_z=p_z, q_z_x=q_z_x, batch_size=B, z_post=z_post, analytical=True)

        # Distortion [S, B]
        log_p_x_z = p_x_z.log_prob(labels)
        assert log_p_x_z.shape == (S, B), f"we assume p_x_z.log_prob shape to be be (S, B), currently {log_p_x_z.shape}"
        # Average over samples and batch: [S, B] -> scalar
        distortion = - log_p_x_z.mean()

        # TODO: self.free_bits_kl(p_z, q_z_x, z_post, free_bits=self.args.free_bits, per_dimension=False)

        # Maximum mean discrepancy
        # z_post at this point is [S, B, D]
        # mmd = scalar tensor
        mmd = self.maximum_mean_discrepancy(z_post)

        elbo = - (distortion + kl_prior_post)

        total_loss, mdr_loss, mdr_multiplier = None, None, None

        if self.args.objective == "AE":
            total_loss = distortion

        elif self.args.objective == "VAE":
            total_loss = -elbo

        elif self.args.objective == "BETA-VAE":
            total_loss = distortion + self.args.beta_beta

        elif self.args.objective == "MDR-VAE":
            # the gradients for the constraints are recorded at some other point
            mdr_loss = self.mdr_constraint(kl_prior_post).squeeze()
            mdr_multiplier = self.mdr_constraint.multiplier
            total_loss = distortion + kl_prior_post + mdr_loss

        elif self.args.objective == "FB-VAE":
            # TODO:
            raise NotImplementedError

        elif self.args.objective == "INFO-VAE":
            # gain = ll - (1 - a)*kl_prior_post - (a + l - 1)*marg_kl
            # loss = distortion + (1 - a)*kl_prior_post + (a + l - 1)*marg_kl
            total_loss = distortion + ((1 - self.args.info_alpha) * kl_prior_post) \
                         + ((self.args.info_alpha + self.args.info_lambda - 1) * mmd)

        elif self.args.objective == "LAG-INFO-VAE":
            # TODO: https://github.com/ermongroup/lagvae/blob/master/methods/lagvae.py
            raise NotImplementedError

        posterior_stats = self.get_posterior_stats(q_z_x, z_post)

        loss_dict = dict(
            total_loss=total_loss,
            mmd=mmd,
            elbo=elbo,
            mdr_loss=mdr_loss,
            mdr_multiplier=mdr_multiplier,
            distortion=distortion,
            kl_prior_post=kl_prior_post
        )

        loss_dict = {**loss_dict, **posterior_stats}

        return loss_dict

    @staticmethod
    def kl_prior_post(p_z, q_z_x, batch_size, z_post=None, analytical=False):
        """Computes the KL from prior to posterior, either analytically or empirically."""

        # A bit of a hack to avoid this kl that raises a NotImplementedError (TODO: make this possible)
        if isinstance(p_z, td.MixtureSameFamily):
            analytical = False

        if analytical:
            kl = kl_divergence(q_z_x, p_z)

        else:
            print("Z_post.shape", z_post.shape)
            print("p_z", p_z)
            print("q_z_x", q_z_x)

            # [S, B] -> [B]
            log_q_z_x = q_z_x.log_prob(z_post).mean(dim=0)

            # else:
            log_p_z = p_z.log_prob(z_post).mean(dim=0)

            kl = (log_q_z_x - log_p_z)

        assert kl.shape == (batch_size,), \
            f"We assume kl_divergence to return one scalar per data point in the batch, current shape: {kl.shape}"

        return kl.mean()

    @staticmethod
    def free_bits_kl(p_z, q_z_x, z_post, free_bits=0.5, per_dimension=True):
        # TODO: come up with solution for the fact that if p_z is of type MixtureSameFamily,
        # TODO: the latent dimensions are automatically reduced due to the component distribution being of type Independent

        if isinstance(q_z_x, AutoRegressiveDistribution):
            (mean, scale) = q_z_x.params
            mean, scale = mean[0, :, :], scale[0, :, :]

        # Independent Normal
        else:
            mean, scale = q_z_x.base_dist.loc, q_z_x.base_dist.scale

        # mean, scale = [B, D]

        # average over the sample dim and latent dim: [S, B, D] - > [B]
        log_q_z_x = td.Normal(mean, scale).log_prob(z_post).mean(dim=0).mean(dim=1)  # TODO: get rid of this second mean
        log_p_z = p_z.log_prob(z_post).mean(dim=0)

        kl = log_q_z_x - log_p_z

        # TODO: reintroduce this if stement
        # # [B, D] -> [B]
        # if not per_dimension:
        #     kl = kl.mean(dim=1)

        # Ignore the dimensions of which the KL-div is already under the
        # threshold, avoiding driving it down even further. Those values do
        # not have to be replaced by the threshold because that would not mean
        # anything to the gradient. That's why they are simply removed. This
        # confused me at first.
        kl_mask = (kl > free_bits).float()

        kl_fb = kl * kl_mask

        return kl_fb.mean()

    @staticmethod
    def maximum_mean_discrepancy(z_post):
        # [S, B, D] -> [B, D]
        z_post = z_post.reshape(-1, z_post.shape[-1])
        prior_sample = torch.randn_like(z_post)  # .to(z_post.device)
        alphas = [0.1 * i for i in range(5)]  # TODO: no clue for these...

        n_1, n_2 = len(z_post), len(prior_sample)
        MMD_stat = MMDStatistic(n_1, n_2)
        tts_mmd = MMD_stat(z_post, prior_sample, alphas, ret_matrix=False)

        return tts_mmd

    @staticmethod
    def get_posterior_stats(q_z_x, z_post):
        if isinstance(q_z_x, td.Independent):
            mean, scale = q_z_x.base_dist.loc, q_z_x.base_dist.scale
        else:
            (mean, scale) = q_z_x.params

        # [1, B, D] -> [B, D]
        mean = mean.squeeze(0)
        scale = scale.squeeze(0)

        mean_mean = mean.mean()
        std_across_x_mean = torch.std(mean, dim=0).mean()
        std_across_z_mean = torch.std(mean, dim=1).mean()

        mean_scale = scale.mean()
        std_across_x_scale = torch.std(scale, dim=0).mean()
        std_across_z_scale = torch.std(scale, dim=1).mean()

        # Log determinant of the covariance matrix of q_z_x
        # metric from InfoVAE paper (Zhao et al.)
        # for np.cov, the observation dimension is expected to be the column dimension
        z_post_np = z_post.squeeze(0).detach().cpu().numpy().transpose()
        cov = np.cov(z_post_np)
        log_det_cov_q_z = np.log(np.linalg.det(cov))

        d = dict(
            mean_mean=mean_mean,
            std_across_x_mean=std_across_x_mean,
            std_across_z_mean=std_across_z_mean,
            mean_scale=mean_scale,
            std_across_x_scale=std_across_x_scale,
            std_across_z_scale=std_across_z_scale,
            log_det_cov_q_z=log_det_cov_q_z
        )

        return d
