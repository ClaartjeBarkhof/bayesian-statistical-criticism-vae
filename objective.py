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

        # MDR-VAE
        self.mdr_constraint = self.get_mdr_constraint()

        # LAG-INFO-VAE
        self.rate_constraint, self.mmd_constraint = self.get_lag_info_vae_constraints()

    def get_mdr_constraint(self):
        return Constraint(self.args.mdr_value, "ge", alpha=0.5).to(
            self.device) if self.args.objective == "MDR-VAE" else None

    def get_lag_info_vae_constraints(self):
        if self.args.objective == "LAG-INFO-VAE":
            rate_constraint = Constraint(self.args.rate_constraint_val,
                                         self.args.rate_constraint_rel, alpha=0.5).to(self.device)
            mmd_constraint = Constraint(self.args.mmd_constraint_val,
                                        self.args.mmd_constraint_rel, alpha=0.5).to(self.device)
        else:
            rate_constraint, mmd_constraint = None, None

        return rate_constraint, mmd_constraint

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

        label_mask, label_length, p_z_l, log_p_l_z = None, None, None, None

        total_loss = None
        loss_dict = dict()

        # Image: get labels for multinomial case
        if self.image_or_language == "image" and self.data_distribution == "multinomial":
            # map 0, 1 range to 0 256 integer range, code taken from:
            # https://github.com/riannevdberg/sylvester-flows/blob/ \
            # 32dde9b7d696fee94f946a338182e542779eecfe/optimization/loss.py#L74
            num_classes = 256

            labels = (x_in * (num_classes - 1)).long()

        # Language: apply some masking
        elif self.image_or_language == "language":


            input_ids, attention_mask = x_in

            # strong decoder: next token prediction
            if self.args.decoder_network_type == "strong_distil_roberta_decoder":
                # [S, B, L-1]
                labels = input_ids[:, 1:].unsqueeze(0).long()  # categorical labels need to be long
                label_mask = attention_mask[:, 1:].unsqueeze(0).float()  # mask values need to be float

            # weak decoder: same token prediction
            else:
                # [S, B, L]
                labels = input_ids.unsqueeze(0).long()  # categorical labels need to be long

                # attention mask has 1 for non-masked, 1 for masked tokens
                # mask in this context means <pad> and has nothing to do with masked language modelling
                # it is just the contour of the sequences in a block of fixed length sequences
                label_mask = attention_mask.unsqueeze(0).float()  # mask values need to be float

                # [S, B]
                label_length = label_mask.sum(dim=-1).long()  # categorical labels need to be long

            # print("label_length.min", label_length.min())
            # print("label_length.max", label_length.max())
            #
            # print("labels", labels.shape)
            # print("label_mask", label_mask.shape)
            # print("label_length", label_length.shape)

        # Image: other cases
        else:
            labels = x_in

        # Language: weak decoder (length model)
        if self.args.decoder_network_type in ["weak_distil_roberta_decoder", "weak_memory_distil_roberta_decoder"]:
            assert type(p_x_z) == tuple, "we expect p_x_z may to be tuple of p_x_z and p_z_l"
            p_x_z, p_z_l = p_x_z

        # Expected KL from prior to posterior (scalar)
        kl_prior_post = self.kl_prior_post(p_z=p_z, q_z_x=q_z_x, batch_size=B, z_post=z_post, analytical=True)

        # Distortion [S, B] or [S, B, L]
        log_p_x_z = p_x_z.log_prob(labels)

        if self.image_or_language == "language":
            # [S, B, L] -> [S, B]
            log_p_x_z = (log_p_x_z * label_mask).sum(dim=-1)

            if self.args.decoder_network_type in ["weak_distil_roberta_decoder", "weak_memory_distil_roberta_decoder"]:

                log_p_l_z = p_z_l.log_prob(label_length)

                loss_dict["log_p_l_z"] = log_p_l_z.mean().item()
                loss_dict["log_p_x_z (without l)"] = log_p_x_z.mean().item()

                # this naming is a bit off but just to match the other code (should have been p_x_z_l)
                log_p_x_z = log_p_x_z + log_p_l_z

        assert log_p_x_z.shape == (1, B), f"we assume p_x_z.log_prob shape to be be (1, B), currently {log_p_x_z.shape}"

        # Average over samples and batch: [S, B] -> scalar (we assume that S=1 actually)
        # if S != 1 we should do a logsumexp(log_p_x_z, dim=0) - log(S)
        distortion = - log_p_x_z.mean()

        # TODO: self.free_bits_kl(p_z, q_z_x, z_post, free_bits=self.args.free_bits, per_dimension=False)

        # Maximum mean discrepancy
        # z_post at this point is [S, B, D]
        # mmd = scalar tensor
        mmd = self.maximum_mean_discrepancy(z_post, p_z)

        elbo = - (distortion + kl_prior_post)

        if self.args.objective == "AE":
            total_loss = distortion

        elif self.args.objective == "VAE":
            total_loss = -elbo

        elif self.args.objective == "BETA-VAE":
            beta_kl = self.args.beta_beta * kl_prior_post
            total_loss = distortion + beta_kl
            loss_dict["beta_kl"] = beta_kl

        elif self.args.objective == "MDR-VAE":
            # the gradients for the constraints are recorded at some other point
            mdr_loss = self.mdr_constraint(kl_prior_post).squeeze()
            mdr_multiplier = self.mdr_constraint.multiplier
            loss_dict["mdr_loss"] = mdr_loss
            loss_dict["mdr_multiplier"] = mdr_multiplier

            total_loss = distortion + kl_prior_post + mdr_loss

        elif self.args.objective == "FB-VAE":
            kl_fb = self.free_bits_kl(p_z=p_z, q_z_x=q_z_x, z_post=z_post, free_bits=self.args.free_bits,
                                      per_dimension=self.args.free_bits_per_dimension)
            loss_dict["kl_fb"] = kl_fb.item()
            total_loss = distortion + kl_fb

        elif self.args.objective == "INFO-VAE":
            # from the original paper:
            # gain = ll - (1 - a)*kl_prior_post - (a + l - 1)*marg_kl
            # loss = distortion + (1 - a)*kl_prior_post + (a + l - 1)*marg_kl
            # total_loss = distortion + ((1 - self.args.info_alpha) * kl_prior_post) \
            #              + ((self.args.info_alpha + self.args.info_lambda - 1) * mmd)

            # rewriting in the LagVAE paper as:
            # MI maximisation: D + l_1 * -ELBO + l_2 * MMD
            lambda_1_Rate = self.args.info_lambda_1_rate * kl_prior_post
            lambda_2_MMD = self.args.info_lambda_2_mmd * mmd
            total_loss = distortion + lambda_1_Rate + lambda_2_MMD

            # Add losses to dict
            loss_dict["lambda_1_Rate"] = lambda_1_Rate
            loss_dict["lambda_2_MMD"] = lambda_2_MMD

        elif self.args.objective == "LAG-INFO-VAE":
            # Assemble loss
            rate_constraint_loss = self.rate_constraint(kl_prior_post)
            rate_constraint_multiplier = self.rate_constraint.multiplier

            mmd_constraint_loss = self.mmd_constraint(mmd)
            mmd_constraint_multiplier = self.mmd_constraint.multiplier

            total_loss = distortion + rate_constraint_loss + mmd_constraint_loss

            # Add losses to dict
            loss_dict["rate_constraint_loss"] = rate_constraint_loss
            loss_dict["rate_constraint_multiplier"] = rate_constraint_multiplier
            loss_dict["mmd_constraint_loss"] = mmd_constraint_loss
            loss_dict["mmd_constraint_multiplier"] = mmd_constraint_multiplier

        posterior_stats = self.get_posterior_stats(q_z_x, z_post)

        loss_dict = {**loss_dict, **dict(
            total_loss=total_loss,
            mmd=mmd,
            elbo=elbo,
            distortion=distortion,
            kl_prior_post=kl_prior_post
        )}

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
        # Per dimension
        if per_dimension:
            # Retrieve the means, scales to make a distribution that does not reduce the latent
            # dimension when calling log_prob (reinterpreted_batch_ndims = 0)
            if isinstance(q_z_x, AutoRegressiveDistribution):
                (mean, scale) = q_z_x.params
                mean, scale = mean[0, :, :], scale[0, :, :]
            elif isinstance(p_z, td.MixtureSameFamily):
                # TODO: not sure how to implement this
                raise NotImplementedError
            # Independent Normal
            else:
                mean, scale = q_z_x.base_dist.loc, q_z_x.base_dist.scale

            # [S, B, D] -> [B, D] (avg. over sample dim.)
            log_p_z = td.Normal(loc=torch.zeros_like(mean), scale=torch.ones_like(scale)).log_prob(z_post).mean(dim=0)

            # [S, B, D] -> [B, D] (avg. over sample dim.)
            log_q_z_x = td.Normal(mean, scale).log_prob(z_post).mean(dim=0)

            # KL ( q(z|x) || p(z) )
            kl = log_q_z_x - log_p_z

            # Free bits operation KL_FB = max(FB, KL), so the KL can not be lower than threshold
            kl_fb = torch.clamp(kl, min=free_bits)

            # Reduce latent dim
            kl_fb = kl_fb.sum(dim=-1)

        # Not per dimension
        else:
            # [S, B] -> [B] (avg. sample dim.)
            log_q_z_x = q_z_x.log_prob(z_post).mean(dim=0)
            log_p_z = p_z.log_prob(z_post).mean(dim=0)

            # [B]
            kl = log_q_z_x - log_p_z

            # Free bits operation KL_FB = max(FB, KL), so the KL can not be lower than threshold
            kl_fb = torch.clamp(kl, min=free_bits)

        return kl_fb.mean()

    @staticmethod
    def maximum_mean_discrepancy(z_post, p_z):
        # [S, B, D] -> [B, D]
        z_post = z_post.reshape(-1, z_post.shape[-1])

        prior_sample = p_z.sample(sample_shape=(z_post.shape[0],))
        # prior_sample = torch.randn_like(z_post)  # .to(z_post.device)

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
        #cov = np.cov(z_post_np)
        #log_det_cov_q_z = np.log(np.linalg.det(cov))

        d = dict(
            mean_mean=mean_mean,
            std_across_x_mean=std_across_x_mean,
            std_across_z_mean=std_across_z_mean,
            mean_scale=mean_scale,
            std_across_x_scale=std_across_x_scale,
            std_across_z_scale=std_across_z_scale,
            # log_det_cov_q_z=log_det_cov_q_z
        )

        return d
