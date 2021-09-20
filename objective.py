from torch.distributions import kl_divergence
import torch
import torch.nn as nn
import torch.distributions as td
from pytorch_constrained_opt.constraint import Constraint
from torch_two_sample import MMDStatistic


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
                a likelihood distribution(-like) object with parameters [S, B, ...X_dims...]
        Returns:
            loss_dict:
                a dictionary containing statistics to log and 'total_loss' which is used for optimisation
        """

        (S, B, D) = z_post.shape

        print("COMPUTE LOSS")
        print("p_x_z", p_x_z)
        print("z_post.shape", z_post.shape)

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
        kl_prior_post = self.kl_prior_post(p_z=p_z, q_z_x=q_z_x, n_samples=S, batch_size=B, z_post=z_post, analytical=True)

        # Reduce all dimensions with sum, except for the batch dimension, average that
        if self.args.decoder_network_type == "conditional_made_decoder":
            # In case of the MADE, the evaluation is in flattened form
            labels = labels.reshape(B, -1)

        print("Labels.shape", labels.shape)

        log_p_x_z = p_x_z.log_prob(labels)
        print("log_p_x_z.shape", log_p_x_z.shape)
        assert log_p_x_z.shape == (S, B), \
            "We assume p_x_z.log_prob to return one scalar sample per per data point in the batch, shape must be (S, B)"
        distortion = - log_p_x_z.mean()

        # Maximum mean discrepancy
        # z_post at this point is [1, B, D]
        mmd = self.maximum_mean_discrepancy(z_post.squeeze(0))

        total_loss, mdr_loss, mdr_multiplier = None, None, None
        if self.args.objective == "AE":
            total_loss = distortion
        elif self.args.objective == "VAE":
            total_loss = kl_prior_post #distortion + kl_prior_post
        elif self.args.objective == "BETA-VAE":
            total_loss = distortion + self.args.beta_beta
        elif self.args.objective == "MDR-VAE":
            # the gradients for the constraints are recorded at some other point
            mdr_loss = self.mdr_constraint(kl_prior_post).squeeze()
            mdr_multiplier = self.mdr_constraint.multiplier
            total_loss = distortion + kl_prior_post + mdr_loss
        elif self.args.objective == "FB-VAE":
            raise NotImplementedError
        elif self.args.objective == "INFO-VAE":
            # gain = ll - (1 - a)*kl_prior_post - (a + l - 1)*marg_kl
            # loss = distortion + (1 - a)*kl_prior_post + (a + l - 1)*marg_kl
            total_loss = distortion + ((1 - self.args.info_alpha) * kl_prior_post) \
                         + ((self.args.info_alpha + self.args.info_lambda - 1) * mmd)
        elif self.args.objective == "LAG-INFO-VAE":
            # https://github.com/ermongroup/lagvae/blob/master/methods/lagvae.py
            raise NotImplementedError

        loss_dict = dict(
            total_loss=total_loss,
            mmd=mmd,
            mdr_loss=mdr_loss,
            mdr_multiplier=mdr_multiplier,
            distortion=distortion,
            kl_prior_post=kl_prior_post
        )

        return loss_dict

    @staticmethod
    def kl_prior_post(p_z, q_z_x, batch_size, n_samples, z_post=None, analytical=False):
        """Computes the KL from prior to posterior, either analytically or empirically."""

        print("kl_prior_post: q_z_x", q_z_x)
        print("kl_prior_post: z_post.shape", z_post.shape)

        # A bit of a hack to avoid this kl that raises a NotImplementedError (TODO: make this possible)
        if isinstance(p_z, td.MixtureSameFamily):
            analytical = False

        if analytical:
            print("ANALYTICAL")

            kl = kl_divergence(q_z_x, p_z)

        else:
            print("NOT ANALYTICAL")
            # [B]
            log_q_z_x = q_z_x.log_prob(z_post)
            log_p_z = p_z.log_prob(z_post)

            kl = (log_q_z_x - log_p_z)

        assert kl.shape == (n_samples, batch_size), "We assume kl_divergence to return one scalar per data point in the batch"

        return kl.mean()

    def maximum_mean_discrepancy(self, z_post):
        prior_sample = torch.randn(z_post.shape).to(z_post.device)
        alphas = [0.1 * i for i in range(5)] # TODO: no clue for these...

        n_1, n_2 = len(z_post), len(prior_sample)
        MMD_stat = MMDStatistic(n_1, n_2)
        tts_mmd = MMD_stat(z_post, prior_sample, alphas, ret_matrix=False)

        return tts_mmd
