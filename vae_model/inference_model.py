import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td

from vae_model.made import MADE
from vae_model.architectures import EncoderGatedConvolutionBlock


class InferenceModel(nn.Module):
    def __init__(self, args):
        super(InferenceModel, self).__init__()

        self.args = args
        self.D = args.latent_dim
        self.L = args.max_seq_len
        self.V = args.vocab_size
        self.B = args.batch_size
        self.image_w_h = args.image_w_h
        self.image_or_language = args.image_or_language
        self.data_distribution = args.data_distribution

        # NETWORK
        # an encoder that maps x to params of q_z_x
        self.encoder_network = self.get_encoder_network()

        # POSTERIOR DISTRIBUTION
        self.q_z_x_type = args.q_z_x_type

        assert self.q_z_x_type in ["independent_gaussian", "conditional_gaussian_made", "iaf"]

        self.q_z_nn = self.get_encoder_network()
        self.q_z_x_block = self.get_posterior_distribution_block()

    def get_encoder_network(self):
        """Retrieves a network block for data set type language (transformer block) or image (convolutional block)"""
        # IMAGE: bernoulli or categorical data distribution
        if self.image_or_language == "image":
            return EncoderGatedConvolutionBlock(args=self.args)
        # LANGUAGE: categorical
        else:
            # TODO: implement for language
            raise NotImplementedError

    def get_posterior_distribution_block(self):
        """Retrieves the right mapper from the NN block to the actual posterior distribution to work with."""
        if self.q_z_x_type == "independent_gaussian":
            return IndependentGaussianBlock(args=self.args)
        elif self.q_z_x_type == "conditional_gaussian_made":
            return ConditionalGaussianBlockMADE(args=self.args)
        elif self.q_z_x_type == "iaf":
            # TODO: return IAF(args=self.args)
            raise NotImplementedError
        else:
            raise ValueError(f"This posterior type q_z_x_type is not recognised: {self.q_z_x_type}")

    def infer_q_z_x(self, x_in):
        """Infers a distribution from encoding the input x_in."""
        # [B, 256]
        q_z_x_params = self.q_z_nn(x_in)
        #print("infer_q_z_x q_z_x_params", q_z_x_params.shape)
        q_z_x = self.q_z_x_block(q_z_x_params)
        #print("infer_q_z_x q_z_x", q_z_x)

        return q_z_x

    def forward(self, x_in, n_samples=1):
        """Infers a distribution and samples from it with the reparameterisation trick."""
        # [S, B, D]
        q_z_x = self.infer_q_z_x(x_in)
        z_post = q_z_x.rsample()  # TODO: sample_shape=(n_samples,)

        return q_z_x, z_post


class IndependentGaussianBlock(nn.Module):
    def __init__(self, args):
        super(IndependentGaussianBlock, self).__init__()
        self.B = args.batch_size
        self.D = args.latent_dim

        self.mean_layer = nn.Linear(256, self.D)
        self.scale_layer = nn.Sequential(nn.Linear(256, self.D), nn.Softplus())

    def forward(self, q_z_x_params):
        mean = self.mean_layer(q_z_x_params)
        scale = self.scale_layer(q_z_x_params)

        # [B, D] (D independent Gaussians)
        q_z_x = td.Independent(td.Normal(loc=mean, scale=scale), 1)

        return q_z_x


class AutoRegressiveGaussianDistribution(nn.Module):
    def __init__(self, q_z_x_params, made, state=None):
        super(AutoRegressiveGaussianDistribution, self).__init__()

        self.q_z_x_params = q_z_x_params
        self.made = made
        self.D = self.made.nin
        self.state = state  # z, mu, scale

    def log_prob(self, z, mean_scale=None):
        if mean_scale is not None:
            mean, scale = mean_scale
        elif self.state is not None:
            mean, scale = self.state[1], self.state[2]
        else:
            output_made = self.made(z, context=self.q_z_x_params)
            params_split = torch.split(output_made, 2, dim=1)
            mean, pre_scale = params_split[0], params_split[1]
            scale = F.softplus(pre_scale)

        self.state = (z, mean, scale)

        # careful that the distribution itself is not independent, but at this point it is valid to use for log_prob
        log_q_z_x = td.Independent(td.Normal(loc=mean, scale=scale), 1).log_prob(z)

        return log_q_z_x

    def rsample(self):
        # TODO: multi sample forward?
        B = self.q_z_x_params.shape[0]

        z_sample = torch.zeros((B, self.D))
        mu_inferred, scale_inferred = [], []

        for i in range(self.D):
            mus_prescales = self.made(z_sample, context=self.q_z_x_params)
            # split in 2 x [B, D]
            mus_prescales = torch.split(mus_prescales, 2, dim=1)
            mus, prescales = mus_prescales[0], mus_prescales[1]

            mu_i = mus[:, i]
            scale_i = F.softplus(prescales[:, i])

            mu_inferred.append(mu_i)
            scale_inferred.append(scale_i)

            z_sample[:, i] = td.Normal(loc=mu_i, scale=scale_i).rsample()

        mu_inferred = torch.stack(mu_inferred, dim=1)
        scale_inferred = torch.stack(scale_inferred, dim=1)

        self.state = (z_sample, mu_inferred, scale_inferred)

        return z_sample


@td.register_kl(AutoRegressiveGaussianDistribution, AutoRegressiveGaussianDistribution)
def _kl(p, q):
    if p.state is None:
        _ = p.rsample()
    p_mean, p_scale = p.state[1], p.state[2]

    if q.state is None:
        _ = q.rsample()
    q_mean, q_scale = q.state[1], q.state[2]

    kl = td.kl_divergence(td.Normal(loc=p_mean, scale=p_scale), td.Normal(loc=q_mean, scale=q_scale)).sum(-1)

    return kl


@td.register_kl(AutoRegressiveGaussianDistribution, td.Normal)
def _kl(p, q):
    if p.state is None:
        _ = p.rsample()
    p_mean, p_scale = p.state[1], p.state[2]

    kl = td.kl_divergence(td.Normal(loc=p_mean, scale=p_scale), q).sum(-1)

    return kl


@td.register_kl(AutoRegressiveGaussianDistribution, td.Distribution)
def _kl(p, q):
    if p.state is None:
        z = p.rsample()
    else:
        z = p.state[0]

    log_p_z = p.log_prob(z)
    log_q_z = q.log_prob(z)

    kl = log_p_z - log_q_z

    return kl


class ConditionalGaussianBlockMADE(nn.Module):
    def __init__(self, args):
        super(ConditionalGaussianBlockMADE, self).__init__()

        self.D = args.latent_dim
        # self.mapping_layer = nn.Linear(256, self.D)

        hiddens = [200, 220]

        natural_ordering = True
        act = nn.ReLU()

        self.made = MADE(self.D, hiddens, int(self.D * 2), natural_ordering=natural_ordering,
                         context_size=256, hidden_activation=act)  # no additional context here

    def forward(self, q_z_x_params):
        # Placeholder distribution object
        q_z_x = AutoRegressiveGaussianDistribution(q_z_x_params=q_z_x_params, made=self.made)

        return q_z_x


class IAF(nn.Module):
    def __init__(self, args):
        super(IAF, self).__init__()
        raise NotImplementedError

    def forward(self, q_z_x_params):
        raise NotImplementedError
