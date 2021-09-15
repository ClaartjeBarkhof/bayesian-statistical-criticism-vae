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
        print("infer_q_z_x q_z_x_params", q_z_x_params.shape)
        q_z_x = self.q_z_x_block(q_z_x_params)
        print("infer_q_z_x q_z_x", q_z_x)

        return q_z_x

    def forward(self, x_in, n_samples=1):
        """Infers a distribution and samples from it with the reparameterisation trick."""
        # [S, B, D]
        q_z_x = self.infer_q_z_x(x_in)
        z_post = q_z_x.rsample() # TODO: sample_shape=(n_samples,)

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
    def __init__(self, z_sample, made_block=None, mu_inferred=None, scale_inferred=None):
        super(AutoRegressiveGaussianDistribution, self).__init__()

        self.z_sample = z_sample

        assert made_block is not None or (mu_inferred is not None and scale_inferred is not None), \
            "Either give a MADE as input or the inferred mus and scales"

        self.made_block = made_block # ConditionalGaussianBlockMADE: careful this is more than the MADE itself
        self.mu_inferred = mu_inferred
        self.scale_inferred = scale_inferred

    def log_prob(self, z):
        if self.mu_inferred is None or self.scale_inferred is None:
            output_made = self.made_block(z)
            params_split = torch.split(output_made, 2, dim=1)
            mean, pre_scale = params_split[0], params_split[1]
            scale = F.softplus(pre_scale)
        else:
            mean, scale = self.mus_inferred, self.scale_inferred

        log_q_z_x = td.Independent(td.Normal(loc=mean, scale=scale), 1).log_prob(z)
        return log_q_z_x

    def rsample(self):
        return self.z_sample

class ConditionalGaussianBlockMADE(nn.Module):
    def __init__(self, args):
        super(ConditionalGaussianBlockMADE, self).__init__()

        self.D = args.latent_dim
        #self.mapping_layer = nn.Linear(256, self.D)

        hiddens = [200, 220]

        natural_ordering = True
        act = nn.ReLU()

        self.made_block = MADE(self.D, hiddens, int(self.D * 2), natural_ordering=natural_ordering,
                               context_size=256, hidden_activation=act)  # no additional context here

    def forward(self, q_z_x_params):
        #made_initial_input = self.mapping_layer(q_z_x_params)

        # All [B, D]
        z_sample, mu_inferred, scale_inferred = self.made_block.auto_regressive_sampling(q_z_x_params)

        # Placeholder distribution object
        q_z_x = AutoRegressiveGaussianDistribution(z_sample=z_sample, mu_inferred=mu_inferred,
                                                   scale_inferred=scale_inferred)

        return q_z_x


class IAF(nn.Module):
    def __init__(self, args):
        super(IAF, self).__init__()
        raise NotImplementedError

    def forward(self, q_z_x_params):
        raise NotImplementedError