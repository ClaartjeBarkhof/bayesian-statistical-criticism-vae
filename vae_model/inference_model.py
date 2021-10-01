import torch.nn as nn
import torch.distributions as td

from vae_model.distributions import AutoRegressiveDistribution
from vae_model.made import MADE

from vae_model.sylvester_flows.models.layers import GatedConv2d


# --------------------------------------------------------------------------------------------------------------------
# INFERENCE MODEL

class InferenceModel(nn.Module):
    def __init__(self, args, device="cpu"):
        super(InferenceModel, self).__init__()

        self.device = device

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
        self.encoder_network_type = args.encoder_network_type
        self.encoder_network = self.get_encoder_network()

        # POSTERIOR DISTRIBUTION
        self.q_z_x_type = args.q_z_x_type

        self.q_z_nn = self.get_encoder_network()
        self.q_z_x_block = self.get_posterior_distribution_block()

    def get_encoder_network(self):
        """Retrieves a network block for data set type language (transformer block) or image (convolutional block)"""
        # IMAGE: bernoulli or categorical data distribution
        if self.image_or_language == "image":
            if self.encoder_network_type == "basic_mlp_encoder":
                return EncoderMLPBlock(args=self.args)
            elif self.encoder_network_type == "basic_conv_encoder":
                return EncoderGatedConvolutionBlock(args=self.args)
            else:
                raise NotImplementedError
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
        q_z_x = self.q_z_x_block(q_z_x_params)

        return q_z_x

    def forward(self, x_in, n_samples=1):
        """Infers a distribution and samples from it with the reparameterisation trick."""
        # [B, D]
        q_z_x = self.infer_q_z_x(x_in)

        # [S, B, D]
        z_post = q_z_x.rsample(sample_shape=(n_samples,))

        return q_z_x, z_post


# --------------------------------------------------------------------------------------------------------------------
# POSTERIOR DISTRIBUTION BLOCKS
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
        assert mean.shape == (q_z_x_params.shape[0], self.D), "mean is supposed to be of shape [B, D]"
        assert scale.shape == (q_z_x_params.shape[0], self.D), "scale is supposed to be of shape [B, D]"

        q_z_x = td.Independent(td.Normal(loc=mean, scale=scale), 1)

        return q_z_x


class ConditionalGaussianBlockMADE(nn.Module):
    def __init__(self, args):
        super(ConditionalGaussianBlockMADE, self).__init__()

        self.D = args.latent_dim

        hiddens = [int(h) for h in args.encoder_MADE_hidden_sizes.split("-")]
        print("Hidden sizes of the encoder made: ", hiddens)

        natural_ordering = True
        act = nn.ReLU()

        self.made = MADE(nin=self.D, hidden_sizes=hiddens, nout=int(self.D * 2), natural_ordering=natural_ordering,
                         context_size=256, hidden_activation=act, gating=False)

    def forward(self, q_z_x_params):
        # Placeholder distribution object
        q_z_x = AutoRegressiveDistribution(context=q_z_x_params, made=self.made, dist_type="gaussian", encoder=True)

        return q_z_x


class IAF(nn.Module):
    def __init__(self, args):
        super(IAF, self).__init__()
        raise NotImplementedError

    def forward(self, q_z_x_params):
        raise NotImplementedError


# --------------------------------------------------------------------------------------------------------------------
# ARCHITECTURE

class EncoderGatedConvolutionBlock(nn.Module):
    """
    Maps x -> q_z_x_params:
        This class implements the basic convolutional neural block that maps an input image to a
        vector q_z_x_params of size 256, which serves as an input to a distribution block (independent_gaussian,
        conditional_gaussian_made or iaf).

    # This code is adapted from Rianne van de Berg's code (sylvester_flows submodule):
    https://github.com/riannevdberg/sylvester-flows/blob/32dde9b7d696fee94f946a338182e542779eecfe/models/VAE.py

    """

    def __init__(self, args):
        super(EncoderGatedConvolutionBlock, self).__init__()

        self.image_w_h = args.image_w_h
        self.D = args.latent_dim
        self.C = args.n_channels
        self.q_z_x_type = args.q_z_x_type
        self.data_distribution = args.data_distribution

        if self.image_w_h == 28:
            last_kernel_size = 7
        else:
            raise ValueError('Only supporting input size 28 now.')

        if self.data_distribution == 'bernoulli':
            self.encoder_gated_cnn_block = nn.Sequential(
                GatedConv2d(self.C, 32, 5, 1, 2),
                GatedConv2d(32, 32, 5, 2, 2),
                GatedConv2d(32, 64, 5, 1, 2),
                GatedConv2d(64, 64, 5, 2, 2),
                GatedConv2d(64, 64, 5, 1, 2),
                GatedConv2d(64, 256, last_kernel_size, 1, 0),
            )

        elif self.data_distribution == "multinomial":
            self.encoder_gated_cnn_block = nn.Sequential(
                GatedConv2d(self.C, 32, 5, 1, 2),
                GatedConv2d(32, 32, 5, 2, 2),
                GatedConv2d(32, 64, 5, 1, 2),
                GatedConv2d(64, 64, 5, 2, 2),
                GatedConv2d(64, 64, 5, 1, 2),
                GatedConv2d(64, 256, last_kernel_size, 1, 0)
            )
        else:
            raise ValueError(f"Data distribution type not implemented: {self.data_distribution}.")

    def forward(self, x):
        """Maps the input to a 256-dimensional vector q_z_x_params, which is the input to a distribution block."""
        # [B, C=256, 1, 1]
        q_z_x_params = self.encoder_gated_cnn_block(x)

        # [B, 256]
        q_z_x_params = q_z_x_params.squeeze(-1).squeeze(-1)

        return q_z_x_params


class EncoderMLPBlock(nn.Module):
    def __init__(self, args):
        super(EncoderMLPBlock, self).__init__()

        self.image_w_h = args.image_w_h
        self.C = args.n_channels

        # in -> 500 --> 256 -> q_z_x block
        self.encoder_mlp_block = nn.Sequential(
            nn.Linear(in_features=self.image_w_h*self.image_w_h*self.C, out_features=500),
            nn.ReLU(),
            nn.Linear(in_features=500, out_features=256),
            nn.ReLU()
        )

    def forward(self, x_in):
        # reshape image to [B, image_w*image_h*C]
        x_in_flat = x_in.reshape(x_in.shape[0], -1)
        q_z_x_params = self.encoder_mlp_block(x_in_flat)

        return q_z_x_params