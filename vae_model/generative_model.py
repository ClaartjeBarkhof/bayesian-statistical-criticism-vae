import torch
import torch.nn as nn
import torch.distributions as td

from sylvester_flows.models.layers import GatedConvTranspose2d
from vae_model.distributions import AutoRegressiveDistribution
from vae_model.made import MADE


class GenerativeModel(nn.Module):
    def __init__(self, args, device="cpu"):
        super(GenerativeModel, self).__init__()

        self.device = device
        self.args = args

        self.D = args.latent_dim
        self.B = args.batch_size

        # LANGUAGE
        self.L = args.max_seq_len
        self.V = args.vocab_size

        # IMAGE
        self.image_w_h = args.image_w_h
        self.C = args.n_channels

        # NETWORK
        # a decoder that maps z (+ x) to params of p_x_z
        self.decoder_network_type = args.decoder_network_type

        self.decoder_network = self.get_decoder_network()

        # PRIOR
        self.p_z_type = args.p_z_type

        # MIXTURE OF GAUSIANS PRIOR
        if self.p_z_type == "mog":
            self.mog_n_components = args.mog_n_components
            self.mix_components = torch.nn.Parameter(torch.rand(self.mog_n_components), requires_grad=True)
            self.component_means = torch.nn.Parameter(torch.randn(self.mog_n_components, self.D), requires_grad=True)
            self.component_scales = torch.nn.Parameter(torch.abs(torch.randn(self.mog_n_components, self.D)), requires_grad=True)

        self.p_z = self.init_p_z()

        # OUTPUT DISTRIBUTION == DATA DISTRIBUTION
        self.p_x_z_type = args.data_distribution

    def sample_generative_model(self, S=1):
        z_prior = self.sample_prior(S=S)
        p_x_z_prior = self.p_x_z(z_prior)
        sample = p_x_z_prior.sample()
        return sample

    def forward(self, z_post, x_in=None):
        """
        Map a sample from the posterior to a generative output distribution: z_post -> p(X|Z=z_post)
        -> Potentially condition on (a part of) x too, e.g. the prefix: z_post -> p(X|Z=z_post, Y=x<i)

        Input:
            z_post: [B, D]
                samples from the posterior q(z|x)
            x_in: [B, L] (language) or [B, C, W, H] (image)
                the original input the posterior conditioned on

        Output:
            p_x_z_post: [B, L] (language) or or [B, C, W, H] (image), generative output distribution p(X|Z=z_post)
        """
        p_x_z_post = self.p_x_z(z_post)
        return p_x_z_post

    def sample_prior(self, S=1):
        """
        Returns a from the prior defined by <p_z_type>.

        Input:
            S: int:
                the number of samples it should return.

        Output:
            z_prior: [S, D]
                samples from the prior of dimensionality of the latent space.
        """
        z_prior = self.p_z.sample(sample_shape=(S,))

        return z_prior

    def p_x_z(self, z):
        if self.decoder_network_type == "conditional_made_decoder":
            p_x_z = self.decoder_network(z)
        else:
            p_x_z_params = self.decoder_network(z)

            if self.p_x_z_type == "bernoulli":
                # p_x_z_params: [B, C, W, H]
                p_x_z = td.Independent(td.Bernoulli(logits=p_x_z_params), 1)

            elif self.p_x_z_type == "multinomial":
                # image: p_x_z_params: [B, C, num_classes, W, H] -> [B, C, W, H, num_classes]
                p_x_z_params = p_x_z_params.permute(0, 1, 3, 4, 2)
                p_x_z = td.Categorical(logits=p_x_z_params)

            else:
                raise ValueError(f"{self.p_x_z_type} is not a valid data_distribution, choices: bernoulli, multinomial")

        return p_x_z

    def init_p_z(self):
        # ISOTROPIC GAUSSIAN
        if self.p_z_type == "isotropic_gaussian":
            return td.Independent(td.Normal(loc=torch.zeros(self.D, device=self.device), scale=torch.ones(self.D, device=self.device)), 1)

        # MIXTURE OF GAUSSIANS
        elif self.p_z_type == "mog":

            mix = td.Categorical(self.mix_components)
            comp = td.Independent(td.Normal(self.component_means, self.component_scales), 1)

            return td.MixtureSameFamily(mix, comp)
        else:
            raise ValueError(f"{self.p_z_type} is not a valid p_z_type, choices: isotropic_gaussian, mog")

    def get_decoder_network(self):
        if self.decoder_network_type == "basic_deconv_decoder":
            return DecoderGatedConvolutionBlock(args=self.args)

        elif self.decoder_network_type == "conditional_made_decoder":
            return ConditionalBernoulliBlockMADE(args=self.args)

        else:
            raise NotImplementedError


class ConditionalBernoulliBlockMADE(nn.Module):
    def __init__(self, args):
        super(ConditionalBernoulliBlockMADE, self).__init__()

        self.X_dim = args.image_w_h * args.image_w_h * args.n_channels
        self.D = args.latent_dim

        hiddens = [200, 220]
        natural_ordering = True
        act = nn.ReLU()

        self.made = MADE(self.X_dim, hiddens, self.X_dim, natural_ordering=natural_ordering,
                         context_size=self.D, hidden_activation=act)  # no additional context here

    def forward(self, z):
        # Placeholder distribution object
        p_x_z = AutoRegressiveDistribution(context=z, made=self.made, dist_type="bernoulli")

        return p_x_z


class DecoderGatedConvolutionBlock(nn.Module):
    """
    Maps z -> p_x_z_params

    # This code is adapted from Rianne van de Berg's code (sylvester_flows submodule):
    https://github.com/riannevdberg/sylvester-flows/blob/32dde9b7d696fee94f946a338182e542779eecfe/models/VAE.py

    """
    def __init__(self, args):
        super(DecoderGatedConvolutionBlock, self).__init__()

        self.num_classes = 256
        self.D = args.latent_dim
        self.C = args.n_channels
        self.data_distribution = args.data_distribution
        self.image_w_h = args.image_w_h

        if self.image_w_h == 28:
            self.last_kernel_size = 7
        else:
            raise ValueError('Only supporting input size 28 now.')

        if self.data_distribution == 'bernoulli':
            self.decoder_gated_cnn_block = nn.Sequential(
                GatedConvTranspose2d(self.D, 64, self.last_kernel_size, 1, 0),
                GatedConvTranspose2d(64, 64, 5, 1, 2),
                GatedConvTranspose2d(64, 32, 5, 2, 2, 1),
                GatedConvTranspose2d(32, 32, 5, 1, 2),
                GatedConvTranspose2d(32, 32, 5, 2, 2, 1),
                GatedConvTranspose2d(32, 32, 5, 1, 2),
                # TODO: why did rianne separate this layer from the rest? (+ a sigmoid), she had extra nn.Sigmoid(),
                nn.Conv2d(32, self.C, 1, 1, 0),

            )

        # output shape: batch_size, num_channels * num_classes, pixel_width, pixel_height
        elif self.data_distribution == 'multinomial':
            self.decoder_gated_cnn_block = nn.Sequential(
                GatedConvTranspose2d(self.D, 64, self.last_kernel_size, 1, 0),
                GatedConvTranspose2d(64, 64, 5, 1, 2),
                GatedConvTranspose2d(64, 32, 5, 2, 2, 1),
                GatedConvTranspose2d(32, 32, 5, 1, 2),
                GatedConvTranspose2d(32, 32, 5, 2, 2, 1),
                GatedConvTranspose2d(32, 32, 5, 1, 2),
                # TODO: why did rianne separate these last two layers from the rest? (+ a sigmoid)
                nn.Conv2d(32, 256, 5, 1, 2),
                nn.Conv2d(256, self.C * self.num_classes, 1, 1, 0),
            )
        else:
            raise ValueError(f"Data distribution type not implemented: {self.data_distribution}.")

    def forward(self, z):
        """z -> p_x_z params"""

        B = z.size(0)
        z = z.view(B, self.D, 1, 1)

        # Multinomial: [B, C*256, image_w_h, image_w_h] (logits, pre-softmax)
        # Bernoulli: [B, C, image_w_h, image_w_h] (logits, pre-sigmoid)

        p_x_z_params = self.decoder_gated_cnn_block(z)

        if self.C == 1 and self.data_distribution == "multinomial":
            p_x_z_params = p_x_z_params.unsqueeze(1)
        elif self.C == 3:
            print("Code not checked for 3-channel input, implement something with reshape here.")
            raise NotImplementedError

        return p_x_z_params