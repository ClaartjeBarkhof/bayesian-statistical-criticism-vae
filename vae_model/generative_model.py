import pytorch_lightning as pl
import numpy as np
import torch
import torch.nn as nn
# from torch.distributions import Categorical, Mul
import torch.distributions as td
from architectures import DCGanDecoder

"""
Config arguments:
- p_z_type: [isotropic_gaussian]
- p_x_z_type: [bernoulli, gaussian, categorical]
- decoder_network_type: [dcgan, pixelcnn, deconvolutional, ...]
- 
"""

class GenerativeModel(nn.Module):
    def __init__(self, args):
        super(GenerativeModel, self).__init__()

        self.D = args.latent_dim
        self.L = args.max_seq_len
        self.V = args.vocab_size
        self.B = args.batch_size
        self.W = args.image_w
        self.H = args.image_h

        # NETWORK
        # a decoder that maps z (+ x) to params of p_x_z
        self.decoder_network_type = args.decoder_network_type
        self.decoder_network = self.get_decoder_network()

        # PRIOR
        self.p_z_type = args.p_z_type
        self.p_z_dist = self.get_p_z_dist()

        # OUTPUT DISTRIBUTION
        self.p_x_z_type = args.p_x_z_type

    def sample(self, S=1):
        z_prior = self.p_z(S=S)
        p_x_z_prior = self.p_x_z(z_prior)
        return p_x_z_prior

    def forward(self, x_in, z_post):
        p_x_z_post = self.p_x_z(z_post)
        ll = p_x_z_post.log_prob(x_in)
        return p_x_z_post, ll

    def p_z(self, S=1):
        return self.prior.sample(sample_shape=(S,))

    def p_x_z(self, z, x=None):
        p_x_z_params = self.decoder_network(z, x)

        p_x_z = None
        if self.p_x_z_type == "bernoulli":
            p_x_z = td.Independent(td.Bernoulli(logits=p_x_z_params), 1)

        # TODO: implement Gaussian output
        elif self.p_x_z_type == "gaussian":
            p_x_z = None

        # TODO: implement Categorical output
        elif self.p_x_z_type == "categorical":
            # p_x_z_params: [B, L, V]
            p_x_z = td.Categorical(logits=p_x_z_params)
        else:
            raise NotImplementedError

        return p_x_z

    def get_p_z_dist(self):
        p_z_dist = None

        if self.p_z_type == "isotropic_gaussian":
            p_z_dist = td.MultivariateNormal(torch.zeros(self.D), torch.eye(self.D))
        # TODO: add other priors
        else:
            raise NotImplementedError

        return p_z_dist

    def get_decoder_network(self):
        decoder_network = None

        # TODO: implement simple deconvolutional
        if self.decoder_network_type == "deconvolutional":
            raise NotImplementedError

        # TODO: implement pixel_cnn
        # check https://github.com/pclucas14/pixel-cnn-pp/blob/master/model.py
        # check modifications in paper Alemi (page 20) https://arxiv.org/pdf/1711.00464.pdf
        elif self.decoder_network_type == "pixel_cnn":
            raise NotImplementedError

        elif self.decoder_network_type == "dcgan":
            decoder_network = DCGanDecoder(z_dim=self.D)

        return decoder_network