import torch.nn as nn
import torch.distributions as td
from vae_model.architectures import DCGanEncoder, DCGanDecoder

"""
config params:
- encoder_network_type: []
- q_x_z_type: [independent_gaussian, ...]
"""


class InferenceModel(nn.Module):
    def __init__(self, args):
        super(InferenceModel, self).__init__()

        self.D = args.latent_dim
        self.L = args.max_seq_len
        self.V = args.vocab_size
        self.B = args.batch_size
        self.W = args.image_w
        self.H = args.image_h

        # NETWORK
        # an encoder that maps x to params of q_z_x
        self.encoder_network_type = args.encoder_network_type
        self.encoder_network = self.get_encoder_network()

        # POSTERIOR DISTRIBUTION
        self.q_z_x_type = args.q_z_x_type

    def get_encoder_network(self):
        encoder_network = None

        # TODO: implement simple deconvolutional
        if self.encoder_network_type == "convolutional":
            raise NotImplementedError

        elif self.decoder_network_type == "dcgan":
            encoder_network = DCGanEncoder(z_dim=self.D)

        return encoder_network

    def encode(self, x_in):
        q_z_x = None
        q_z_x_params = self.encoder_network(x_in)

        if self.q_x_z_type == "independent_gaussian":
            # TODO: check whether positivitvy of scale is ensured
            loc, scale = q_z_x_params[:, :self.D, :, self.D:]
            # [B, D] (D independent Gaussians)
            q_z_x = td.Independent(td.Normal(loc=loc, scale=scale), 1)
        else:
            raise NotImplementedError

        return q_z_x

    def forward(self, x_in, n_samples=1):
        # [S, B, D]
        q_z_x = self.encode(x_in)
        z_post = q_z_x.rsample(sample_shape=(n_samples,))
        return q_z_x, z_post
