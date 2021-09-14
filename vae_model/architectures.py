# Claartje: this code is adapted from Rianne van de Berg's code (sylvester_flows submodule)
# https://github.com/riannevdberg/sylvester-flows/blob/32dde9b7d696fee94f946a338182e542779eecfe/models/VAE.py

import torch.nn as nn
from vae_model.sylvester_flows.models.layers import GatedConv2d, GatedConvTranspose2d


class EncoderGatedConvolutionBlock(nn.Module):
    """
    Maps x -> q_z_x_params:
        This class implements the basic convolutional neural block that maps an input image to a
        vector q_z_x_params of size 256, which serves as an input to a distribution block (independent_gaussian,
        conditional_gaussian_made or iaf).
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
            # TODO: Rianne has a Hardtanh actiation for her q_z_var:
            #     q_z_var = nn.Sequential(
            #         nn.Linear(256, self.D),
            #         nn.Softplus(),
            #         nn.Hardtanh(min_val=0.01, max_val=7.)
            #     ), why was that there?
        else:
            raise ValueError(f"Data distribution type not implemented: {self.data_distribution}.")

    def forward(self, x):
        """Maps the input to a 256-dimensional vector q_z_x_params, which is the input to a distribution block."""
        # [B, C=256, 1, 1]
        q_z_x_params = self.encoder_gated_cnn_block(x)

        # [B, 256]
        q_z_x_params = q_z_x_params.squeeze(-1).squeeze(-1)

        return q_z_x_params


class DecoderGatedConvolutionBlock(nn.Module):
    """
    Maps z -> p_x_z_params
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

        return p_x_z_params