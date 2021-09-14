import torch.nn as nn
import torch
import numpy as np
from vae_model.sylvester_flows.models.layers import GatedConv2d, GatedConvTranspose2d
import torch.distributions as td


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

        elif self.data_distribution == "categorical":
            self.encoder_gated_cnn_block = nn.Sequential(
                GatedConv2d(self.C, 32, 5, 1, 2),
                GatedConv2d(32, 32, 5, 2, 2),
                GatedConv2d(32, 64, 5, 1, 2),
                GatedConv2d(64, 64, 5, 2, 2),
                GatedConv2d(64, 64, 5, 1, 2),
                GatedConv2d(64, 256, last_kernel_size, 1, 0)
            )
            # q_z_mean = nn.Linear(256, self.D)
            #     q_z_var = nn.Sequential(
            #         nn.Linear(256, self.D),
            #         nn.Softplus(),
            #         nn.Hardtanh(min_val=0.01, max_val=7.) # TODO: why is this here?
            #     )

    def forward(self, x):
        # [B, C=256, 1, 1]
        q_z_x_params = self.encoder_gated_cnn_block(x)
        # [B, 256]
        q_z_x_params = q_z_x_params.squeeze(-1).squeeze(-1)
        print("q_z_x_params.shape", q_z_x_params.shape)
        return q_z_x_params


class DecoderGatedConvolutionBlock(nn.Module):
    """
    Maps z -> p_z_z_params
    """

    def __init__(self, args):
        super(DecoderGatedConvolutionBlock, self).__init__()