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

        num_classes = 256

        if self.data_distribution == 'bernoulli':
            p_x_nn = nn.Sequential(
                GatedConvTranspose2d(self.z_size, 64, self.last_kernel_size, 1, 0),
                GatedConvTranspose2d(64, 64, 5, 1, 2),
                GatedConvTranspose2d(64, 32, 5, 2, 2, 1),
                GatedConvTranspose2d(32, 32, 5, 1, 2),
                GatedConvTranspose2d(32, 32, 5, 2, 2, 1),
                GatedConvTranspose2d(32, 32, 5, 1, 2)
            )

            p_x_mean = nn.Sequential(
                nn.Conv2d(32, self.input_size[0], 1, 1, 0),
                nn.Sigmoid()
            )
            return p_x_nn, p_x_mean

        elif self.input_type == 'multinomial':
            act = None
            p_x_nn = nn.Sequential(
                GatedConvTranspose2d(self.z_size, 64, self.last_kernel_size, 1, 0, activation=act),
                GatedConvTranspose2d(64, 64, 5, 1, 2, activation=act),
                GatedConvTranspose2d(64, 32, 5, 2, 2, 1, activation=act),
                GatedConvTranspose2d(32, 32, 5, 1, 2, activation=act),
                GatedConvTranspose2d(32, 32, 5, 2, 2, 1, activation=act),
                GatedConvTranspose2d(32, 32, 5, 1, 2, activation=act)
            )

            p_x_mean = nn.Sequential(
                nn.Conv2d(32, 256, 5, 1, 2),
                nn.Conv2d(256, self.input_size[0] * num_classes, 1, 1, 0),
                # output shape: batch_size, num_channels * num_classes, pixel_width, pixel_height
            )

            return p_x_nn, p_x_mean