import torch.nn as nn
# import torch
# import numpy as np



# import torch.nn as nn
# import torch
# import numpy as np
#
# # --------------------------------------------------------------------------------------------------------------------
# # TODO: MLP Architecture
# # from: https://github.com/probabll/mixed-rv-vae/blob/master/components.py
# # Encoder
# # nn.Sequential(
# #                 nn.Dropout(p_drop),
# #                 nn.Linear(data_dim if mean_field else y_dim + data_dim, hidden_enc_size),
# #                 nn.ReLU(),
# #                 nn.Dropout(p_drop),
# #                 nn.Linear(hidden_enc_size, hidden_enc_size),
# #                 nn.ReLU(),
# #                 nn.Linear(hidden_enc_size, z_num_params)
# #             )
# # Decoder
# # nn.Sequential(
# #             nn.Dropout(p_drop),
# #             nn.Linear(z_dim + y_dim, hidden_dec_size),
# #             nn.ReLU(),
# #             nn.Dropout(p_drop),
# #             nn.Linear(hidden_dec_size, hidden_dec_size),
# #             nn.ReLU(),
# #             nn.Dropout(p_drop),
# #             nn.Linear(hidden_dec_size, data_dim),
# #         )
#
# # --------------------------------------------------------------------------------------------------------------------
# # DCGAN Architecture
#
# # from: https://github.com/pratikm141/MMD-Variational-Autoencoder-Pytorch-InfoVAE/blob/master/
# # mmd_vae_pytorchver_norunlognoimg.ipynb
#
# class ChannelsToLinear(nn.Linear):
#     """Flatten a Variable to 2d and apply Linear layer"""
#
#     def forward(self, x):
#         b = x.size(0)
#         return super().forward(x.view(b, -1))
#
#
# class DCGanEncoder(nn.Module):
#     def __init__(self, z_dim):
#         super(DCGanEncoder, self).__init__()
#         n_filters = 64
#         self.conv1 = nn.Conv2d(1, n_filters, 4, 2, 1)
#         self.conv2 = nn.Conv2d(n_filters, n_filters * 2, 4, 2, 1)
#
#         self.toLinear1 = ChannelsToLinear(n_filters * 2 * 7 * 7, 1024)
#         self.fc1 = nn.Linear(1024, z_dim*2) # added * 2 for mu, sigma
#
#         self.lrelu = nn.LeakyReLU(negative_slope=0.1)
#
#     def forward(self, x):
#         h1 = self.lrelu(self.conv1(x))
#         h2 = self.lrelu(self.conv2(h1))
#         h3 = self.lrelu(self.toLinear1(h2))
#         h4 = self.fc1(h3)
#
#         return h4
#
#
# class DCGanDecoder(nn.Module):
#     def __init__(self, D):
#         super(DCGanDecoder, self).__init__()
#
#         self.D = D
#         ngf = 32  # number of filters
#         nc = 1  # number of channels, change for colour to 3
#
#         self.network = nn.Sequential(
#             nn.ConvTranspose2d(D, ngf * 4, 4, 1, 0, bias=False),
#             nn.BatchNorm2d(ngf * 4),
#             nn.ReLU(True),
#
#             nn.ConvTranspose2d(ngf * 4, ngf * 2, 3, 2, 1, bias=False),
#             nn.BatchNorm2d(ngf * 2),
#             nn.ReLU(True),
#
#             nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ngf),
#             nn.ReLU(True),
#
#             nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
#             nn.Tanh()
#         )
#
#     def forward(self, z):
#         z = z.reshape(-1, self.D, 1, 1) # TODO: check this
#         logits = self.network(z)
#         return logits
#
#
# # --------------------------------------------------------------------------------------------------------------------
# # Encoder / decoder architectures from Alemi et al. (2018)
#
# class SamePadding(nn.Module):
#     def __init__(self, input_size, kernel_size, stride):
#         super(SamePadding, self).__init__()
#         self.input_size = input_size
#         output_size = input_size
#         padding_size = (((input_size - kernel_size) / stride) + 1 - output_size) * - (stride / 2)
#         # print("padding size", padding_size)
#         # print("padding tuple", (np.floor(padding_size), np.ceil(padding_size), np.floor(padding_size), np.ceil(padding_size)))
#         self.same_padding = torch.nn.ZeroPad2d((int(np.floor(padding_size)), int(np.ceil(padding_size)),
#                                                 int(np.floor(padding_size)), int(np.ceil(padding_size))))
#         # self.same_padding = nn.ZeroPad2d((15, 16, 15, 16))
#
#     def forward(self, x):
#         assert x.shape[-1] == self.input_size, "input size not correct"
#         assert x.shape[-2] == self.input_size, "input size not correct"
#
#         x_padded = self.same_padding(x)
#
#         return x_padded
#
#
# class GatedLinearUnit(nn.Module):
#     def __init__(self):
#         super(GatedLinearUnit, self).__init__()
#
#     def forward(self, x):
#         activation = torch.nn.functional.glu(torch.stack([x, x], dim=-1), dim=-1).squeeze(-1)
#         return activation
#
#
# class AlemiSimpleEncoder(nn.Module):
#     """
#     This class implements the simple (image) encoder from Alemi et al. (2018), Appendix F.
#
#     They describe the architecture as follows
#     -> Conv (depth, kernel size, stride, padding):
#
#     Unless otherwise specified, all layers used a linearly gated activation function
#     activation function (Dauphin et al., 2017), h(x) = (W1x + b2)σ(W2x + b2)
#
#     • Input (28, 28, 1)
#     • Conv (32, 5, 1, same)
#     • Conv (32, 5, 2, same)
#     • Conv (64, 5, 1, same)
#     • Conv (64, 5, 2, same)
#     • Conv (256, 7, 1, valid)
#     • Gauss (Linear (64), Softplus (Linear (64)))
#
#     """
#
#     def __init__(self, D=64, C=1, W=28, H=28, q_z_x_type="gaussian"):
#         super(AlemiSimpleEncoder, self).__init__()
#
#         assert W == H, "This encoder only accepts square input"
#         self.in_shape = W
#         self.D = D
#         self.C = C
#         self.q_z_x_type = q_z_x_type
#
#         self.conv1 = nn.Conv2d(in_channels=C, out_channels=32, kernel_size=5, stride=1, padding='same')
#         self.gate1 = GatedLinearUnit()
#
#         # pad output of conv1 so that it keeps the original shape (weight and height)
#         self.pad2 = SamePadding(self.in_shape, kernel_size=5, stride=2)
#         self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=2)
#         self.gate2 = GatedLinearUnit()
#
#         self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding='same')
#         self.gate3 = GatedLinearUnit()
#
#         # pad output of conv3 so that it keeps the original shape (weight and height)
#         self.pad4 = SamePadding(self.in_shape, kernel_size=5, stride=2)
#         self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=2)
#         self.gate4 = GatedLinearUnit()
#
#         self.conv5 = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=7, stride=1, padding='valid')
#         self.gate5 = GatedLinearUnit()
#
#         # w * h * channels
#         out_5_shape = int((self.in_shape - 7) + 1) * int((self.in_shape - 7) + 1) * 256
#
#         if self.q_z_x_type == "gaussian":
#             # torch.nn.Linear(in_features, out_features, bias=True, device=None, dtype=None)
#             self.linear_mean = nn.Linear(out_5_shape, self.D)
#             self.linear_logvar = nn.Linear(out_5_shape, self.D)
#         else:
#             raise NotImplementedError
#
#     def forward(self, x):
#         conv1_out = self.gate1(self.conv1(x))
#         # same padding with stride =/= 1
#         conv2_out = self.gate2(self.conv2(self.pad2(conv1_out)))
#         conv3_out = self.gate3(self.conv3(conv2_out))
#         # same padding with stride =/= 1
#         conv4_out = self.gate4(self.conv4(self.pad4(conv3_out)))
#         conv5_out = self.gate5(self.conv5(conv4_out))
#
#         conv5_out_flat = torch.flatten(conv5_out, start_dim=1)
#
#         q_z_x_params = None
#         if self.q_z_x_type:
#             mean = self.linear_mean(conv5_out_flat)
#             logvar = self.linear_logvar(conv5_out_flat)
#             q_z_x_params = (mean, logvar)
#
#         return q_z_x_params
#
#
# class AlemiSimpleDecoder(nn.Module):
#     """
#     This class implements the simple (image) encoder from Alemi et al. (2018), Appendix F.
#
#     They describe the architecture as follows
#     -> Deconv (depth, kernel size, stride, padding):
#
#     Unless otherwise specified, all layers used a linearly gated activation function
#     activation function (Dauphin et al., 2017), h(x) = (W1x + b2)σ(W2x + b2)
#
#     • Input (1, 1, 64)
#     • Deconv (64, 7, 1, valid)
#     • Deconv (64, 5, 1, same)
#     • Deconv (64, 5, 2, same)
#     • Deconv (32, 5, 1, same)
#     • Deconv (32, 5, 2, same)
#     • Deconv (32, 4, 1, same)
#     • Bernoulli (Linear Conv (1, 5, 1, same))
#
#     """
#
#     def __init__(self, z_dim=64):
#         super(AlemiSimpleDecoder, self).__init__()
#
#         self.fc = nn.Linear(, 64)
#         self.deconv1 = nn.ConvTranspose2d(32, 64, kgiternel_size=3, stride=2, padding=1, output_padding=1)
#         self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
#         self.deconv3 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
#         self.deconv4 = nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1)
#
#         deconv_1 = nn.ConvTranspose2d

