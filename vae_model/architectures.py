import torch.nn as nn
import torch

# --------------------------------------------------------------------------------------------------------------------
# TODO: MLP Architecture
# from: https://github.com/probabll/mixed-rv-vae/blob/master/components.py
# Encoder
# nn.Sequential(
#                 nn.Dropout(p_drop),
#                 nn.Linear(data_dim if mean_field else y_dim + data_dim, hidden_enc_size),
#                 nn.ReLU(),
#                 nn.Dropout(p_drop),
#                 nn.Linear(hidden_enc_size, hidden_enc_size),
#                 nn.ReLU(),
#                 nn.Linear(hidden_enc_size, z_num_params)
#             )
# Decoder
# nn.Sequential(
#             nn.Dropout(p_drop),
#             nn.Linear(z_dim + y_dim, hidden_dec_size),
#             nn.ReLU(),
#             nn.Dropout(p_drop),
#             nn.Linear(hidden_dec_size, hidden_dec_size),
#             nn.ReLU(),
#             nn.Dropout(p_drop),
#             nn.Linear(hidden_dec_size, data_dim),
#         )

# --------------------------------------------------------------------------------------------------------------------
# DCGAN Architecture

# from: https://github.com/pratikm141/MMD-Variational-Autoencoder-Pytorch-InfoVAE/blob/master/
# mmd_vae_pytorchver_norunlognoimg.ipynb

class ChannelsToLinear(nn.Linear):
    """Flatten a Variable to 2d and apply Linear layer"""

    def forward(self, x):
        b = x.size(0)
        return super().forward(x.view(b, -1))


class DCGanEncoder(nn.Module):
    def __init__(self, z_dim):
        super(DCGanEncoder, self).__init__()
        n_filters = 64
        self.conv1 = nn.Conv2d(1, n_filters, 4, 2, 1)
        self.conv2 = nn.Conv2d(n_filters, n_filters * 2, 4, 2, 1)

        self.toLinear1 = ChannelsToLinear(n_filters * 2 * 7 * 7, 1024)
        self.fc1 = nn.Linear(1024, z_dim*2) # added * 2 for mu, sigma

        self.lrelu = nn.LeakyReLU(negative_slope=0.1)

    def forward(self, x):
        h1 = self.lrelu(self.conv1(x))
        h2 = self.lrelu(self.conv2(h1))
        h3 = self.lrelu(self.toLinear1(h2))
        h4 = self.fc1(h3)

        return h4


class DCGanDecoder(nn.Module):
    def __init__(self, D):
        super(DCGanDecoder, self).__init__()

        self.D = D
        ngf = 32  # number of filters
        nc = 1  # number of channels, change for colour to 3

        self.network = nn.Sequential(
            nn.ConvTranspose2d(D, ngf * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        z = z.reshape(-1, self.D, 1, 1) # TODO: check this
        logits = self.network(z)
        return logits
