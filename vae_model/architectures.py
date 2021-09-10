import torch.nn as nn

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


class LinearToChannels2d(nn.Linear):
    """Reshape 2d Variable to 4d after Linear layer"""

    def __init__(self, m, n, w=1, h=None, **kw):
        h = h or w
        super().__init__(m, n * w * h, **kw)
        self.w = w
        self.h = h

    def forward(self, x):
        b = x.size(0)
        return super().forward(x).view(b, -1, self.w, self.h)


class DCGanDecoder(nn.Module):
    def __init__(self, z_dim):
        super(DCGanDecoder, self).__init__()
        n_filters = 64

        self.fc1 = nn.Linear(z_dim, 1024)
        self.LineartoChannel = LinearToChannels2d(1024, n_filters * 2, 7, 7)
        self.conv1 = nn.ConvTranspose2d(n_filters * 2, n_filters, 4, 2, 1)
        self.conv2 = nn.ConvTranspose2d(n_filters, 1, 4, 2, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        h1 = self.relu(self.fc1(z))
        h2 = self.relu(self.LineartoChannel(h1))

        h3 = self.relu(self.conv1(h2))
        h4 = self.sigmoid(self.conv2(h3)) # logits

        return h4
