import pytorch_lightning as pl
import torch
from torch.distributions import MultivariateNormal


class LanguageEncoder(pl.LightningModule):
    def __init__(self, D, B, network):
        super().__init__()

        self.D = D  # latent dimensionality D
        self.B = B  # batch size B
        self.network = network

    def encode(self, x_batch):
        """
        Returns a batched multivariate distribution object with independent dimensions.

        """

        # TODO: do an actual encode
        # [B, D]
        mus, sigmas = self.network(self.B, self.D)

        # Make diagonal covariance matrices: [B, D, D]
        sigmas = torch.diag_embed(sigmas)

        # [B, D]
        q_z_x = MultivariateNormal(mus, sigmas)

        return q_z_x

    def forward(self, x_batch, n_samples=1):
        # [S, B, D]
        return self.encode(x_batch).rsample(sample_shape=(n_samples,))

def fake_encoder_network(B, D):
    # [B, D]
    mus = torch.randn((B, D))
    sigmas = torch.abs(torch.randn((B, D)))
    return mus, sigmas

if __name__ == "__main__":
    e = Encoder(2, 4, fake_network)
    print(e.encode("bullshit"))
    print(e("more_bullshit").shape)
