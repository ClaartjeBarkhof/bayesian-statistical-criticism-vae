import torch
import pytorch_lightning as pl
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np
from encoder import LanguageEncoder, fake_encoder_network
from decoder import LanguageDecoder, fake_decoder_network


class LanguageVaeModel(pl.LightningModule):
    def __init__(self, encoder, decoder, loss_term_manager, prior, B, D, L, V):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

        self.D = D
        self.L = L
        self.V = V
        self.B = B

        self.loss_term_manager = loss_term_manager

        self.prior = prior

    def forward(self, x_batch):
        # Get the posterior (multivariate Gaussian with diagonal covariance)
        # [B, D]
        q_z_x = self.encoder.encode(x_batch)

        # [B, D], Sample from the posterior w. reparameterisation
        z_post = q_z_x.rsample()

        # [B, L]
        p_x_z = self.decoder.forward(x_batch, z_post)

        return q_z_x, z_post, p_x_z

    def training_step(self, x_batch):
        q_z_x, z_post, p_x_z, logits = self.forward(x_batch)

        # TODO: should be something along the lines of x[:, 1:] (cutting of the start token)
        labels = torch.tensor(np.random.randint(0, self.V, size=(self.B, self.L)))


        loss = self.loss_term_manager.compute_loss(labels, q_z_x, z_post, p_x_z)

        return loss

    def sample(self, S):
        """Sample z from prior and decode auto-regressively."""

        # [S, D]
        z_prior = MultivariateNormal(torch.zeros(self.D), torch.eye(self.D)).sample(sample_shape=(S,))
        self.decoder.sample(z_prior)

if __name__ == "__main__":
    D = 2
    L = 10
    V = 100
    B = 5

    encoder = LanguageEncoder(D=D, B=B, network=fake_encoder_network)
    decoder = LanguageDecoder(B=B, L=L, V=V, network=fake_decoder_network)

    vae = LanguageVaeModel(encoder=encoder, decoder=decoder, loss_term_manager=None)