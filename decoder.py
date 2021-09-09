import pytorch_lightning as pl

import numpy as np
import torch
from torch.distributions import Categorical

# TO DO:
# class ImageDecoder(pl.LightningModule):
#     def __init__(self, B, L, W, H, network):
#         super().__init__()

class LanguageDecoder(pl.LightningModule):
    """
    A language decoder is one that takes a latent representation and a pre-fix sequence and maps
    it to the parameters of a Categorical distribution over a sequence.
    """
    def __init__(self, B, L, V, network):
        super().__init__()

        self.B = B  # batch size B
        self.L = L  # max sequence length L
        self.V = V  # vocabulary size

        self.network = network  # the deep neural network that does the actual mapping

    def forward(self, x_batch, z):
        """This function maps the pre-fix and latent code to a set of categoricals over the sequence."""
        # [B, L, V]
        logits = self.network(x_batch, z, self.B, self.L, self.V)

        # Output categorical distributions at the decoder [B, L]
        p_x_z = Categorical(logits=logits)

        return p_x_z

def fake_decoder_network(x_batch, z, B, L, V):
    # [B, L, V]
    return torch.randn((B, L, V))

if __name__ == "__main__":
    V = 20
    L = 10
    B = 3

    d = Decoder(B, L, V, fake_decoder_network)
    x_batch = torch.tensor(np.random.randint(0, V, size=(B, L)))

    p_x_z, log_probs = d.forward(x_batch, "bullshit_z")
    print(p_x_z)
    print(log_probs.shape)

