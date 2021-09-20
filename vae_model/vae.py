from .generative_model import GenerativeModel
from .inference_model import InferenceModel
import torch.nn as nn
import torch
import numpy as np


class VaeModel(nn.Module):
    def __init__(self, args, device="cpu"):
        super().__init__()
        self.D = args.latent_dim
        self.L = args.max_seq_len
        self.V = args.vocab_size
        self.B = args.batch_size
        self.image_w_h = args.image_w_h

        self.args = args

        # VAE = Inference model (encoder) + generative model (decoder)
        self.inf_model = InferenceModel(args=args, device=device)
        self.gen_model = GenerativeModel(args=args, device=device)

    def forward(self, x_in):
        # Infer the approximate posterior q(z|x) and sample from it to obtain z_post
        # [B, D], [S, B, D]
        print("vae.forward x_in.shape", x_in.shape)
        q_z_x, z_post = self.inf_model(x_in=x_in, n_samples=1)
        assert z_post.dim() == 3, "samples from the posterior must be 3D (S, B, D)"

        # Make predictions / generate based on the inferred latent
        p_x_z = self.gen_model(x_in=x_in, z_post=z_post)  # distribution-like object

        # Get the prior of the generative model
        p_z = self.gen_model.p_z  # distribution object

        return q_z_x, z_post, p_z, p_x_z

    def estimate_log_likelihood(self, data_loader):
        self.vae_model.eval()

        with torch.no_grad():
            for batch in data_loader:
                x_in = batch[0]
                x_in = x_in.to(self.device)

                q_z_x, z_post, p_z, p_x_z = self.vae_model.forward(x_in)

    @staticmethod
    def iw_log_p_x(log_p_x_z, log_p_z, log_q_z_x):
        """
        Importance weighted likelihood.
        log_p_x_z, log_p_z, log_q_z_x: [batch, n_samples]
        """
        n_samples = log_p_x_z.shape[1]
        iw_frac = log_p_x_z + log_p_z - log_q_z_x

        # Reduce the sample dimension with logsumexp, leaves shape [batch_size]
        iw_likelihood = torch.logsumexp(iw_frac, dim=-1) - np.log(n_samples)

        return iw_likelihood