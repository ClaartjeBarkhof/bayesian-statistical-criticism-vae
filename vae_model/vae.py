from .generative_model import GenerativeModel
from .inference_model import InferenceModel
import torch.nn as nn
import torch
import numpy as np


class VaeModel(nn.Module):
    def __init__(self, args, device="cpu"):
        super().__init__()

        self.device = device

        self.D = args.latent_dim
        self.L = args.max_seq_len
        self.V = args.vocab_size
        self.B = args.batch_size
        self.image_w_h = args.image_w_h

        self.args = args

        # VAE = Inference model (encoder) + generative model (decoder)
        self.inf_model = InferenceModel(args=args, device=device)
        self.gen_model = GenerativeModel(args=args, device=device)

    def forward(self, x_in, n_samples=1):
        # Infer the approximate posterior q(z|x) and sample from it to obtain z_post
        # [B, D], [S, B, D]
        q_z_x, z_post = self.inf_model(x_in=x_in, n_samples=n_samples)
        assert z_post.dim() == 3, "samples from the posterior must be 3D (S, B, D)"

        # Make predictions / generate based on the inferred latent
        p_x_z = self.gen_model(x_in=x_in, z_post=z_post)  # distribution-like object

        # Get the prior of the generative model
        p_z = self.gen_model.p_z  # distribution object

        return q_z_x, z_post, p_z, p_x_z

    def estimate_log_likelihood(self, data_loader, n_samples=10):
        self.eval()

        iw_lls = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                print(f"... calculating importance weighted (n_samples={n_samples})"
                      f" log likelihood {batch_idx+1:3d}/{len(data_loader)}", end="\r")
                x_in = batch[0]
                x_in = x_in.to(self.device)

                # dist, [S, B, D], dist, dist
                q_z_x, z_post, p_z, p_x_z = self(x_in, n_samples=n_samples)

                # [S, B]
                log_q_z_x = q_z_x.log_prob(z_post)
                log_p_z = p_z.log_prob(z_post)
                if self.args.decoder_network_type == "conditional_made_decoder":
                    x_in = x_in.reshape(x_in.shape[0], -1)
                log_p_x_z = p_x_z.log_prob(x_in)

                n_samples = log_p_x_z.shape[0]
                iw_frac = log_p_x_z + log_p_z - log_q_z_x

                # Reduce the sample dimension with logsumexp
                # [B]
                iw_ll = torch.logsumexp(iw_frac, dim=0) - np.log(n_samples)

                iw_lls.append(iw_ll)

            iw_lls = torch.cat(iw_lls)

            iw_ll_mean, iw_ll_std = iw_lls.mean(), iw_lls.std()

            return iw_ll_mean, iw_ll_std