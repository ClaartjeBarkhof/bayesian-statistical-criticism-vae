from .generative_model import GenerativeModel
from .inference_model import InferenceModel
import torch.nn as nn
import torch
import numpy as np
import torch.distributions as td


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

    def reconstruct(self, x_in, Sx=10, Sz=1):
        q_z_x, z_post, p_z, p_x_z = self.forward(x_in, n_samples=Sz)
        return p_x_z.sample(sample_shape=(Sx,))

    def estimate_log_likelihood_batch(self, x_in, n_samples=10, per_bit=False):
        # dist, [S, B, D], dist, dist
        q_z_x, z_post, p_z, p_x_z = self(x_in, n_samples=n_samples)

        # [S, B]
        log_q_z_x = q_z_x.log_prob(z_post)
        log_p_z = p_z.log_prob(z_post)

        n_samples = log_p_z.shape[0]

        # [B, X_dim]
        if per_bit:
            # Independent Bernoulli
            if isinstance(p_x_z, td.Independent):
                logits = p_x_z.base_dist.logits
            # AutoregressiveDist
            else:
                # just to get the logits
                if p_x_z.params is None:
                    _ = p_x_z.log_prob(x_in)
                logits = p_x_z.params

            # Logits shape [S, B,  C, H, W]
            log_p_x_z = td.Independent(td.Bernoulli(logits=logits), 0).log_prob(x_in)
            iw_frac = log_p_x_z + log_p_z.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) - log_q_z_x.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

            iw_ll = torch.logsumexp(iw_frac, dim=0) - np.log(n_samples)
        # [B]
        else:
            log_p_x_z = p_x_z.log_prob(x_in)
            iw_frac = log_p_x_z + log_p_z - log_q_z_x

            # Reduce the sample dimension with logsumexp
            # [B]

            iw_ll = torch.logsumexp(iw_frac, dim=0) - np.log(n_samples)

        return iw_ll

    def estimate_log_likelihood_dataset(self, data_loader, n_samples=10, max_batches=None, short_dev_run=False):
        self.eval()

        iw_lls = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                print(f"... calculating importance weighted (n_samples={n_samples})"
                      f" log likelihood {batch_idx+1:3d}/{len(data_loader)}", end="\r")

                x_in = batch[0]
                x_in = x_in.to(self.device)

                iw_ll = self.estimate_log_likelihood_batch(x_in=x_in, n_samples=n_samples)

                iw_lls.append(iw_ll)

                if max_batches is not None and batch_idx + 1 == max_batches:
                    break

                if short_dev_run and batch_idx == 1:
                    break

            iw_lls = torch.cat(iw_lls)

            return iw_lls.tolist()