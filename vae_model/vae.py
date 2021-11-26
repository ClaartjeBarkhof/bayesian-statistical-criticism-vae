from .generative_model import GenerativeModel
from .inference_model import InferenceModel
import torch.nn as nn
import torch
import numpy as np
import torch.distributions as td

from .tie_weights import tie_weights_fn

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

        # Weight tying / sharing between encoder and decoder RoBERTa part
        # print("Tying encoder decoder RoBERTa checkpoint weights!")
        # base_model_prefix = self.gen_model.decoder_network.roberta_model.base_model_prefix
        # tie_weights_fn(self.inf_model.encoder_network.roberta_model,
        #                self.gen_model.decoder_network.roberta_model._modules[base_model_prefix],
        #                base_model_prefix)

    def forward(self, x_in, n_samples=1):
        # Infer the approximate posterior q(z|x) and sample from it to obtain z_post
        # [B, D], [S, B, D]
        q_z_x, z_post = self.inf_model(x_in=x_in, n_samples=n_samples)
        assert z_post.dim() == 3, "samples from the posterior must be 3D (S, B, D)"

        # Make predictions / generate based on the inferred latent
        # distribution-like object or tuple with (dist, dist), where the second dist = p_l_z
        p_x_z = self.gen_model(x_in=x_in, z_post=z_post)

        # Get the prior of the generative model
        p_z = self.gen_model.get_p_z()  # distribution object

        return q_z_x, z_post, p_z, p_x_z

    def reconstruct(self, x_in, Sx=10, Sz=1):
        q_z_x, z_post, p_z, p_x_z = self.forward(x_in, n_samples=Sz)
        return p_x_z.sample(sample_shape=(Sx,))

    def estimate_log_likelihood_batch(self, x_in, n_samples=10, image_or_language="image", per_bit=False):
        # dist, [S, B, D], dist, dist (or tuple of dist, dist)
        q_z_x, z_post, p_z, p_x_z = self(x_in, n_samples=n_samples)

        # [S, B]
        log_q_z_x = q_z_x.log_prob(z_post)
        log_p_z = p_z.log_prob(z_post)

        n_samples = log_p_z.shape[0]

        # IMAGE
        if image_or_language == "image":
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

        # LANGUAGE
        else:
            input_ids, attention_mask = x_in

            # [1, B, L]
            labels = input_ids[:, 1:].unsqueeze(0)
            label_mask = attention_mask[:, 1:].unsqueeze(0)

            # [1, B]
            label_length = label_mask.sum(dim=-1).long()

            # weak decoder
            p_l_z = None
            weak_decoder = False
            if type(p_x_z) == tuple:
                weak_decoder = True
                # unpack
                p_x_z, p_l_z = p_x_z

            log_p_x_z = p_x_z.log_prob(labels)
            log_p_x_z = (log_p_x_z * label_mask).sum(dim=-1)

            if weak_decoder:
                log_p_l_z = p_l_z.log_prob(label_length)
                # this naming is a bit off but just to match the other code
                log_p_x_z = log_p_x_z + log_p_l_z

            iw_frac = log_p_x_z + log_p_z - log_q_z_x

            # Reduce the sample dimension with logsumexp
            # [B]
            iw_ll = torch.logsumexp(iw_frac, dim=0) - np.log(n_samples)

        return iw_ll

    def estimate_log_likelihood_dataset(self, data_loader, n_samples=10, max_batches=None, image_or_language="image", short_dev_run=False):
        print(f"Calculating importance weighted log likelihood "
              f"(n_samples={n_samples}, batch_size={data_loader.batch_size})!")
        self.eval()

        iw_lls = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                print(f"... calculating importance weighted (n_samples={n_samples})"
                      f" log likelihood {batch_idx+1:3d}/{len(data_loader)}", end="\r")

                # language
                if type(batch) == dict:
                    x_in = (batch["input_ids"].to(self.device), batch["attention_mask"].to(self.device))

                # image
                else:
                    x_in = batch[0]
                    x_in = x_in.to(self.device)

                iw_ll = self.estimate_log_likelihood_batch(x_in=x_in, n_samples=n_samples, image_or_language=image_or_language)

                iw_lls.append(iw_ll)

                if max_batches is not None and batch_idx + 1 == max_batches:
                    break

                if short_dev_run and batch_idx == 1:
                    break

            iw_lls = torch.cat(iw_lls)

            return iw_lls.tolist()