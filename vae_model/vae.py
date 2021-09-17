import pytorch_lightning as pl
from .generative_model import GenerativeModel
from .inference_model import InferenceModel
from objective import Objective
import torch
import torch.nn as nn
from lagrangian_opt.constraint import Constraint, ConstraintOptimizer


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
        # [B, D]
        q_z_x, z_post = self.inf_model(x_in=x_in)

        # Make predictions / generate based on the inferred latent
        # Language: Categorical of [B, L]
        # Image: Bernoulli or Gaussian of [W, H]
        p_x_z = self.gen_model(x_in=x_in, z_post=z_post)  # distribution object

        # Get the prior of the generative model
        p_z = self.gen_model.p_z  # distribution object

        return q_z_x, z_post, p_z, p_x_z