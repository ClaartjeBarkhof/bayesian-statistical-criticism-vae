import pytorch_lightning as pl
from .generative_model import GenerativeModel
from .inference_model import InferenceModel
from loss_and_optimisation import Objective
import torch


class VaeModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()

        self.D = args.latent_dim
        self.L = args.max_seq_len
        self.V = args.vocab_size
        self.B = args.batch_size
        self.image_w_h = args.image_w_h

        self.args = args

        # VAE = Inference model (encoder) + generative model (decoder)
        self.inf_model = InferenceModel(args=args)
        self.gen_model = GenerativeModel(args=args)

        # Objective
        self.objective = Objective(args=args)

    def forward(self, x_in):
        # Infer the approximate posterior q(z|x) and sample from it to obtain z_post
        # [B, D]
        q_z_x, z_post = self.inf_model(x_in=x_in)

        # Make predictions / generate based on the inferred latent
        # Language: Categorical of [B, L]
        # Image: Bernoulli or Gaussian of [W, H]
        p_x_z = self.gen_model(x_in=x_in, z_post=z_post)

        # Get the prior of the generative model
        p_z = self.gen_model.p_z()

        return q_z_x, z_post, p_z, p_x_z

    def training_step(self, x_in_labels_in, batch_idx):
        x_in, labels_in = x_in_labels_in[0], x_in_labels_in[1]

        q_z_x, z_post, p_z, p_x_z = self.forward(x_in)

        loss_dict = self.objective.compute_loss(x_in, q_z_x, z_post, p_z, p_x_z)

        self.log("train_stats", loss_dict)

        return loss_dict["total_loss"]

    def configure_optimizers(self):
        optimiser = torch.optim.AdamW(self.parameters(), lr=self.args.lr)
        return optimiser

