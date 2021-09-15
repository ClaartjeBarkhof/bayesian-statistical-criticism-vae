import pytorch_lightning as pl
from .generative_model import GenerativeModel
from .inference_model import InferenceModel
from loss_and_optimisation import Objective
import torch
from lagrangian_opt.constraint import Constraint, ConstraintOptimizer


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

        # For the rate constraint
        self.mdr_constraint = self.get_mdr_constraint()
        self.mdr_loss = None
        # TODO: self.lag_vae_constraint = ...

    def get_mdr_constraint(self):
        if self.objective == "MDR-VAE":
            return Constraint(self.args.mdr_value, "ge", alpha=0.5)
        else:
            return None

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

    def shared_step(self, batch):
        x_in, labels = batch[0], batch[1]

        q_z_x, z_post, p_z, p_x_z = self.forward(x_in)

        loss_dict = self.objective.compute_loss(x_in, q_z_x, z_post, p_z, p_x_z)

        return loss_dict

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        if optimizer_idx == 0 or optimizer_idx is None:
            loss_dict = self.shared_step(batch)

            self.log("train_stats", loss_dict)

            return loss_dict["total_loss"]

        elif optimizer_idx == 1:
            if self.objective == "MDR-VAE":
                # do constraint forward
                constraint_loss = self.mdr_constraint

    def validation_step(self, batch, batch_idx):
        loss_dict = self.shared_step(batch)

        self.log("valid_stats", loss_dict)

    def configure_optimizers(self):
        # examples for multiple optimisers

        vae_optimiser = torch.optim.AdamW(self.parameters(), lr=self.args.lr)

        #return (vae_optimiser, torch.optim.AdamW(self.parameters(), lr=self.args.lr))
        return vae_optimiser