import pytorch_lightning as pl
from .generative_model import GenerativeModel
from .inference_model import InferenceModel
from objective import Objective
import torch
from lagrangian_opt.constraint import Constraint, ConstraintOptimizer


class VaeModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()

        # saves the arguments passed to the PL module in the checkpoint
        self.save_hyperparameters()

        self.automatic_optimization = False

        self.D = args.latent_dim
        self.L = args.max_seq_len
        self.V = args.vocab_size
        self.B = args.batch_size
        self.image_w_h = args.image_w_h

        self.args = args

        # VAE = Inference model (encoder) + generative model (decoder)
        self.inf_model = InferenceModel(args=args)
        self.gen_model = GenerativeModel(args=args)

        # For the rate constraint
        self.mdr_constraint = self.get_mdr_constraint()
        self.rate = None
        # TODO: self.lag_vae_constraint = ...

        # Objective
        self.objective = args.objective
        self.objective_module = Objective(args=args, mdr_constraint=self.mdr_constraint)
        self.vae_optimiser, self.mdr_constraint_optimiser = None, None # will be set in configure_optimizers

    def get_mdr_constraint(self):
        if self.args.objective == "MDR-VAE":
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

        print("vae forward p_x_z", p_x_z)

        # Get the prior of the generative model
        p_z = self.gen_model.p_z  # distribution object

        return q_z_x, z_post, p_z, p_x_z

    def shared_step(self, batch):
        x_in, labels = batch[0], batch[1]

        #print("x_in type shape", type(x_in), x_in.shape)

        q_z_x, z_post, p_z, p_x_z = self.forward(x_in)

        print("before compute loss")

        loss_dict = self.objective_module.compute_loss(x_in, q_z_x, z_post, p_z, p_x_z)

        return loss_dict

    def training_step(self, batch, batch_idx, optimizer_idx=None):

        # Get optimizers + zero grads
        if self.objective == "MDR-VAE":
            vae_optimiser, mdr_optimiser = self.optimizers()
            mdr_optimiser.zero_grad()
        else:
            vae_optimiser = self.optimizers()

        vae_optimiser.zero_grad()

        # Forward
        loss_dict = self.shared_step(batch)

        # Backward
        self.manual_backward(loss_dict["total_loss"], vae_optimiser)

        # if self.objective == "MDR-VAE":
        #     self.manual_backward(loss_dict["mdr_loss"], mdr_optimiser)

        # Step
        if self.objective == "MDR-VAE":
            mdr_optimiser.step()

        vae_optimiser.step()

        self.custom_log(loss_dict, "train")

        if self.objective == "MDR-VAE":
            self.log("train mdr_constraint_lambda", self.mdr_constraint.multiplier,
                     on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def validation_step(self, batch, batch_idx):
        loss_dict = self.shared_step(batch)
        self.custom_log(loss_dict, "valid")


    def custom_log(self, loss_dict, phase):
        for k, v in loss_dict.items():
            if v is None:
                continue
            else:
                if type(v) is float:
                    log_val = v
                else:
                    log_val = v.item()
                self.log(f"{phase} {k}", log_val, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        # examples for multiple optimisers

        vae_optimiser = torch.optim.Adam(self.parameters(), lr=self.args.lr)
        if self.objective == "MDR-VAE":
            print("RETURN MULTIPLE OPTIMIZERS")
            # noinspection PyTypeChecker
            mdr_constraint_optimiser = ConstraintOptimizer(torch.optim.RMSprop, self.mdr_constraint.parameters(),
                                                           self.args.mdr_constraint_optim_lr)
            self.vae_optimiser = vae_optimiser
            self.mdr_constraint_optimiser = mdr_constraint_optimiser
            return [vae_optimiser, mdr_constraint_optimiser]

        else:
            print("RETURN SINGLE OPTIMISER")
            self.vae_optimiser = vae_optimiser
            return vae_optimiser
