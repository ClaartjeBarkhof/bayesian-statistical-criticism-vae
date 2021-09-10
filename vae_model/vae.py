import pytorch_lightning as pl
from generative_model import GenerativeModel
from inference_model import InferenceModel
from loss_and_optimisation import Objective


class VaeModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()

        self.D = args.latent_dim
        self.L = args.max_seq_len
        self.V = args.vocab_size
        self.B = args.batch_size
        self.W = args.image_w
        self.H = args.image_h

        # Inference + generative model (encoder + decoder)
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

        return q_z_x, z_post, p_x_z

    def training_step(self, x_in):
        q_z_x, z_post, p_x_z = self.forward(x_in)

        loss = self.objective.compute_loss(x_in, q_z_x, z_post, p_x_z)

        return loss