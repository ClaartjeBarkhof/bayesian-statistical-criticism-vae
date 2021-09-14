import torch
import torch.nn as nn
import torch.distributions as td
from vae_model.architectures import DecoderGatedConvolutionBlock


class GenerativeModel(nn.Module):
    def __init__(self, args):
        super(GenerativeModel, self).__init__()

        self.args = args

        self.D = args.latent_dim
        self.B = args.batch_size

        # LANGUAGE
        self.L = args.max_seq_len
        self.V = args.vocab_size

        # IMAGE
        self.image_w_h = args.image_w_h
        self.C = args.n_channels

        # NETWORK
        # a decoder that maps z (+ x) to params of p_x_z
        self.decoder_network_type = args.decoder_network_type
        assert self.decoder_network_type in ["basic_decoder", "conditional_decoder"]
        self.decoder_network = self.get_decoder_network()

        # PRIOR
        self.p_z_type = args.p_z_type

        # OUTPUT DISTRIBUTION
        self.p_x_z_type = args.p_x_z_type #TODO: is this necessary? it is already defined by args.data_distribtion

    def sample_generative_model(self, S=1):
        z_prior = self.sample_prior(S=S)
        p_x_z_prior = self.p_x_z(z_prior)
        return p_x_z_prior

    def forward(self, z_post, x_in=None):
        """
        Map a sample from the posterior to a generative output distribution: z_post -> p(X|Z=z_post)
        -> Potentially condition on (a part of) x too, e.g. the prefix: z_post -> p(X|Z=z_post, Y=x<i)

        Input:
            z_post: [B, D]
                samples from the posterior q(z|x)
            x_in: [B, L] (language) or [B, C, W, H] (image)
                the original input the posterior conditioned on

        Output:
            p_x_z_post: [B, L] (language) or or [B, C, W, H] (image), generative output distribution p(X|Z=z_post)
        """
        p_x_z_post = self.p_x_z(z_post)
        return p_x_z_post

    def sample_prior(self, S=1):
        """
        Returns a from the prior defined by <p_z_type>.

        Input:
            S: int:
                the number of samples it should return.

        Output:
            z_prior: [S, D]
                samples from the prior of dimensionality of the latent space.
        """
        z_prior = self.p_z.sample(sample_shape=(S,))

        return z_prior

    def p_x_z(self, z, x=None):
        p_x_z_params = self.decoder_network(z)  # TODO: some decoder networks take x as additional input

        p_x_z = None
        if self.p_x_z_type == "bernoulli":
            # TODO: apply sigmoid?, as the output of DCGAN is tanh
            p_x_z = td.Independent(td.Bernoulli(logits=p_x_z_params), 1)

        elif self.p_x_z_type == "gaussian":
            p_x_z = td.Independent(td.Normal(loc=p_x_z_params, scale=torch.ones_like(p_x_z_params)), 1)

        elif self.p_x_z_type == "categorical":
            # TODO: implement Categorical output
            # p_x_z_params: [B, L, V]
            p_x_z = td.Categorical(logits=p_x_z_params)

        else:
            # TODO: implement other options
            raise NotImplementedError

        return p_x_z

    def p_z(self):
        if self.p_z_type == "isotropic_gaussian":
            return td.Independent(td.Normal(loc=torch.zeros(self.D), scale=torch.ones(self.D)), 1)
        # TODO: add other priors
        else:
            raise NotImplementedError

    def get_decoder_network(self):
        if self.decoder_network_type == "basic_decoder":
            return DecoderGatedConvolutionBlock(args=self.args)
        elif self.decoder_network_type == "conditional_decoder":
            raise NotImplementedError
        else:
            raise ValueError(f"{self.decoder_network_type} is not a valid decoder_network_type, choose: "
                             f"'basic_decoder' or 'conditional_decoder'")