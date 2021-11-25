import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td

from vae_model.sylvester_flows.models.layers import GatedConvTranspose2d
from vae_model.distributions import AutoRegressiveDistribution, AutoRegressiveRobertaStrongDecoder
from vae_model.made import MADE

from vae_model.pixel_cnn_pp.model import PixelCNN

from vae_model.roberta.roberta_strong_decoder import VaeStrongDecoderRobertaForCausalLM
from vae_model.roberta.roberta import RobertaForMaskedLM
from transformers.models.roberta.configuration_roberta import RobertaConfig


class GenerativeModel(nn.Module):
    def __init__(self, args, device="cpu"):
        super(GenerativeModel, self).__init__()

        self.device = device
        self.args = args

        self.D = args.latent_dim
        self.B = args.batch_size

        self.image_or_language = args.image_or_language

        # LANGUAGE
        self.L = args.max_seq_len
        self.V = args.vocab_size

        # IMAGE
        self.image_w_h = args.image_w_h
        self.C = args.n_channels

        # NETWORK
        # a decoder that maps z (+ x) to params of p_x_z
        self.decoder_network_type = args.decoder_network_type

        self.decoder_network = self.get_decoder_network()

        # PRIOR
        self.p_z_type = args.p_z_type

        # MIXTURE OF GAUSIANS PRIOR
        if self.p_z_type == "mog":
            self.mog_n_components = args.mog_n_components
            self.register_parameter("mix_components", torch.nn.Parameter(torch.rand(self.mog_n_components)))
            self.register_parameter("component_means", torch.nn.Parameter(torch.randn(self.mog_n_components, self.D)))
            self.register_parameter("component_pre_scales", torch.nn.Parameter(torch.abs(
                torch.randn(self.mog_n_components, self.D))))

        # ISOTROPIC GAUSSIAN PRIOR
        else:
            self.register_buffer("prior_z_means", torch.zeros(self.D, requires_grad=False))
            self.register_buffer("prior_z_scales", torch.ones(self.D, requires_grad=False))

        # OUTPUT DISTRIBUTION == DATA DISTRIBUTION
        self.p_x_z_type = args.data_distribution

    def sample_generative_model(self, z=None, Sx=1, Sz=1, return_z=False, device="cuda:0"):
        # Unconditional generation
        if z is None:
            # [S, 1, D]
            z = self.sample_prior(S=Sz).to(device)
        # Conditional generation
        else:
            assert z.dim() == 3, f"we expect z always to be 3d [S, B, D], shape received {z.shape}"

        p_x_z = self.p_x_z(z)

        # Because of its autoregressiveness, it is better to do it this way
        # a bit of a hack: make some sort of autoregressive distribution here as well
        if self.decoder_network_type == "cond_pixel_cnn_pp":
            sampled_x = self.decoder_network.sample
        else:
            sampled_x = p_x_z.sample(sample_shape=(Sx,))

        # sampled_x is of shape: [Sx, Sz, B, C, W, H]

        if return_z:
            return z, sampled_x
        else:
            return sampled_x

    def forward(self, z_post, x_in=None):
        """
        Map a sample from the posterior to a generative output distribution: z_post -> p(X|Z=z_post)
        -> Potentially condition on (a part of) x too, e.g. the prefix: z_post -> p(X|Z=z_post, Y=x<i)

        Input:
            z_post: [S, B, D]
                samples from the posterior q(z|x)
            x_in: [B, L] (language) or [B, C, W, H] (image)
                the original input the posterior conditioned on

        Output:
            p_x_z_post: [B, L] (language) or or [B, C, W, H] (image), generative output distribution p(X|Z=z_post)
        """

        p_x_z_post = self.p_x_z(z_post, x_in=x_in)

        return p_x_z_post

    def sample_prior(self, S=1):
        """
        Returns a from the prior defined by <p_z_type>.

        Input:
            S: int:
                the number of samples it should return.

        Output:
            z_prior: [S, 1, D]
                samples from the prior of dimensionality of the latent space.
        """
        z_prior = self.get_p_z().sample(sample_shape=(S, 1))

        return z_prior

    def p_x_z(self, z, x_in=None):
        """
        Maps z to a distribution p(x|z) via a decoder network.

        Input:
            z: [S, B, D]

        Returns a distribution-like object with parameters [S, B, ...], reducing ... as dimensions for log_prob
        """

        p_x_z = None
        if self.decoder_network_type == "conditional_made_decoder":
            # Autoregressive distribution object
            # the true forward is only called with log_prob
            p_x_z = self.decoder_network(z)

        else:
            # [S, B, D]
            (S, B, _) = z.shape

            # Catch the case of the auto-regressive forward of strong language decoder
            # this should return a special wrapper object of type AutoRegressiveRobertaStrongDecoder
            if self.decoder_network_type == "strong_distil_roberta_decoder" and x_in is None:
                p_x_z = self.decoder_network.auto_regressive_forward(z=z)
                return p_x_z

            # Flattening / reshaping for MLP or Conv happens in respective modules
            p_x_z_params = self.decoder_network(z, x_in=x_in)

            # DATA DISTRIBUTION (bernoulli or categorical)
            if self.image_or_language == "image":
                if self.p_x_z_type == "bernoulli":
                    # [S, B, C, W, H]
                    assert p_x_z_params.shape == (S, B, self.C, self.image_w_h, self.image_w_h), \
                        f"bernoulli logits should be of shape [S, B, C, W, H], currently of shape {p_x_z_params.shape}"
                    p_x_z = td.Independent(td.Bernoulli(logits=p_x_z_params), 3)  # reduce last 3 dimensions with log_prob

                elif self.p_x_z_type == "multinomial":
                    # [S, B, C, W, H, num_classes]
                    assert p_x_z_params.shape == (S, B, self.C, self.image_w_h, self.image_w_h), \
                        f"multinomial logits should be of shape [S, B, C, W, H, num_classes], currently of shape {p_x_z_params.shape}"
                    p_x_z = td.Categorical(logits=p_x_z_params)

            # LANGUAGE (categorical)
            else:
                # strong decoder
                if self.decoder_network_type == "strong_distil_roberta_decoder":
                    assert p_x_z_params.shape == (S, B, self.L-1, self.V), \
                        f"categorical logits should be of shape (S, B, L, V), currently: {p_x_z_params.shape}"
                    # no independent because masking needs to happen still, so the seq. dim. should stay
                    p_x_z = td.Categorical(logits=p_x_z_params)
                # weak decoder
                else:
                    p_x_z_params, length_logits = p_x_z_params
                    assert p_x_z_params.shape == (S, B, self.L - 1, self.V), \
                        f"categorical logits should be of shape (S, B, L-1, V), currently: {p_x_z_params.shape}"
                    assert length_logits.shape == (S, B, self.L), \
                        f"we expect length_logits to be of shape (S, B, L), currently: {length_logits.shape}"
                    # no independent because masking needs to happen still, so the seq. dim. should stay
                    p_x_z = td.Categorical(logits=p_x_z_params)
                    p_z_l = td.Categorical(logits=length_logits)
                    p_x_z = (p_x_z, p_z_l)

        return p_x_z

    def get_p_z(self):
        # ISOTROPIC GAUSSIAN
        if self.p_z_type == "isotropic_gaussian":
            return td.Independent(td.Normal(loc=self.prior_z_means, scale=self.prior_z_scales), 1)

        # MIXTURE OF GAUSSIANS
        elif self.p_z_type == "mog":
            mix = td.Categorical(logits=self.mix_components)
            comp = td.Independent(td.Normal(self.component_means, F.softplus(self.component_pre_scales)), 1)
            return td.MixtureSameFamily(mix, comp)
        else:
            raise ValueError(f"{self.p_z_type} is not a valid p_z_type, choices: isotropic_gaussian, mog")

    def get_decoder_network(self):
        # Image
        if self.decoder_network_type == "basic_deconv_decoder":
            return DecoderGatedConvolutionBlock(args=self.args)
        elif self.decoder_network_type == "basic_mlp_decoder":
            return DecoderMLPBlock(args=self.args)
        elif self.decoder_network_type == "conditional_made_decoder":
            return ConditionalBernoulliBlockMADE(args=self.args)
        elif self.decoder_network_type == "cond_pixel_cnn_pp":
            return DecoderPixelCNNppBlock(args=self.args)

        # Language
        elif self.decoder_network_type == "strong_distil_roberta_decoder":
            return DecoderStrongDistilRoberta(args=self.args)
        elif self.decoder_network_type == "weak_distil_roberta_decoder":
            return DecoderWeakDistilRoberta(args=self.args)
        else:
            raise NotImplementedError


class ConditionalBernoulliBlockMADE(nn.Module):
    def __init__(self, args):
        super(ConditionalBernoulliBlockMADE, self).__init__()

        self.X_shape = (args.n_channels, args.image_w_h, args.image_w_h)
        self.X_dim = args.image_w_h * args.image_w_h * args.n_channels
        self.D = args.latent_dim

        if hasattr(args, 'decoder_MADE_hidden_sizes'):
            hiddens = [int(h) for h in args.decoder_MADE_hidden_sizes.split("-")]
            mech = args.decoder_MADE_gating_mechanism
        else:
            hiddens = [200, 220]
            mech = 0

        print("Hidden sizes of the decoder made: ", hiddens)
        natural_ordering = True
        act = nn.ReLU()

        self.made = MADE(self.X_dim, hiddens, self.X_dim, natural_ordering=natural_ordering,
                         context_size=self.D, hidden_activation=act, gating=args.decoder_MADE_gating,
                         gating_mechanism=mech)

    def forward(self, z):
        # Placeholder distribution object
        p_x_z = AutoRegressiveDistribution(context=z, made=self.made, dist_type="bernoulli",
                                           encoder=False, X_shape=self.X_shape)

        return p_x_z


class DecoderGatedConvolutionBlock(nn.Module):
    """
    Maps z -> p_x_z_params
        In case of Bernoulli these are logits [B, C, W, H]
        In case of Multinomial these are logits [B, C, num_classes=256, W, H]

    # This code is adapted from Rianne van de Berg's code (sylvester_flows submodule):
    https://github.com/riannevdberg/sylvester-flows/blob/32dde9b7d696fee94f946a338182e542779eecfe/models/VAE.py

    """

    def __init__(self, args):
        super(DecoderGatedConvolutionBlock, self).__init__()

        self.num_classes = 256
        self.D = args.latent_dim
        self.C = args.n_channels
        self.data_distribution = args.data_distribution
        self.image_w_h = args.image_w_h

        if self.image_w_h == 28:
            self.last_kernel_size = 7
        else:
            raise ValueError('Only supporting input size 28 now.')

        if self.data_distribution == 'bernoulli':
            self.decoder_gated_cnn_block = nn.Sequential(
                GatedConvTranspose2d(self.D, 64, self.last_kernel_size, 1, 0),
                GatedConvTranspose2d(64, 64, 5, 1, 2),
                GatedConvTranspose2d(64, 32, 5, 2, 2, 1),
                GatedConvTranspose2d(32, 32, 5, 1, 2),
                GatedConvTranspose2d(32, 32, 5, 2, 2, 1),
                GatedConvTranspose2d(32, 32, 5, 1, 2),
                # TODO: why did rianne separate this layer from the rest? (+ a sigmoid), she had extra nn.Sigmoid(),
                nn.Conv2d(32, self.C, 1, 1, 0),

            )

        # output shape: batch_size, num_channels * num_classes, pixel_width, pixel_height
        elif self.data_distribution == 'multinomial':
            self.decoder_gated_cnn_block = nn.Sequential(
                GatedConvTranspose2d(self.D, 64, self.last_kernel_size, 1, 0),
                GatedConvTranspose2d(64, 64, 5, 1, 2),
                GatedConvTranspose2d(64, 32, 5, 2, 2, 1),
                GatedConvTranspose2d(32, 32, 5, 1, 2),
                GatedConvTranspose2d(32, 32, 5, 2, 2, 1),
                GatedConvTranspose2d(32, 32, 5, 1, 2),
                # TODO: why did rianne separate these last two layers from the rest? (+ a sigmoid)
                nn.Conv2d(32, 256, 5, 1, 2),
                nn.Conv2d(256, self.C * self.num_classes, 1, 1, 0),
            )
        else:
            raise ValueError(f"Data distribution type not implemented: {self.data_distribution}.")

    def forward(self, z, x_in=None):
        """z -> p_x_z params"""
        assert z.dim() == 3, f"we assume z to be 3D, current shape {z.shape}"

        (S, B, D) = z.shape

        # [S, B, D] -> [S*B, D, 1, 1]
        z = z.reshape(-1, D, 1, 1)

        # Multinomial: [S*B, C*256, image_w_h, image_w_h] (logits, pre-softmax)
        # Bernoulli: [S*B, C, image_w_h, image_w_h] (logits, pre-sigmoid)
        p_x_z_params = self.decoder_gated_cnn_block(z)

        if self.data_distribution == "multinomial":
            # [S*B, C*256, image_w_h, image_w_h] -> [S, B, C, 256, image_w_h, image_w_h] (6 dim)
            p_x_z_params = p_x_z_params.reshape(S, B, self.C, self.num_classes, self.image_w_h, self.image_w_h)
            # [S, B, C, num_classes, W, H] -> [S, B, C, W, H, num_classes]
            p_x_z_params = p_x_z_params.permute(0, 1, 2, 4, 5, 3)
        else:
            # [S*B, C, image_w_h, image_w_h] -> [S, B, C, image_w_h, image_w_h] (5 dim)
            p_x_z_params = p_x_z_params.reshape(S, B, self.C, self.image_w_h, self.image_w_h)

        return p_x_z_params


class DecoderMLPBlock(nn.Module):
    """
        Maps z -> p_x_z_params.
            In case of Bernoulli these are logits [B, C, W, H]
            In case of Multinomial these are logits [B, C, num_classes=256, W, H]

    """

    def __init__(self, args):
        super(DecoderMLPBlock, self).__init__()

        self.data_distribution = args.data_distribution

        if self.data_distribution == "bernoulli":
            self.num_classes = 1
        elif self.data_distribution == "multinomial":
            self.num_classes = 256
        else:
            raise NotImplementedError
        self.image_w_h = args.image_w_h
        self.D = args.latent_dim
        self.C = args.n_channels

        # z_in -> 500 --> 500 -> image_w*image_h*C
        self.decoder_mlp_block = nn.Sequential(
            nn.Linear(in_features=self.D, out_features=500),
            nn.ReLU(),
            nn.Linear(in_features=500, out_features=500),
            nn.ReLU(),
            nn.Linear(500, self.image_w_h * self.image_w_h * self.C * self.num_classes)
        )

    def forward(self, z, x_in=None):
        assert z.dim() == 3, f"we assume z to be 3D, current shape {z.shape}"

        (S, B, D) = z.shape

        # [S, B, D] -> [S*B, D]
        z = z.reshape(-1, D)

        # Reshape to bernoulli [S*B, image_w*image_h*C] or multinomial [S*B, C*num_classes*W*H]
        pred_flat = self.decoder_mlp_block(z)

        # Bernoulli: [S*B, image_w*image_h*C] -> [S, B, C, image_w_h, image_w_h] (5 dim)
        if self.data_distribution == "bernoulli":
            assert pred_flat.shape == (S * B, self.C * self.image_w_h * self.image_w_h), \
                "Expected the predictions to be of shape [B, C*W*H]"
            p_x_z_params = pred_flat.reshape(S, B, self.C, self.image_w_h, self.image_w_h)

        # Multinomial: [S*B, C*num_classes*W*H] -> [S, B, C, num_classes, W, H] (6 dim)
        else:
            assert pred_flat.shape == (S * B, self.C * self.image_w_h * self.image_w_h * self.num_classes), \
                "Expected the predictions to be of shape [B, C*num_classes*W*H]"
            p_x_z_params = pred_flat.reshape(S, B, self.C, self.num_classes, self.image_w_h, self.image_w_h)
            # [S, B, C, num_classes, W, H] -> [S, B, C, W, H, num_classes]
            p_x_z_params = p_x_z_params.permute(0, 1, 2, 4, 5, 3)

        return p_x_z_params


class DecoderPixelCNNppBlock(nn.Module):
    def __init__(self, args):
        super(DecoderPixelCNNppBlock, self).__init__()

        self.data_distribution = args.data_distribution

        if self.data_distribution == "bernoulli":
            self.num_classes = 1
        elif self.data_distribution == "multinomial":
            self.num_classes = 256
        else:
            raise NotImplementedError

        self.image_w_h = args.image_w_h
        self.D = args.latent_dim
        self.C = args.n_channels

        self.pixel_cnn = PixelCNN(nr_resnet=2, nr_filters=64, nr_logistic_mix=1, num_classes=self.num_classes,
                                  resnet_nonlinearity='concat_elu', input_channels=self.C,
                                  dim_in=28, conditional=True, h_dim=self.D)

        self.sample = None

    def forward(self, z, x_in=None):
        assert z.dim() == 3, f"we assume z to be 3D, current shape {z.shape}"
        # x_in: [B, C, W, H]
        # z_post: [B, D]

        (S, B, D) = z.shape

        # Teacher-forcing forward
        if x_in is not None:
            (B, C, W, H) = x_in.shape

            z_2d = z.reshape(S * B, D)
            # in case we have a multi sample forward, we need to repeat X to match shape with z
            x_exp_2d = x_in.repeat(S, 1, 1, 1)

            # both z_2d and x_exp_2d have a sample dimension integrated in the first "batch" dimension = S*B

            # [B*S, num_classes, W, H]
            p_x_z_params = self.pixel_cnn(x_exp_2d, sample=False, h=z_2d)

            if self.data_distribution == "multinomial":
                # [S*B, C*256, image_w_h, image_w_h] -> [S, B, C, 256, image_w_h, image_w_h] (6 dim)
                p_x_z_params = p_x_z_params.reshape(S, B, self.C, self.num_classes, self.image_w_h, self.image_w_h)
                # [S, B, C, num_classes, W, H] -> [S, B, C, W, H, num_classes]
                p_x_z_params = p_x_z_params.permute(0, 1, 2, 4, 5, 3)
            else:
                # [S*B, C, image_w_h, image_w_h] -> [S, B, C, image_w_h, image_w_h] (5 dim)
                p_x_z_params = p_x_z_params.reshape(S, B, self.C, self.image_w_h, self.image_w_h)

        # SAMPLE
        else:
            p_x_z_params = self.auto_regressive_forward(z)

        return p_x_z_params

    def auto_regressive_forward(self, z):
        assert self.data_distribution == "bernoulli", "PixelCNN++ is only implemented for Bernoulli case now."

        (S, B, D) = z.shape
        z_2d = z.reshape(S * B, D)

        p_x_z_params = torch.zeros(S*B, self.C, self.image_w_h, self.image_w_h).to(z.device)
        x_pred = torch.zeros(S*B, self.C, self.image_w_h, self.image_w_h).to(z.device)

        for i in range(self.image_w_h):
            for j in range(self.image_w_h):
                print(f"{i:2d},{j:2d}", end='\r')

                # pixel_cnn returns bernoulli parameters [S*B, C, W, H]
                p_x_z_params_ij = self.pixel_cnn(x_pred, sample=True, h=z_2d)
                x_ij = td.Bernoulli(logits=p_x_z_params_ij).sample()

                x_pred[:, :, i, j] = x_ij[:, :, i, j]
                p_x_z_params[:, :, i, j] = p_x_z_params_ij[:, :, i, j]

        # [S*B, C, image_w_h, image_w_h] -> [S, B, C, image_w_h, image_w_h] (5 dim)
        p_x_z_params = p_x_z_params.reshape(S, B, self.C, self.image_w_h, self.image_w_h)
        x_pred = x_pred.reshape(S, B, self.C, self.image_w_h, self.image_w_h)
        self.sample = x_pred

        return p_x_z_params


class DecoderStrongDistilRoberta(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.L = args.max_seq_len
        self.V = args.vocab_size
        self.D = args.latent_dim

        checkpoint_name = "distilroberta-base"
        self.config = RobertaConfig.from_pretrained(checkpoint_name)

        # make some important settings explicit
        self.config.is_decoder = True  # adds LM head and masks auto-regressively
        self.config.add_cross_attention = False  # not a classic seq2seq model
        self.config.max_length = self.L

        self.roberta_model = VaeStrongDecoderRobertaForCausalLM(config=self.config).from_pretrained(
            pretrained_model_name_or_path=checkpoint_name,
            config=self.config)

        self.latent_to_memory_projection = nn.Linear(self.D, self.config.hidden_size * self.config.num_hidden_layers)
        self.latent_to_memory_projection.weight.data.normal_(mean=0.0, std=self.config.initializer_range)

    def forward(self, z, x_in=None):
        assert x_in is None, "we assume x_in to be None always, otherwise auto_regressive_forward should be called"
        assert z.dim() == 3, f"we assume z to be 3D, current shape {z.shape}"
        # x_in: [B, L, H]
        # z_post: [B, D]

        (S, B, D) = z.shape
        # Both z_2d and x_exp_2d need to have a sample dimension integrated in the first "batch" dimension = S*B
        z_2d = z.reshape(S * B, D)

        z_proj = self.latent_to_memory_projection(z_2d)
        # Makes tuple of equally sized tensors of (batch x 1 x hidden_size)
        z_proj = torch.split(z_proj.unsqueeze(1), self.config.hidden_size, dim=2)

        input_ids, attention_mask = x_in
        # In case we have a multi sample forward, we need to repeat X to match shape with z
        input_ids_2d_exp = input_ids.repeat(S, 1, 1).reshape(S*B, -1)
        attention_mask_2d_exp = attention_mask.repeat(S, 1, 1).reshape(S*B, -1)

        out = self.roberta_model(z=z_proj, input_ids=input_ids_2d_exp,
                                 attention_mask=attention_mask_2d_exp, return_dict=True)

        p_x_z_params = out.logits
        p_x_z_params = p_x_z_params.reshape(S, B, self.L, self.V)
        # cut of prediction for last token
        p_x_z_params = p_x_z_params[:, :, :-1, :]

        return p_x_z_params

    def auto_regressive_forward(self, z, Sx=1):
        Sz, B, D = z.shape
        DEVICE = z.device

        # [Sx*Sz*B, D], with the dimensions being of order Sx, S
        z_2d = z.reshape(Sz*B, D).repeat(Sx, 1)
        z_proj = self.latent_to_memory_projection(z_2d)
        # Makes tuple of equally sized tensors of (batch x 1 x hidden_size)
        z_proj = torch.split(z_proj.unsqueeze(1), self.config.hidden_size, dim=2)

        # Add <s>
        bos_token_id = 0

        # [Sx*Sz*B, 1]
        input_ids = torch.tensor([[bos_token_id] for _ in range(Sz*Sx*B)])
        input_ids = input_ids.to(DEVICE)
        generated_so_far = []

        # Init with nothing
        past_key_values = None

        # Sequence length includes start and end token
        for i in range(self.L - 1):
            decoder_outs = self.roberta_model(z=z_proj,
                                              input_ids=input_ids,
                                              attention_mask=None,
                                              past_key_values=past_key_values)

            past_key_values = decoder_outs.past_key_values
            sample = td.Categorical(logits=decoder_outs.logits[:, -1, :]).sample().unsqueeze(1)
            input_ids = sample

            generated_so_far.append(sample)

        generated_so_far = torch.cat(generated_so_far, dim=1)
        generated_so_far = generated_so_far.reshape(Sx, Sz, B, -1)

        p_x_z = AutoRegressiveRobertaStrongDecoder(strong_decoder=self, z=z, sample=generated_so_far)

        return p_x_z


class DecoderWeakDistilRoberta(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.D = args.latent_dim
        self.L = args.max_seq_len
        self.V = args.vocab_size

        checkpoint_name = "distilroberta-base"
        self.config = RobertaConfig.from_pretrained(checkpoint_name)

        # make some important settings explicit
        self.config.is_decoder = False  # no auto-regressive masking
        self.config.add_cross_attention = False  # not a classic seq2seq model
        self.config.max_length = self.L

        # use the RobertaForMaskedLM model as it is a bidirectional model with no shift between input and ouput
        self.roberta_model = RobertaForMaskedLM(config=self.config).from_pretrained(
            pretrained_model_name_or_path=checkpoint_name,
            config=self.config)

        self.latent_to_h0_proj = nn.Linear(self.D, self.config.hidden_size)
        # you make max. 63 preds
        self.latent_to_length = nn.Linear(self.D, self.L)

    def forward(self, z, x_in=None):
        assert z.dim() == 3, f"we assume z to be 3D, current shape {z.shape}"
        # x_in is ignored!
        # z_post: [B, D]

        (S, B, D) = z.shape
        # Both z_2d and x_exp_2d need to have a sample dimension integrated in the first "batch" dimension = S*B
        z_2d = z.reshape(S * B, D)

        # [B, D] -> [B, H]
        h0 = self.latent_to_h0_proj(z_2d)
        input_embeds = torch.stack([h0 for _ in range(self.L)], dim=1)

        # [S, B, max_len - 1]
        length_logits = self.latent_to_length(z_2d).reshape(S, B, -1)

        out = self.roberta_model(input_ids=None, attention_mask=None, inputs_embeds=input_embeds, return_dict=True)

        p_x_z_params = out.logits
        p_x_z_params = p_x_z_params.reshape(S, B, self.L, self.V)

        # cut of prediction for last token
        p_x_z_params = p_x_z_params[:, :, :-1, :]

        return p_x_z_params, length_logits


