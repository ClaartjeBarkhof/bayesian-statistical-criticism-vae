import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td


class AutoRegressiveDistribution(nn.Module):
    def __init__(self, context, made, dist_type="gaussian", sample=None, params=None, encoder=True):
        super(AutoRegressiveDistribution, self).__init__()

        assert dist_type in ["gaussian", "bernoulli"], \
            "AutoRegressiveDistribution only supports gaussian and bernoulli for now"

        # Gaussian encoder: context x [B, Hidden_dim]
        # Bernoulli decoder: context z [S, B, D]
        self.context = context
        print("context.shape")

        self.dist_type = dist_type  # gaussian or bernoulli
        self.encoder = encoder  # encoder or decoder
        assert (self.encoder and dist_type == "gaussian") or (not self.encoder and dist_type == "bernoulli"), \
            "Either Gaussian encoder or Bernoulli decoder, multinomial decoder is not implemented yet."

        self.made = made
        # Sample dim: latent_dim (D) or image dim (C x H x W)
        self.x_z_dim = self.made.nin

        # Gaussian: sample z [S, B, D], params (mu, scale): ([S, B, D], [S, B, D])
        # Bernoulli sample x [B, X_dim], params (logits): [S, B, X_dim]
        self.sample = sample
        self.params = params

    def log_prob(self, x_z, mean_reduce_sample_dim=False):
        print("-> in log_prob")
        # GAUSSIAN ENCODER CASE
        # If encoder, we expect the context X to be 2D [B, Hidden_dim]
        # and we expect the samples Z to be 3D [S, B, D]

        if self.encoder:
            # (B, H_dim) = self.context.shape
            (S, B, D) = x_z.shape
            input_z = x_z

            print("input_z.shape", input_z.shape)

            if self.params is not None:
                # [B, D]
                mean, scale = self.params
                print("mean.shape, scale.shape", mean.shape, scale.shape)
            else:
                context_x = self.context

                # [S*B, H_dim]
                context_x = context_x.repeat(S, 1, 1).reshape(S * B, -1)

                # [S*B, D*2]
                params = self.made(input_z, context=context_x)
                params_split = torch.split(params, 2, dim=1)
                mean, pre_scale = params_split[0], params_split[1]
                scale = F.softplus(pre_scale)

                print("mean.shape, scale.shape", mean.shape, scale.shape)

            # [S, B]
            log_prob_z_x = td.Independent(td.Normal(loc=mean, scale=scale), 1).log_prob(input_z)
            print("log_prob_z_x.shape", log_prob_z_x.shape)

            # [S, B] -> [B,]
            if mean_reduce_sample_dim:
                log_prob_z_x = log_prob_z_x.mean(dim=0)

            return log_prob_z_x

        # BERNOULLI DECODER CASE
        # If decoder, we expect the context Z to be 3D [S, B, D]
        # and we expect the samples X to be 2D or 3D (not implemented yet)
        else:
            (S, B, D) = self.context.shape
            input_x = x_z

            if self.params is not None:
                bern_logits = self.params.reshape(S*B, -1)

            else:
                # If input X is 3D as well, we assume one sample per data point
                assert input_x.dim() == 2, "did not implement the 3D input case yet..."

                # If input X is 2D, we assume the samples to correspond to multiple samples per data point
                # Merge the sample dimension into the batch dimension:
                # [S, B, D] -> [S*B, D]
                context_z = self.context.reshape(-1, D)

                # Copy the context to match the sample dimension
                # [B, X_dim] -> [S*B, X_dim]
                # Eg: [x_1, x_2, x_1, x_2, x_1, x_2] for 3 samples of batch of 2 elements
                input_x = x_z.repeat(S, 1, 1).reshape(S * B, -1)

                # [S*B, X_dim]
                bern_logits = self.made(input_x, context=context_z)
                self.params = bern_logits.reshape(S, B, -1)

            # [S*B] reinterpreted_batch_ndims=1 because we are working with flattened input
            log_prob_x_z = td.Independent(td.Bernoulli(logits=bern_logits), 1).log_prob(input_x)

            # [S, B]
            log_prob_x_z = log_prob_x_z.reshape(S, B)

            # [S, B] -> [B,]
            if mean_reduce_sample_dim:
                log_prob_x_z = log_prob_x_z.mean(dim=0)

            return log_prob_x_z

    def rsample(self, sample_shape=(1,)):
        # Context X: [B, D]
        # Sample Z: [S, B, D]
        # Params (mean, scale): 2 x [S, B, D]

        assert len(sample_shape) == 1, "only accepting 1 dimensional shape for sample_shape (S,)"
        assert self.dist_type == "gaussian", "This functionality is only implemented for the Gaussian case."

        B = self.context.shape[0]
        D = self.x_z_dim
        S = sample_shape[0]

        print("S B D", S, B, D)

        z_sample = torch.zeros((S, B, D), device=self.context.device)
        print("z_sample.shape")

        mu_inferred, scale_inferred = [], []

        for i in range(self.x_z_dim):
            # [S*B, D*2]
            z_reshape = z_sample.reshape(S*B, D)
            context_x = self.context.repeat(S, 1, 1).reshape(S * B, -1)
            print("z_reshape.shape, self.context.shape, context_x.shape", z_reshape.shape, context_x.shape)
            mus_prescales = self.made(z_reshape, context=context_x)
            # mus_prescales = torch.randn((z_reshape.shape[0], int(z_reshape.shape[1]*2)))
            print("mu_prescales.shape", mus_prescales.shape)
            mus_prescales = mus_prescales.reshape(S, B, D*2)
            print("mu_prescales.shape (after reshape)", mus_prescales.shape)

            # split in 2 x [S, B, D]
            mus_prescales = torch.split(mus_prescales, 2, dim=-1)
            mus, prescales = mus_prescales[0], mus_prescales[1]

            print("mus, prescales shapes", mus.shape, prescales.shape)

            # i-th dimension [S, B, i] = shape (S, B)
            mu_i = mus[:, :, i]
            scale_i = F.softplus(prescales[:, :, i])

            print("mu_i.shape", mu_i.shape)
            print("scale_i.shape", scale_i.shape)

            mu_inferred.append(mu_i)
            scale_inferred.append(scale_i)

            # [S, B]
            # don't pass sample shape because it is already implicit through set-up of shapes (S, B, D)
            assert mu_i.shape == scale_i.shape == (S, B), "expecting mu_i and scale_i to be (S, B) - for one dimension"

            z_i = td.Normal(loc=mu_i, scale=scale_i).rsample()

            z_sample[:, :, i] = z_i

        # [S, B, D]
        mu_inferred = torch.stack(mu_inferred, dim=-1)
        scale_inferred = torch.stack(scale_inferred, dim=-1)

        # [S, B, D], [S, B, D], [S, B, D]
        self.sample = z_sample
        self.params = (mu_inferred, scale_inferred)

        print("z_sample.shape", z_sample.shape)
        print("mu_inferred.shape", mu_inferred.shape)
        print("scale_inderred.shape", scale_inferred.shape)

        return z_sample

    def sample(self, C, W, sample_shape=None):
        raise NotImplementedError
        # if self.context.shape[0] > 1:
        #     print("Not implemented multi sample decoder for multi-sample Z context")
        #     raise NotImplementedError
        #
        # context_z = self.context.squeeze(0)
        #
        # if sample_shape is not None:
        #     assert len(sample_shape) == 1, "only accepting 1 dimensional shape for sample_shape (S,)"
        #
        # assert self.dist_type == "bernoulli", "This functionality has only been implemented for the Bernoulli case so far."
        #
        # B = self.context.shape[0]
        #
        # if sample_shape is not None:
        #     x_sample = torch.zeros((sample_shape[0], B, self.x_z_dim))
        # else:
        #     x_sample = torch.zeros((B, self.x_z_dim))
        #
        # logits = torch.zeros((B, self.x_z_dim))
        #
        # for i in range(self.x_z_dim):
        #     logits = self.made(logits, context=context_z)
        #     logits_i = logits[:, i]
        #
        #     x_sample[:, :, i] = td.Bernoulli(logits=logits_i).sample()
        #
        # # logits_inferred = torch.stack(logits_inferred, dim=1)
        # # TODO: should this change the state?
        #
        # x_sample = x_sample.reshape(B, C, W, H)  # TOD: check this
        #
        # return x_sample


@td.register_kl(AutoRegressiveDistribution, AutoRegressiveDistribution)
def _kl(p, q):
    print("XXXX KL 1")

    assert p.dist_type == q.dist_type == "gaussian", "Both AutoRegressiveDistributions should be Gaussian."

    if p.params is None:
        _ = p.rsample()
    p_mean, p_scale = p.params[0], p.params[1]

    if q.params is None:
        _ = q.rsample()

    # [S, B, D]
    q_mean, q_scale = q.params[0], q.params[1]

    # [S, B]
    kl = td.kl_divergence(td.Normal(loc=p_mean, scale=p_scale), td.Normal(loc=q_mean, scale=q_scale)).sum(-1)

    return kl


@td.register_kl(AutoRegressiveDistribution, td.Normal)
def _kl(p, q):
    print("XXXX KL 2")
    assert p.dist_type == "gaussian", "The AutoRegressiveDistributions should be Gaussian."

    if p.params is None:
        _ = p.rsample()

    p_mean, p_scale = p.params[0], p.params[1]

    kl = td.kl_divergence(td.Normal(loc=p_mean, scale=p_scale), q).sum(-1)

    return kl


@td.register_kl(AutoRegressiveDistribution, td.Distribution)
def _kl(p, q):
    print("XXXX KL 3")
    if p.params is None:
        print("r_sample")
        z = p.rsample()
    else:
        print("get sample")
        z = p.sample

    print("z", z.shape)
    print("p", type(p))

    log_p_z = p.log_prob(z)
    log_q_z = q.log_prob(z)

    kl = log_p_z - log_q_z

    return kl
