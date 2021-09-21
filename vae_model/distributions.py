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
        # GAUSSIAN ENCODER CASE
        # If encoder, we expect the context X to be 2D [B, Hidden_dim]
        # and we expect the samples Z to be 3D [S, B, D]

        if self.encoder:
            # (B, H_dim) = self.context.shape
            (S, B, D) = x_z.shape
            input_z = x_z

            if self.params is not None:
                # [S, B, D]
                mean, scale = self.params
            else:
                # [B, D]
                context_x = self.context

                # Loop for sample dimension
                means, scales = [], []
                for s in range(S):
                    # [B, D*2]
                    params = self.made(input_z[s, :, :], context=context_x)
                    params_split = torch.chunk(params, 2, dim=1)
                    mean, pre_scale = params_split[0], params_split[1]
                    scale = F.softplus(pre_scale)

                    means.append(mean)
                    scales.append(scale)

                mean, scale = torch.stack(means, dim=0), torch.stack(scales, dim=0)

            # [S, B]
            log_prob_z_x = td.Independent(td.Normal(loc=mean, scale=scale), 1).log_prob(input_z)

            # [S, B] -> [B,]
            if mean_reduce_sample_dim:
                log_prob_z_x = log_prob_z_x.mean(dim=0)

            return log_prob_z_x

        # BERNOULLI DECODER CASE
        # If decoder, we expect the context Z to be 3D [S, B, D]
        # and we expect the samples X to be 2D or 3D (not implemented yet)
        else:
            (S, B, D) = self.context.shape
            input_x = x_z  # [B, X_dim]

            if self.params is not None:
                bern_logits = self.params

            else:
                # If input X is 2D, we assume the samples to correspond to multiple samples per data point
                # [S, B, D]
                context_z = self.context

                bern_logits = []
                for s in range(S):

                    # [B, X_dim]
                    bern_logits_s = self.made(input_x, context=context_z[s, :, :])
                    bern_logits.append(bern_logits_s)

                # [S, B, X_dim]
                bern_logits = torch.stack(bern_logits, dim=0)
                self.params = bern_logits

            # [S, B] reinterpreted_batch_ndims=1 because we are working with flattened input
            log_prob_x_z = td.Independent(td.Bernoulli(logits=bern_logits), 1).log_prob(input_x)

            # [S, B] -> [B,]
            if mean_reduce_sample_dim:
                log_prob_x_z = log_prob_x_z.mean(dim=0)

            return log_prob_x_z

    def rsample(self, sample_shape=(1,)):
        S = sample_shape[0]
        B = self.context.shape[0]
        D = self.x_z_dim

        z_samples, mus, scales = [], [], []

        for s in range(S):

            # [B, D]
            z_sample_s = torch.zeros((B, D), device=self.context.device)
            mu_s, scale_s = torch.zeros((B, D)), torch.zeros((B, D))

            for d in range(D):
                # [B, D*2]
                mean_scale = self.made(z_sample_s, context=self.context)

                # 2 x [B, D]
                mu, scale = torch.chunk(mean_scale, 2, dim=-1)
                scale = F.softplus(scale)

                # [B]
                mu_d = mu[:, d]

                scale_d = scale[:, d]
                mu_s[:, d] = mu_d
                scale_s[:, d] = scale_d

                z_d = td.Normal(loc=mu_d, scale=scale_d).rsample()
                z_sample_s[:, d] = z_d

            z_samples.append(z_sample_s)
            mus.append(mu_s)
            scales.append(scale_s)

        # [S, B, D]
        mus = torch.stack(mus, dim=0)
        scales = torch.stack(scales, dim=0)
        z_samples = torch.stack(z_samples, dim=0)

        # [S, B, D], [S, B, D], [S, B, D]
        self.sample = z_samples
        self.params = (mus, scales)

        return z_samples

    # def rsample(self, sample_shape=(1,)):
    #     # Context X: [B, D]
    #     # Sample Z: [S, B, D]
    #     # Params (mean, scale): 2 x [S, B, D]
    #
    #     assert len(sample_shape) == 1, "only accepting 1 dimensional shape for sample_shape (S,)"
    #     assert self.dist_type == "gaussian", "This functionality is only implemented for the Gaussian case."
    #
    #     B = self.context.shape[0]
    #     D = self.x_z_dim
    #     S = sample_shape[0]
    #
    #     z_sample = torch.zeros((S, B, D), device=self.context.device)
    #     mu_inferred, scale_inferred = [], []
    #
    #     for i in range(self.x_z_dim):
    #         # [S*B, D]
    #         z_reshape = z_sample.reshape(S*B, D)
    #         context_x = self.context.repeat(S, 1, 1).reshape(S * B, -1)
    #
    #         print("z_reshape, context_x", z_reshape.shape, context_x.shape)
    #         # 2 chunks [S*B, D]
    #         mu, prescale = torch.chunk(self.made(z_reshape, context=context_x), 2, dim=-1)
    #         mu, prescale = mu.reshape((S, B, D)), prescale.reshape((S, B, D))
    #         scale = F.softplus(prescale)
    #
    #         # i-th dimension [S, B, i] = shape (S, B)
    #         mu_i = mu[:, :, i]
    #         scale_i = scale[:, :, i]
    #
    #         print("mu_i.shape", mu_i.shape)
    #         print("scale_i.shape", scale_i.shape)
    #
    #         mu_inferred.append(mu_i)
    #         scale_inferred.append(scale_i)
    #
    #         # [S, B]
    #         # don't pass sample shape because it is already implicit through set-up of shapes (S, B, D)
    #         assert mu_i.shape == scale_i.shape == (S, B), "expecting mu_i and scale_i to be (S, B) - for one dimension"
    #
    #         z_i = td.Independent(td.Normal(loc=mu_i, scale=scale_i), 1).rsample()
    #         print("z_i.shape", z_i.shape)  # S, B
    #
    #         z_sample[:, :, i] = z_i
    #
    #     # [S, B, D]
    #     mu_inferred = torch.stack(mu_inferred, dim=-1)
    #     scale_inferred = torch.stack(scale_inferred, dim=-1)
    #
    #     # [S, B, D], [S, B, D], [S, B, D]
    #     self.sample = z_sample
    #     self.params = (mu_inferred, scale_inferred)
    #
    #     print("z_sample.shape", z_sample.shape)
    #     print("mu_inferred.shape", mu_inferred.shape)
    #     print("scale_inderred.shape", scale_inferred.shape)
    #
    #     return z_sample

    def sample(self, C, W, sample_shape=None):
        raise NotImplementedError
        # if self.context.shape[0] > 1:
        #     print("Not implemented multi sample decoder for multi-sample Z context")
        #     raise NotImplementedError
        # comm
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
    assert p.dist_type == q.dist_type == "gaussian", "Both AutoRegressiveDistributions should be Gaussian."

    if p.params is None:
        _ = p.rsample()
    p_mean, p_scale = p.params[0], p.params[1]

    if q.params is None:
        _ = q.rsample()

    # [S, B, D]
    q_mean, q_scale = q.params[0], q.params[1]

    # [B]
    kl = td.kl_divergence(td.Normal(loc=p_mean, scale=p_scale), td.Normal(loc=q_mean, scale=q_scale)).sum(-1).mean(0)

    return kl


@td.register_kl(AutoRegressiveDistribution, td.Normal)
def _kl(p, q):
    assert p.dist_type == "gaussian", "The AutoRegressiveDistributions should be Gaussian."

    if p.params is None:
        _ = p.rsample()

    p_mean, p_scale = p.params[0], p.params[1]

    # [B]
    kl = td.kl_divergence(td.Normal(loc=p_mean, scale=p_scale), q).sum(-1).mean(0)

    return kl


@td.register_kl(AutoRegressiveDistribution, td.Distribution)
def _kl(p, q):
    if p.params is None:
        z = p.rsample()
    else:
        z = p.sample

    log_p_z = p.log_prob(z)
    log_q_z = q.log_prob(z)

    kl = log_p_z - log_q_z

    # [S, B] -> [B]
    if kl.dim() == 2:
        kl = kl.mean(dim=0)

    return kl
