import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td


class AutoRegressiveDistribution(nn.Module):
    def __init__(self, context, made, dist_type="gaussian", samples=None, params=None, encoder=True, X_shape=(1, 28, 28)):
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
        self.X_shape = X_shape

        # Gaussian: sample z [S, B, D], params (mu, scale): ([S, B, D], [S, B, D])
        # Bernoulli sample x [B, X_dim], params (logits): [S, B, X_dim]
        self.samples = samples
        self.params = params

    def log_prob(self, x_z, mean_reduce_sample_dim=False):
        # GAUSSIAN ENCODER CASE
        # If encoder, we expect the context X to be 2D [B, Hidden_dim]
        # and we expect the samples Z to be 3D [S, B, D]

        if self.encoder:
            input_z = x_z
            assert input_z.dim() == 3, f"We expect Z to be 3D (S, B, D), current shape {input_z.shape}"

            # (B, H_dim) = self.context.shape
            (S, B, D) = input_z.shape

            if self.params is not None:
                # [S, B, D]
                mean, scale = self.params
            else:
                # [B, D]
                context_x = self.context

                # [S*B, D]
                input_z_2d = input_z.reshape(-1, D)
                context_x_2d = context_x.repeat(S, 1)

                assert len(input_z_2d) == len(context_x_2d), "the shapes of x and z should match along dim = 0"

                # [S*B, D*2]
                params = self.made(input_z_2d, context=context_x_2d)
                params_split = torch.chunk(params, 2, dim=1)
                mean, pre_scale = params_split[0], params_split[1]
                scale = F.softplus(pre_scale)

                # Re-introduce the sample dimension
                mean, scale = mean.reshape(S, B, D), scale.reshape(S, B, D)

            # [S, B]
            log_prob_z_x = td.Independent(td.Normal(loc=mean, scale=scale), 1).log_prob(input_z)

            # [S, B] -> [B,]
            if mean_reduce_sample_dim:
                log_prob_z_x = log_prob_z_x.mean(dim=0)

            return log_prob_z_x

        # BERNOULLI DECODER CASE
        # If decoder, we expect the context Z to be 3D [S, B, D]
        # and we expect the samples X to be [B, C, W, H]
        else:
            input_x = x_z  # [B, X_dim]

            assert input_x.dim() == 4, f"We expect X to be 4D (B, C, W, H), current shape {input_x.shape}"
            assert self.context.dim() == 3, f"We expect context Z to be 3D (S, B, D), current shape {self.context.shape}"

            (B, C, W, H) = input_x.shape
            (S, Bz, D) = self.context.shape

            assert B == Bz, "Here we assume multiple samples z per data point x, so the B dimension should correspond."

            # flatten x [B, C, W, H] -> [B, C*W*H]
            input_x_flat = input_x.reshape(B, -1)

            if self.params is not None:
                bern_logits = self.params
            else:
                # If input X is 2D, we assume the samples to correspond to multiple samples per data point
                # [S, B, D]
                context_z = self.context
                context_z_2d = context_z.reshape(-1, D)
                input_x_2d = input_x_flat.repeat(S, 1)

                assert len(input_x_2d) == len(context_z_2d), "the shapes of x and z should match along dim = 0"

                # [S*B, C*W*H]
                bern_logits = self.made(input_x_2d, context=context_z_2d)

                # [S*B, C*W*H] -> [S, B, C, W, H]
                bern_logits = bern_logits.reshape(S, B, C, W, H)

                self.params = bern_logits

            # [S, B, C, W, H]
            assert bern_logits.shape == (S, B, C, W, H), \
                f"bernoulli logits must be of shape  (S, B, C, W, H), currently pf shape {bern_logits.shape}"

            # [S, B]
            log_prob_x_z = td.Independent(td.Bernoulli(logits=bern_logits), 3).log_prob(input_x)
            assert log_prob_x_z.shape == (S, B), \
                f"log_prob_x_z should be of shape (S, B), currently of shape {log_prob_x_z.shape}"

            # [S, B] -> [B,]
            if mean_reduce_sample_dim:
                log_prob_x_z = log_prob_x_z.mean(dim=0)

            return log_prob_x_z

    def rsample(self, sample_shape=(1,)):
        assert self.encoder and self.dist_type == "gaussian", "rsample() can only be used for a Gaussian encoder MADE"

        S = sample_shape[0]
        # Context x = [B, 256]
        B = self.context.shape[0]
        D = self.x_z_dim

        z_samples, mus, scales = [], [], []

        for s in range(S):

            # All [B, D]
            z_sample_s = torch.zeros((B, D), device=self.context.device)
            mu_s = torch.zeros((B, D), device=self.context.device)
            scale_s = torch.zeros((B, D), device=self.context.device)

            for d in range(D):
                # [B, D*2]
                mean_scale = self.made(z_sample_s, context=self.context)

                # 2 x [B, D]
                mu, scale = torch.chunk(mean_scale, 2, dim=-1)
                scale = F.softplus(scale)

                # prepare for index_put operation
                indices = torch.LongTensor([[i, d] for i in range(B)])
                indices = tuple(indices.t())

                # [B]
                mu_d = mu[:, d]
                scale_d = scale[:, d]

                # [B, D]
                mu_s = mu_s.index_put(indices, mu_d)
                scale_s = scale_s.index_put(indices, scale_d)

                # [B]
                z_d = td.Normal(loc=mu_d, scale=scale_d).rsample()

                # [B, D]
                z_sample_s = z_sample_s.index_put(indices, z_d)

            z_samples.append(z_sample_s)
            mus.append(mu_s)
            scales.append(scale_s)

        # [S, B, D]
        mus = torch.stack(mus, dim=0)
        scales = torch.stack(scales, dim=0)
        z_samples = torch.stack(z_samples, dim=0)

        # All [S, B, D]
        self.samples = z_samples
        self.params = (mus, scales)

        return z_samples

    def sample(self, sample_shape=(1,)):
        # context z is (S, B, D)
        # we are going to sample x, at least 1 per S*B

        assert len(self.X_shape) == 3, f"we expected X_shape to be (C, W, H), currently given {self.X_shape}"
        assert not self.encoder and self.dist_type != "gaussian", "sample() can only be used by non Gaussian decoders"

        # [S, B, D]
        C, W, H = self.X_shape
        x_dim_flat = int(C * W * H)
        context_z = self.context
        (S, B, D) = context_z.shape
        Sx = sample_shape[0]

        # [S, B, D] -> [Sx, S, B, D] -> [Sx*B*S, D]
        context_z_2d = context_z.unsqueeze(2).repeat(Sx, 1, 1, 1).reshape(-1, D)

        # [Sx, B, S, X_dim] -> [Sx*B*D, X_dim]
        x_sample = torch.zeros((Sx, S, B, x_dim_flat), device=self.context.device)
        x_sample_2d = x_sample.reshape(-1, x_dim_flat)

        for d in range(x_dim_flat):
            # x [Sx*B*S, X_dim] + context z [Sx*B*S, D] -> logits [Sx*B*S, X_dim]
            logits = self.made(x_sample_2d, context=context_z_2d)

            # [Sx*B*S, X_dim]
            x_sample_dim = td.Bernoulli(logits=logits).sample()
            x_sample_2d[:, d] = x_sample_dim[:, d]

        # [Sx, S, B, X_dim_flat]
        x_sample_2d = x_sample_2d.reshape(Sx, S, B, x_dim_flat)

        # [Sx*S*B, X_dim_flat] -> [Sx, S, B, C, W, H]
        x_samples = x_sample_2d.reshape(Sx, S, B, C, W, H)

        return x_samples


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
        z = p.samples

    log_p_z = p.log_prob(z)
    log_q_z = q.log_prob(z)

    kl = log_p_z - log_q_z

    # [S, B] -> [B]
    if kl.dim() == 2:
        kl = kl.mean(dim=0)

    return kl
