import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td


class AutoRegressiveDistribution(nn.Module):
    def __init__(self, context, made, dist_type="gaussian", state=None):
        super(AutoRegressiveDistribution, self).__init__()

        assert dist_type in ["gaussian", "bernoulli"]

        self.context = context
        self.dist_type = dist_type
        self.made = made
        # latent_dim or image dim (C x H x W)
        self.x_z_dim = self.made.nin

        # Gaussian: state = (z, mu, scale)
        # Bernoulli state = (x, logits)
        self.state = state

    def log_prob(self, x_z):

        #print("log_prob x_z.shape", x_z.shape)

        if self.state is not None:
            params = self.state[1:]
        else:
            made_out_params = self.made(x_z, context=self.context)
            if self.dist_type == "gaussian":
                params_split = torch.split(made_out_params, 2, dim=1)
                mean, pre_scale = params_split[0], params_split[1]
                scale = F.softplus(pre_scale)
                params = (mean, scale)  # mean, scale for Gaussian
            else:
                params = (made_out_params,)  # logits for Bernoulli
                #print("params", params)

        self.state = (x_z,) + params
        #print("self.state", self.state)

        if self.dist_type == "gaussian":
            mean, scale = params
            # careful that the distribution itself is not independent, but at this point it is valid to use for log_prob
            # log_prob = log q_z_x
            log_prob = td.Independent(td.Normal(loc=mean, scale=scale), 1).log_prob(x_z)
        else:
            logits = params[0]
            #print("logits.shape", logits.shape)
            # log_prob = log p_x_z
            log_prob = td.Independent(td.Bernoulli(logits=logits), 1).log_prob(x_z)

        return log_prob


    def rsample(self):
        # TODO: multi rsample?
        assert self.dist_type == "gaussian", "This functionality is only implemented for the Gaussian case."

        B = self.context.shape[0]

        z_sample = torch.zeros((B, self.x_z_dim), device=self.context.device)
        mu_inferred, scale_inferred = [], []

        for i in range(self.x_z_dim):
            mus_prescales = self.made(z_sample, context=self.context)
            # split in 2 x [B, D]
            mus_prescales = torch.split(mus_prescales, 2, dim=1)
            mus, prescales = mus_prescales[0], mus_prescales[1]

            mu_i = mus[:, i]
            scale_i = F.softplus(prescales[:, i])

            mu_inferred.append(mu_i)
            scale_inferred.append(scale_i)

            z_sample[:, i] = td.Normal(loc=mu_i, scale=scale_i).rsample()

        mu_inferred = torch.stack(mu_inferred, dim=1)
        scale_inferred = torch.stack(scale_inferred, dim=1)

        self.state = (z_sample, mu_inferred, scale_inferred)

        return z_sample

    def sample(self, C, W, H):
        # TODO: multi sample?
        assert self.dist_type == "bernoulli", "This functionality has only been implemented for the Bernoulli case so far."

        B = self.context.shape[0]
        x_sample = torch.zeros((B, self.x_z_dim))

        # logits_inferred = []

        for i in range(self.x_z_dim):
            logits = self.made(x_sample, context=self.context)
            logits_i = logits[:, i]

            x_sample[:, i] = td.Bernoulli(logits=logits_i).sample()

        # logits_inferred = torch.stack(logits_inferred, dim=1)
        # TODO: should this change the state?

        x_sample = x_sample.reshape(B, C, W, H)  # TOD: check this

        return x_sample


@td.register_kl(AutoRegressiveDistribution, AutoRegressiveDistribution)
def _kl(p, q):
    assert p.dist_type == q.dist_type == "gaussian", "Both AutoRegressiveDistributions should be Gaussian."

    if p.state is None:
        _ = p.rsample()
    p_mean, p_scale = p.state[1], p.state[2]

    if q.state is None:
        _ = q.rsample()
    q_mean, q_scale = q.state[1], q.state[2]

    kl = td.kl_divergence(td.Normal(loc=p_mean, scale=p_scale), td.Normal(loc=q_mean, scale=q_scale)).sum(-1)

    return kl


@td.register_kl(AutoRegressiveDistribution, td.Normal)
def _kl(p, q):
    assert p.dist_type == "gaussian", "The AutoRegressiveDistributions should be Gaussian."

    if p.state is None:
        _ = p.rsample()
    p_mean, p_scale = p.state[1], p.state[2]

    kl = td.kl_divergence(td.Normal(loc=p_mean, scale=p_scale), q).sum(-1)

    return kl


@td.register_kl(AutoRegressiveDistribution, td.Distribution)
def _kl(p, q):
    if p.state is None:
        z = p.rsample()
    else:
        z = p.state[0]

    log_p_z = p.log_prob(z)
    log_q_z = q.log_prob(z)

    kl = log_p_z - log_q_z

    return kl
