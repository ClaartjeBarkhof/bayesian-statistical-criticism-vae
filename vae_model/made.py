# Claartje: taken from Wilker's code https://github.com/probabll/dgm.pt/blob/master/probabll/dgm/conditioners/ffnn.py

"""
Implements Masked AutoEncoder for Density Estimation, by Germain et al. 2015
Re-implementation by Andrej Karpathy based on https://arxiv.org/abs/1502.03509
- wilkeraziz: I've only extended the implementation to also take a number of context units
    which can be conditioned on without restrictions
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td


# ------------------------------------------------------------------------------


class MaskedLinear(nn.Linear):
    """same as Linear except has a configurable mask on the weights"""

    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        self.register_buffer("mask", torch.ones(out_features, in_features))

    def set_mask(self, mask):
        self.mask.data.copy_(torch.from_numpy(mask.astype(np.uint8).T))

    def forward(self, input):
        return F.linear(input, self.mask * self.weight, self.bias)


class MADE(nn.Module):
    def __init__(
            self,
            nin,
            hidden_sizes,
            nout,
            num_masks=1,
            natural_ordering=False,
            context_size=0,
            hidden_activation=nn.ReLU(),
            gating=False,
            gating_mechanism=0,
    ):
        """
        nin: integer; number of inputs
        hidden sizes: a list of integers; number of units in hidden layers
        nout: integer; number of outputs, which usually collectively parameterize some kind of 1D distribution
              note: if nout is e.g. 2x larger than nin (perhaps the mean and std), then the first nin
              will be all the means and the second nin will be stds. i.e. output dimensions depend on the
              same input dimensions in "chunks" and should be carefully decoded downstream appropriately.
              the output of running the tests for this file makes this a bit more clear with examples.
        num_masks: can be used to train ensemble over orderings/connections
        natural_ordering: force natural ordering of dimensions, don't use random permutations
        context_size: allows to condition on an additional input (without restrictions)
        hidden_activation: nonlinearity for hidden layers
        """

        super().__init__()
        self.nin = nin
        self.nout = nout
        self.hidden_sizes = hidden_sizes
        self.hidden_activation = hidden_activation
        assert self.nout % self.nin == 0, "nout must be integer multiple of nin"
        self.context_size = context_size

        if context_size == 0:  # no need to deal with context (original code)
            self.net = []
            hs = [nin] + hidden_sizes + [nout]
            for h0, h1 in zip(hs, hs[1:]):
                self.net.extend(
                    [
                        MaskedLinear(h0, h1),
                        hidden_activation,
                    ]
                )
            self.net.pop()  # pop the last ReLU for the output layer
            self.net = nn.Sequential(*self.net)
            self.ctxt_net = None
        else:  # can condition on context (modified code)
            self.net = []
            hs = [nin] + hidden_sizes + [nout]
            self.net.extend([MaskedLinear(h0, h1) for h0, h1 in zip(hs, hs[1:])])
            self.net = nn.ModuleList(self.net)
            self.ctxt_net = [nn.Linear(context_size, h) for h in hidden_sizes]
            self.gating = gating
            self.gating_mechanism = gating_mechanism
            if gating:
                # First gating option is a learned gated context addition
                if self.gating_mechanism == 0:
                    for h_idx, h_size in enumerate(hidden_sizes):
                        self.register_parameter(f"gate_h_{h_idx}", torch.nn.Parameter(torch.randn(h_size)))
                # Gated Pixel CNN like gating
                # act( lin_1(z) + lin_2(x) ) * sigm( lin_3(x, z) )
                elif self.gating_mechanism == 1:
                    # same as self.net (masked linear layers)
                    self.gate_x_net = nn.ModuleList([MaskedLinear(h0, h1) for h0, h1 in zip(hs, hs[1:])])
                    self.gate_z_net = nn.ModuleList([nn.Linear(in_features=context_size, out_features=h) for h in hidden_sizes])

            self.ctxt_net = nn.ModuleList(self.ctxt_net)

        # seeds for orders/connectivities of the model ensemble
        self.natural_ordering = natural_ordering
        self.num_masks = num_masks
        self.seed = 0  # for cycling through num_masks orderings

        self.m = {}
        self.update_masks()  # builds the initial self.m connectivity
        # note, we could also precompute the masks and cache them, but this
        # could get memory expensive for large number of masks.

        # print("Print parameters of the MADE")
        # for n, p in self.named_parameters():
        #     print(n, p.shape)

    def update_masks(self):
        if self.m and self.num_masks == 1:
            return  # only a single seed, skip for efficiency
        L = len(self.hidden_sizes)

        # fetch the next seed and construct a random stream
        rng = np.random.RandomState(self.seed)
        self.seed = (self.seed + 1) % self.num_masks

        # sample the order of the inputs and the connectivity of all neurons
        self.m[-1] = (
            np.arange(self.nin) if self.natural_ordering else rng.permutation(self.nin)
        )
        for l in range(L):
            self.m[l] = rng.randint(
                self.m[l - 1].min(), self.nin - 1, size=self.hidden_sizes[l]
            )

        # construct the mask matrices
        masks = [self.m[l - 1][:, None] <= self.m[l][None, :] for l in range(L)]
        masks.append(self.m[L - 1][:, None] < self.m[-1][None, :])

        # handle the case where nout = nin * k, for integer k > 1
        if self.nout > self.nin:
            k = int(self.nout / self.nin)
            # replicate the mask across the other outputs
            masks[-1] = np.concatenate([masks[-1]] * k, axis=1)

        # set the masks in all MaskedLinear layers
        layers = [l for l in self.net.modules() if isinstance(l, MaskedLinear)]
        for l, m in zip(layers, masks):
            l.set_mask(m)

        if self.gating_mechanism == 1:
            layers = [l for l in self.gate_x_net.modules() if isinstance(l, MaskedLinear)]
            for l, m in zip(layers, masks):
                l.set_mask(m)

    def forward(self, x, context=None):
        if self.ctxt_net is None:
            return self.net(x)
        else:
            h = x
            for i, (t, c) in enumerate(zip(self.net, self.ctxt_net)):  # hidden layers
                if self.gating:
                    if self.gating_mechanism == 0:
                        gate_i = self.__getattr__(f"gate_h_{i}")
                        h = self.hidden_activation(t(h) + torch.mul(c(context), F.sigmoid(gate_i)))
                    else:
                        # act ( lin(z) + m_lin(x) ) * sigm ( lin(z) + m_lin(x) )
                        # act ( z_1 + x_1 ) + sigm ( z_2 + x_2 )

                        # print(f"made layer {i}")
                        # print("context", context.shape)
                        # print("h", h.shape)

                        z_1 = c(context)
                        z_2 = self.gate_z_net[i](context)
                        x_1 = t(h)
                        x_2 = self.gate_x_net[i](h)

                        # print("z_1", z_1.shape)
                        # print("z_2", z_2.shape)
                        # print("x_1.shape", x_1.shape)
                        # print("x_2.shape", x_2.shape)

                        h = self.hidden_activation(x_1 + z_1) * F.sigmoid(x_2 + z_2)
                else:
                    h = self.hidden_activation(t(h) + c(context))
            return self.net[-1](h)  # output layer


# ------------------------------------------------------------------------------

if __name__ == "__main__":
    from torch.autograd import Variable

    # run a quick and dirty test for the autoregressive property
    D = 10
    rng = np.random.RandomState(14)
    x = (rng.rand(1, D) > 0.5).astype(np.float32)

    configs = [
        (D, [], D, False),  # test various hidden sizes
        (D, [200], D, False),
        (D, [200, 220], D, False),
        (D, [200, 220, 230], D, False),
        (D, [200, 220], D, True),  # natural ordering test
        (D, [200, 220], 2 * D, True),  # test nout > nin
        (D, [200, 220], 3 * D, False),  # test nout > nin
    ]

    for nin, hiddens, nout, natural_ordering in configs:

        print(
            "checking nin %d, hiddens %s, nout %d, natural %s"
            % (nin, hiddens, nout, natural_ordering)
        )
        model = MADE(nin, hiddens, nout, natural_ordering=natural_ordering)

        # run backpropagation for each dimension to compute what other
        # dimensions it depends on.
        res = []
        for k in range(nout):
            xtr = Variable(torch.from_numpy(x), requires_grad=True)
            xtrhat = model(xtr)
            loss = xtrhat[0, k]
            loss.backward()

            depends = (xtr.grad[0].numpy() != 0).astype(np.uint8)
            depends_ix = list(np.where(depends)[0])
            isok = k % nin not in depends_ix

            res.append((len(depends_ix), k, depends_ix, isok))

        # pretty print the dependencies
        res.sort()
        for nl, k, ix, isok in res:
            print(
                "output %2d depends on inputs: %30s : %s"
                % (k, ix, "OK" if isok else "NOTOK")
            )
