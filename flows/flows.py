import flows.maf as maf
import flows.argmax as argmax
import torch
import torch.nn as nn
import torch.distributions as dist
import math


class FlowBlock(nn.Module):
    def __init__(self, input_dim, n_hidden, cond_nn_factory=None, noise_condn_nn_in_dim=0, aug_noise_dim=0):
        super().__init__()
        self.input_dim = input_dim
        self.n_hidden = n_hidden
        self.noise_cond_nn_in_dim = noise_condn_nn_in_dim
        self.aug_noise_dim = aug_noise_dim
        self.cond_nn = cond_nn_factory(self.n_hidden)
        if noise_condn_nn_in_dim:
            self.noise_cond_nn = nn.Sequential(
                nn.Linear(noise_condn_nn_in_dim, 100),
                nn.ReLU(),
                nn.Linear(100, 2*aug_noise_dim)
            )
        self.flow_modules = nn.ModuleList([
            maf.MADE(self.input_dim, num_hidden=self.n_hidden, act='relu'),
            maf.BatchNormFlow(self.input_dim),
            maf.Reverse(self.input_dim)
        ])

    def forward(self, inputs, cond_inputs=None, prev_inputs=None):
        if cond_inputs is not None:
            assert self.cond_nn is not None, 'Conditional NN not defined for conditional inputs!'

        if self.noise_cond_nn_in_dim:
            noise_dist = dist.normal.Normal(
                torch.zeros(inputs.shape[0], self.aug_noise_dim).cuda(),
                torch.ones(inputs.shape[0], self.aug_noise_dim).cuda(),
            )
            noise = noise_dist.sample()
            log_prob = noise_dist.log_prob(noise).sum(dim=1, keepdim=True)
            # noise_cond_nn_out = self.noise_cond_nn(cond_inputs).view(-1, 2, self.input_dim//2)
            # mean, log_std = noise_cond_nn_out.permute(1, 0, 2)
            noise_cond_nn_input = torch.cat(prev_inputs, dim=1)
            mean, log_std = self.noise_cond_nn(noise_cond_nn_input).chunk(2, dim=1)
            log_std = 2. * torch.tanh(log_std / 2.)
            noise = (noise + mean) * torch.exp(log_std)
            inputs = torch.cat((inputs, noise), dim=1)
            logp_z_e = (log_prob + log_std).sum(-1, keepdim=True)

        logdets = 0.0
        outputs = inputs
        for f in self.flow_modules:
            outputs, logdets_ = f(outputs, self.cond_nn(cond_inputs))
            #outputs, logdets_ = f(outputs)
            logdets += logdets_

        if self.noise_cond_nn_in_dim:
            logdets = logp_z_e - logdets
        return outputs, logdets

    def reverse(self, inputs, cond_inputs=None):
        outputs = inputs
        for f in reversed(self.flow_modules):
            outputs = f.reverse(outputs, self.cond_nn(cond_inputs))
            #outputs = f.reverse(outputs)
        if self.noise_cond_nn_in_dim:
            outputs = outputs[:, :-self.aug_noise_dim]
        return outputs


def _get_noise_condn_nn_in_dims(n_blocks, input_dim, growth_rate):
    if growth_rate == 0:
        return n_blocks * [0]
    in_dims = [0, input_dim]
    cur_dim, dim_sum = input_dim, input_dim
    for i in range(n_blocks - 2):
        dim_sum += cur_dim
        in_dims.append(dim_sum)
        cur_dim += growth_rate
    return in_dims[:n_blocks]


class Flow(nn.Module):
    """ A sequential container for flows.
    In addition to a forward pass it implements a backward pass and
    computes log jacobians.
    """
    def __init__(self, n_classes, n_blocks, n_hidden, cond_nn_factory=None, noise_growth_rate=0, reject_sampling=False):
        super().__init__()
        self.n_classes = n_classes
        self.n_blocks = n_blocks
        self.n_hidden = n_hidden
        self.cond_nn_factory=cond_nn_factory
        self.noise_growth_rate = noise_growth_rate
        self.flow_blocks = nn.ModuleList()
        self.flow_blocks.append(argmax.ArgmaxLayer(self.n_classes))
        input_dim = self.n_classes
        noise_cond_nn_in_dim = _get_noise_condn_nn_in_dims(n_blocks, n_classes, noise_growth_rate)
        for i in range(self.n_blocks):
            self.flow_blocks.append(
                FlowBlock(input_dim, self.n_hidden, self.cond_nn_factory, noise_cond_nn_in_dim[i], noise_growth_rate)
            )
            if noise_growth_rate:
                input_dim += noise_growth_rate

    def forward(self, inputs, cond_inputs=None):
        """ Performs a forward or backward pass for flow modules.
        Args:
            inputs: a tuple of inputs and logdets
            mode: to run direct computation or inverse
        """
        logdets = torch.zeros(inputs.size(0), 1, device=inputs.device)
        inputs, logdet = self.flow_blocks[0](inputs)
        logdets += logdet
        prev_inputs = []
        for module in self.flow_blocks[1:]:
            prev_inputs.append(inputs)
            inputs, logdet = module(inputs, cond_inputs, prev_inputs[:-1])
            logdets += logdet
        return inputs, logdets

    def reverse(self, inputs, cond_inputs=None):
        """ Performs a forward or backward pass for flow modules.
        Args:
            inputs: a tuple of inputs and logdets
            mode: to run direct computation or inverse
        """
        for module in reversed(self.flow_blocks):
            inputs = module.reverse(inputs, cond_inputs)
        return inputs

    def log_probs(self, inputs, mean=None, log_var=None, cond_inputs=None):
        u, log_jacob = self(inputs, cond_inputs)
        if mean is not None and log_var is not None:
            std = torch.sqrt(torch.exp(log_var))
            u = (u - mean) / std
        log_probs = (-0.5 * u.pow(2) - 0.5 * math.log(2 * math.pi)).sum(-1, keepdim=True)
        return (log_probs + log_jacob).sum(-1, keepdim=True)

    def sample_with_log_prob(self, n_samples, mean=None, log_var=None, cond_inputs=None):
        input_dim = self.n_classes + self.noise_growth_rate * (self.n_blocks - 1)
        noise_dist = torch.distributions.normal.Normal(
            torch.zeros(n_samples, input_dim).cuda(),
            torch.ones(n_samples, input_dim).cuda()
        )
        noise = noise_dist.sample()
        log_prob = noise_dist.log_prob(noise).sum(dim=1, keepdim=True)
        device = next(self.parameters()).device
        noise = noise.to(device)
        if mean is not None and log_var is not None:
            std = torch.sqrt(torch.exp(log_var))
            noise = noise * std + mean

        log_prob = log_prob.to(device)
        if cond_inputs is not None:
            cond_inputs = cond_inputs.to(device)
        samples = self.reverse(noise, cond_inputs)

        return samples, log_prob
