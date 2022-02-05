import flows.maf as maf
import flows.argmax as argmax
import torch
import torch.nn as nn
import torch.distributions as dist
import math

class FlowBlock(nn.Module):
    def __init__(self, input_dim, n_hidden, cond_nn_factory=None, augment_noise=False):
        super().__init__()
        self.input_dim = input_dim
        self.n_hidden = n_hidden
        self.augment_noise = augment_noise
        self.cond_nn = cond_nn_factory(self.n_hidden)
        if augment_noise:
            self.noise_cond_nn = cond_nn_factory(input_dim)
        self.flow_modules = nn.ModuleList([
            maf.MADE(self.input_dim, num_hidden=self.n_hidden, act='relu'),
            maf.BatchNormFlow(self.input_dim),
            maf.Reverse(self.input_dim)
        ])

    def forward(self, inputs, cond_inputs=None):
        if cond_inputs is not None:
            assert self.cond_nn is not None, 'Conditional NN not defined for conditional inputs!'

        if self.augment_noise:
            noise_dist = dist.normal.Normal(torch.zeros_like(inputs), torch.ones_like(inputs))
            noise = noise_dist.sample()
            log_prob = noise_dist.log_prob(noise).sum(dim=1, keepdim=True)
            noise_cond_nn_out = self.noise_cond_nn(cond_inputs).view(-1, 2, self.input_dim//2)
            mean, log_std = noise_cond_nn_out.permute(1, 0, 2)
            noise = (noise + mean) * torch.exp(log_std)
            inputs = torch.cat((inputs, noise), dim=1)
            logp_z_e = (log_prob + log_std).sum(-1, keepdim=True)

        logdets = 0.0
        outputs = inputs
        for f in self.flow_modules:
            #outputs, logdets_ = f(outputs, self.cond_nn(cond_inputs))
            outputs, logdets_ = f(outputs)
            logdets += logdets_

        if self.augment_noise:
            logdets = logp_z_e - logdets
        return outputs, logdets

    def reverse(self, inputs, cond_inputs=None):
        if self.augment_noise:
            inputs = inputs[:, :inputs.shape[-1]/2]
        outputs = inputs
        for f in reversed(self.flow_modules):
            #outputs = f.reverse(outputs, self.cond_nn(cond_inputs))
            outputs = f.reverse(outputs)
        return outputs

class Flow(nn.Module):
    """ A sequential container for flows.
    In addition to a forward pass it implements a backward pass and
    computes log jacobians.
    """
    def __init__(self, n_classes, n_blocks, n_hidden, cond_nn_factory=None, augment_noise=False, reject_sampling=False):
        super().__init__()
        self.n_classes = n_classes
        self.n_blocks = n_blocks
        self.n_hidden = n_hidden
        self.cond_nn_factory=cond_nn_factory
        self.flow_blocks = nn.ModuleList()
        self.flow_blocks.append(argmax.ArgmaxLayer(self.n_classes))
        input_dim = self.n_classes
        self.flow_blocks.append(FlowBlock(input_dim, self.n_hidden, self.cond_nn_factory, False))
        for _ in range(self.n_blocks - 1):
            if augment_noise:
                input_dim *= 2
            self.flow_blocks.append(FlowBlock(input_dim, self.n_hidden, self.cond_nn_factory, augment_noise))

    def forward(self, inputs, cond_inputs=None):
        """ Performs a forward or backward pass for flow modules.
        Args:
            inputs: a tuple of inputs and logdets
            mode: to run direct computation or inverse
        """
        logdets = torch.zeros(inputs.size(0), 1, device=inputs.device)
        for module in self.flow_blocks:
            inputs, logdet = module(inputs, cond_inputs)
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

    def sample_with_log_prob(self, sample_shape=None, mean=None, log_var=None, cond_inputs=None):
        noise_dist = torch.distributions.normal.Normal(torch.zeros(sample_shape), torch.ones(sample_shape))
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
