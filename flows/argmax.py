import torch
import torch.nn as nn
import torch.nn.functional as F

class ArgmaxLayer(nn.Module):
    def __init__(self, dim):
        super(ArgmaxLayer, self).__init__()
        #self.noise_dist = noise_dist
        self.dim = dim
        self.cond_nn = nn.Sequential(
            nn.Linear(dim, 2 * dim)
        )

    def threshold(self, T, inputs):
        return T - nn.Softplus()(T - inputs)

    def sample_with_log_prob(self, inputs):
        cond_out = self.cond_nn(inputs).view(-1, 2, self.dim)
        mean, log_var = cond_out.permute(1, 0, 2)
        noise_dist = torch.distributions.normal.Normal(
            torch.zeros(inputs.shape),
            torch.ones(inputs.shape)
        )
        noise = noise_dist.sample()
        log_prob = noise_dist.log_prob(noise).sum(dim=1, keepdim=True).cuda()
        std = torch.sqrt(torch.exp(log_var))
        noise = noise.cuda() * std + mean
        return noise, log_prob

    def forward(self, inputs=None, cond_inputs=None):
        max_idx = torch.argmax(inputs, dim=1)
        u, log_pu = self.sample_with_log_prob(inputs)
        T = u[torch.arange(u.shape[0]), max_idx]
        v = self.threshold(T.unsqueeze(1), u)
        v[torch.arange(u.shape[0]), max_idx] = T
        log_jacob = F.logsigmoid(T.unsqueeze(1) - u)
        log_jacob[torch.arange(u.shape[0]), max_idx] = 0
        log_det = log_jacob.sum(dim=1, keepdim=True)

        # -(log_pu - log_det)
        return v, -(log_pu - log_det)

    def reverse(self, inputs=None, cond_inputs=None):
        max_idx = torch.argmax(inputs, dim=1)
        outputs = torch.zeros_like(inputs)
        outputs[torch.arange(inputs.shape[0]), max_idx] = 1
        return outputs