"""A pure-PyTorch AdamATan2 (Everett et al. 2024, "Scaling Exponents Across Parameterizations and Optimizers"), standing
in for the CUDA package adam-atan2 that TRM imports: the Adam step with atan2 in place of m / (sqrt(v) + eps), and decoupled
weight decay. Same update as the package's kernel: p -= lr / (1 - b1^t) * atan2(m, sqrt(v) / sqrt(1 - b2^t)).
The job scripts put this folder on PYTHONPATH, so `import adam_atan2` in the authors' code resolves to this module."""
import torch


class AdamATan2(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), weight_decay=1e-2):
        super().__init__(params, dict(lr=lr, betas=betas, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            b1, b2 = group['betas']
            for p in group['params']:
                if p.grad is None:
                    continue
                st = self.state[p]
                if not st:
                    st['step'] = 0
                    st['m'] = torch.zeros_like(p)
                    st['v'] = torch.zeros_like(p)
                st['step'] += 1
                t = st['step']
                p.mul_(1 - group['lr'] * group['weight_decay'])
                st['m'].lerp_(p.grad, 1 - b1)
                st['v'].lerp_(p.grad * p.grad, 1 - b2)
                # As the CUDA kernel (csrc/adam_atan2.cu, 0.0.3): the first moment's bias correction scales the step,
                # outside atan2; the second moment's divides the denominator.
                denom = st['v'].sqrt() / (1 - b2 ** t) ** 0.5
                p.add_(torch.atan2(st['m'], denom), alpha=-group['lr'] / (1 - b1 ** t))
