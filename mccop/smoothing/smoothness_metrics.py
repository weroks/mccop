import torch
from torch import nn


def avg_input_grad_norm(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Compute the average L2 norm of the input gradients as a proxy of model smoothness.

    Args:
        model: The neural network model.
        x: Input tensor of shape (batch_size, ...).

    Returns:
        A tensor containing the average L2 norm of the input gradients.
    """
    x = x.detach().requires_grad_(True)
    y = model(x).squeeze(-1)
    grads = torch.autograd.grad(
        outputs=y,
        inputs=x,
        grad_outputs=torch.ones_like(y),
        create_graph=False,
        retain_graph=False,
    )[0]

    return grads.reshape(grads.size(0), -1).norm(p=2, dim=1).mean()


def local_lipschitz_adv(model: nn.Module, x: torch.Tensor, eps: float = 1e-3, iters: int = 5) -> torch.Tensor:
    """Estimate the worst-case local Lipschitz constant using a mini-PGD approach."""
    model.eval()
    x = x.detach()

    noise = torch.randn_like(x).detach()

    for _ in range(iters):
        noise.requires_grad_(True)

        n_flat = noise.view(noise.size(0), -1)
        n_norm = n_flat.norm(p=2, dim=1).view(-1, 1, 1, 1)

        perturbed_noise = (noise / (n_norm + 1e-12)) * eps

        with torch.enable_grad():
            y_diff = (model(x + perturbed_noise).squeeze(-1) - model(x).squeeze(-1)).abs().sum()

        grad_n = torch.autograd.grad(y_diff, noise)[0]

        noise = grad_n.detach()

    with torch.no_grad():
        n_flat = noise.view(noise.size(0), -1)
        n_norm = n_flat.norm(p=2, dim=1).view(-1, 1, 1, 1)
        final_noise = (noise / (n_norm + 1e-12)) * eps

        y_orig = model(x).squeeze(-1)
        y_pert = model(x + final_noise).squeeze(-1)
        return (y_pert - y_orig).abs().mean() / eps
