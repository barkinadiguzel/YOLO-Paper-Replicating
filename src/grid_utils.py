import torch

def create_grid(S, device='cpu'):
    lin = torch.linspace(0, 1, S, device=device)
    x, y = torch.meshgrid(lin, lin, indexing='ij')
    grid = torch.stack([x, y], dim=-1)  # (S, S, 2)
    return grid
