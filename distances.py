import torch

def euclidean_distance(x: torch.Tensor, squared: bool = True) -> torch.Tensor:
    dot_product = torch.matmul(x, torch.transpose(x, 0, 1))
    vectors_norms = torch.diag(dot_product)
    distances_ = torch.unsqueeze(vectors_norms, dim=1) - 2.0 * dot_product + torch.unsqueeze(vectors_norms, dim=0)
    distances_ = torch.maximum(distances_, torch.tensor(0.0)) # to(device)
    if not squared:
        mask = torch.eq(distances_, torch.tensor(0.0)).to(torch.float32)
        distances_ = distances_ + mask * 1e-16
        distances_ = torch.sqrt(distances_)
        distances_ = distances_ * (1.0 - mask)
        
    return distances_


class EuclideanDistance(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, squared: bool = False):
        with torch.no_grad():
            dot_product = torch.matmul(x, torch.transpose(x, 0, 1))
            vectors_norms = torch.diag(dot_product)
            distances_ = torch.unsqueeze(vectors_norms, dim=1) - 2.0 * dot_product + torch.unsqueeze(vectors_norms, dim=0)
            distances_ = torch.maximum(distances_, torch.tensor(0.0))
            if not squared:
                mask = torch.eq(distances_, torch.tensor(0.0)).to(torch.float32)
                distances_ = distances_ + mask * 1e-16
                distances_ = torch.sqrt(distances_)
                distances_ = distances_ * (1.0 - mask)
        return distances_

custom_pytorch_distance = EuclideanDistance()