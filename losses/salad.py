import math
import torch
import torch.nn as nn

# Code adapted from OpenGlue, MIT license
# https://github.com/ucuapps/OpenGlue/blob/main/models/superglue/optimal_transport.py
def log_otp_solver(log_a, log_b, M, num_iters: int = 20, reg: float = 1.0) -> torch.Tensor:
    r"""Sinkhorn matrix scaling algorithm for Differentiable Optimal Transport problem.
    This function solves the optimization problem and returns the OT matrix for the given parameters.
    Args:
        log_a : torch.Tensor
            Source weights
        log_b : torch.Tensor
            Target weights
        M : torch.Tensor
            metric cost matrix
        num_iters : int, default=100
            The number of iterations.
        reg : float, default=1.0
            regularization value
    """
    M = M / reg  # regularization

    u, v = torch.zeros_like(log_a), torch.zeros_like(log_b)

    for _ in range(num_iters):
        u = log_a - torch.logsumexp(M + v.unsqueeze(1), dim=2).squeeze()
        v = log_b - torch.logsumexp(M + u.unsqueeze(2), dim=1).squeeze()

    return M + u.unsqueeze(2) + v.unsqueeze(1)

# Code adapted from OpenGlue, MIT license
# https://github.com/ucuapps/OpenGlue/blob/main/models/superglue/superglue.py
def get_matching_probs(S, dustbin_score = 1.0, num_iters=3, reg=1.0):
    """sinkhorn"""
    batch_size, m, n = S.size()
    # augment scores matrix
    S_aug = torch.empty(batch_size, m + 1, n, dtype=S.dtype, device=S.device)
    S_aug[:, :m, :n] = S
    S_aug[:, m, :] = dustbin_score

    # prepare normalized source and target log-weights
    norm = -torch.tensor(math.log(n + m), device=S.device)
    log_a, log_b = norm.expand(m + 1).contiguous(), norm.expand(n).contiguous()
    log_a[-1] = log_a[-1] + math.log(n-m)
    log_a, log_b = log_a.expand(batch_size, -1), log_b.expand(batch_size, -1)
    log_P = log_otp_solver(
        log_a,
        log_b,
        S_aug,
        num_iters=num_iters,
        reg=reg
    )
    return log_P - norm


class SALAD(nn.Module):
    """
    This class represents the Sinkhorn Algorithm for Locally Aggregated Descriptors (SALAD) model.

    Attributes:
        num_channels (int): The number of channels of the inputs (d).
        num_clusters (int): The number of clusters in the model (m).
        cluster_dim (int): The number of channels of the clusters (l).
        token_dim (int): The dimension of the global scene token (g).
        dropout (float): The dropout rate.
    """
    def __init__(self,
            num_channels=1536,
            num_clusters=64,
            cluster_dim=128,
            token_dim=256,
            dropout=0.3,
        ) -> None:
        super().__init__()

        self.num_channels = num_channels
        self.num_clusters= num_clusters
        self.cluster_dim = cluster_dim
        self.token_dim = token_dim
        
        if dropout > 0:
            dropout = nn.Dropout(dropout)
        else:
            dropout = nn.Identity()

        # MLP for global scene token g
        self.token_features = nn.Sequential(
            nn.Linear(self.num_channels, 512),
            nn.ReLU(),
            nn.Linear(512, self.token_dim)
        )
        # MLP for local features f_i
        self.cluster_features = nn.Sequential(
            nn.Conv2d(self.num_channels, 512, 1),
            dropout,
            nn.ReLU(),
            nn.Conv2d(512, self.cluster_dim, 1)
        )
        # MLP for score matrix S
        self.score = nn.Sequential(
            nn.Conv2d(self.num_channels, 512, 1),
            dropout,
            nn.ReLU(),
            nn.Conv2d(512, self.num_clusters, 1),
        )
        # Dustbin parameter z
        self.dust_bin = nn.Parameter(torch.tensor(1.))


    def forward(self, x):
        """
        x (tuple): A tuple containing two elements, f and t. 
            (torch.Tensor): The feature tensors (t_i) [B, C, H // 14, W // 14].
            (torch.Tensor): The token tensor (t_{n+1}) [B, C].

        Returns:
            f (torch.Tensor): The global descriptor [B, m*l + g]
        """
        x, t = x  # Extract features and token
        # x: [B, C, H // 14, W // 14]
        # t: [B, C]

        # Cluster features (local descriptors) - f
        f = self.cluster_features(x).flatten(2)
        # f after cluster_features: [B, l, (H // 14) * (W // 14)]

        # Score matrix - p
        p = self.score(x).flatten(2)
        # p after score: [B, m, (H // 14) * (W // 14)]

        # Token features (global descriptor) - t
        t = self.token_features(t)
        # t after token_features: [B, g]

        # Sinkhorn algorithm
        p = get_matching_probs(p, self.dust_bin, 3)
        # p after get_matching_probs: [B, m + 1, (H // 14) * (W // 14)]

        p = torch.exp(p)
        # p after exponentiation: [B, m + 1, (H // 14) * (W // 14)]

        # Remove dustbin row to match dimensions
        p = p[:, :-1, :]
        # p after removing dustbin row: [B, m, (H // 14) * (W // 14)]

        # Expand dimensions to match feature maps
        p = p.unsqueeze(1).repeat(1, self.cluster_dim, 1, 1)
        # p after unsqueeze and repeat: [B, l, m, (H // 14) * (W // 14)]

        f = f.unsqueeze(2).repeat(1, 1, self.num_clusters, 1)
        # f after unsqueeze and repeat: [B, l, m, (H // 14) * (W // 14)]

        # Aggregated feature descriptors
        f = torch.cat([
            nn.functional.normalize(t, p=2, dim=-1),
            nn.functional.normalize((f * p).sum(dim=-1), p=2, dim=1).flatten(1)
        ], dim=-1)
        # f after normalization and concatenation: [B, m * l + g]

        return nn.functional.normalize(f, p=2, dim=-1)
        # Final output: [B, m * l + g]

    

# Set random seed for reproducibility
torch.manual_seed(0)

# Create dummy data
batch_size = 8
num_channels = 768
height, width = 16, 16  # Assume that the input feature map is 14x14 for simplicity
token_dim = num_channels

# Initialize model
model = SALAD(num_channels=num_channels)

# Generate random feature map and token for testing
feature_map = torch.randn(batch_size, num_channels, height, width)
token = torch.randn(batch_size, num_channels)

# Forward pass
output = model((feature_map, token))

# Print output shape
print("Output shape:", output.shape)

# Check expected shape
expected_shape = (batch_size, model.num_clusters * model.cluster_dim + model.token_dim)
assert output.shape == expected_shape, f"Expected shape {expected_shape}, but got {output.shape}"

print("Test passed!")