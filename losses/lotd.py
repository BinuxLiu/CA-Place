import torch
from torch import nn
import torch.nn.functional as F
import math


def gem(x, p=torch.ones(1)*3, eps: float = 1e-6, k = 1):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2)//k, x.size(-1)//k)).pow(1./p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, k=1):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.eps = eps
        self.k = k
    
    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps, k = self.k)
    
    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p.data.tolist()[0]:.4f}, eps={self.eps})"


class Flatten(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        assert x.shape[2] == x.shape[3] == 1, f"{x.shape[2]} != {x.shape[3]} != 1"
        return x[:, :, 0, 0]
    
class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim
    
    def forward(self, x):
        return F.normalize(x, p=2.0, dim=self.dim)


def log_otp_solver(log_a, log_b, M, num_iters: int = 20, reg: float = 1.0) -> torch.Tensor:
    """
    Sinkhorn algorithm in the log-domain for Differentiable Optimal Transport.
    """
    M = M / reg  # Regularization

    u, v = torch.zeros_like(log_a), torch.zeros_like(log_b)

    for _ in range(num_iters):
        u = log_a - torch.logsumexp(M + v.unsqueeze(1), dim=2)
        v = log_b - torch.logsumexp(M + u.unsqueeze(2), dim=1)

    return M + u.unsqueeze(2) + v.unsqueeze(1)

class LoTD(nn.Module): 
    # Optimal Transport for knowledge Distillation in a Low-rank space
    def __init__(self, channel_s=640, channel_t=768, num_hiddens=64, embedding=False): 
        super(LoTD, self).__init__() 
        
        self.channel_s = channel_s
        self.channel_t = channel_t
        self.num_hiddens = num_hiddens
        self.embedding = embedding
        
        self.proj_s = nn.Sequential( 
            nn.Linear(self.channel_s, self.num_hiddens),
            L2Norm(),
        ).to("cuda")
        
        self.proj_t = nn.Sequential(
            nn.Linear(self.channel_t, self.num_hiddens),
            L2Norm(),
        ).to("cuda")

    def forward(self, feat_s, feat_t):
        # Step 1: Match the teacher's spatial dimensions
        if feat_s.shape[2:] != feat_t.shape[2:]:
            feat_s = F.interpolate(feat_s, size=feat_t.shape[2:], mode='bilinear', align_corners=False)
        
        # Step 2: Flatten feature maps
        BS, C_s, H, W = feat_s.shape
        N_s = H * W
        feat_s = feat_s.view(BS, C_s, N_s).permute(0, 2, 1)  # [BS, N_s, C_s]

        BS, C_t, H_t, W_t = feat_t.shape
        N_t = H_t * W_t
        feat_t = feat_t.view(BS, C_t, N_t).permute(0, 2, 1)  # [BS, N_t, C_t]
        
        # Step 3: Apply projections
        feat_s = self.proj_s(feat_s)  # [BS, N_s, num_hiddens]
        feat_t = self.proj_t(feat_t)  # [BS, N_t, num_hiddens]
        
        # Step 4: Compute cost matrix
        M = torch.cdist(feat_s, feat_t, p=2) ** 2  # [BS, N_s, N_t]
        
        # Step 5: Define uniform weights
        log_a = torch.full((BS, N_s), -math.log(N_s), device=feat_s.device)
        log_b = torch.full((BS, N_t), -math.log(N_t), device=feat_t.device)
        
        # Step 6: Apply differentiable optimal transport
        log_T = log_otp_solver(log_a, log_b, M, num_iters=50, reg=0.1)
        T = torch.exp(log_T)  # [BS, N_s, N_t]
        
        # Step 7: Compute loss
        loss = torch.sum(T * M) / BS

        return loss

    

class APFNorm(nn.Module):
    def __init__(self):
        super(APFNorm, self).__init__()
        
        self.avgpool=nn.AdaptiveAvgPool2d(1)
        self.norm = L2Norm()
        
    def forward(self, feat):
        feat = self.avgpool(feat).flatten(1)
        feat = self.norm(feat)
        
        return feat
    
    
def test():

    # Set random seed for reproducibility
    torch.manual_seed(0)

    # Define the batch size and spatial dimensions (example)
    batch_size = 2
    teacher_channels = 768
    student_channels = 160

    # Generate dummy teacher and student feature maps
    teacher_feat = torch.randn(batch_size, teacher_channels, 16, 16)
    student_feat = torch.randn(batch_size, student_channels, 28, 28)

    # Print shapes
    print(f"Teacher feature map shape: {teacher_feat.shape}")
    print(f"Student feature map shape: {student_feat.shape}")

    # Run the compute_distillation_loss function
    loss = LoTD(channel_s=student_channels, channel_t=teacher_channels, num_hiddens=64, embedding=False)
    distillation_loss = loss(student_feat, teacher_feat)
    
    # Print the distillation loss
    print(f"Distillation loss: {distillation_loss.item()}")
    
if __name__ == "__main__":
    test()