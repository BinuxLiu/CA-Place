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
        
def log_otp_solver(log_a, log_b, M, num_iters: int = 3, reg: float = 1.0) -> torch.Tensor:
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
    # Optimal Transport or KL-Div for knowledge Distillation in a Low-rank space
    def __init__(self, channel_s=640, channel_t=768, num_hiddens=128, embedding=False, metric = "cosine", T = 4): 
        super(LoTD, self).__init__() 
        
        self.channel_s = channel_s
        self.channel_t = channel_t
        self.num_hiddens = num_hiddens
        self.embedding = embedding
        self.metric = metric
        self.T = T
        
        self.proj_s = nn.Sequential(
            # nn.Linear(self.channel_s, self.channel_t),
            # nn.ReLU(),
            # nn.Dropout(0.05),
            # nn.Linear(self.num_hiddens, self.channel_t),
            # L2Norm(),
            # nn.Sigmoid(),
        ).to("cuda")
        
        self.proj_t = nn.Sequential(
            # nn.Linear(self.channel_t, self.channel_s),
            # nn.Sigmoid(),
            # L2Norm(),
        ).to("cuda")
        
        self.flatten = Flatten()
        
        self.norm = L2Norm()
        
        self.avgpool=nn.AdaptiveAvgPool2d(1)

    def forward(self, feat_s, feat_t):
        # Step 1: Match the teacher's spatial dimensions
        if not self.embedding:
            feat_s = self.norm(feat_s)
            feat_t = self.norm(feat_t)
            feat_s = self.avgpool(feat_s).to("cuda")
            
            if self.metric == "ot":
                # Step 2: Flatten feature maps
                BS, C_s, H, W = feat_s.shape
                N_s = H * W
                feat_s = feat_s.view(BS, C_s, N_s).permute(0, 2, 1)  # [BS, N_s, C_s]

                BS, C_t, H_t, W_t = feat_t.shape
                N_t = H_t * W_t
                feat_t = feat_t.view(BS, C_t, N_t).permute(0, 2, 1)  # [BS, N_t, C_t]
            else:
                feat_s = self.flatten(feat_s)
                feat_t = self.flatten(feat_t)
        else:
            if self.metric == "ot":
                BS, C_s = feat_s.shape
                BS, C_t = feat_t.shape
                N_s = 1
                N_t = 1
                feat_s = feat_s.view(BS, C_s, N_s).permute(0, 2, 1)  # [BS, N_s, C_s]
                feat_t = feat_t.view(BS, C_t, N_t).permute(0, 2, 1)  # [BS, N_t, C_t]
        
        # Step 3: Apply projections
        feat_s = self.proj_s(feat_s)  # [BS, N_s, num_hiddens]
        feat_t = self.proj_t(feat_t)  # [BS, N_t, num_hiddens]
        
        if self.metric == "ot":
            # Step 4: Compute cost matrix
            M = torch.cdist(feat_s, feat_t, p=2) ** 2  # [BS, N_s, N_t]
            
            # Step 5: Define uniform weights
            log_a = torch.full((BS, N_s), -math.log(N_s), device=feat_s.device)
            log_b = torch.full((BS, N_t), -math.log(N_t), device=feat_t.device)
            
            # Step 6: Apply differentiable optimal transport
            log_T = log_otp_solver(log_a, log_b, M, num_iters=20, reg=3)
            T = torch.exp(log_T)  # [BS, N_s, N_t]
            
            # Step 7: Compute loss
            loss = torch.sum(T * M) / BS
            
        elif self.metric == "kl":
            loss = F.kl_div(F.log_softmax(feat_s/self.T, dim = 1), F.softmax(feat_t/self.T, dim = 1), reduction="batchmean") * self.T * self.T
            
        elif self.metric == "mse":
            loss = F.mse_loss(feat_s, feat_t, reduction='mean')
            
        elif self.metric == "cosine":
            loss = 1 - F.cosine_similarity(feat_s, feat_t, dim=-1).mean()

        return loss    
    
def test():

    # Define the batch size and spatial dimensions (example)
    batch_size = 2
    teacher_channels = 768
    student_channels = 640
    hidden_channels = 16

    # Generate dummy teacher and student feature maps
    teacher_feat = torch.randn(batch_size, teacher_channels, 16, 16)
    student_feat = torch.randn(batch_size, student_channels, 28, 28)

    avgpool = torch.nn.AdaptiveAvgPool2d(1)
    teacher_feat = avgpool(teacher_feat).to("cuda")
    # Print shapes
    print(f"Teacher feature map shape: {teacher_feat.shape}")
    print(f"Student feature map shape: {student_feat.shape}")

    # Run the compute_distillation_loss function
    loss = LoTD(channel_s=student_channels, channel_t=teacher_channels, embedding=False)
    distillation_loss = loss(student_feat, teacher_feat)
    
    # Print the distillation loss
    print(f"Distillation loss: {distillation_loss.item()}")
    
    # Generate dummy teacher and student feature maps
    teacher_feat = torch.randn(batch_size, teacher_channels).cuda()
    student_feat = torch.randn(batch_size, student_channels).cuda()

    # Print shapes
    print(f"Teacher feature map shape: {teacher_feat.shape}")
    print(f"Student feature map shape: {student_feat.shape}")

    # Run the compute_distillation_loss function
    loss = LoTD(channel_s=student_channels, channel_t=teacher_channels, embedding=True)
    distillation_loss = loss(student_feat, teacher_feat)
    
    # Print the distillation loss
    print(f"Distillation loss: {distillation_loss.item()}")
    
if __name__ == "__main__":
    test()