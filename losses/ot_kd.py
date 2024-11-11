import torch
import torch.nn.functional as F
import torch.nn as nn
import math

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

class OT_KD(nn.Module):
    def __init__(self, teacher_channels = 768, student_channels = 640, compressed_channels=64, num_iters=5, reg=1.0):
        """
        Initializes the OT_KD module with learnable channel compression layers.

        Args:
            teacher_channels (int): Number of channels in the teacher feature map.
            student_channels (int): Number of channels in the student feature map.
            compressed_channels (int): Target number of channels after compression.
            num_iters (int): Number of iterations for the Sinkhorn algorithm.
            reg (float): Regularization parameter for the Sinkhorn algorithm.
        """
        super(OT_KD, self).__init__()
        
        self.num_iters = num_iters
        self.reg = reg

        # Learnable linear layers to compress channels
        self.teacher_proj = nn.Linear(teacher_channels, compressed_channels).to("cuda")
        self.student_proj = nn.Linear(student_channels, compressed_channels).to("cuda")

    def forward(self, teacher_feat, student_feat):
        """
        Computes the loss between compressed teacher and student feature maps with optimal transport.

        Args:
            teacher_feat (torch.Tensor): Teacher's feature map of shape [B, C_teacher, H_teacher, W_teacher].
            student_feat (torch.Tensor): Student's feature map of shape [B, C_student, H_student, W_student].

        Returns:
            torch.Tensor: Distillation loss.
        """

        # teacher_feat = (teacher_feat - teacher_feat.mean()) / (teacher_feat.std() + 1e-8)
        # student_feat = (student_feat - student_feat.mean()) / (student_feat.std() + 1e-8)

        # Apply global average pooling to reduce feature maps to shape [B, C, 1, 1]
        teacher_feat_pooled = F.adaptive_avg_pool2d(teacher_feat, output_size=(1, 1)).squeeze(-1).squeeze(-1)
        student_feat_pooled = F.adaptive_avg_pool2d(student_feat, output_size=(1, 1)).squeeze(-1).squeeze(-1)

        # Compress channels using learnable linear layers
        teacher_feat_compressed = self.teacher_proj(teacher_feat_pooled)  # [B, compressed_channels]
        student_feat_compressed = self.student_proj(student_feat_pooled)  # [B, compressed_channels]

        # Reshape for optimal transport cost calculation
        teacher_feat_flat = teacher_feat_compressed.unsqueeze(-1)  # [B, compressed_channels, 1]
        student_feat_flat = student_feat_compressed.unsqueeze(-1)  # [B, compressed_channels, 1]

        # Compute cost matrix (squared Euclidean distance)
        cost_matrix = torch.cdist(teacher_feat_flat, student_feat_flat, p=2) ** 2

        # Normalize channel descriptors
        log_a = torch.log(F.softmax(teacher_feat_flat.squeeze(-1), dim=1))
        log_b = torch.log(F.softmax(student_feat_flat.squeeze(-1), dim=1))

        # Optimal transport to map teacher channels to student channels
        log_P = log_otp_solver(log_a, log_b, cost_matrix, num_iters=self.num_iters, reg=self.reg)
        P = torch.exp(log_P)  # Optimal transport plan

        # Apply the transport plan to map teacher features
        teacher_feat_mapped = torch.bmm(P.permute(0, 2, 1), teacher_feat_flat).squeeze(-1)

        # Compute distillation loss on compressed features
        loss = F.mse_loss(teacher_feat_mapped, student_feat_compressed)

        return loss
    
def test():

    # Set random seed for reproducibility
    torch.manual_seed(0)

    # Define the batch size and spatial dimensions (example)
    batch_size = 2
    height, width = 224, 224  # Assuming spatial dimensions for student features
    teacher_channels = 768
    student_channels = 160
    K = 4

    # Generate dummy teacher and student feature maps
    teacher_feat = torch.randn(batch_size, teacher_channels, 16, 16)  # [B, 768, 8, 8]
    student_feat = torch.randn(batch_size, student_channels, height // K, width // K)  # [B, 160, 16, 16]

    # Print shapes
    print(f"Teacher feature map shape: {teacher_feat.shape}")
    print(f"Student feature map shape: {student_feat.shape}")

    # Run the compute_distillation_loss function
    distillation_loss = ot_kd(teacher_feat, student_feat, num_iters=20, reg=1.0)

    # Print the distillation loss
    print(f"Distillation loss: {distillation_loss.item()}")
    
if __name__ == "__main__":
    test()
