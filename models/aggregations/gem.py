import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import numpy as np


def gem(x, p=torch.ones(1)*3, eps: float = 1e-6, k = 1):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2)//k, x.size(-1)//k)).pow(1./p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, k=1):
        super().__init__()
        self.p = Parameter(torch.ones(1)*p)
        self.eps = eps
        self.k = k
    
    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps, k = self.k)
    
    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p.data.tolist()[0]:.4f}, eps={self.eps})"


class Flatten(torch.nn.Module):
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
    
class CosGeM(nn.Module):
    def __init__(self, features_dim, fc_output_dim):
        super().__init__()
        self.norm1 = L2Norm()
        self.gem = GeM()
        self.flatten = Flatten()
        self.fc = nn.Linear(features_dim, fc_output_dim)
        self.norm2 = L2Norm()

    def forward(self, x):
        x, _ = x
        # x = self.norm1(x)
        x = self.gem(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.norm2(x)
        return x

class CBAM_CA(nn.Module):
    def __init__(self, channel, reduction = 12):
        super().__init__()
        self.maxpool=nn.AdaptiveMaxPool2d(1)
        self.avgpool=nn.AdaptiveAvgPool2d(1)
        self.channel_attention = nn.Sequential(
            nn.Conv2d(channel,channel//reduction,1,bias=False),
            nn.ReLU(),
            nn.Conv2d(channel//reduction,channel,1,bias=False))
        self.sigmoid=nn.Sigmoid()
    
    def forward(self, x) :
        max_result=self.maxpool(x)
        avg_result=self.avgpool(x)
        max_out=self.channel_attention(max_result)
        avg_out=self.channel_attention(avg_result)
        output=self.sigmoid(max_out+avg_out)
        return output.flatten(1)

class SE_CA(nn.Module):
    def __init__(self, channel, reduction=12):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_attention = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid())
    
    def forward(self, x):
        x=self.avg_pool(x)
        x=self.channel_attention(x.flatten(1))
        return x

class GCA(nn.Module):
    def __init__(self, num_channels, num_hiddens=3):
        super().__init__()

        self.channel_attention = nn.Sequential(
            GeM(),
            nn.Flatten(),
            nn.Linear(num_channels, num_hiddens),
            nn.GELU(),
            nn.Linear(num_hiddens, num_channels),
            nn.Sigmoid())
    
    def forward(self, x) :

        x=self.channel_attention(x)
        
        return x

class G2M(nn.Module):
    def __init__(
        self,
        num_channels=768,
        fc_output_dim=768,
        num_hiddens=3,
        use_cls=False,
        use_ca=False,
        pooling_method="gem",
    ):
        super().__init__()

        self.use_cls = use_cls
        self.use_ca = use_ca

        self.num_channels = num_channels
        self.fc_output_dim = fc_output_dim
        self.cls_channels = self.fc_output_dim


        self.gem = GeM()

        if self.use_ca:
            if pooling_method == "gem":
                self.channel_attention = GCA(self.num_channels, num_hiddens)
            elif pooling_method == "avg":
                self.channel_attention = SE_CA(channel=self.num_channels)
            elif pooling_method == "cba":
                self.channel_attention = CBAM_CA(channel=self.num_channels)

        self.feat_proj = nn.Sequential(
            nn.Linear(self.num_channels, self.fc_output_dim), L2Norm()
        )

        if self.use_cls:
            self.cls_proj = nn.Sequential(
                nn.Linear(self.num_channels, self.cls_channels), L2Norm()
            )

        self.norm = L2Norm()

    def forward(self, x):
        
        if self.use_cls:
            x_feat, x_cls = x
        else:
            x_feat = x

        if self.use_ca:
            x_atte = self.channel_attention(x_feat)

        x_feat = self.gem(x_feat).flatten(1)

        if self.use_ca:
            x_feat = self.feat_proj(x_feat * x_atte)
        else:
            x_feat = self.feat_proj(x_feat)

        if self.use_cls:
            x_cls = self.cls_proj(x_cls)
            x_feat = self.norm(torch.cat([x_cls, x_feat], dim=-1))

        return x_feat


class CLS(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        _, x = x
        return x


def print_nb_params(m):
    model_parameters = filter(lambda p: p.requires_grad, m.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f"Trainable parameters: {params/1e6:.3}M")


def main():
    import torch.cuda.amp as amp  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    x = (torch.randn(1, 768, 16, 16, device=device), torch.randn(1, 768, device=device))  

    agg = G2M(
        num_channels=768,
        fc_output_dim=768,
        num_hiddens=64,
        use_cls=False,
        use_ca=True,
        pooling_method="gem",
    ).to(device)

    import time
    
    print_nb_params(agg)
    
    start_time = time.time()
    for _ in range(3000): 
        with torch.cuda.amp.autocast():  
            output = agg(x)  
    end_time = time.time()
   
    average_time = (end_time - start_time) / 3000  
    print(f'Average time per pass: {average_time:.6f} seconds')
    print(output.shape)

if __name__ == "__main__":
    main()
