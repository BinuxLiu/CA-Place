from torch import nn

from models import dinov2_network
import models.aggregations as aggregations
from models import mamba_vision

import torchvision.transforms as transforms
import time


class MambaVGL(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.backbone = mamba_vision.mamba_vision_T(pretrained=True, model_path = "/root/autodl-tmp/MambaVision/weights/mambavision_tiny_1k.pth.tar")
        
        self.aggregation = get_aggregation(args)
        
    def forward(self, x):
        x = self.backbone.patch_embed(x)
        for lev in self.backbone.levels:
            x = lev(x)
        x = self.backbone.norm(x)
        x = self.aggregation(x)
        return x
    
class VGLNet(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.backbone = dinov2_network.DINOv2(backbone=args.backbone,
                               trainable_layers=args.trainable_layers)
        
        self.aggregation = get_aggregation(args)
        
    def forward(self, x):

        x = self.backbone(x)
        x = self.aggregation(x)
        return x, x
    
class VGLNet_Test(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.backbone = dinov2_network.DINOv2(backbone=args.backbone,
                               trainable_layers=args.trainable_layers)
        
        self.aggregation = get_aggregation(args)
        
        self.all_time = 0
        
    def forward(self, x):

        if not self.training:
            b, c, h, w = x.shape
            h = round(h / 14) * 14
            w = round(w / 14) * 14
            x = transforms.functional.resize(x, [h, w], antialias=True)

        x = self.backbone(x)
        
        # start_time = time.time()
        
        x = self.aggregation(x)
        
        # end_time = time.time()
        # self.all_time = self.all_time + (end_time - start_time)

        return x
    

def get_aggregation(args):
    if args.aggregation == "salad":
        return aggregations.SALAD(num_channels = dinov2_network.CHANNELS_NUM[args.backbone])
    elif args.aggregation == "netvlad":
        return aggregations.NetVLAD(clusters_num=args.clusters, dim=dinov2_network.CHANNELS_NUM[args.backbone], work_with_tokens=args.use_cls, linear_dim = args.linear_dim, work_with_linear = args.use_linear)
    elif args.aggregation == "cosgem":
        return aggregations.CosGeM(features_dim=dinov2_network.CHANNELS_NUM[args.backbone], fc_output_dim=args.features_dim)
    elif args.aggregation == "cls":
        return aggregations.CLS()
    elif args.aggregation == "g2m":
        return aggregations.G2M(
            num_channels=640,
            # num_channels=dinov2_network.CHANNELS_NUM[args.backbone],
            fc_output_dim=args.features_dim,
            num_hiddens=args.num_hiddens,
            use_cls=args.use_cls,
            use_ca=args.use_ca,
            pooling_method=args.ca_method,
        )
