from torch import nn

from models import dinov2_network
import models.aggregations as aggregations
from models import mamba_vision

import torchvision.transforms as transforms
import time

class MambaVGL(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.backbone = mamba_vision.mamba_vision_T(
            pretrained=True, 
            model_path = "/root/autodl-tmp/MambaVision/weights/mambavision_tiny_1k.pth.tar")
        
        self.aggregation = get_aggregation(args, channels=640)
        
    def forward(self, x):
        x = self.backbone.patch_embed(x)
        for lev in self.backbone.levels:
            x = lev(x)
        feats_s = x
        x = self.backbone.norm(x)
        x = self.aggregation(x)
        return x, feats_s
    
class VGLNet(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.backbone = dinov2_network.DINOv2(
            backbone=args.backbone_t,
            trainable_layers=args.trainable_layers,
            return_token=args.use_cls)
        
        self.aggregation = get_aggregation(args, channels = dinov2_network.CHANNELS_NUM[args.backbone_t])
        
    def forward(self, x):

        x = self.backbone(x)
        feats_t = x
        x = self.aggregation(x)
        return x, feats_t
    
class VGLNet_Test(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.backbone = dinov2_network.DINOv2(backbone=args.backbone_t,
                               trainable_layers=args.trainable_layers)
        
        self.aggregation = get_aggregation(args, channels = dinov2_network.CHANNELS_NUM[args.backbone_t])
        
        self.all_time = 0
        
    def forward(self, x):

        if not self.training:
            b, c, h, w = x.shape
            h = round(h / 14) * 14
            w = round(w / 14) * 14
            x = transforms.functional.resize(x, [h, w], antialias=True)

        x = self.backbone(x)
        feats_t = x
        x = self.aggregation(x)

        return x, feats_t
    

def get_aggregation(args, channels = None):
    if args.aggregation == "salad":
        return aggregations.SALAD(num_channels = channels)
    elif args.aggregation == "cosgem":
        return aggregations.CosGeM(features_dim= channels, fc_output_dim=args.features_dim)
    elif args.aggregation == "cls":
        return aggregations.CLS()
    elif args.aggregation == "g2m":
        return aggregations.G2M(
            # num_channels=640,
            num_channels=channels,
            fc_output_dim=args.features_dim,
            num_hiddens=args.num_hiddens,
            use_cls=args.use_cls,
            use_ca=args.use_ca,
            pooling_method=args.ca_method,
        )
