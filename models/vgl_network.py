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
            pretrained=False, 
            model_path = "/root/autodl-tmp/MambaVision/weights/mambavision_tiny_1k.pth.tar")
        
        self.aggregation = get_aggregation(args, channels=640, fc_output_dim = 640)
        
    def forward(self, x):
        fs = []
        x = self.backbone.patch_embed(x)
        
        for lev in self.backbone.levels:
            x = lev(x)
            fs.append(x)
            
        x = self.backbone.norm(x)
        x = self.aggregation(x)
        
        return x, fs
    
class VGLNet(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.backbone = dinov2_network.DINOv2(
            backbone=args.backbone_t,
            trainable_layers=args.trainable_layers,
            return_token=args.use_cls)
        self.channels_num = dinov2_network.CHANNELS_NUM[args.backbone_t]
        self.aggregation = get_aggregation(args, channels = self.channels_num, fc_output_dim = 768)
     
    def forward(self, x):
        
        B, C, H, W = x.shape
        fs = []
        x = self.backbone.model.prepare_tokens_with_masks(x)

        for blk in self.backbone.model.blocks:
            x = blk(x)
            f = x[:, 1:]
            f = f.reshape((B, H // 14, W // 14, self.channels_num)).permute(0, 3, 1, 2)
            fs.append(f)

        x = self.backbone.model.norm(x)
        f = x[:, 1:]
        f = f.reshape((B, H // 14, W // 14, self.channels_num)).permute(0, 3, 1, 2)
        x = self.aggregation(f)
        
        return x, fs
    
class VGLNet_Test(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.backbone = dinov2_network.DINOv2(backbone=args.backbone_t,
                               return_token=args.use_cls)
        
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
    

def get_aggregation(args, channels = None , fc_output_dim = None):
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
            fc_output_dim=fc_output_dim,
            num_hiddens=args.num_hiddens,
            use_cls=args.use_cls,
            use_ca=args.use_ca,
            pooling_method=args.ca_method,
        )
