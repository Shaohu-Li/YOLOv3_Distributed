#=================================================#
#   用来存放当前网络的 backbone                    #
#=================================================#

import torch
import torch.nn as nn
from module import Create_config_layer

class BackBone(nn.Module):
    def __init__(self, in_channels, backbone_conf):
        super().__init__()
        self.in_channels    = in_channels
        self.backbone_conf  = backbone_conf
        self.backbone       = Create_config_layer(backbone_conf, in_channels)

    def forward(self):
        return self.backbone
