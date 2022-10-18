#=================================================#
#   用来存放当前网络的 不同的"卷积 " 模块            #
#=================================================#

import torch
import torch.nn as nn

#-------------------------------------------------#
#   标准的卷积模块：
#     bn_act=True   -> 卷积 + BN + 激活函数
#     bn_act=false  -> 卷积 
#-------------------------------------------------#
class CNNBlock(nn.Module):

    def __init__( self, in_channels, out_channels, bn_act=True, **kwargs ):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, bias= not bn_act, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(0.1)
        self.use_bn_act = bn_act

    def forward(self, x):
        if self.use_bn_act:
            return self.leaky(self.bn(self.conv(x)))
        else:
            return self.conv(x)


#-------------------------------------------------#
#   标准的残差模块：
#     use_residual=True   -> 使用的标准卷积为：卷积 + BN + 激活函数
#     use_residual=false  -> 使用的标准卷积为：卷积 
#
#     num_repeats         -> 残差块重复的次数
#-------------------------------------------------#
class ResidualBlock(nn.Module):
    def __init__(self, channels, use_residual=True, num_repeats=1):
        super().__init__()
        self.layers =  nn.ModuleList()
        for _ in range(num_repeats):
            self.layers += [
                nn.Sequential(
                    CNNBlock(channels, channels // 2, kernel_size=1),
                    CNNBlock(channels // 2, channels, kernel_size=3, padding=1),
                )
            ]
        self.use_residual = use_residual
        self.num_repeats = num_repeats

    def forward(self, x):
        for layer in self.layers:
            x = layer(x) + x if self.use_residual else layer(x)
        
        return x

#-------------------------------------------------#
#   根据配置参数快速构建网络架构：
#-------------------------------------------------#
class Create_config_layer(nn.Module):
    def __init__(self, layer_config, in_channels):
        super().__init__()
        self.layer_config         = layer_config
        self.in_channels          = in_channels
    
    def forward(self):
        layers = nn.ModuleList()
        in_channels = self.in_channels

        for conf in self.layer_config:
            if isinstance(conf, tuple):
                out_channels, kernel_size, stride = conf
                layers.append(
                        CNNBlock(
                            in_channels, 
                            out_channels, 
                            kernel_size=kernel_size, 
                            stride=stride, 
                            padding=1 if kernel_size == 3 else 0,
                            )
                    )
                in_channels = out_channels
            elif isinstance(conf, list):
                num_repeats = conf[1]
                layers.append(
                    ResidualBlock(
                        in_channels,
                        num_repeats=num_repeats,
                    )
                )
            elif isinstance(conf, str):
                if conf == "S":
                    layers += [
                        ResidualBlock(in_channels, use_residual=False, num_repeats=1),
                        CNNBlock(in_channels, in_channels //2, kernel_size=1),
                        ScalePrediction(in_channels // 2, self.num_classes)
                    ]
                    in_channels = in_channels // 2

                elif conf == "U":
                    layers.append(nn.Upsample(scale_factor=2),)
                    # 因为将再上采样层之后的进行拼接操作，又因为现在的通道数目为将要哦拼接的原始通道数的一半，所以我们才需要3倍数的通道数目
                    in_channels = in_channels * 3 
        return layers

#-------------------------------------------------#
#   构建yolo多头模块：
#   根据输入通道的不同就可以构建 yolo 的多头
#-------------------------------------------------#
class ScalePrediction(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.pred = nn.Sequential(
            CNNBlock(in_channels, 2 * in_channels, kernel_size=3, padding=1),
            CNNBlock(2 * in_channels, 3 * (num_classes + 5 ),  bn_act=False, kernel_size=1)
        )
        self.num_classes = num_classes

    def forward(self, x):
        return (
            self.pred(x)
            .reshape(x.shape[0], 3, self.num_classes + 5, x.shape[2], x.shape[3])
            .permute(0, 1, 3, 4, 2) # 将分数和框的大小放到后面
        )