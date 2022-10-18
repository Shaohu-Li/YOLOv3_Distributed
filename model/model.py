#=================================================#
#   用来存放当前网络的 完整的网络模型                #
#=================================================#
import torch
import torch.nn as nn
from torchsummary import summary
from model.module import Create_config_layer, ScalePrediction, ResidualBlock, CNNBlock

class YOLOv3(nn.Module):
    def __init__(self, model_config, in_channels=3, num_classes=80):
        super().__init__()
        self.num_classes =num_classes
        self.in_channels = in_channels
        self.model_config = model_config
        self.layers = self._create_conv_layers()
        


    def forward(self, x):
        outputs = []
        route_connection = []

        for layer in self.layers:
            if  isinstance(layer, ScalePrediction):
                outputs.append(layer(x))
                continue

            x = layer(x)

            if isinstance(layer, ResidualBlock) and layer.num_repeats == 8:
                route_connection.append(x)
            
            elif isinstance(layer, nn.Upsample):
                x = torch.cat([x, route_connection[-1]], dim=1)
                route_connection.pop()
        
        return outputs


    def _create_conv_layers(self):

        layers = nn.ModuleList()
        in_channels = self.in_channels
        models = self.model_config

        for module in models:
            if isinstance(module, tuple):
                out_channels, kernel_size, stride = module
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
            elif isinstance(module, list):
                num_repeats = module[1]
                layers.append(
                    ResidualBlock(
                        in_channels,
                        num_repeats=num_repeats,
                    )
                )
            elif isinstance(module, str):
                if module == "S":
                    layers += [
                        ResidualBlock(in_channels, use_residual=False, num_repeats=1),
                        CNNBlock(in_channels, in_channels //2, kernel_size=1),
                        ScalePrediction(in_channels // 2, self.num_classes)
                    ]
                    in_channels = in_channels // 2

                elif module == "U":
                    layers.append(nn.Upsample(scale_factor=2),)
                    # 因为将再上采样层之后的进行拼接操作，又因为现在的通道数目为将要哦拼接的原始通道数的一半，所以我们才需要3倍数的通道数目
                    in_channels = in_channels * 3 
        return layers
if __name__ == "__main__":

    #-----------------------------------#
    #   解决vscdoe 导入不了自己包的问题   #
    #-----------------------------------#
    #
    import sys
    sys.path.append("./")
    # print(sys.path)
    #-----------------------------------#

    from config.config import model_config
    num_classes = 20
    IMAGE_SIZE = 416
    model = YOLOv3(model_config=model_config, num_classes=num_classes)
    x = torch.randn((2, 3, IMAGE_SIZE, IMAGE_SIZE))
    # print(model)
    summary(model,(3, 416, 416))