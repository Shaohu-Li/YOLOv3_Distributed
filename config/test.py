import config
import torch

if __name__ == "__main__":
    # import sys
    # sys.path.append("./")
    # scaled_anchors = (
    #     torch.tensor(config.ANCHORS).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    # ).to(config.DEVICE)
    print( torch.tensor(config.ANCHORS).shape)
    print(torch.tensor(config.IMG_GRID).shape)