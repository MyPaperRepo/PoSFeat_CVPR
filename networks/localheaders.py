import torch
import torch.nn as nn
import torch.nn.functional as F

__all__=['SpHeader_HR_Deep_IN']

class SpHeader_HR_Deep_IN(nn.Module):
    """
    spatical attention header
    """
    def __init__(self, in_channels, out_channels=1, act='Sigmoid'):
        super(SpHeader_HR_Deep_IN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        self.norm1 = nn.InstanceNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels+64, 128, 3, 1, 1)
        self.norm2 = nn.InstanceNorm2d(128)
        self.conv3 = nn.Conv2d(128, out_channels, 1, 1, 0)
        self.norm3 = nn.InstanceNorm2d(out_channels)
        self.relu = nn.PReLU()
        self.act = getattr(nn, act)()

        self.convimg = nn.Conv2d(3, 64, 3, 1, 1)
        self.normimg = nn.InstanceNorm2d(64)

    def forward(self, fine_maps):
        fine_map = fine_maps[0]
        img_tensor = fine_maps[1]

        x = self.relu(self.norm1(self.conv1(fine_map)))
        x = F.interpolate(x, img_tensor.shape[2:], align_corners=False, mode='bilinear')
        img_tensor = self.normimg(self.convimg(img_tensor))
        x = torch.cat([x, img_tensor], dim=1)
        x = self.relu(self.norm2(self.conv2(x)))
        score = self.act(self.norm3(self.conv3(x)))

        return score