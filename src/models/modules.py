import torch
import torch.nn as nn
import torch.nn.functional as F

class LateralBlock(nn.Module):

    def __init__(self, in_ch, out_ch, shortcut_conv = False, name = 'lateral'):
        super(LateralBlock, self).__init__()
        self.shortcut_conv = shortcut_conv
        self.name = name

        self.conv = nn.Sequential(
            nn.PReLU(),     # can be called with n_channels
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.PReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1)
        )
        if self.shortcut_conv is True:
            self.conv2 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
       
    def forward(self, x):
        if self.shortcut_conv is True: # should be called first or final lateral block
            return self.conv(x) + self.conv2(x)
        else: 
            return self.conv(x)

            

class DownSamplingBlock(nn.Module):

    def __init__(self, in_ch, out_ch, name = 'down'):
        super(DownSamplingBlock, self).__init__()
        self.name = name
        self.conv = nn.Sequential(
            nn.PReLU(),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
            nn.PReLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return self.conv(x)

class UpSamplingBlock(nn.Module):

    def __init__(self, in_ch, out_ch, name = 'up'):
        super(UpSamplingBlock, self).__init__()
        self.name = name
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.PReLU(),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.PReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1)
        )
    
    def forward(self, x):
        return self.up(x)



'''
auto-infering the x-y dimensions.
'''
class AddCoords(nn.Module):

    def __init__(self):
        super().__init__()
        self.xx_channel = (torch.arange(256).repeat(1, 256, 1).cuda().float() / 255) * 2 - 1
        self.yy_channel = self.xx_channel.transpose(1, 2).cuda()

    def forward(self, input_tensor):
        """
        Args:
            input_tensor: shape(batch, channel, x_dim, y_dim)
        """
        batch_size, _, x_dim, y_dim = input_tensor.size()

        # xx_channel = torch.arange(x_dim).repeat(1, y_dim, 1)
        # yy_channel = torch.arange(y_dim).repeat(1, x_dim, 1).transpose(1, 2)

        # xx_channel = xx_channel.float() / (x_dim - 1)
        # yy_channel = yy_channel.float() / (y_dim - 1)

        # xx_channel = xx_channel * 2 - 1
        # yy_channel = yy_channel * 2 - 1

        xx_channel = self.xx_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
        yy_channel = self.yy_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)

        ret = torch.cat([
            input_tensor,
            xx_channel.type_as(input_tensor),
            yy_channel.type_as(input_tensor)], dim=1)

        return ret


class CoordConv(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.addcoords = AddCoords()
        in_size = in_channels+2
        self.conv = nn.Conv2d(in_size, out_channels, **kwargs)

    def forward(self, x):
        ret = self.addcoords(x)
        ret = self.conv(ret)
        return ret




class CoordLateralBlock(nn.Module):

    def __init__(self, in_ch, out_ch, shortcut_conv = False, name = 'coord_lateral'):
        super(CoordLateralBlock, self).__init__()
        self.shortcut_conv = shortcut_conv
        self.name = name

        self.conv = nn.Sequential(
            # nn.PReLU(),     # can be called with n_channels
            CoordConv(in_ch, out_ch, kernel_size=3, padding=1),
            nn.PReLU(),
            CoordConv(out_ch, out_ch, kernel_size=3, padding=1)
        )
        if self.shortcut_conv is True:
            self.conv2 = CoordConv(in_ch, out_ch, kernel_size=3, padding=1)
       
    def forward(self, x):
        if self.shortcut_conv is True: # should be called first or final lateral block
            return self.conv(x) + self.conv2(x)
        else: 
            return self.conv(x)

            

class CoordDownSamplingBlock(nn.Module):

    def __init__(self, in_ch, out_ch, name = 'coord_down'):
        super(CoordDownSamplingBlock, self).__init__()
        self.name = name
        self.conv = nn.Sequential(
            nn.PReLU(),
            CoordConv(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
            nn.PReLU(),
            CoordConv(out_ch, out_ch, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return self.conv(x)

class CoordUpSamplingBlock(nn.Module):

    def __init__(self, in_ch, out_ch, name = 'coord_up'):
        super(CoordUpSamplingBlock, self).__init__()
        self.name = name
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.PReLU(),
            CoordConv(in_ch, out_ch, kernel_size=3, padding=1),
            nn.PReLU(),
            CoordConv(out_ch, out_ch, kernel_size=3, padding=1)
        )
    
    def forward(self, x):
        return self.up(x)
