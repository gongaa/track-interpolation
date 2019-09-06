import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import LateralBlock, DownSamplingBlock, UpSamplingBlock
from .modules import CoordLateralBlock, CoordDownSamplingBlock, CoordUpSamplingBlock

class GridNet(nn.Module):

    def __init__(self, n_channels, seg_out = 20, img_out = 3, filters_level = [32, 64, 96],
                 name = 'gridnet'):
        super(GridNet, self).__init__()
        self.n_row = 3
        self.n_col = 6
        self.seg_out = seg_out
        self.img_out = img_out
        self.f_level = filters_level
        self.name = name

        self.lateral_in = LateralBlock(in_ch=n_channels, out_ch=self.f_level[0],
                                       shortcut_conv=True, name='lateral_in')
        self.lateral_out_seg = LateralBlock(in_ch=self.f_level[0], out_ch=seg_out,
                                       shortcut_conv=False, name='lateral_out_seg')
        self.lateral_out_img = LateralBlock(in_ch=self.f_level[0], out_ch=img_out,
                                       shortcut_conv=False, name='lateral_out_img') 
        setattr(self, 'down_00', DownSamplingBlock(in_ch=self.f_level[0], out_ch=self.f_level[1], name='down_00'))
        setattr(self, 'down_10', DownSamplingBlock(in_ch=self.f_level[1], out_ch=self.f_level[2], name='down_10'))

        for i in range(1, int(self.n_col/2)):
            setattr(self, f'lateral_0{i-1}',LateralBlock(in_ch=self.f_level[0], out_ch=self.f_level[0], name=f'lateral_0{i-1}'))
            setattr(self, f'down_0{i}', DownSamplingBlock(in_ch=self.f_level[0], out_ch=self.f_level[1], name=f'down_0{i}'))
            setattr(self, f'down_1{i}', DownSamplingBlock(in_ch=self.f_level[1], out_ch=self.f_level[2], name=f'down_1{i}'))
            setattr(self, f'lateral_1{i-1}', LateralBlock(in_ch=self.f_level[1], out_ch=self.f_level[1], name=f'lateral_1{i-1}'))
            setattr(self, f'lateral_2{i-1}', LateralBlock(in_ch=self.f_level[2], out_ch=self.f_level[2], name=f'lateral_2{i-1}'))

        for i in range(int(self.n_col/2), self.n_col):
            setattr(self, f'lateral_2{i-1}', LateralBlock(in_ch=self.f_level[2], out_ch=self.f_level[2], name = f'lateral_2{i-1}'))
            setattr(self, f'lateral_1{i-1}', LateralBlock(in_ch=self.f_level[1], out_ch=self.f_level[1], name = f'lateral_1{i-1}'))
            setattr(self, f'lateral_0{i-1}', LateralBlock(in_ch=self.f_level[0], out_ch=self.f_level[0], name = f'lateral_0{i-1}'))
            setattr(self, f'up_1{i}', UpSamplingBlock(in_ch=self.f_level[2], out_ch=self.f_level[1], name=f'up_1{i}')) # when use, use self.up_1[i-int(self.n_col/2)]
            setattr(self, f'up_0{i}', UpSamplingBlock(in_ch=self.f_level[1], out_ch=self.f_level[0], name=f'up_0{i}')) # when use, use self.up_0[i-int(self.n_col/2)]


    def forward(self, x):
        x_0 = self.lateral_in(x)
        x_1 = getattr(self, 'down_00')(x_0)
        x_2 = getattr(self, 'down_10')(x_1)

        for i in range(1, self.n_col):
            if i < self.n_col/2:
                x_0 = getattr(self, f'lateral_0{i-1}')(x_0)
                x_1 = getattr(self, f'down_0{i}')(x_0) + getattr(self, f'lateral_1{i-1}')(x_1)
                x_2 = getattr(self, f'down_1{i}')(x_1) + getattr(self, f'lateral_2{i-1}')(x_2)
            else:
                x_2 = getattr(self, f'lateral_2{i-1}')(x_2)
                x_1 = getattr(self, f'up_1{i}')(x_2) + getattr(self, f'lateral_1{i-1}')(x_1)
                x_0 = getattr(self, f'up_0{i}')(x_1) + getattr(self, f'lateral_0{i-1}')(x_0)

        return self.lateral_out_seg(x_0), self.lateral_out_img(x_0)




class CoordGridNet(nn.Module):

    def __init__(self, n_channels, seg_out = 20, img_out = 3, filters_level = [32, 64, 96],
                 name = 'coordgridnet'):
        super(CoordGridNet, self).__init__()
        self.n_row = 3
        self.n_col = 6
        self.seg_out = seg_out
        self.img_out = img_out
        self.f_level = filters_level
        self.name = name

        self.lateral_in = CoordLateralBlock(in_ch=n_channels, out_ch=self.f_level[0],
                                       shortcut_conv=True, name='lateral_in')
        self.lateral_out_seg = LateralBlock(in_ch=self.f_level[0], out_ch=seg_out,
                                       shortcut_conv=False, name='lateral_out_seg')
        self.lateral_out_img = LateralBlock(in_ch=self.f_level[0], out_ch=img_out,
                                       shortcut_conv=False, name='lateral_out_img') 
        setattr(self, 'down_00', DownSamplingBlock(in_ch=self.f_level[0], out_ch=self.f_level[1], name='down_00'))
        setattr(self, 'down_10', DownSamplingBlock(in_ch=self.f_level[1], out_ch=self.f_level[2], name='down_10'))

        for i in range(1, int(self.n_col/2)):
            setattr(self, f'lateral_0{i-1}',LateralBlock(in_ch=self.f_level[0], out_ch=self.f_level[0], name=f'lateral_0{i-1}'))
            setattr(self, f'down_0{i}', DownSamplingBlock(in_ch=self.f_level[0], out_ch=self.f_level[1], name=f'down_0{i}'))
            setattr(self, f'down_1{i}', DownSamplingBlock(in_ch=self.f_level[1], out_ch=self.f_level[2], name=f'down_1{i}'))
            setattr(self, f'lateral_1{i-1}', LateralBlock(in_ch=self.f_level[1], out_ch=self.f_level[1], name=f'lateral_1{i-1}'))
            setattr(self, f'lateral_2{i-1}', LateralBlock(in_ch=self.f_level[2], out_ch=self.f_level[2], name=f'lateral_2{i-1}'))

        for i in range(int(self.n_col/2), self.n_col):
            setattr(self, f'lateral_2{i-1}', LateralBlock(in_ch=self.f_level[2], out_ch=self.f_level[2], name = f'lateral_2{i-1}'))
            setattr(self, f'lateral_1{i-1}', LateralBlock(in_ch=self.f_level[1], out_ch=self.f_level[1], name = f'lateral_1{i-1}'))
            setattr(self, f'lateral_0{i-1}', LateralBlock(in_ch=self.f_level[0], out_ch=self.f_level[0], name = f'lateral_0{i-1}'))
            setattr(self, f'up_1{i}', UpSamplingBlock(in_ch=self.f_level[2], out_ch=self.f_level[1], name=f'up_1{i}')) # when use, use self.up_1[i-int(self.n_col/2)]
            setattr(self, f'up_0{i}', UpSamplingBlock(in_ch=self.f_level[1], out_ch=self.f_level[0], name=f'up_0{i}')) # when use, use self.up_0[i-int(self.n_col/2)]


    def forward(self, x):
        x_0 = self.lateral_in(x)
        x_1 = getattr(self, 'down_00')(x_0)
        x_2 = getattr(self, 'down_10')(x_1)

        for i in range(1, self.n_col):
            if i < self.n_col/2:
                x_0 = getattr(self, f'lateral_0{i-1}')(x_0)
                x_1 = getattr(self, f'down_0{i}')(x_0) + getattr(self, f'lateral_1{i-1}')(x_1)
                x_2 = getattr(self, f'down_1{i}')(x_1) + getattr(self, f'lateral_2{i-1}')(x_2)
            else:
                x_2 = getattr(self, f'lateral_2{i-1}')(x_2)
                x_1 = getattr(self, f'up_1{i}')(x_2) + getattr(self, f'lateral_1{i-1}')(x_1)
                x_0 = getattr(self, f'up_0{i}')(x_1) + getattr(self, f'lateral_0{i-1}')(x_0)

        return self.lateral_out_seg(x_0), self.lateral_out_img(x_0)


