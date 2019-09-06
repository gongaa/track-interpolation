# The VGG loss in this file is copied from
# https://github.com/ekgibbons/pytorch-sepconv/blob/master/python/_support/VggLoss.py
# The SsimLoss loss in this file is copied (with minor modifications) from
# https://github.com/Po-Hsun-Su/pytorch-ssim/blob/master/pytorch_ssim/__init__.py

from math import exp

import torch
import torch.nn.functional as F
import torchvision
# import src.config as config
from torch import nn
from torch.autograd import Variable


class GradientLoss(nn.Module):
    def __init__(self):
        super(GradientLoss, self).__init__()

    def forward(self, a, b):    # a is output, b is target
        xloss = torch.sum(
            torch.abs(torch.abs(a[:, :, 1:, :] - a[:, :, :-1, :]) - torch.abs(b[:, :, 1:, :] - b[:, :, :-1, :])))
        yloss = torch.sum(
            torch.abs(torch.abs(a[:, :, :, 1:] - a[:, :, :, :-1]) - torch.abs(b[:, :, :, 1:] - b[:, :, :, :-1])))
        return (xloss + yloss) / (a.size()[0] * a.size()[1] * a.size()[2] * a.size()[3])



class VggLoss(nn.Module):
    def __init__(self):
        super(VggLoss, self).__init__()

        model = torchvision.models.vgg19(pretrained=True).cuda()

        self.features = nn.Sequential(
            # stop at relu4_4 (-10)
            *list(model.features.children())[:-10]
        )
        # self.mse = nn.MSELoss(reduction='mean')
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, output, target):
        outputFeatures = self.features(output)
        targetFeatures = self.features(target)
        
        loss = (outputFeatures - targetFeatures).abs().mean()

        return loss
        # return config.VGG_FACTOR * loss
        # return loss + reg_loss


class CombinedLoss(nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.vgg = VggLoss()
        self.ssim = SsimLoss()
        self.gd = GradientLoss()

    def forward(self, output, target) -> torch.Tensor:
        return self.vgg(output, target) + self.gd(output, target) + self.ssim(output, target)

class SsimLoss(torch.nn.Module):
    def __init__(self):
        super(SsimLoss, self).__init__()
        # self.alpha_recon_image = 0.8
    def SSIM(self, x, y):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_x = F.avg_pool2d(x, 3, 1)
        mu_y = F.avg_pool2d(y, 3, 1)

        sigma_x = F.avg_pool2d(x ** 2, 3, 1) - mu_x ** 2
        sigma_y = F.avg_pool2d(y ** 2, 3, 1) - mu_y ** 2
        sigma_xy = F.avg_pool2d(x * y, 3, 1) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

        SSIM = SSIM_n / SSIM_d

        return torch.clamp((1 - SSIM) / 2, 0, 1).mean()

    def forward(self, x, y, opt=None):
        sim = 0
        # for ii in range(opt.num_predicted_frames):
        for ii in range(x.size()[1]):
            sim += self.SSIM(x[:, ii, ...], y[:, ii, ...])
        return sim
