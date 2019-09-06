# full assembly of the sub-parts to form the complete net
import torch
import torch.nn as nn
import torch.nn.functional as F

		
class EncoderDecoder(nn.Module):
	def __init__(self, n_channels, n_classes):
		super(EncoderDecoder, self).__init__()
		self.n_classes = n_classes
		self.encoder = nn.Sequential(
			nn.Conv2d(in_channels=n_channels, out_channels=32, kernel_size=5, stride=1, padding=2), # n_channels Y
				nn.ReLU(),
			nn.Conv2d(32, 64, 3, 2, 1),
				nn.ReLU(),
			nn.Conv2d(64, 64, 3, 1, 1),
				nn.ReLU(),
			nn.Conv2d(64, 128, 3, 2, 1),
				nn.ReLU(),
			nn.Conv2d(128, 128, 3, 1, 1),
				nn.ReLU(),
			nn.Conv2d(128, 128, 3, 1, 1),
				nn.ReLU()
		)
		self.dilated_layer = nn.Sequential(
			nn.Conv2d(128, 128, 3, 1, 2, dilation=2),
				nn.ReLU(),
			nn.Conv2d(128, 128, 3, 1, 4, dilation=4),
				nn.ReLU(),
			nn.Conv2d(128, 128, 3, 1, 8, dilation=8),
				nn.ReLU(),
			nn.Conv2d(128, 128, 3, 1, 16, dilation=16),
				nn.ReLU()
		)
		self.bottle_neck = nn.Sequential(
			nn.Conv2d(128, 128, 3, 1, 1),
				nn.ReLU(),
			nn.Conv2d(128, 128, 3, 1, 1),
				nn.ReLU()
		)
		self.decoder = nn.Sequential(
			nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
				nn.Conv2d(128, 128, 3, 1, 1),   nn.ReLU(),
				nn.Conv2d(128, 128, 3, 1, 1),   nn.ReLU(),
			nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
				nn.Conv2d(128, 64, 3, 1, 1),    nn.ReLU(),
				nn.Conv2d(64, 64, 3, 1, 1),     nn.ReLU(),
			nn.Conv2d(64, self.n_classes, 3, 1, 1)
		)


	def forward(self, seg):
		# bs,c,h,w = img.size()
		# x = torch.cat([img, seg], dim=1)


		x = self.encoder(seg)
		x = self.dilated_layer(x)
		x = self.bottle_neck(x)
		x = self.decoder(x)
		
		# x = torch.clamp(x, -1, 1)

		# x = x*(1-mask) + seg
		# assert x.size() == [bs, self.n_classes, h, w], [ img.size(), x.size() ]
		# try my probability map first
		# x = x.view(bs, self.n_classes, h, w)
		# x = F.softmax(x, 1)
		# x = x.view(bs, self.n_classes, h, w)
		return x


	def _init_weights(self):
		def normal_init(m, mean, stddev, truncated=False):
			"""
			weight initalizer: truncated normal and random normal.
			"""
			# x is a parameter
			if truncated:
				m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
			else:
				m.weight.data.normal_(mean, stddev)
				m.bias.data.zero_()
		for i in self.encoder:
			if "conv" in i.__class__.__name__:
				normal_init(i, 0, 0.01)
		for i in self.decoder:
			if "conv" in i.__class__.__name__:
				normal_init(i, 0, 0.01)

