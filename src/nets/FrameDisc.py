import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
import numpy as np
from nets.SpectralNorm import SpectralNorm

class ResnetBlock(nn.Module):
	def __init__(self, in_dim, out_dim, ks):
		super(ResnetBlock, self).__init__()
		self.conv = nn.Sequential(
				nn.Conv2d(in_dim, out_dim, ks, stride=1, padding=ks//2),
				nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(out_dim, out_dim, ks, stride=1, padding=ks//2)
			)
	
	def forward(self, input):
		conv_out = self.conv(input)
		return  conv_out + input

class ResnetSNBlock(nn.Module):
	def __init__(self, in_dim, out_dim, ks):
		super(ResnetSNBlock, self).__init__()
		self.conv = nn.Sequential(
				SpectralNorm(nn.Conv2d(in_dim, out_dim, ks, stride=1, padding=ks//2)),
				nn.LeakyReLU(0.2, inplace=True),
				SpectralNorm(nn.Conv2d(out_dim, out_dim, ks, stride=1, padding=ks//2))
			)
	
	def forward(self, input):
		conv_out = self.conv(input)
		return  conv_out + input


class FrameGlobalDiscriminator(nn.Module):
	def __init__(self, args):
		super(FrameGlobalDiscriminator, self).__init__()
		self.args=args
		self.w = args.input_w
		self.h = args.input_h
		self.input_dim = 3 
		self.layer = nn.Sequential(
				nn.Conv2d(self.input_dim, 16, 3, 1, 1),          # 16,w,h
				nn.LeakyReLU(0.2,inplace=False),

				nn.Conv2d(16, 32, 5, 1, 2),						 # 32,w,h
				nn.BatchNorm2d(32),
				nn.LeakyReLU(0.2,inplace=False),

				# downsize 1 64*64*64
				nn.Conv2d(32, 64, 3, 2, 1),						 # 64,w//2,h//2
				nn.LeakyReLU(0.2, inplace=True),
				ResnetBlock(64, 64, 3),							
				# downsize 2 96*32*32
				nn.Conv2d(64, 96, 3, 2, 1),						 # 96,w//4,h//4
				nn.LeakyReLU(0.2, inplace=True),
				ResnetBlock(96, 96, 3),
				# downsize 3 128*16*16
				nn.Conv2d(96, 128, 3, 2, 1),					 # 128,w//8,h//8
				nn.LeakyReLU(0.2, inplace=True),
				ResnetBlock(128, 128, 3),
				# downsize 4 192*8*8
				nn.Conv2d(128, 192, 3, 2, 1),					 # 192,w//16,h//16
				nn.LeakyReLU(0.2, inplace=True),
				ResnetBlock(192, 192, 3),
				# out layer
			)

		self.linear = nn.Linear(192*self.h*self.w//256, 1)
		self.sigmoid = nn.Sigmoid()

	def forward(self, x):
		input = x
		bs = input.size()[0]
		output = self.layer(input)
		output = self.linear(output.view(bs, -1))
		return self.sigmoid(output)
	

class InstanceSNDiscriminator(nn.Module):
	def __init__(self, args):
		super(InstanceSNDiscriminator, self).__init__()
		self.args=args
		self.w = args.fast_input_w
		self.h = args.fast_input_h
		self.input_dim = 3
		self.layer = nn.Sequential(
				SpectralNorm(nn.Conv2d(self.input_dim, 16, 3, 1, 1)),  # 16,w,h
				nn.LeakyReLU(0.2,inplace=False),

				SpectralNorm(nn.Conv2d(16, 32, 5, 1, 2)),			   # 32,w,h
				nn.LeakyReLU(0.2,inplace=False),

				# downsize 1 64*64*64
				SpectralNorm(nn.Conv2d(32, 64, 3, 2, 1)),			   # 64,w//2,h//2
				nn.LeakyReLU(0.2, inplace=True),
				ResnetSNBlock(64, 64, 3),
				# downsize 2 96*32*32
				SpectralNorm(nn.Conv2d(64, 96, 3, 2, 1)),			   # 96,w//4,h//4
				nn.LeakyReLU(0.2, inplace=True),
				ResnetSNBlock(96, 96, 3),
				# downsize 3 128*16*16
				SpectralNorm(nn.Conv2d(96, 128, 3, 2, 1)),			   # 128,w//8,h//8
				nn.LeakyReLU(0.2, inplace=True),
				ResnetSNBlock(128, 128, 3),
			)
		self.linear = nn.Linear(128*self.h*self.w//64, 1)
		self.sigmoid = nn.Sigmoid()

	def forward(self, x):
		input = x
		bs = input.size()[0]
		output = self.layer(input)
		output = self.linear(output.view(bs, -1))
		return self.sigmoid(output)


class VideoSNDiscriminator(nn.Module):
	def __init__(self, args):
		super(VideoSNDiscriminator, self).__init__()
		self.args=args
		self.w = args.fast_input_w
		self.h = args.fast_input_h
		self.input_dim = 3
		self.layer = nn.Sequential(
				SpectralNorm(nn.Conv2d(3*self.input_dim, 32, 3, 1, 1)),
				nn.LeakyReLU(0.2,inplace=False),

				SpectralNorm(nn.Conv2d(32, 64, 5, 1, 2)),        			# 64,w,h
				nn.LeakyReLU(0.2,inplace=False),
				SpectralNorm(nn.Conv2d(64, 32, 3, 1, 1)),					# 32,w,h
				nn.LeakyReLU(0.2,inplace=False),

				# downsize 1 32*64*64
				SpectralNorm(nn.Conv2d(32, 32, 3, 2, 1)),					# 32,w//2,h//2
				nn.LeakyReLU(0.2, inplace=True),
				ResnetSNBlock(32, 32, 3),
				# ResnetBlock(32, 32, 3),

				# downsize 2 64*32*32
				SpectralNorm(nn.Conv2d(32, 64, 3, 2, 1)),					# 64,w//4,h//4
				nn.LeakyReLU(0.2, inplace=True),
				ResnetSNBlock(64, 64, 3),
				# downsize 3 128*16*16
				SpectralNorm(nn.Conv2d(64, 128, 3, 2, 1)),					# 128,w//8,h//8
				nn.LeakyReLU(0.2, inplace=True),
				ResnetSNBlock(128, 128, 3),
			)
		self.linear = nn.Linear(128*self.h*self.w//64, 1)
		self.sigmoid = nn.Sigmoid()

	def forward(self, middle, first, last):
		input = torch.cat([first, middle, last], dim=1)
		bs = input.size()[0]
		output = self.layer(input)
		output = self.linear(output.view(bs, -1))
		return self.sigmoid(output)

# class FrameLocalDiscriminator(nn.Module):
# 	def __init__(self, args):
# 		super(FrameLocalDiscriminator, self).__init__()
# 		self.output_shape = (1024,)
# 		self.img_c = 3
# 		self.img_h = args.input_h
# 		self.img_w = args.input_w
# 		self.layer = nn.Sequential(
# 			# input_shape: (None, img_c, img_h, img_w)
# 			nn.Conv2d(self.img_c, 16, 5, 2, 2),
# 			nn.BatchNorm2d(16),
# 			nn.LeakyReLU(0.2, inplace=False),
# 			# input_shape: (None, 16, img_h//2, img_w//2)

# 			nn.Conv2d(16, 32, 5, 2, 2),
# 			nn.BatchNorm2d(32),
# 			nn.LeakyReLU(0.2, inplace=False),
# 			# input_shape: (None, 32, img_h//4, img_w//4)

# 			nn.Conv2d(32, 64, 5, 2, 2),
# 			nn.BatchNorm2d(64),
# 			nn.LeakyReLU(0.2, inplace=False),
# 			# input_shape: (None, 64, img_h//8, img_w//8)

# 			nn.Conv2d(64, 128, 5, 2, 2),
# 			nn.BatchNorm2d(128),
# 			nn.LeakyReLU(0.2, inplace=False),
# 			# input_shape: (None, 128, img_h//16, img_w//16)

# 			nn.Conv2d(128, 128, 5, 2, 2),
# 			nn.BatchNorm2d(128),
# 			nn.LeakyReLU(0.2, inplace=False),
# 			# input_shape: (None, 128, img_h//32, img_w//32)
# 		)
	
# 		in_features = 128 * (self.img_h//32) * (self.img_w//32)
# 		self.linear = nn.Linear(in_features, 256)
# 		self.act = nn.ReLU()
# 		# output_shape: (None, 256)

# 	def forward(self, x):
# 		x = self.layer(x)
# 		x = self.linear(x.view(-1))
# 		return x


# class FrameContextDiscriminator(nn.Module):
# 	def __init__(self, args):
# 		super(FrameContextDiscriminator, self).__init__()
# 		self.arc = arc
# 		self.output_shape = (1,)
# 		self.model_ld = FrameLocalDiscriminator(args)
# 		self.model_gd = FrameGlobalDiscriminator(args)
# 		# input_shape: [(None, 1024), (None, 1024)]
# 		in_features = self.model_ld.output_shape[-1] + self.model_gd.output_shape[-1]
# 		# input_shape: (None, 2048)
# 		self.linear = nn.Linear(in_features, 1)
# 		self.act = nn.Sigmoid()
# 		# output_shape: (None, 1)

# 	def forward(self, x):
# 		x_ld, x_gd = x
# 		x_ld = self.model_ld(x_ld)
# 		x_gd = self.model_gd(x_gd)
# 		out = self.act(self.linear(torch.cat([x_ld, x_gd], dim=1)))
# 		return out

# class InstanceSNLocalDiscriminator(nn.Module):
# 	def __init__(self, args):
# 		super(InstanceSNLocalDiscriminator, self).__init__()
# 		self.args=args
# 		self.input_dim = 23 if self.args.seg_disc else 3
# 		self.layer = nn.Sequential(
# 				SpectralNorm(nn.Conv2d(self.input_dim, 16, 3, 1, 1)),			# 1,3
# 				nn.LeakyReLU(0.2,inplace=False),
# 				SpectralNorm(nn.Conv2d(16, 32, 5, 1, 2)),						# 1,7
# 				nn.LeakyReLU(0.2,inplace=False),

# 				# downsize 1  64*64*64
# 				SpectralNorm(nn.Conv2d(32, 64, 3, 2, 1)),			# 2,9
# 				nn.LeakyReLU(0.2, inplace=True),
# 				SpectralNorm(nn.Conv2d(64, 64, 3, 1, 1)),			# 2,13
# 				nn.LeakyReLU(0.2, inplace=True),
# 				# downsize 2 128*32*32
# 				SpectralNorm(nn.Conv2d(64, 128, 3, 2, 1)),		# 4,17
# 				nn.LeakyReLU(0.2, inplace=True),
# 				SpectralNorm(nn.Conv2d(128, 128, 3, 1, 1)),		# 4,25
# 				nn.LeakyReLU(0.2, inplace=True),
# 				SpectralNorm(nn.Conv2d(128, 64, 3, 1, 1)),		# 4,33
# 				nn.LeakyReLU(0.2, inplace=True),
# 				SpectralNorm(nn.Conv2d(64,  1, 1, 1, 0))			# 8,33
# 			)

# 	def forward(self, x, seg, bboxes=None):
# 		input = x
# 		if self.args.seg_disc:
# 			input = torch.cat([x, seg], dim=1)
# 		output = self.layer(input)
# 		return output
