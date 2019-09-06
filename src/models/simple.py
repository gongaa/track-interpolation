import torch
import torch.nn as nn
import torch.nn.functional as F

from .u_net import UNet
from .encoder_decoder import EncoderDecoder
from .net_utils import *

__all__ = ['simple29_unet', 'simple29_encoderdecoder']

class Simple(nn.Module):
	'''
	'''
	def __init__(self, n_classes, embedding_dim, model_name="u_net"):
		super(Simple, self).__init__()
		self.n_classes = n_classes
		self.reconst_loss = None 
		self.model_name = model_name
		self.weight = torch.tensor([0.19, 0.45, 0.29, 0.13, 0.2, 0.33, 0.48, 0.14, 0.36, 0.34, 1.0, 0.43, 0.66, 0.33, 0.51, 0.41, 0.17, 0.31, 0.19, 0.33, 0.57, 0.21, 0.48, 0.49, 0.75, 0.88, 0.49, 0.61, 0.42]).cuda()
		if embedding_dim is None:
			self.embedding = lambda x : x
		else:
			self.embedding = torch.nn.Embedding(num_embeddings=30, embedding_dim=embedding_dim) # 29+1(cropped)
		if self.model_name == "u_net":
			self.layer = UNet(3, n_classes)
		elif self.model_name == "encoder_decoder":
			self.layer = EncoderDecoder(n_channels=embedding_dim, n_classes=n_classes)

	def forward(self, mask, onehot, img=None, seg_gt=None):
		# assert img.size(1) == 3, img.size()
		# mask (N,H,W), seg_gt (N,H,W), onehot (N,cls)
		# assert len(seg.size()) == 4, seg.size()
		# assert len(mask.size()) == 3, mask.size()
		num_cls = onehot.size(-1)
		assert num_cls == 29, 'number of class not equal to onehot last dimension'
		# seg = torch.zeros(seg_gt.size())
		# seg.copy_(seg_gt)
		# seg = seg.long().cuda()
		with torch.no_grad():
			seg = seg_gt.clone().long()
			seg[mask] = self.n_classes	# set where mask=1(cropped) to indice num_cls=29
		x2 = self.embedding(seg)		# x2 in shape (N,H,W,embedding_dim)
		# x2.transpose_(1,3).transpose_(2,3)		# x2 in shape (N,embedding_dim, H,W)
		x2 = x2.permute(0, 3, 1, 2).contiguous()
		x3 = self.layer(x2)			# x3 in shape (N,num_class,H,W)

		# print(output[0,:,200,200])
		assert not mask.requires_grad
		assert not seg_gt.requires_grad
		seg_gt = seg_gt.long()
		seg_one_hot = torch.eye(self.n_classes)[seg_gt].permute(0,3,1,2).cuda()
		assert not seg_one_hot.requires_grad
		output = x3*mask.unsqueeze(1).float() + seg_one_hot #transform_seg_one_hot(seg_gt, self.n_classes)*mask

		if self.training:
			self.reconst_loss = F.cross_entropy(input=output, weight=self.weight, target=seg_gt, reduction='sum')
			# target of shape (N,H,W), where each element in [0,C-1], input of shape (N,C,H,W)
			elems = (1-mask).nonzero().size(0)
			self.reconst_loss = self.reconst_loss / elems

		# return squeeze_seg(output, self.n_classes), self.reconst_loss
		return output, self.reconst_loss


def simple29_unet(embedding_dim=15):
	return Simple(29, embedding_dim=embedding_dim, model_name='u_net')

def simple29_encoderdecoder(embedding_dim=15):
	return Simple(29, embedding_dim=embedding_dim, model_name='encoder_decoder')
