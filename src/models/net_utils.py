import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F


def adjust_learning_rate(optimizer, decay=0.1):
    """Sets the learning rate to the initial LR decayed by 0.5 every 20 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = decay * param_group['lr']

def transform_seg_one_hot(seg, n_cls):
	'''
		input tensor:
			seg  (bs, h, w) Long Tensors
			n_clss
		output tensor
			seg_one_hot  (bs, n_cls, h, w) float tensor

	'''
	seg_one_hot = torch.eye(n_cls)[seg.long()].permute(0,3,1,2).cuda()
	return seg_one_hot


def mask2box(mask):
	'''
		input: mask of shape (bs, h, w) outer region is 1
		output: bbox (bs, 4) of (h1, w1, h2, w2) 
	'''
	inner_mask = (1-mask)
	bs = mask.size(0)
	output = []
	for i in range(bs):
		nonzero_indices = torch.nonzero(inner_mask[i]) # inner region indices of shape (k, 2) 
		min_hw,_ = torch.min(nonzero_indices, dim=0)
		max_hw,_ = torch.max(nonzero_indices, dim=0)
		i_hw = torch.cat([min_hw.view(1,2), max_hw.view(1,2)], dim=1)
		output.append(i_hw)
	return torch.cat(output, dim=0)


