import os
import sys
import time
import math
import argparse

import numpy as np
import scipy.misc
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.autograd import Variable

from data_loader import Samplerian, DataLoaderian
from data_utils import get_data
from models import *
from net_utils import *
from cfg import cfg
from cityscape_utils import *
import warnings
warnings.filterwarnings("ignore")

from subprocess import call


def parse_args():
	"""
	Parse input arguments
	"""
	parser = argparse.ArgumentParser(description='Train a segmentation completion network')
	parser.add_argument('--dataset', dest='dataset',
											help='training dataset',
											choices=['cityscape'],
											default='cityscape')
	parser.add_argument('--model', dest='model',
											help='model to use',
											choices=['u_net', 'wgan', 'encoder_decoder'],
											default='u_net')
	parser.add_argument('--load_dir', dest='load_dir',
											help='directory to load models', default="models",
											type=str)
	parser.add_argument('--save_dir', dest='save_dir',
											help='directory to save results', default="results",
											type=str)
	parser.add_argument('--nw', dest='num_workers',
											help='number of worker to load data',
											default=0, type=int)
	parser.add_argument('--cuda', dest='cuda',
											help='whether use CUDA',
											action='store_true')                    
	parser.add_argument('--mGPUs', dest='mGPUs',
											help='whether use multiple GPUs',
											action='store_true')
	parser.add_argument('--bs', dest='batch_size',
											help='batch_size',
											default=1, type=int)

	parser.add_argument('--checksession', dest='checksession',
											help='checksession to load model',
											default=1, type=int)
	parser.add_argument('--checkepoch', dest='checkepoch',
											help='checkepoch to load network',
											default=1, type=int)
	parser.add_argument('--checkpoint', dest='checkpoint',
											help='checkpoint to load network',
											default=1487, type=int)

	args = parser.parse_args()
	return args


if __name__ == '__main__':
	args = parse_args()

	# torch.cuda.set_device(0)

	print('Called with args:')
	print(args)

	####################### set up data loader for train and val ######################
	n_classes, train_imgs, train_segs, train_masks, \
	val_imgs, val_segs, val_masks = get_data(args.dataset, flip=True, val=True)

	val_size = val_imgs.shape[0]
	dataset_val     = DataLoaderian(val_imgs, val_segs, val_masks, n_classes, training=False)
	dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size)

	##########################  check some minor stuff ########################
	if val_size % args.batch_size != 0:
		raise Exception("batch size must be divided by val size ({})".format(val_size))

	###################### set up model #####################
	if args.model == "u_net":
		model = Model(n_classes, "u_net")
	elif args.model == "encoder_decoder":
		model = Model(n_classes, "encoder_decoder")

	# model.create_architecture()

	load_name = os.path.join(args.load_dir,
		'{}_{}_{}_{}.pth'.format(args.model, args.checksession, args.checkepoch, args.checkpoint))

	# print("loading checkpoint %s" % (load_name))
	# checkpoint = torch.load(load_name)
	# model.load_state_dict(checkpoint['model'])
	# print("load checkpoint successfully !")


	###################### init variable #####################
	im_data = torch.FloatTensor(1)
	seg_gt_data = torch.LongTensor(1)
	seg_data = torch.FloatTensor(1)
	mask_data = torch.FloatTensor(1)


	if args.cuda:
		im_data = im_data.cuda()
		seg_data = seg_data.cuda()
		mask_data = mask_data.cuda()
		seg_gt_data = seg_gt_data.cuda()
		model.cuda()

	im_data =Variable(im_data)
	seg_data =Variable(seg_data)
	mask_data =Variable(mask_data)
	seg_gt_data = Variable(seg_gt_data)

	if args.mGPUs:
		model = nn.DataParallel(model)

	#################### minor stuff ########################
	seg_index2color = { v : np.array(list(k)) for k, v in seg_color2index.items()}
	seg_index2color_hash = np.zeros((n_classes, 3))
	for i in range(n_classes):
		seg_index2color_hash[i] = seg_index2color[i]
	save_folder = args.save_dir+"/{}_{}_{}_{}".format(args.model, args.checksession, args.checkepoch, args.checkepoch)
	if not os.path.exists(save_folder):
		os.makedirs(save_folder)

	##################### training begins #######################
	iters_per_epoch = int(val_size / args.batch_size) 

	print("evaluation start")
	sys.stdout.flush()
	# setting to train mode
	model.eval()

	data_iter = iter(dataloader_val)

	img_count = 0
	for step in range(iters_per_epoch):
		data = next(data_iter)
		im_data.data.resize_(data[0].size()).copy_(data[0])
		seg_gt_data.data.resize_(data[1].size()).copy_(data[1])
		mask_data.data.resize_(data[2].size()).copy_(data[2])
		
		
		
		# seg_input = transform_seg_one_hot(seg_gt_data, n_classes)
		seg_input = data[1].unsqueeze(1)
		seg_data.data.resize_(seg_input.size()).copy_(seg_input)

		tic = time.time()
		output_segs, rec_loss = model(im_data, seg_data, mask_data) 
		eval_time = time.time() - tic
		tic = time.time()
		
		# print(output_segs[0,:,200,200])	

		output_segs = torch.argmax(output_segs, dim=1)

		# print(output_segs[0,195:200,195:200])	

		assert list(output_segs.size()) == [args.batch_size, 1024, 2048], output_segs.size()

		output_segs = seg_index2color_hash[output_segs.cpu()].astype(np.uint8)

		# output_segs = np.vectorize(seg_index2color.get)(output_segs.cpu()).astype(np.uint8)
		draw_time = time.time() - tic
		tic = time.time()

		for i in range(args.batch_size):
			save_dir = save_folder + "/{}.png".format(img_count)
			scipy.misc.imsave(save_dir, output_segs[i])
			img_count+=1
		save_time = time.time()-tic
		sys.stdout.write("\rsaving image {}/{} \ttime cost: {:.2f}\teval cost: {:.2f}".format(img_count+1, val_size, eval_time+draw_time+save_time, eval_time))
		sys.stdout.flush()


