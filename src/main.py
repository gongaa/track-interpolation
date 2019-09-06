import argparse
import datetime
import logging
import pathlib
import random
import socket
import sys
import pickle

import torch
import torch.distributed as dist 
import torch.multiprocessing as mp 
import numpy as np 

from utils import *
from HRNetTrainer import HRNetTrainer
from options.InterpolationOptions import InterpolationOptions
from subprocess import call

from time import time

import warnings
warnings.filterwarnings("ignore")

from trainer import Trainer 

def get_exp_path():
	'''Retrun new experiment path.'''
	return '../log/exp-{0}'.format(
		datetime.datetime.now().strftime('%m-%d-%H:%M:%S'))


def get_logger(path, rank=None):
	'''Get logger for experiment.'''
	logger = logging.getLogger(__name__)
	logger.setLevel(logging.DEBUG)

	if rank is None:
		formatter = logging.Formatter('%(asctime)s-%(message)s')
	else:
		formatter = logging.Formatter('%(asctime)s - [worker '
			+ str(rank) +'] - %(message)s')
	
	# stderr log
	handler = logging.StreamHandler(sys.stderr)
	handler.setLevel(logging.DEBUG)
	handler.setFormatter(formatter)
	logger.addHandler(handler)

	# file log
	handler = logging.FileHandler(path)
	handler.setLevel(logging.DEBUG)
	handler.setFormatter(formatter)
	logger.addHandler(handler)

	return logger


def worker(rank, args):
	logger = get_logger(args.path + '/experiment.log',
						rank) # process specific logger
	args.logger = logger
	args.rank = rank
	dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:%d' % args.port,
		world_size=args.gpus, rank=args.rank)

	# seed
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	torch.cuda.manual_seed_all(args.seed)

	trainer = HRNetTrainer(args)

	if args.split=='val':
  		trainer.validate()
	else:
		if args.rank == 0:
			pathlib.Path('../checkpoint').mkdir(
					parents=True, exist_ok=False)

		for epoch in range(args.start_epoch, args.epochs):
			trainer.set_epoch(epoch)
			trainer.train()
			# metrics = trainer.validate()

			if args.rank == 0:	# gpu id
				trainer.save_checkpoint()
				# trainer.save_checkpoint(metrics)


def main():
	parser = InterpolationOptions()
	args = parser.parse()
	
	# exp path
	# if args.path is None:
	args.path = get_exp_path()
	pathlib.Path(args.path).mkdir(parents=True, exist_ok=False)
	(pathlib.Path(args.path) / 'checkpoint').mkdir(parents=True, exist_ok=False)

	# find free port
	if args.port is None:
		with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
			s.bind(('', 0))
			s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
			args.port = int(s.getsockname()[1])

	# logger
	logger = get_logger(args.path + '/experiment.log')
	logger.info('Start of experiment')
	logger.info('=========== Initilized logger =============')
	logger.info('\n\t' + '\n\t'.join('%s: %s' % (k, str(v))
		for k, v in sorted(dict(vars(args)).items())))
	
	# distributed training
	args.gpus = torch.cuda.device_count()
	logger.info('Total number of gpus: %d' % args.gpus)
	mp.spawn(worker, args=(args,), nprocs=args.gpus)

if __name__ == '__main__':
	main()
