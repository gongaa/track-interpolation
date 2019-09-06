import argparse
import os
import torch

class InterpolationOptions():
	def __init__(self):
		self.parser = argparse.ArgumentParser()
		self.initialized = False
    
	def initialize(self):                
		self.parser.add_argument('--dataset', dest='dataset',
												default='cityscape',
												help='training dataset', 
												choices=['cityscape', 'ucf101','vimeo'])
		self.parser.add_argument('--split', dest='split', 
												help='whether eval after each training ', 
												default='train',
												choices=['train','val','test','cycgen', 'mycycgen'])
		self.parser.add_argument('--img_dir', dest='img_dir',
												help='directory to load models', default=None,
												type=str)
		self.parser.add_argument('--seg_dir', dest='seg_dir',
												help='directory to load models', default=None,
												type=str)
		self.parser.add_argument('--cycgen_load_dir', dest='cycgen_load_dir',
												help='directory to load cycgen inputs', default=None,
												type=str)

		self.parser.add_argument('--input_h',
										default=128,
										type=int,
										help='input image size')
		self.parser.add_argument('--input_w',
										default=256,
										type=int,
										help='input image size')	

		self.parser.add_argument('--fast_train', dest='fast_train',
												help='whether eval after each training ', 
												action='store_true')		
		self.parser.add_argument('--fast_input_h',
										default=64,
										type=int,
										help='input image size')
		self.parser.add_argument('--fast_input_w',
										default=64,
										type=int,
										help='input image size')	


		self.parser.add_argument('--syn_type', dest='syn_type',
												help='synthesize method',
												choices=['inter', 'extra'],
												default='inter') 
		self.parser.add_argument('--mode', dest='mode',
												help='mode to use',
												choices=['xs2xs', 'xx2x'],
												default='xs2xs')
		self.parser.add_argument('--bs', dest='batch_size', 
												type=int,
												default=1, 
												help='Batch size (over multiple gpu)')
		self.parser.add_argument('--epochs', dest='epochs', 
												type=int,
												default=30, 
												help='Number of training epochs')      
		self.parser.add_argument('--interval', dest='interval',
												help='training optimizer loss weigh of feat',
												type=float,
												default=9)      # given frame 0, 18, predict 9
		# distributed training
		self.parser.add_argument('--nw',  dest='num_workers', 
												type=int, default=4,
												help='Number of data loading workers')
		self.parser.add_argument('--port', dest='port',
												type=int, default=None, 
												help='Port for distributed training')
		self.parser.add_argument('--seed', type=int,
												default=1024, help='Random seed')

		self.parser.add_argument('--start_epoch', dest='start_epoch',
												help='starting epoch',
												default=0, type=int)
		self.parser.add_argument('--disp_interval', dest='disp_interval',
												help='number of iterations to display',
												default=100, type=int)
		# config optimization
		self.parser.add_argument('--lr_decay_step', dest='lr_decay_step', 
												help='step to do learning rate decay, unit is epoch',
												default=5, type=int)
		self.parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
												help='learning rate decay ratio', 
												default=1, type=float)
		self.parser.add_argument('--save_dir', dest='save_dir',
												help='directory to load models', default="log",
												type=str)
		self.parser.add_argument('--one_hot_seg', dest='one_hot_seg',
												help='whether segmenatation use one hot label', 
												action='store_true')
		self.parser.add_argument('--ef', dest='effec_flow',
												help='effective flow', 
												action='store_true')

		self.parser.add_argument('--norm', dest='norm',
												help='whether normalize', 
												action='store_true')
		self.parser.add_argument('--s', dest='session',
												help='training session',
												default=0, type=int)

		self.parser.add_argument('--r', dest='resume',
												help='whether resume ', 
												action='store_true')
		self.parser.add_argument('--checksession', dest='checksession',
												help='checksession to load model',
												default=1, type=int)
		self.parser.add_argument('--checkepoch', dest='checkepoch',
												help='checkepoch to load model',
												default=0, type=int)
		self.parser.add_argument('--checkepoch_range', dest='checkepoch_range',
												help='whether eval multiple epochs', 
												action='store_true')
		self.parser.add_argument('--checkepoch_low', dest='checkepoch_low',
												help='when checkepoch_range is true, inclusive starting epoch',
												default=1, type=int)    
		self.parser.add_argument('--checkepoch_up', dest='checkepoch_up',
												help='when checkepoch_range is true, inclusive ending epoch',
												default=20, type=int)
		self.parser.add_argument('--checkpoint', dest='checkpoint',
												help='checkpoint to load model',
												default=0, type=int)
		self.parser.add_argument('--load_dir', dest='load_dir',
												help='directory to load models', default="models",
												type=str)
		# weight of losses
		self.parser.add_argument('--l1_w', dest='l1_weight',
												help='training optimizer loss weigh of l1',
												type=float,
												default=80)
		self.parser.add_argument('--vgg_w', dest='vgg_weight',
												help='training optimizer loss weigh of vgg',
												type=float,
												default=20)
		self.parser.add_argument('--ce_w', dest='ce_weight',
												help='training optimizer loss weigh of vgg',
												type=float,
												default=30)     
		# weight of obj losses
		self.parser.add_argument('--obj_l1_w', dest='obj_l1_weight',
												help='training optimizer loss weigh of l1',
												type=float,
												default=80)

		self.parser.add_argument('--vid_len', dest='vid_length', 
												type=int,
												default=1, 
												help='predicted video length')
		self.parser.add_argument('--n_track', dest='num_track_per_img', 
												type=int,
												default=4, 
												help='predicted video length')
		
		self.parser.add_argument('--low_res', dest='low_res',
										help='whether load coarse model ', 
										action='store_true')

		self.parser.add_argument('--highres_large', dest='highres_large',
										help='whether use four levels of hrnet ', 
										action='store_true')

		self.parser.add_argument('--model', dest='model', 
										default='InterGANNet', 
										help='model to use',
										choices=['InterNet', 'InterRefineNet', 'InterStage3Net', 'InterGANNet', 'InterTrackNet'])  
		self.parser.add_argument('--load_model', dest='load_model', 
										default='InterGANNet', 
										help='model to use',
										choices=['InterNet', 'InterRefineNet', 'InterStage3Net', 'InterGANNet', 'InterTrackNet'])  
		self.parser.add_argument('--n_sc', dest='n_scales', 
										help='scales of output',
										default=1, type=int)	


		### coarse model settings ###
		self.parser.add_argument('--coarse_model', dest='coarse_model', 
										default='HRNet', 
										help='model to use',
										choices=['HRNet', 'VAEHRNet', 'TrackGenV4Test'])  					  
		self.parser.add_argument('--frame_global_disc_model', dest='frame_global_disc_model', 
										default='FrameGlobalDiscriminator', 
										help='model to use',
										choices=['FrameGlobalDiscriminator'])  					  
		self.parser.add_argument('--ins_global_disc_model', dest='ins_global_disc_model', 
										default='InstanceSNDiscriminator', 
										help='model to use',
										choices=['InstanceSNDiscriminator'])  					  
		self.parser.add_argument('--ins_video_disc_model', dest='ins_videl_disc_model', 
										default='VideoSNDiscriminator', 
										help='model to use',
										choices=['VideoSNDiscriminator'])  					  
		self.parser.add_argument('--pretrained_coarse', dest='pretrained_coarse',
								help='whether train coarse model ', 
								action='store_true')
		self.parser.add_argument('--pretrained_coarse_model', dest='pretrained_coarse_model',
										help='directory to load models', default="log",
										type=str)
		self.parser.add_argument('--coarse_o', dest='coarse_optimizer', 
										help='training coarse optimizer',
										choices =['adamax','adam', 'sgd'], 
										default="adamax")
		self.parser.add_argument('--coarse_lr', dest='coarse_learning_rate', 
										help='coarse learning rate',
										default=0.001, type=float)	
		self.parser.add_argument('--disc_frame_lr', dest='frame_global_disc_learning_rate', 
										help='coarse learning rate',
										default=0.001, type=float)	
		self.parser.add_argument('--disc_ins_lr', dest='ins_global_disc_learning_rate', 
										help='coarse learning rate',
										default=0.001, type=float)	
		self.parser.add_argument('--load_coarse', dest='load_coarse',
												help='whether load coarse model ', 
												action='store_true')
		self.parser.add_argument('--train_coarse', dest='train_coarse',
												help='whether train coarse model ', 
												action='store_false')


		self.initialized = True


	def parse(self, save=True):
		if not self.initialized:
			self.initialize()
		self.opt = self.parser.parse_args()
		return self.opt