import torch.utils.data as data
from PIL import Image
import cv2
import os
import os.path
import sys
from random import randint
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
import math

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', 'webp']

def pil_loader_rgb(path):
	# open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
	with open(path, 'rb') as f:
		img = Image.open(f)
		return img.convert('RGB')

def pil_loader_seg(path):
	# open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
	with open(path, 'rb') as f:
		img = Image.open(f)
		return img.convert('L')


def rgb_load(img_list):
	return [pil_loader_rgb(i) for i in img_list]

def seg_load(seg_list):
	l = []
	for i in seg_list:
		seg_img = pil_loader_seg(i)
		l.append(seg_img)
	return l

class Transform(object):
	def __init__(self):
		super(Transform, self).__init__()
		# self.transforms = tf.transforms
		self.crop_params = None

	def __call__(self, image, record=False, scale=1, tf=None, pre_crop_params=None):
		'''
			pre_crop_params: (h_min, w_min, height, width)
		'''
		self.transforms = tf.transforms
		if record:
			assert self.crop_params is None
			assert scale == 1
		for t in self.transforms:
			if isinstance(t, transforms.RandomCrop):
				assert not (pre_crop_params is not None and record), 'pre_crop_params and record can not be both used'
				if pre_crop_params is None:
					if record:
						self.crop_params = t.get_params(image, output_size=t.size)
					crop_params = ( int(scale*crop_param) for crop_param in self.crop_params )  
				else:
					crop_params = pre_crop_params
				image = transforms.functional.crop(image, *crop_params)
			else:
				image = t(image)
		return image

	def derecord(self):
		self.crop_params = None

	def __repr__(self):
		format_string = self.__class__.__name__ + '('
		for t in self.transforms:
			format_string += '\n'
			format_string += '    {0}'.format(t)
		format_string += '\n)'
		return format_string

class DatasetFolder(data.Dataset):
	def __init__(self, args, root, transform=None, target_transform=None, bboxes=None):
		self.root = root
		self.tfs = transform
		self.transform = Transform()
		self.target_transform = target_transform
		self.args = args
		self.img_height = 128 #if self.args.split == 'train' else 256
		self.img_width  = 256 #if self.args.split == 'train' else 512
		# self.mean = (0.287, 0.3253, 0.284)
		# self.std = (0.1792, 0.18213, 0.1799898)
		self.mean = (0.5, 0.5, 0.5)
		self.std = (0.5, 0.5, 0.5)
		if self.args.dataset=='cityscape':
			if self.args.split == 'train':
				self.img_dir = args.img_dir if args.img_dir is not None else "/data/linz/proj/Dataset/Cityscape/leftImg_sequence/leftImg8bit_sequence_128x256/"   # 256x512
				self.seg_dir = args.seg_dir if args.seg_dir is not None else "/data/linz/proj/Dataset/Cityscape/leftImg_sequence/gtFine_myseg_id_sequence_128x256/"  # 256x512
				self.bboxes = self.convert(bboxes)
			elif self.args.split == 'val':
				self.img_dir = args.img_dir if args.img_dir is not None else "/data/linz/proj/Dataset/Cityscape/leftImg_sequence/leftImg8bit_sequence_256x512/"
				self.seg_dir = args.seg_dir if args.seg_dir is not None else "/data/linz/proj/Dataset/Cityscape/leftImg_sequence/gtFine_myseg_id_sequence_256x512/"
				if bboxes is not None:
					self.bboxes = self.convert(bboxes)
			self.img_ext = "_leftImg8bit.png"
			self.seg_ext = "_gtFine_myseg_id.png"
			self.edge_dir = "/data/linz/proj/Dataset/Cityscape/leftImg_sequence/edges256/"
			self.edge_ext = "_edge.png"
			self.disparity_dir = '/data/linz/proj/Dataset/Cityscape/disparity_sequence/'
			self.disparity_ext = '_disparity.png'

		self.n_classes = 20
		self.vid_len = len(self.root[0])

	# example of clips_boxes[0]:
	# [[[0.005786895751953125, 65, 208, 147, 245], [0.01348114013671875, 325, 201, 439, 263], [0.01348114013671875, 325, 201, 439, 263], [0.005786895751953125, 65, 208, 147, 245]],
	#  [[0.005786895751953125, 87, 210, 169, 247], [0.0121307373046875, 326, 203, 432, 263], [0.0121307373046875, 326, 203, 432, 263], [0.005786895751953125, 87, 210, 169, 247]],
	#  [[0.005123138427734375, 81, 213, 160, 247], [0.0107879638671875, 305, 210, 406, 266], [0.0107879638671875, 305, 210, 406, 266], [0.005123138427734375, 81, 213, 160, 247]]]
	def convert(self,clips_boxes):
		img_width = self.img_width
		img_height = self.img_height
		for i, clip_boxes in enumerate(clips_boxes):
			for j,frame_boxes in enumerate(clip_boxes):
				for k,frame_box in enumerate(frame_boxes):
					if frame_box is not None:
						clips_boxes[i][j][k][1] = math.floor(clips_boxes[i][j][k][1]*img_width/1024.) # x1
						clips_boxes[i][j][k][2] = math.floor(clips_boxes[i][j][k][2]*img_height/512.)  # y1 	
						clips_boxes[i][j][k][3] = math.floor(clips_boxes[i][j][k][3]*img_width/1024.) # x2
						clips_boxes[i][j][k][4] = math.floor(clips_boxes[i][j][k][4]*img_height/512.)  # y2
						clips_boxes[i][j][k][2], clips_boxes[i][j][k][1] =  clips_boxes[i][j][k][1], clips_boxes[i][j][k][2]
						clips_boxes[i][j][k][4], clips_boxes[i][j][k][3] =  clips_boxes[i][j][k][3], clips_boxes[i][j][k][4] # y1, x1, y2, x2
						if clips_boxes[i][j][k][3] <= clips_boxes[i][j][k][1] or clips_boxes[i][j][k][4] <= clips_boxes[i][j][k][2]:
							clips_boxes[i][j][k] = None
				assert len(frame_boxes) == self.args.num_track_per_img, len(frame_boxes)
		return clips_boxes   # in [y1, x1, y2, x2], where y<=128, x<=256
	

	def __getitem__(self, index):
		if self.args.dataset != 'vimeo':
			img_files     = [self.img_dir+self.root[index][i]+self.img_ext for i in range(self.vid_len)]
		else:
			assert self.args.mode == 'xx2x'
			img_files = [self.img_dir+self.root[index]+ "/im{}".format(i+1) + self.img_ext for i in range(self.vid_len)]

		ori_imgs = rgb_load(img_files)

		if self.args.dataset == 'cityscape':
			seg_files = [self.seg_dir+self.root[index][i]+self.seg_ext for i in range(self.vid_len)]
			seg_imgs = seg_load(seg_files)
			seg_imgs_ori = []
		try:
			clip_boxes = self.bboxes[index].copy()
		except:
			clip_boxes = torch.zeros(3,self.args.num_track_per_img,4)
		isHorflip = randint(0,2)
		# isVerflip = randint(0,2)
		# isReverse = randint(0,2)
		if isHorflip and self.args.split == 'train':
			ori_imgs = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in ori_imgs]
			seg_imgs = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in seg_imgs]
			assert len(clip_boxes) == 3
			assert len(clip_boxes[0]) == len(clip_boxes[1]) and len(clip_boxes[1]) == len(clip_boxes[2])
			for i,frame_boxes in enumerate(clip_boxes):
				for j, frame_box in enumerate(frame_boxes):
					if frame_box is not None:
						t = clip_boxes[i][j][2]
						clip_boxes[i][j][2] = self.img_width-1-clip_boxes[i][j][4]
						clip_boxes[i][j][4] = self.img_width-1-t
						# [y1, x1, y2, x2], where y<=128, x<=256
						# becomes [y1, 255-x2, y2, 255-x1]

		# sample crop images
		if self.args.split == 'train':
			# get rid of fast train
			for i in range(self.vid_len):
				ori_imgs[i] = transforms.functional.normalize( 
										transforms.functional.to_tensor( ori_imgs[i]      # no cropping now
												# self.transform(ori_imgs[i], tf=self.tfs[0], pre_crop_params=seq_crop_params[i])
										),  mean=self.mean, std=self.std
									)
				if self.args.dataset == 'cityscape':
					np_seg = np.array(  seg_imgs[i]     
										#self.transform(seg_imgs[i], scale=1,\
										#  tf=self.tfs[1], pre_crop_params=seq_crop_params[i])
										)
					# np_seg = np.eye(20)[np_seg]
					# seg_imgs[i] = torch.from_numpy(np.transpose(np_seg, (2,0,1))).float() 
					seg_imgs[i] = torch.from_numpy(np_seg).float().unsqueeze_(0)
				
		else: # validation
			# seq_crop_params = [(0,0,self.args.input_h,self.args.input_w)]*3
			for i in range(self.vid_len):
				ori_imgs[i] = transforms.functional.normalize( 
										transforms.functional.to_tensor( ori_imgs[1]
												# self.transform(ori_imgs[i], record=(i==0), tf=self.tfs[0])
										),  mean=self.mean, std=self.std
									)
				if self.args.dataset == 'cityscape':
					np_seg = np.array(self.transform(seg_imgs[i], scale= 1,\
										 tf=self.tfs[1]))
					# np_seg = np.eye(20)[np_seg]
					# seg_imgs[i] = torch.from_numpy(np.transpose(np_seg, (2,0,1))).float() #* 2 - 1
					seg_imgs[i] = torch.from_numpy(np_seg)/float().unsqueeze_(0)


		assert 	len(clip_boxes[0]) == self.args.num_track_per_img and \
				len(clip_boxes[1]) == self.args.num_track_per_img and \
				len(clip_boxes[2]) == self.args.num_track_per_img, [len(clip_boxes[0]),len(clip_boxes[1]),len(clip_boxes[2])]

		
		if len(clip_boxes[1]) < self.args.num_track_per_img:   # repeat instance bbox to ensure 4 instance per triplet
			existed_track_num = len(clip_boxes[1])
			while len(clip_boxes[1]) < self.args.num_track_per_img:
				rand_ind = np.random.randint(existed_track_num)
				for i in range(3):
					clip_boxes[i].append(clip_boxes[i][rand_ind].copy())


		for j in range(self.args.num_track_per_img):
			for i in range(3):
				bbox = clip_boxes[i][j] 
				assert  bbox[3] >= bbox[1] and bbox[4] >= bbox[2], clip_boxes
	
		clip_boxes = torch.tensor(np.array(clip_boxes)).float() # (3, 10, 4)
		self.transform.derecord()
		if self.args.split == 'train':
			if not self.check_clip_boxes(clip_boxes):
				print('meet none_have bboxes')
				return self.__getitem__(index+1)
		
		if self.args.dataset == 'cityscape':
			return_dict = {}
			for i in range(self.vid_len):
				return_dict['frame'+str(i+1)] = ori_imgs[i]		
				return_dict['seg'+str(i+1)] = seg_imgs[i]	
				return_dict['bboxes'] = clip_boxes	
				# how to read [y1, 255-x2, y2, 255-x1]  forward(first image) example
				# for i in range(bs): for j in range(TRACK_NUM):
				# for_box = bboxes[i, 0, j]    # 2 for middle, 3 for backward
				# for_patch = for_img[i, :, int(for_box[1]):int(for_box[3])+1, int(for_box[2]):int(for_box[4])+1]
				# for_patch = F.interpolate(for_patch.unsqueeze(0), size=(self.H, self.W), mode='bilinear', align_corners=True)
			return return_dict
		else:
			return {'frame1':ori_imgs[0],
					'frame2':ori_imgs[1],
					'frame3':ori_imgs[2],
					'seg1'  :torch.zeros(1,1),
					'seg2'  :torch.zeros(1,1),
					'seg3'  :torch.zeros(1,1)}

	def __len__(self):
		return len(self.root)

	def check_clip_boxes(self, clip_boxes):
		if clip_boxes[0].sum() == 0 and clip_boxes[2].sum() ==0:
			return False
		return True

	def __repr__(self):
		fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
		fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
		fmt_str += '    Root Location: {}\n'.format(self.root)
		tmp = '    Transforms (if any): '
		fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
		tmp = '    Target Transforms (if any): '
		fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
		return fmt_str

class ImageFolder(DatasetFolder):
	def __init__(self, args, root, transform=None, bboxes=None):
		super(ImageFolder, self).__init__(args, root, transform, bboxes=bboxes)
