import torchvision.transforms as transforms
from folder import ImageFolder
from PIL import Image
# import pickle
# from cityscape_utils import *
'''
	input:			 dataset name(str)

	return np data:	
		n_classes	: int

		train_imgs	: (n_t, h, w, 3)
		train_segs	: (n_t, h, w)
		train_masks	: (n_t, h, w)  missing region is 0, known region is 1 

		val_imgs	: (n_v, h, w, 3)
		val_segs	: (n_v, h, w)
		val_masks	: (n_v, h, w)
'''

def get_dataset(args):
	### explicitly set flip = True #######
	if args.dataset == "cityscape":
		# if 'Det' in args.frame_disc_model or 'Det' in args.video_disc_model or args.frame_det_disc or args.video_det_disc:
		clip_file = "/data/linz/proj/Dataset/Cityscape/load_files/int_{}_len_3_max_{}bb_area_3000_extra_panet_lsclip.pkl".format(int(args.interval), int(args.num_track_per_img))
		# if not args.track_gen and args.split == 'val':
		# 	clip_file = "/data/linz/proj/Dataset/Cityscape/load_files/int_{}_len_3_extra_lsclip.pkl".format(int(args.interval))
		obj_coord_file = "/data/linz/proj/Dataset/Cityscape/obj_coords/int_{}_len_3_extra_512x1024_max_{}bb_area_3000_panet_lsclip.pkl".format(int(args.interval), int(args.num_track_per_img))
		# if args.syn_type == 'extra' and args.vid_length != 1:
		# 	clip_file = "/data/linz/proj/Dataset/Cityscape/load_files/int_{}_len_{}_extra_lsclip.pkl".format(int(args.interval), args.vid_length+2)
		# if args.effec_flow:
		# 	clip_file = "/data/linz/proj/Dataset/Cityscape/load_files/effec_flow_int_{}_len_3_extra_lsclip.pkl".format(int(args.interval))
		import pickle
		with open(clip_file, 'rb') as f:
			load_f = pickle.load(f)
			if args.split == 'train':
				clips_train_file = load_f['train'] 
			elif args.split == 'val':
				clips_val_file = load_f['val'] 
		with open(obj_coord_file, 'rb') as f:
			load_f = pickle.load(f)
			if args.split == 'train':
				coords_train_file = load_f['train'] 
			if args.split == 'val':
				coords_val_file = load_f['val']
			# else:
			# 	coords_val_file = None

		crop_size = (args.input_h, args.input_w)
		if args.split == 'train':
			# train 
			tfs = []
			tfs.append(transforms.Compose([transforms.RandomCrop(crop_size)]))
			# tfs.append(transforms.Compose([		transforms.Resize((150, 300), interpolation=Image.NEAREST),
			# 											transforms.RandomCrop((128, 256))	]))
			tfs.append(transforms.Compose([transforms.RandomCrop(crop_size)]))

			train_dataset = ImageFolder(args, clips_train_file, transform=tfs, bboxes=coords_train_file)	
		else:
			train_dataset=None	

		if args.split == 'val':
			# val
			tfs = []
			tfs.append(transforms.Compose([]))
			tfs.append(transforms.Compose([]))

			val_dataset   = ImageFolder(args, clips_val_file, transform=tfs, bboxes = coords_val_file)
		else:
			val_dataset=None

	return train_dataset, val_dataset


