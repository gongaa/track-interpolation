import os
import sys
from time import time
import math
import argparse
import itertools
import shutil
from collections import OrderedDict
import cv2 
import numpy as np
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.distributed as dist 
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets 
from tensorboardX import SummaryWriter 
from torchvision.utils import make_grid, save_image
import random
# from utils import AverageMeter
# from loss import CombinedLoss
from losses import VGGLoss, RGBLoss, PSNR, SSIM, IoU, IOULoss, GANLoss, VGGCosineLoss, KLDLoss, GANScalarLoss, TrackObjLoss
import nets

from data import get_dataset
from folder import rgb_load, seg_load
from utils import *
import pickle

# def get_model(args):
#     # build model
#     model = nets.__dict__[args.model](args)
#     return model

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class HRNetTrainer:
    def __init__(self, args):
        self.args = args

        args.logger.info('Initializing trainer')
        self.coarse_model = nets.__dict__[args.coarse_model](args)
        self.frame_global_disc_model = nets.__dict__[args.frame_global_disc_model](args)
        self.ins_global_disc_model = nets.__dict__[args.ins_global_disc_model](args)
        self.ins_video_disc_model = nets.__dict__[args.ins_video_disc_model](args)
        if not self.args.train_coarse:
            self.set_net_grad(self.coarse_model, False)
       
        params_cnt = count_parameters(self.coarse_model)
        args.logger.info("coarse params "+str(params_cnt))

        ########################## put models on gpu ##############################
        torch.cuda.set_device(args.rank)
        self.coarse_model.cuda(args.rank)
        self.coarse_model = torch.nn.parallel.DistributedDataParallel(self.coarse_model,
                device_ids=[args.rank])
        self.frame_global_disc_model.cuda(args.rank)
        self.frame_global_disc_model= torch.nn.parallel.DistributedDataParallel(self.frame_global_disc_model,
                device_ids=[args.rank])
        self.ins_global_disc_model.cuda(args.rank)
        self.ins_global_disc_model= torch.nn.parallel.DistributedDataParallel(self.ins_global_disc_model,
                device_ids=[args.rank])
        self.ins_video_disc_model.cuda(args.rank)
        self.ins_video_disc_model= torch.nn.parallel.DistributedDataParallel(self.ins_video_disc_model,
                device_ids=[args.rank])
        ############################################################################ 

        if self.args.split in ['train', 'val']:
            train_dataset, val_dataset = get_dataset(args)

        if args.split == 'train':
            # train loss
            # RGB: global VGG loss
            # RGB: instance VGG loss (after)
            # RGB: L1 loss * mask out instance (after)(union of two instance mask)
            # seg: CE loss * mask out instance (after)
            # seg: instance CE loss (after)
            self.L1Loss = nn.L1Loss().cuda(args.rank)
            self.CELoss = nn.CrossEntropyLoss().cuda(args.rank)
            self.CELoss_no_reduction = nn.CrossEntropyLoss(reduction='none').cuda(args.rank)
            self.VGGLoss = VGGLoss().cuda(args.rank)
            self.IOULoss = IOULoss().cuda(args.rank)

            self.FrameDisc_DLoss = GANScalarLoss(weight=1).cuda(args.rank)
            self.FrameDisc_GLoss = GANScalarLoss(weight=1).cuda(args.rank)

            self.VideoDisc_DLoss = GANScalarLoss(weight=1).cuda(args.rank)
            self.VideoDisc_GLoss = GANScalarLoss(weight=1).cuda(args.rank)
            
            self.InsDisc_DLoss = GANScalarLoss(weight=1).cuda(args.rank)
            self.InsDisc_GLoss = GANScalarLoss(weight=1).cuda(args.rank)

            self.coarse_opt = torch.optim.Adamax(list(self.coarse_model.parameters()), lr = args.coarse_learning_rate) 
            self.frame_global_disc_opt = torch.optim.Adam(list(self.frame_global_disc_model.parameters()), lr = args.frame_global_disc_learning_rate)
            self.ins_global_disc_opt = torch.optim.Adam(list(self.ins_global_disc_model.parameters()), lr=args.ins_global_disc_learning_rate) 
            self.ins_video_disc_opt = torch.optim.Adam(list(self.ins_video_disc_model.parameters()), lr=args.ins_global_disc_learning_rate) 
            # self.coarse_opt = torch.optim.Adamax(list(self.model.module.coarse_model.parameters()), lr=args.coarse_learning_rate)
            

            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
            self.train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=args.batch_size//args.gpus, shuffle=False,
                num_workers=args.num_workers, pin_memory=True, sampler=train_sampler)

        elif args.split == 'val':
            # val criteria
            self.L1Loss  = nn.L1Loss().cuda(args.rank)
            self.PSNRLoss = PSNR().cuda(args.rank)
            self.SSIMLoss = SSIM().cuda(args.rank)
            self.IoULoss = IoU().cuda(args.rank)
            self.VGGCosLoss = VGGCosineLoss().cuda(args.rank)

            val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
            self.val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=args.batch_size//args.gpus, shuffle=False,
                num_workers=args.num_workers, pin_memory=True, sampler=val_sampler)
            self.val_dataset = val_dataset

        torch.backends.cudnn.benchmark = True
        self.global_step = 0
        self.epoch=1

        self.mean = torch.tensor([0.287, 0.3253, 0.284])[:, None, None]#.cuda()
        self.std = torch.tensor([0.1792, 0.18213, 0.1799898])[:, None, None]#.cuda()

        #################TODO: wrap this into self.load_checkpoint #################
        self.load_coarse_model()

        if args.rank == 0:
            writer_name = args.path+'/{}_int_{}_len_{}_{}_logs'.format(self.args.split, int(self.args.interval), self.args.vid_length, self.args.dataset)
            self.writer = SummaryWriter(writer_name)

    def set_net_grad(self, net, flag=True):
        for p in net.parameters():
            p.requires_grad = flag

    def set_epoch(self, epoch):
        self.args.logger.info("Start of epoch %d" % (epoch+1))
        self.epoch = epoch + 1
        self.train_loader.sampler.set_epoch(epoch)


    def normalize(self, img):
        return (img+1)/2

    def denormalize(self, img):
        ############### TODO: add self.mean, self.std #############################
        return torch.clamp(img * self.std + self.mean, 0, 1)

    def draw_bbox(self, img, bboxes):
        '''
            img (c, h, w)
            bboxes(4,4) 4 objects only (y1, x1, y2, x2)
        '''
        img_np = img.permute(1,2,0).contiguous().numpy()
        colors = [  (32,32,240),# red
                    (240,32,53),# blue
                    (74,240,32),# green
                    (32,157,240), # orange
                    (80,55,19),    # dark blue
                    (157,161,156), # grey
                    ]
        for i in range(self.args.num_track_per_img):
            bbox = bboxes[i]
            color = colors[i%len(colors)]
            cv2.rectangle(img_np, (int(bbox[1]), int(bbox[0])), (int(bbox[3]), int(bbox[2])), color, 2)
        return torch.tensor(img_np).permute(2,0,1).contiguous()

    def prepare_obj_image_seg(self, for_rgb, for_seg, back_rgb, back_seg, gt_rgb, gt_seg, out_rgb, out_seg):
        num_pred_imgs = 3
        self.args.logger.info('for_rgb size')
        self.args.logger.info(for_rgb.size())
        view_gt_rgbs = [ self.denormalize(for_rgb), self.denormalize(gt_rgb), self.denormalize(back_rgb)]
        view_gt_segs = [    vis_seg_mask(for_seg.unsqueeze(0), 20).squeeze(0),
                            vis_seg_mask(gt_seg.unsqueeze(0), 20).squeeze(0),
                            vis_seg_mask(back_seg.unsqueeze(0), 20).squeeze(0)   ]

        black_img = torch.zeros_like(view_gt_rgbs[0])

        n_rows = 4
        view_gt_rgbs.insert(2, self.denormalize(out_rgb))
        view_gt_segs.insert(2, vis_seg_mask(out_seg.unsqueeze(0), 20).squeeze(0))

        view_imgs = view_gt_rgbs + view_gt_segs
        view_imgs = [F.interpolate(img.unsqueeze(0), size=(128, 128), mode='bilinear', align_corners=True)[0] for img in view_imgs]

        write_in_img = make_grid(view_imgs, nrow=n_rows)

        return write_in_img

    def prepare_image_set(self, data, coarse_img, coarse_seg, for_bbox, gt_bbox, back_bbox, gen_bbox=None, track_gt_rgb=None, track_gt_seg=None, track_rgb=None, track_seg=None, ori_coarse_img=None, ori_coarse_seg=None):
        '''
            preparing output for tensorboard: denormalized img and seg cpu 
        '''
        # coarse_img should be of size 128x256 now
        # coarse_img = F.interpolate(coarse_img.unsqueeze(0), size=(256, 512), mode='bilinear', align_corners=True).squeeze(0)
        num_pred_imgs = self.args.num_pred_step*self.args.num_pred_once if self.args.syn_type == 'extra' else 3
        view_gt_rgbs = [ self.denormalize(data['frame'+str(i+1)][0].cpu().detach()) for i in range(3)]
        view_gt_segs = [ vis_seg_mask(data['seg'+str(i+1)][0].unsqueeze(0), 20, seg_id=True).squeeze(0) for i in range(3)]

        black_img = torch.zeros_like(view_gt_rgbs[0])

        n_rows = 4
        view_gt_rgbs.insert(2, self.denormalize(coarse_img))
        view_gt_segs.insert(2, vis_seg_mask(coarse_seg.unsqueeze(0), 20, seg_id=False).squeeze(0))

        view_imgs = view_gt_rgbs + view_gt_segs
        
        bboxes = data['bboxes'][0].cpu().numpy()
        view_bboxes = [
            # for_box (4, 4) first 4 is four object, later four is four vertices
            self.draw_bbox(view_gt_rgbs[0].cpu().detach(), for_bbox),
            self.draw_bbox(view_gt_rgbs[1].cpu().detach(), gt_bbox),
            self.draw_bbox(view_gt_rgbs[3].cpu().detach(), back_bbox)
        ]
        if gen_bbox is None:
            view_bboxes.insert(2, self.draw_bbox(view_gt_rgbs[2].cpu().detach(), bboxes[1, :, 1:]))
        else:
            view_bboxes.insert(2, self.draw_bbox(view_gt_rgbs[2].cpu().detach(), gen_bbox))

        view_imgs = view_imgs[:4] + view_bboxes + view_imgs[4:]

        view_imgs = [F.interpolate(img.unsqueeze(0), size=(128, 256), mode='bilinear', align_corners=True)[0] for img in view_imgs]

        write_in_img = make_grid(view_imgs, nrow=n_rows) 

        return write_in_img

    def get_loss_record_dict(self):

        D = {'data_cnt':0,
            'all_loss_record':0}

        d = {
        'coarse_iou_loss_record'     :0,      # iou loss of bbox
        'coarse_l1_glb_loss_record'  :0,      # global l1 (with mask later)
        'coarse_vgg_ins_loss_record' :0,      # instance
        'coarse_vgg_glb_loss_record' :0,      # global
        'coarse_ce_glb_loss_record'  :0,      # global ce (with mask later)
        # 'coarse_ce_ins_loss_record'  :0,      # instance ce
        'coarse_all_loss_record'     :0,

        'disc_frame_loss_record'     :0,      # generator wnats to maximize
        'disc_frame_fake_loss_record':0,
        'disc_frame_real_loss_record':0,

        'disc_video_loss_record'     :0,      
        'disc_video_fake_loss_record':0,
        'disc_video_real_loss_record':0,

        'disc_ins_loss_record'       :0,
        'disc_ins_fake_loss_record:' :0,
        'disc_ins_real_loss_record:' :0,
        }

        D.update(d)

        return D

    def update_loss_record_dict(self, record_dict, loss_dict, batch_size):
        record_dict['data_cnt']+=batch_size

        loss_name_list = ['iou', 'l1_glb', 'vgg_ins', 'vgg_glb', 'ce_glb']#, 'ce_ins']
        for loss_name in loss_name_list:
            record_dict['coarse_{}_loss_record'.format(loss_name)] += \
                            batch_size*loss_dict['coarse_{}_loss'.format(loss_name)].item()
            record_dict['coarse_all_loss_record'] += batch_size*loss_dict['coarse_{}_loss'.format(loss_name)].item()
        
        ###########TODO: video discriminator ##############
        disc_name_list = ['frame', 'video', 'ins']
        for disc_name in disc_name_list:
            record_dict['disc_{}_loss_record'.format(loss_name)] += \
                            batch_size*loss_dict['disc_{}_loss'.format(disc_name)].item()
            for loss_name in ['real', 'fake']:
                record_dict['disc_{}_{}_loss_record'.format(disc_name, loss_name)] += \
                    batch_size*loss_dict['disc_{}_{}_loss'.format(disc_name, loss_name)].item()


        record_dict['all_loss_record']+=batch_size*loss_dict['loss_all'].item()
        return record_dict

    def make_instance_masks(self, for_bbox, back_bbox, gt_x, bs):
        for_instance_masks = []
        back_instance_masks = []
        for i in range(bs):
            for_masks = []
            back_masks = []
            for j in range(self.args.num_track_per_img):
                for_coord = for_bbox[i,j,:]
                back_coord = back_bbox[i,j,:]
                for_mask = torch.zeros_like(gt_x[0,0,:,:])
                back_mask = torch.zeros_like(for_mask)
                for_mask[int(for_coord[0]):int(for_coord[2])+1, int(for_coord[1]):int(for_coord[3])+1] = 1
                back_mask[int(back_coord[0]):int(back_coord[2])+1, int(back_coord[1]):int(back_coord[3])+1] = 1
                for_masks.append(for_mask.unsqueeze(0))
                back_masks.append(back_mask.unsqueeze(0))
            for_instance_masks.append(torch.cat(for_masks, dim=0).unsqueeze(0))
            back_instance_masks.append(torch.cat(back_masks, dim=0).unsqueeze(0))
        for_instance_masks = torch.cat(for_instance_masks, dim=0)    # (bs,num_track,input_h, input_w)
        back_instance_masks = torch.cat(back_instance_masks, dim=0)
        return for_instance_masks, back_instance_masks

    def get_coarse_instance(self, rgb, bbox):
        # (bs,3,H,W) (bs,1,H,W) (bs, num_track,4)
        # for_patch = for_img[i, :, int(for_box[1]):int(for_box[3])+1, int(for_box[2]):int(for_box[4])+1]
        # for_patch = F.interpolate(for_patch.unsqueeze(0), size=(self.H, self.W), mode='bilinear', align_corners=True)
        bs,_,h,w = rgb.size()
        num_track = bbox.size()[1]
        instance_rgb = []
        # instance_seg = []
        for i in range(bs):
            for j in range(num_track):
                with torch.no_grad():
                    square_bbox, padding = self.get_square_bbox_and_padding(bbox=bbox[i,j], h=h, w=w)
                ins_rgb = rgb[i,:,square_bbox[0]:square_bbox[2]+1, square_bbox[1]:square_bbox[3]+1]
                # ins_seg = seg[i,:,square_bbox[0]:square_bbox[2]+1, square_bbox[1]:square_bbox[3]+1]
                pad = nn.ZeroPad2d(padding)
                instance_rgb.append(F.interpolate(pad(ins_rgb).unsqueeze(0), size=(64,64), mode='bilinear', align_corners=True))
                # instance_seg.append(F.interpolate(pad(ins_seg).unsqueeze(0), size=(64,64), mode='nearest'))

        return torch.cat(instance_rgb, dim=0)#, torch.cat(instance_seg, dim=0)  # (bs*num_track,3,64,64)   (bs*num_track,1,64,64)

    def get_square_bbox_and_padding(self, bbox, h, w):
        # 0<=bbox[0]<=bbox[2]<=127 (h-1)
        # 0<=bbox[1]<=bbox[3]<=255 (w-1)
        padding = [0,0,0,0]
        hh, ww = bbox[2]-bbox[0], bbox[3]-bbox[1]
        half_e = float(max(hh, ww))/2.0
        if half_e < 2:
            half_e = 4
        mid_h, mid_w = bbox[0] + float(hh)/2.0, bbox[1] + float(ww)/2.0
        square_bbox = [int(mid_h-half_e), int(mid_w-half_e), int(mid_h+half_e), int(mid_w+half_e)]
        padding[0] = 0 if (square_bbox[0]>=0) else -square_bbox[0]
        padding[1] = 0 if (square_bbox[1]>=0) else -square_bbox[1]
        padding[2] = 0 if (square_bbox[2]<h) else square_bbox[2]-h+1
        padding[3] = 0 if (square_bbox[3]<w) else square_bbox[3]-w+1
        square_bbox[0] = max(0, square_bbox[0])
        square_bbox[1] = max(0, square_bbox[1])
        square_bbox[2] = min(h-1, square_bbox[2])
        square_bbox[3] = min(w-1, square_bbox[3])

        return square_bbox, [padding[1], padding[3], padding[0], padding[2]] # (left right top bottom)

    def train(self):
        if self.args.rank == 0:
            self.args.logger.info('Training started')
            step_loss_record_dict = self.get_loss_record_dict()
            epoch_loss_record_dict = self.get_loss_record_dict()
        self.coarse_model.train()
        end = time()
        load_time = 0
        comp_time = 0
        GAN_TRAIN_STEP = 0
        
        for step, data in enumerate(self.train_loader):
            # if step < 1500:
            #     if self.args.rank==0 and step%100 == 0:
            #         self.args.logger.info('skip step {}'.format(step))
            #     continue
            self.step = step
            load_time += time() - end
            end = time()
            self.global_step += 1

            batch_size = data['frame1'].size(0)
            loss_dict = OrderedDict()
            # 1. get input
            gt_x = data['frame2'].cuda(self.args.rank, non_blocking=True)
            gt_seg = data['seg2'].cuda(self.args.rank, non_blocking=True)
            bboxes = data['bboxes'].cuda(self.args.rank,non_blocking=True)    # (bs, 3, num_track, 5)

            for_bbox = bboxes[:,0,:,1:].contiguous() # (bs, num_track, 4)  4 coordinates
            gt_bbox = bboxes[:,1,:,1:].contiguous()
            back_bbox = bboxes[:,2,:,1:].contiguous()
            gen_bbox = ((for_bbox+back_bbox)/2).long().float()

            ###################### TODO: wrap extract mask accoring to coordinate to a function ###################################
            with torch.no_grad():
                for_instance_masks, back_instance_masks = self.make_instance_masks(for_bbox=for_bbox, back_bbox=back_bbox, gt_x=gt_x, bs=batch_size)
            input_without_mask = torch.cat([data['seg1'], data['frame1'], data['frame3'], data['seg3']], dim=1).cuda(self.args.rank, non_blocking=True)
            input = torch.cat([for_instance_masks, input_without_mask, back_instance_masks], dim=1)
            ##################### TODO: my hrnet forward ###########################
            # 1. get coarse rgb and seg and (y1, 255-x2, y2, 255-x1)
            coarse_rgb, coarse_seg, coarse_gen_bbox = self.coarse_model(input=input) 
            # (bs, 3, 128, 256), (bs, 1, 128, 256)
            # expect coarse_gen_bbox (bs, num_track, 4)
            # 2. calculate IOU, if >0.3 then use generated bbox, otherwise use average
            iou = self.IOULoss(gt_bbox=gt_bbox, gen_bbox=coarse_gen_bbox)  # (bs, num_track)
            flattened_coarse_gen_bbox = coarse_gen_bbox.view(-1, 4)
            flattened_gen_bbox = gen_bbox.view(-1, 4)
            ###########TODO#############
            # big_iou = iou.view(-1).gt(0)
            # flattened_coarse_gen_bbox[big_iou] = flattened_coarse_gen_bbox[big_iou]
            if self.epoch>1:
                for i, iou_score in enumerate(iou.view(-1).tolist()):
                    if iou_score>0.5:
                        flattened_gen_bbox[i] = flattened_coarse_gen_bbox[i]     # python will modify gen_bbox
            # 3. calculate mask (union of gen_bbox with gt_bbox)
            with torch.no_grad():
                gen_instance_masks, gt_instance_masks = self.make_instance_masks(for_bbox=gen_bbox, back_bbox=gt_bbox, gt_x=gt_x, bs=batch_size)
                union_mask = gen_instance_masks.byte() | gt_instance_masks.byte()
                merged_union_mask = union_mask[:,0,:,:] | union_mask[:,1,:,:] | union_mask[:,2,:,:] | union_mask[:,3,:,:]
                outside_mask = (1-merged_union_mask)[:,None,:,:]        # expect (bs, 1, 128, 256)
            # 4. calculate other losses
            ##################################################################################################
            ################# TODO: think carefully whether we need instance cross entropy ####################
            # coarse_instance_rgb, coarse_instance_seg = self.get_coarse_instance(rgb=coarse_rgb, seg=coarse_seg, bbox=coarse_gen_bbox)
            # gt_instance_rgb, gt_instance_seg = self.get_coarse_instance(rgb=gt_x, seg=gt_seg, bbox=gt_bbox) # (bs,3,H,W) (bs,1,H,W) (bs, num_track,4)
            coarse_instance_rgb = self.get_coarse_instance(rgb=coarse_rgb, bbox=coarse_gen_bbox)
            gt_instance_rgb = self.get_coarse_instance(rgb=gt_x, bbox=gt_bbox) # (bs,3,H,W) (bs,1,H,W) (bs, num_track,4)
            with torch.no_grad():
                for_instance_rgb = self.get_coarse_instance(rgb=input_without_mask[:,1:4,:,:], bbox=for_bbox)
                back_instance_rgb = self.get_coarse_instance(rgb=input_without_mask[:,4:7,:,:], bbox=back_bbox)
            # (bs*num_track,3,64,64), (bs*num_track,20,64,64)
            # 3. update outputs and store them
            prefix = 'coarse'
            loss_dict[prefix+'_iou_loss'] = -torch.mean(iou)    # minimize -iou, expect iou to be high
            if self.epoch<=30:
                loss_dict[prefix+'_l1_glb_loss'] = self.L1Loss(coarse_rgb, gt_x)
                # loss_dict[prefix+'_ce_glb_loss']  = 0.1 * self.CELoss(coarse_seg, (gt_seg).long().squeeze(1))    # from (bs, 1, H, W) to (bs, H, W)
            else:
                loss_dict[prefix+'_l1_glb_loss']  = self.L1Loss(torch.masked_select(coarse_rgb,outside_mask), torch.masked_select(gt_x,outside_mask))
                # loss_dict[prefix+'_ce_glb_loss']  = 0.1 * torch.mean(torch.masked_select(self.CELoss_no_reduction(coarse_seg, gt_seg.long().squeeze(1)), outside_mask.squeeze(1)))    # from (bs, 1, H, W) to (bs, H, W)
            loss_dict[prefix+'_vgg_ins_loss'] = self.VGGLoss(output=coarse_instance_rgb, target=gt_instance_rgb)
            loss_dict[prefix+'_vgg_glb_loss'] = self.VGGLoss(output=coarse_rgb, target=gt_x)
            loss_dict[prefix+'_ce_glb_loss']  = 0.1 * self.CELoss(coarse_seg, gt_seg.long().squeeze(1))    # from (bs, 1, H, W) to (bs, H, W)
            # loss_dict[prefix+'_ce_ins_loss']  = 0.1 * self.CELoss(coarse_instance_seg, gt_instance_seg.long().squeeze(1))

            ######################################### GAN LOSS #####################################################################
            D_fake_frame_prob, D_real_frame_prob = self.frame_global_disc_model(coarse_rgb.detach()), self.frame_global_disc_model(gt_x)
            D_fake_ins_prob, D_real_ins_prob = self.ins_global_disc_model(coarse_instance_rgb.detach()), self.ins_global_disc_model(gt_instance_rgb)
            D_fake_video_prob = self.ins_video_disc_model(middle=coarse_instance_rgb.detach(), first=for_instance_rgb, last=back_instance_rgb)
            D_real_video_prob = self.ins_video_disc_model(middle=gt_instance_rgb.detach(), first=for_instance_rgb, last=back_instance_rgb)
            G_fake_frame_prob, G_fake_ins_prob = self.frame_global_disc_model(coarse_rgb), self.ins_global_disc_model(coarse_instance_rgb)
            G_fake_video_prob = self.ins_video_disc_model(middle=coarse_instance_rgb, first=for_instance_rgb, last=back_instance_rgb)


            loss_dict['disc_frame_loss']    = self.FrameDisc_GLoss(G_fake_frame_prob, True) if self.global_step > GAN_TRAIN_STEP else \
                                                self.FrameDisc_GLoss(G_fake_frame_prob, True)*0
            loss_dict['disc_frame_real_loss'] = self.FrameDisc_DLoss(D_real_frame_prob, True) if self.global_step > GAN_TRAIN_STEP else \
                                                self.FrameDisc_DLoss(D_real_frame_prob, True)*0
            loss_dict['disc_frame_fake_loss'] = self.FrameDisc_DLoss(D_fake_frame_prob, False) if self.global_step > GAN_TRAIN_STEP else \
                                                self.FrameDisc_DLoss(D_fake_frame_prob, False)*0

                                                    
            loss_dict['disc_video_loss']    = self.VideoDisc_GLoss(G_fake_video_prob, True) if self.global_step > GAN_TRAIN_STEP else \
                                                self.VideoDisc_GLoss(G_fake_video_prob, True)*0
            loss_dict['disc_video_real_loss'] = self.VideoDisc_DLoss(D_real_video_prob, True) if self.global_step > GAN_TRAIN_STEP else \
                                                self.VideoDisc_DLoss(D_real_video_prob, True)*0
            loss_dict['disc_video_fake_loss'] = self.VideoDisc_DLoss(D_fake_video_prob, False) if self.global_step > GAN_TRAIN_STEP else \
                                                self.VideoDisc_DLoss(D_fake_video_prob, False)*0

            loss_dict['disc_ins_loss']    = self.InsDisc_GLoss(G_fake_ins_prob, True) if self.global_step > GAN_TRAIN_STEP else \
                                                self.InsDisc_GLoss(G_fake_ins_prob, True)*0
            loss_dict['disc_ins_real_loss'] = self.InsDisc_DLoss(D_real_ins_prob, True) if self.global_step > GAN_TRAIN_STEP else \
                                                self.InsDisc_DLoss(D_real_ins_prob, True)*0
            loss_dict['disc_ins_fake_loss'] = self.InsDisc_DLoss(D_fake_ins_prob, False) if self.global_step > GAN_TRAIN_STEP else \
                                                self.InsDisc_DLoss(D_fake_ins_prob, False)*0
            ##########################################################################################################################
            loss = 0
            for i in loss_dict.values():
                loss += torch.mean(i)
            loss_dict['loss_all'] = loss
            self.sync(loss_dict)
            ####################### backward pass ###################################
            self.coarse_opt.zero_grad()     if self.args.train_coarse  else None
            self.frame_global_disc_opt.zero_grad()
            self.ins_global_disc_opt.zero_grad()
            self.ins_video_disc_opt.zero_grad()

            loss_dict['loss_all'].backward()
            self.coarse_opt.step()  if self.args.train_coarse  else None
            self.frame_global_disc_opt.step()
            self.ins_global_disc_opt.step()
            self.ins_video_disc_opt.step()
            ##########################################################################
            comp_time += time() - end
            end = time()

            if self.args.rank == 0:
                step_loss_record_dict = self.update_loss_record_dict(step_loss_record_dict, loss_dict, batch_size)
                # add info to tensorboard
                info = {key:value.item() for key,value in loss_dict.items()}
                self.writer.add_scalars("losses", info, self.global_step)

                if self.step % self.args.disp_interval == 0:
                    for key, value in step_loss_record_dict.items():
                        epoch_loss_record_dict[key]+=value

                    if step_loss_record_dict['data_cnt'] != 0:
                        for key, value in step_loss_record_dict.items():
                            if key!='data_cnt':
                                step_loss_record_dict[key] /= step_loss_record_dict['data_cnt']

                    log_main = 'Epoch [{epoch:d}/{tot_epoch:d}][{cur_batch:d}/{tot_batch:d}] load [{load_time:.3f}s] comp [{comp_time:.3f}s] '.format(epoch=self.epoch, tot_epoch=self.args.epochs,
                            cur_batch=self.step+1, tot_batch=len(self.train_loader),
                            load_time=load_time, comp_time=comp_time)
                    # log = '\n\tcoarse l1_glb [{l1_glb:.3f}] vgg_ins [{vgg_ins:.3f}] vgg_glb [{vgg_glb:.3f}] ce_glb [{ce_glb:.3f}] ce_ins [{ce_ins:.3f}] -iou [{iou:.3f}]'.format(
                    log = '\n\tcoarse l1_glb [{l1_glb:.3f}] vgg_ins [{vgg_ins:.3f}] vgg_glb [{vgg_glb:.3f}] ce_glb [{ce_glb:.3f}] -iou [{iou:.3f}]'.format(
                            l1_glb=step_loss_record_dict['coarse_l1_glb_loss_record'],
                            vgg_glb=step_loss_record_dict['coarse_vgg_glb_loss_record'],
                            vgg_ins=step_loss_record_dict['coarse_vgg_ins_loss_record'],
                            ce_glb=step_loss_record_dict['coarse_ce_glb_loss_record'],
                            # ce_ins=step_loss_record_dict['coarse_ce_ins_loss_record'],
                            iou=step_loss_record_dict['coarse_iou_loss_record']
                        )
                    log+= ' all [{all:.3f}]'.format(all=step_loss_record_dict['coarse_all_loss_record'])
                    log_main+=log
                    log_main += '\n\t\t\tloss total [{:.3f}]'.format(step_loss_record_dict['all_loss_record'])

                    self.args.logger.info(log_main)
                    # self.args.logger.info('hhhhhhhhhhhh')
                    comp_time = 0
                    load_time = 0

                    if step_loss_record_dict['data_cnt'] != 0:
                        for key, value in step_loss_record_dict.items():
                            step_loss_record_dict[key] = 0

                if self.step % 30 == 0: 
                    # only show the first image of the batch
                    image_set = self.prepare_image_set(data, coarse_rgb[0].cpu(), coarse_seg[0].cpu(),
                                                            for_bbox=for_bbox[0].cpu(), gt_bbox=gt_bbox[0].cpu(),
                                                            back_bbox=back_bbox[0].cpu(), gen_bbox=gen_bbox[0].cpu())
                                                        
                    img_name = 'image_{}_glob '.format(
                                                self.global_step)

                    self.writer.add_image(img_name, image_set, self.global_step)


        # if self.args.rank == 0:
        #     for key, value in step_loss_record_dict.items():
        #         epoch_loss_record_dict[key]+=value

        #     if epoch_loss_record_dict['data_cnt'] != 0:
        #         for key, value in epoch_loss_record_dict.items():
        #             if key!='data_cnt':
        #                 epoch_loss_record_dict[key] /= epoch_loss_record_dict['data_cnt']

        #     log_main = 'Epoch [{epoch:d}/{tot_epoch:d}]'.format(epoch=self.epoch, tot_epoch=self.args.epochs)

        #     log = '\n\tcoarse l1_glb [{l1_glb:.3f}] vgg_ins [{vgg_ins:.3f}] vgg_glb [{vgg_glb:.3f}] ce_glb [{ce_glb:.3f}] ce_ins [{ce_ins:.3f}]'.format(
        #             l1_glb=step_loss_record_dict['coarse_l1_glb_loss_record'],
        #             vgg_glb=step_loss_record_dict['coarse_vgg_glb_loss_record'],
        #             vgg_ins=step_loss_record_dict['coarse_vgg_ins_loss_record'],
        #             ce_glb=step_loss_record_dict['coarse_ce_glb_loss_record'],
        #             ce_ins=step_loss_record_dict['coarse_ce_ins_loss_record']
        #         )
        #     log_main+=log
        #     log_main += '\n\t\t\t\t\t\t\tloss total [{:.3f}]'.format(epoch_loss_record_dict['all_loss_record'])
        #     self.args.logger.info(
        #         log_main
        #     )


    def validate(self):
        self.args.logger.info('Validation epoch {} started'.format(self.epoch))
        self.coarse_model.eval()

        val_criteria = {}
        criteria_list = ['l1', 'psnr', 'ssim', 'vgg']
        if self.args.mode == 'xs2xs':
            criteria_list.append('iou')
        for crit in criteria_list:
            val_criteria[crit] = AverageMeter()

        step_losses = OrderedDict()

        with torch.no_grad():
            end = time()
            load_time = 0
            comp_time = 0
            for i, data in enumerate(self.val_loader):
                root_name = self.val_dataset.root[i][0].split('/')[-1]
                root_name = root_name[:-6] + "{:0>6d}".format(int(root_name[-6:])-1)
                name = root_name
                load_time += time()-end
                end = time()
                self.step=i
                # name = root[i]
                if self.step>400:
                    break


                batch_size = data['frame1'].size(0)
                loss_dict = OrderedDict()
                # 1. get input
                gt_x = data['frame2'].cuda(self.args.rank, non_blocking=True)
                gt_seg = data['seg2'].cuda(self.args.rank, non_blocking=True)
                bboxes = data['bboxes'].cuda(self.args.rank,non_blocking=True)    # (bs, 3, num_track, 5)

                for_bbox = bboxes[:,0,:,1:] # (bs, num_track, 4)  4 coordinates
                gt_bbox = bboxes[:,1,:,1:]
                back_bbox = bboxes[:,2,:,1:]
                gen_bbox = ((for_bbox+back_bbox)/2).long().float()

                for_instance_masks, back_instance_masks = self.make_instance_masks(for_bbox=for_bbox, back_bbox=back_bbox, gt_x=gt_x, bs=batch_size)
                input = torch.cat([data['seg1'], data['frame1'], data['frame3'], data['seg3']], dim=1).cuda(self.args.rank, non_blocking=True)
                input = torch.cat([for_instance_masks, input, back_instance_masks], dim=1)
                coarse_rgb, coarse_seg, coarse_gen_bbox = self.coarse_model(input=input) 
 
                # rgb criteria
                step_losses['l1']    = self.L1Loss(coarse_rgb, gt_x)
                step_losses['psnr']  = self.PSNRLoss(coarse_rgb, gt_x)
                step_losses['ssim']  = 1-self.SSIMLoss(coarse_rgb, gt_x)
                step_losses['iou']   =  self.IoULoss(torch.argmax(coarse_seg, dim=1), gt_seg.long().squeeze(1))
                step_losses['vgg']   =  self.VGGCosLoss(coarse_rgb, gt_x)

                self.sync(step_losses) # sum

                comp_time += time() - end
                end = time()

                # print
                if self.args.rank == 0:
                    for crit in criteria_list:
                        val_criteria[crit].update(step_losses[crit].cpu().item(), batch_size*self.args.gpus)

                    if self.step % self.args.disp_interval == 0:
                        self.args.logger.info(
                            'Epoch [{epoch:d}][{cur_batch:d}/{tot_batch:d}] '
                            'load [{load_time:.3f}s] comp [{comp_time:.3f}s]'.format(
                                epoch=self.epoch, cur_batch=self.step+1, tot_batch=len(self.val_loader),
                                load_time=load_time, comp_time=comp_time
                            )
                        )
                        comp_time = 0
                        load_time = 0
                    if self.step % 1 == 0: 
                        image_set = self.prepare_image_set(data, coarse_rgb[0].cpu(), coarse_seg[0].cpu(),
                                                            for_bbox=for_bbox[0].cpu(), gt_bbox=gt_bbox[0].cpu(),
                                                            back_bbox=back_bbox[0].cpu(), gen_bbox=gen_bbox[0].cpu())

                        img_name = 'e{}_{}_glob'.format(self.epoch, self.step)
                        self.writer.add_image(img_name, image_set, self.step)

        if self.args.rank == 0:
            log_main = '\n######################### Epoch [{epoch:d}] Evaluation Results #########################'.format(epoch=self.epoch)

            log = '\n\tcoarse l1 [{l1:.3f}] vgg [{vgg:.3f}] psnr [{psnr:.3f}] ssim [{ssim:.3f}] iou [{iou:.3f}]'.format(
                    l1  =val_criteria['l1'].avg,
                    vgg =val_criteria['vgg'].avg,
                    psnr=val_criteria['psnr'].avg,
                    ssim=val_criteria['ssim'].avg,
                    iou =val_criteria['iou'].avg
                )
            log_main+=log
            log_main += '\n#####################################################################################\n'

            self.args.logger.info(log_main)

            tfb_info = {key:value.avg for key,value in val_criteria.items()}
            self.writer.add_scalars('val/score', tfb_info, self.epoch)

    def sync(self, loss_dict, mean=True):
        '''Synchronize all tensors given using mean or sum.'''
        for tensor in loss_dict.values():
            dist.all_reduce(tensor)
            if mean:
                tensor.div_(self.args.gpus)

    def save_checkpoint(self):
        save_md_dir = '{}_{}_{}_{}'.format(self.args.model, self.args.mode, self.args.syn_type, self.args.session)
        save_name = os.path.join(self.args.path, 
                                'checkpoint',
                                save_md_dir + '_{}_{}.pth'.format(self.epoch, self.step))
        self.args.logger.info('Saving checkpoint..')
        save_dict = {
            'session': self.args.session,
            'epoch': self.epoch + 1,
            'coarse_model': self.coarse_model.state_dict(),
            'coarse_opt': self.coarse_opt.state_dict(),
            'frame_global_disc_model': self.frame_global_disc_model.state_dict(),
            'frame_global_disc_opt': self.frame_global_disc_opt.state_dict(),
            'ins_global_disc_model': self.ins_global_disc_model.state_dict(),
            'ins_global_disc_opt': self.ins_global_disc_opt.state_dict(),
            'ins_video_disc_model': self.ins_video_disc_model.state_dict(),
            'ins_video_disc_opt': self.ins_video_disc_opt.state_dict()
        }
        
        torch.save(save_dict, save_name)
        self.args.logger.info('save model: {}'.format(save_name))

    def load_coarse_model(self):
        if self.args.pretrained_coarse:
            new_ckpt = OrderedDict()
            device = torch.device('cpu')
            # coarse_model_dict = self.model.module.coarse_model.state_dict()
            coarse_model_dict = self.coarse_model.state_dict()
            ckpt = torch.load(self.args.pretrained_coarse_model, map_location=device)
            for key,item in ckpt['coarse_model'].items():
                new_ckpt[key] = item
            coarse_model_dict.update(new_ckpt)
            self.coarse_model.load_state_dict(coarse_model_dict)
            self.coarse_opt.load_state_dict(ckpt['coarse_opt'])
            self.epoch = ckpt['epoch']+1
            print('successful loading pretrained coarse model')


