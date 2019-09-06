import os
import sys
from time import time
import math
import argparse
import itertools
import shutil

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist 
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets 
from tensorboardX import SummaryWriter 
from torchvision.utils import make_grid
from utils import AverageMeter
from loss import CombinedLoss
import models
from models.networks import init_net
from models.networks import get_norm_layer
from models.networks import GANLoss
import cv2
import random
from data import get_dataset
from models.net_utils import *
# from cfg import cfg

color_map = [
[128, 64, 128] ,   # road
[244, 35, 232]  ,  # sidewald
[70, 70, 70] ,  # building
[102, 102, 156] , # wall
[190, 153, 153] , # fence
[153, 153, 153] , # pole
[250, 170, 30] , # traffic light
[220, 220, 0] ,  # traffic sign
[107, 142, 35] , # vegetation
[152, 251, 152] , # terrain
[70, 130, 180]  , # sky
[220, 20, 60] , # person
[255, 0, 0]  , # rider
[0, 0, 142]   , # car
[0, 0, 70]  ,  # truck
[0, 60, 100] ,  # bus
[0, 80, 100] ,  # on rails / train
[0, 0, 230]  , # motorcycle
[119, 11, 32] , # bicycle
[0, 0, 0]   # None
]

def get_model(args):
    # build model
    norm_layer = get_norm_layer(norm_type=args.norm)
    generator = models.__dict__[args.generator](args.input_nc, args.output_nc, args.ngf, norm_layer=norm_layer, use_dropout=not args.no_dropout, n_blocks=9).cuda(args.rank)
    discriminator = models.__dict__[args.discriminator](9, args.ndf, n_layers=3, norm_layer=norm_layer).cuda(args.rank)
    init_net(generator, init_type=args.init_type, init_gain=args.init_gain)
    init_net(discriminator, init_type=args.init_type, init_gain=args.init_gain)

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    # load model
    if getattr(args, 'ckpt', None) is not None:
        args.logger.info('Loading from ckpt %s' % args.ckpt)
        ckpt = torch.load(args.ckpt,
            map_location=torch.device('cpu')) # load to cpu
        if 'generator' in ckpt:
            generator.load_state_dict(ckpt['generator'])
        if 'discriminator' in ckpt:
            discriminator.load_state_dict(ckpt['discriminator'])
        if 'optimizer_G' in ckpt:
            optimizer_G.load_state_dict(ckpt['optimizer_G'])
        if 'optimizer_D' in ckpt:
            optimizer_D.load_state_dict(ckpt['optimizer_D'])

    return generator, discriminator, optimizer_G, optimizer_D


def get_gridnet(args):
    cgn = models.__dict__[args.arch](n_channels=10).cuda(args.rank)
    optim = torch.optim.Adamax(cgn.parameters(), lr = args.lr, betas=(args.beta1, 0.999))

    if getattr(args, 'ckpt', None) is not None:
        args.logger.info('Loading from ckpt %s' % args.ckpt)
        ckpt = torch.load(args.ckpt,
            map_location=torch.device('cpu')) # load to cpu
        if 'gridnet' in ckpt:
            generator.load_state_dict(ckpt['gridnet'])
        if 'optimizer' in ckpt:
            optim.load_state_dict(ckpt['optimizer'])
    
    return cgn, optim

def get_HNED(args):
    modelPath = '/home/agong/HED-on-Cityscapes/playground/train_edge_110/checkpoint/020.pth'
    hed = models.__dict__['HNED']().cuda(args.rank)
    hed.load_state_dict(torch.load(modelPath)['generator'])
    return hed


class Trainer:

    def __init__(self, args):
        args.logger.info('Initializing trainer')
        if not os.path.isdir('../predict'):
            os.makedirs('../predict')

        torch.cuda.set_device(args.rank)

        self.gridnet, self.optimizer = get_gridnet(args)
        self.gridnet = torch.nn.parallel.DistributedDataParallel(self.gridnet, device_ids=[args.rank])
        self.hed = get_HNED(args)
        self.hed = torch.nn.parallel.DistributedDataParallel(self.hed, device_ids=[args.rank])
        # self.netG, self.netD, self.optimizer_G, self.optimizer_D = get_model(args)
        # self.model.cuda(args.rank)
        # self.netG = torch.nn.parallel.DistributedDataParallel(self.netG, device_ids=[args.rank])
        # self.netD = torch.nn.parallel.DistributedDataParallel(self.netD, device_ids=[args.rank])
        self.mean_arr = torch.tensor([-0.03,-0.088,-0.188])[None,:,None,None].cuda(args.rank)
        self.std_arr = torch.tensor([0.448,0.448,0.450])[None,:,None,None].cuda(args.rank)
        self.img_std_arr = torch.tensor([0.229,0.224,0.225])[None,:,None,None].cuda()
        self.img_mean_arr = torch.tensor([0.485,0.456,0.406])[None,:,None,None].cuda()
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction='mean')
        self.loss = CombinedLoss()
        self.cross_entropy_loss.cuda(args.rank)
        self.loss.cuda(args.rank)
        self.criterionGAN = GANLoss(args.gan_mode)
        self.criterionGAN.cuda(args.rank)
        self.criterionL1 = torch.nn.L1Loss()
        self.criterionL1.cuda(args.rank)
        # args.logger.info('Model info\n%s' % str(self.model))


        torch.backends.cudnn.benchmark = True
        self.global_step = 0

        if args.resume is not None:
            self.load_checkpoint(args.resume)

        if args.rank == 0:
            self.writer = SummaryWriter(args.path)

        train_dataset, val_dataset = get_dataset(args)
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size//args.gpus, shuffle=False,
            num_workers=args.workers, pin_memory=True, sampler=train_sampler)
        self.val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size//args.gpus, shuffle=False,
            num_workers=args.workers, pin_memory=True, sampler=val_sampler)

        self.args = args
        self.epoch = 0
        self.args.logger.debug('Finish init trainer')

    def set_epoch(self, epoch):
        self.args.logger.info("Start of epoch %d" % (epoch+1))
        self.epoch = epoch + 1
        self.train_loader.sampler.set_epoch(epoch)
        self.val_loader.sampler.set_epoch(epoch)
        if self.args.optimizer == 'sgd':
            self.lr_scheduler.step(epoch)
            if self.args.rank == 0:
                self.writer.add_scalar('other/lr-epoch', self.optimizer.param_groups[0]['lr'], self.epoch)

    def train(self):
        self.args.logger.info('Training started')
        # self.netG.train()
        # self.netD.train()
        self.gridnet.train()
        self.hed.train()
        end = time()
        for i, (frame1, seg1, frame2, seg2, frame3, seg3) in enumerate(self.train_loader):
            load_time = time() -end
            end = time()
            # for tensorboard
            self.global_step += 1

            # forward pass
            # x = torch.cat([seg1, frame1, frame2, seg2], dim=1) # zeroth is batch size
            # x = x.cuda(self.args.rank, non_blocking=True)
            frame1 = frame1.cuda(self.args.rank, non_blocking=True)
            frame2 = frame2.cuda(self.args.rank, non_blocking=True)
            frame3 = frame3.cuda(self.args.rank, non_blocking=True)
            seg1 = seg1.cuda(self.args.rank, non_blocking=True)
            seg2 = seg2.cuda(self.args.rank, non_blocking=True)
            seg3 = seg3.cuda(self.args.rank, non_blocking=True)

            # get edge
            with torch.no_grad():
                e1 = self.hed(frame1)
                e2 = self.hed(frame2)
                frame1 = (frame1 - self.img_mean_arr) / self.img_std_arr
                frame2 = (frame2 - self.img_mean_arr) / self.img_std_arr
                frame3 = (frame3 - self.img_mean_arr) / self.img_std_arr

            x = torch.cat([e1, seg1, frame1, frame2, seg2, e2], dim=1).cuda(self.args.rank, non_blocking=True)

            # random flipping
            if random.random() < 0.5:
                with torch.no_grad():
                    x = torch.flip(x, [3])
                    frame1 = torch.flip(frame1, [3])
                    frame2 = torch.flip(frame2, [3])
                    frame3 = torch.flip(frame3, [3])
                    seg3 = torch.flip(seg3, [2]) # N*H*W

            # seg, img = self.netG(x)
            seg, img = self.gridnet(x)
            # normalize img from range [-1,1] to range [(0-0.485)/0.229, (1-0.485)/0.229] for the first channel
            img  = F.tanh(img)
            img = (img - self.mean_arr) / self.std_arr

            with torch.no_grad():
                g_img = img * self.img_std_arr + self.img_mean_arr
                e3 = self.hed(g_img)
            
            
            ########################################################
            # self.set_requires_grad(self.netD, True)  # enable backprop for D
            # for param in self.netD.parameters():
            #     param.requires_grad = True
            # self.optimizer_D.zero_grad()     # set D's gradients to zero
            # # self.backward_D()                # calculate gradients for D
            # fake_AB = torch.cat((frame1, frame2, img), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
            # pred_fake = self.netD(fake_AB.detach())
            # self.loss_D_fake = self.criterionGAN(pred_fake, False)
            # # Real
            # real_AB = torch.cat((frame1, frame2, frame3), 1)
            # pred_real = self.netD(real_AB)
            # self.loss_D_real = self.criterionGAN(pred_real, True)
            # # combine loss and calculate gradients
            # self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
            # self.sync([self.loss_D])
            # self.loss_D.backward()
            # self.optimizer_D.step()          # update D's weights
            ##########################################################
            # update G
            # self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
            # for param in self.netD.parameters():
            #     param.requires_grad = False
            # self.optimizer_G.zero_grad()        # set G's gradients to zero
            # fake_AB = torch.cat((frame1, frame2, img), 1)
            # pred_fake = self.netD(fake_AB)
            # self.loss_G_GAN = self.criterionGAN(pred_fake, True)
            # Second, G(A) = B
            self.optimizer.zero_grad()
            self.loss_G_L1 = self.criterionL1(img, frame3) * 40
            self.style_loss = self.loss(output=img, target=frame3) * 20
            self.seg_loss = self.cross_entropy_loss(input=seg, target=seg3) * 10
            self.loss_G = self.loss_G_L1 + self.style_loss + self.seg_loss
            # self.loss_G_NOTGAN
            self.args.logger.debug(self.loss_G)
            # combine loss and calculate gradients
            # self.loss_G = self.loss_G_GAN + self.loss_G_NOTGAN
            self.sync([self.loss_G])
            self.loss_G.backward()
            self.optimizer.step()
            # self.optimizer_G.step()             # udpate G's weights

            comp_time = time() - end
            end = time()

            seg = self.vis_seg_mask(seg, 20, argmax=True)
            seg_gt = self.vis_seg_mask(seg3, 20, argmax=False)

            # print
            if self.args.rank == 0 and i % self.args.print_freq == 0:
                self.args.logger.info(
                    'Epoch [{epoch:d}/{tot_epoch:d}][{cur_batch:d}/{tot_batch:d}] '
                    'load [{load_time:.3f}s] comp [{comp_time:.3f}s] '
                    'loss [{loss:.4f}]'.format(
                        epoch=self.epoch, tot_epoch=self.args.epochs,
                        cur_batch=i+1, tot_batch=len(self.train_loader),
                        load_time=load_time, comp_time=comp_time,
                        loss=self.loss_G.item()
                    )
                )
                # self.writer.add_scalar('train/disc loss', self.loss_D.item(), self.global_step)
                self.writer.add_scalar('train/gen loss GAN', self.loss_G.item(), self.global_step)
                # self.writer.add_scalar('train/gen loss NOTGAN', self.loss_G_NOTGAN.item(), self.global_step)
                self.writer.add_image('train/img gt', make_grid(frame3, normalize=True), self.global_step)
                self.writer.add_image('train/img', make_grid(img, normalize=True), self.global_step)
                self.writer.add_image('train/seg gt', make_grid(seg_gt, normalize=True), self.global_step)
                self.writer.add_image('train/seg', make_grid(seg, normalize=True), self.global_step)
                self.writer.add_image('train/edge', make_grid(e3, normalize=True), self.global_step)


    def validate(self):
        self.args.logger.info('Validation started')
        # self.netG.eval()
        # self.netD.eval()
        self.hed.eval()
        self.gridnet.eval()

        val_loss = AverageMeter()

        with torch.no_grad():
            end = time()
            for i, (frame1, seg1, frame2, seg2, frame3, seg3) in enumerate(self.val_loader):
                load_time = time()-end
                end = time()

                # forward pass
                # x = torch.cat([seg1, frame1, frame2, seg2], dim=1) # zeroth is batch size, first is channel
                # x = x.cuda(self.args.rank, non_blocking=True)
                frame1 = frame1.cuda(self.args.rank, non_blocking=True)
                frame2 = frame2.cuda(self.args.rank, non_blocking=True)
                frame3 = frame3.cuda(self.args.rank, non_blocking=True)
                seg1 = seg1.cuda(self.args.rank, non_blocking=True)
                seg2 = seg2.cuda(self.args.rank, non_blocking=True)
                seg3 = seg3.cuda(self.args.rank, non_blocking=True)
                # self.args.logger.debug(x.shape)
                # get edge
                with torch.no_grad():
                    e1 = self.hed(frame1)
                    e2 = self.hed(frame2)
                    frame1 = (frame1 - self.img_mean_arr) / self.img_std_arr
                    frame2 = (frame2 - self.img_mean_arr) / self.img_std_arr
                    frame3 = (frame3 - self.img_mean_arr) / self.img_std_arr

                x = torch.cat([e1, seg1, frame1, frame2, seg2, e2], dim=1).cuda(self.args.rank, non_blocking=True)

                seg, img = self.gridnet(x)
                # seg, img = self.netG(x)
                # normalize img from range [-1,1] to range [(0-0.485)/0.229, (1-0.485)/0.229] for the first channel
                img = F.tanh(img)
                img = (img - self.mean_arr) / self.std_arr
                

                # img = 2.5 * F.tanh(img)   # if normalize gt image
                # img = F.sigmoid(img)
                self.loss_G_L1 = self.criterionL1(img, frame3) * 40
                self.style_loss = self.loss(output=img, target=frame3) * 20
                self.seg_loss = self.cross_entropy_loss(input=seg, target=seg3) * 10
                loss = self.loss_G_L1 + self.style_loss + self.seg_loss
                self.args.logger.debug(loss)
                # loss and accuracy
                # img.size(0) should be batch size
                size = torch.tensor(float(img.size(0))).cuda(self.args.rank) # pylint: disable=not-callable
                loss.mul_(size)
                self.sync([loss], mean=False) # sum
                loss.div_(size)
                val_loss.update(loss.item(), size.item())

                seg = torch.argmax(seg, dim=1).unsqueeze_(1).float() # from NCHW to N1HW
                seg3 = seg3.unsqueeze_(1).float() # from [N,H,W] to [N,1,H,W]

                # img_gt = frame3 * std_arr + mean_arr
                # img_gen = img * std_arr + mean_arr
                # save validate result
                if self.epoch % 1 == 0 and self.args.rank == 0 and i % 100 == 0:
                    p = torch.cat([frame1.cuda(), frame2.cuda(), frame3, img, seg1.cuda(), seg2.cuda(), seg3, seg], dim=1)
                    p = p.cpu().detach().numpy()
                    np.save('../predict/val_'+str(end)+'_'+str(i).zfill(6)+'.npy', p)

                # if self.epoch % 5 == 0 and self.args.rank == 0 and i % 100 == 0:
                #     self.generate_sequence(frame1, frame2, seg1, seg2)
                
                comp_time = time() - end
                end = time()

                # print
                if self.args.rank == 0 and i % self.args.print_freq == 0:
                    self.args.logger.info(
                        'Epoch [{epoch:d}/{tot_epoch:d}][{cur_batch:d}/{tot_batch:d}] '
                        'load [{load_time:.3f}s] comp [{comp_time:.3f}s]'.format(
                            epoch=self.epoch, tot_epoch=self.args.epochs,
                            cur_batch=i+1, tot_batch=len(self.val_loader),
                            load_time=load_time, comp_time=comp_time,
                        )
                    )

        if self.args.rank == 0:
            self.args.logger.info(
                'Epoch [{epoch:d}/{tot_epoch:d}] loss [{loss:.4f}] '.format(
                    epoch=self.epoch, tot_epoch=self.args.epochs,
                    loss=val_loss.avg,
                )
            )
            self.writer.add_scalar('val/loss', val_loss.avg, self.epoch)

        return {'loss': val_loss.avg}

    def sync(self, tensors, mean=True):
        '''Synchronize all tensors given using mean or sum.'''
        for tensor in tensors:
            dist.all_reduce(tensor)
            if mean:
                tensor.div_(self.args.gpus)

    

    def save_checkpoint(self, metrics):
        self.args.logger.info('Saving checkpoint..')
        prefix = '../checkpoint'
        torch.save({
            'epoch': self.epoch,
            'arch': self.args.arch,
            'generator': self.gridnet.module.state_dict(), # data parallel
            'optimizer': self.optimizer.state_dict(),
        }, '%s/%03d.pth' % (prefix, self.epoch))
        shutil.copy('%s/%03d.pth' % (prefix, self.epoch),
            '%s/latest.pth' % prefix)

    def load_checkpoint(self, resume):
        self.args.logger.info('Resuming checkpoint %s' % resume)
        ckpt = torch.load(resume)
        assert ckpt['arch'] == self.args.arch, ('Architecture mismatch: ckpt %s, config %s'
            % (ckpt['arch'], self.args.arch))

        self.epoch = ckpt['epoch']
        self.gridnet.load_state_dict(ckpt['generator'])
        self.optimizer.load_state_dict(ckpt['optimizer'])

        self.args.logger.info('Checkpoint loaded')

    def vis_seg_mask(self, seg, n_classes, argmax=False):
        '''
        mask (bs, c,h,w) into normed rgb (bs, 3,h,w)
        all tensors
        '''
        global color_map
        if argmax:
            id_seg = torch.argmax(seg, dim=1)
        else: id_seg = seg
        color_mapp = torch.tensor(color_map)
        rgb_seg = color_mapp[id_seg].permute(0,3,1,2).contiguous().float()
        return rgb_seg/255

    def eval_generate_sequence(self, img1, img2, seg1, seg2):
        seg1 = cv2.imread(seg1, cv2.IMREAD_GRAYSCALE)
        seg2 = cv2.imread(seg2, cv2.IMREAD_GRAYSCALE)
        img1 = cv2.imread(img1)
        img2 = cv2.imread(img2)
        if seg1 is None or seg2 is None or img1 is None or img2 is None:
            self.args.logger.debug('path name not exists')
            return
        seg1 = cv2.resize(seg1, dsize=(256,256), interpolation=cv2.INTER_NEAREST)
        seg2 = cv2.resize(seg2, dsize=(256,256), interpolation=cv2.INTER_NEAREST)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB) 
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB) 
        to_normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
        ])
        img1, img2 = to_normalize(img1).unsqueeze_(0), to_normalize(img2).unsqueeze_(0)
        seg1, seg2 = torch.from_numpy(seg1), torch.from_numpy(seg2)
        seg1, seg2 = seg1.float().unsqueeze_(0).unsqueeze_(0), seg2.float().unsqueeze_(0).unsqueeze_(0)
        self.args.logger.debug(seg1.shape)
        self.args.logger.debug(img1.shape)
        self.generate_sequence(img1, img2, seg1, seg2)


    def generate_sequence(self, img1, img2, seg1, seg2):
        img, seg = [], []
        img.append(img1.cuda(self.args.rank, non_blocking=True))
        img.append(img2.cuda(self.args.rank, non_blocking=True))
        seg.append(seg1.cuda(self.args.rank, non_blocking=True))
        seg.append(seg2.cuda(self.args.rank, non_blocking=True))
        with torch.no_grad():
            for i in range(8):
                x = torch.cat([seg[-2], img[-2], img[-1], seg[-1]], dim=1) # zeroth is batch size, first is channel
                x = x.cuda(self.args.rank, non_blocking=True)
                # self.args.logger.debug(x.shape)
                seg_next, img_next = self.netG(x)
                # img_next = F.tanh(img_next)
                img_next = (img_next - self.mean_arr) / self.std_arr
                seg_next = torch.argmax(seg_next, dim=1).unsqueeze_(1).float() # from NCHW to N1HW
                img.append(img_next)
                seg.append(seg_next)
            p = torch.cat(img, dim=1)
            q = torch.cat(seg, dim=1)
            p = p.cpu().detach().numpy()
            q = q.cpu().detach().numpy()
            t = time()
            np.save('../predict/val_'+str(t)+'_'+'img'+'.npy', p)
            np.save('../predict/val_'+str(t)+'_'+'seg'+'.npy', q)







