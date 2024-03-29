import torch

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def vis_seg_mask(seg, n_classes, seg_id=False):
	'''
		mask (bs, c,h,w) into normed rgb (bs, 3,h,w)
		all tensors
	'''
	global color_map
	assert len(seg.size()) == 4
	if not seg_id:
		id_seg = torch.argmax(seg, dim=1)
	else:
		id_seg = seg.squeeze(1).long()
	color_mapp = torch.tensor(color_map)
	rgb_seg = color_mapp[id_seg].permute(0,3,1,2).contiguous().float()
	return rgb_seg/255

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

class LambdaLR:
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert (n_epochs - decay_start_epoch) > 0, "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)