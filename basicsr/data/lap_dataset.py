import numpy as np
import torch
from pathlib import Path
from torch.utils import data as data
from basicsr.data.transforms import paired_random_crop, four_random_crop
from basicsr.utils import img2tensor
import glob
import os
import torchvision.transforms as transforms
import torch.nn.functional as F

def unsqueeze_twice(x):
    return x.unsqueeze(0).unsqueeze(0)

def warp(img,jit):
    jit = torch.from_numpy(-jit)
    # img = torch.from_numpy(img).float() # error: uint16 convert don't support
    img = torch.from_numpy(img.astype(np.int32)).float() # val psnr error, np.int32 pr np.float32 instead of np.int16
    h, w = img.shape
    grid_y, grid_x = torch.meshgrid(torch.arange(0, h), torch.arange(0, w))
    grid = torch.stack((grid_x, grid_y), 2).type_as(img)
    grid.requires_grad = False

    grid_flow = grid + jit
    grid_flow = grid_flow.unsqueeze(0)
    grid_flow = grid_flow[:, :h, :w, :]
    grid_flow_x = 2.0 * grid_flow[:, :, :, 0] / max(w - 1, 1) - 1.0  
    grid_flow_y = 2.0 * grid_flow[:, :, :, 1] / max(h - 1, 1) - 1.0
    grid_flow = torch.stack((grid_flow_x, grid_flow_y), dim = 3)

    img_tensor = unsqueeze_twice(img)
    img_subdivision = F.grid_sample(img_tensor, grid_flow, 
        mode = 'bilinear', padding_mode = "reflection", align_corners = True) # nearest
    img = np.array(img_subdivision).astype(int)[0,0,:,:]
    return img

class LAPDataSet(data.Dataset):
    def __init__(self, opt):
        super(LAPDataSet, self).__init__()
        self.opt = opt
        print(opt) 
        self.path = sorted(glob.glob(os.path.join(Path(opt['dataroot_gt'])) + "/*.*"))
        self.transform = transforms.Compose( [transforms.ToTensor(),])
    
    def __getitem__(self, index):
        index = int(index)
        data_dict = np.load(self.path[index % len(self.path)], allow_pickle=True )
        data_dict = data_dict.item()

        img = data_dict["img_LAP"]
        gt = data_dict["img_gt"]
        flow = data_dict["jit_information_noise"]
        flow_gt = data_dict["jit_information"]

        img = np.expand_dims(img, axis=2)
        gt = np.expand_dims(gt, axis=2)

        # normalization
        img = img.astype(np.float32) / 255. / 255.
        gt = gt.astype(np.float32) / 255. / 255.
     
        if self.opt['name'] == 'TrainSet':
            scale = self.opt['scale']
            gt_size = self.opt['gt_size']
            # gt, img = paired_random_crop(gt, img, gt_size, scale, None) 
            gt, img, flow, flow_gt = four_random_crop(gt, img, flow, flow_gt, gt_size, scale, None) 

        img = img2tensor(img)
        gt = img2tensor(gt)
        flow = img2tensor(flow)
        flow_gt = img2tensor(flow_gt)

        return {'lq': img, 'gt': gt, 'flow': flow,"flow_gt": flow_gt,"lq_path": self.path[index % len(self.path)]}

    def __len__(self):
        return len(self.path)


class LAPDataSetNoWarp(data.Dataset):
    def __init__(self, opt):
        super(LAPDataSetNoWarp, self).__init__()
        self.opt = opt
        print(opt) 
        self.path = sorted(glob.glob(os.path.join(Path(opt['dataroot_gt'])) + "/*.*"))
        self.transform = transforms.Compose( [transforms.ToTensor(),])
    
    def __getitem__(self, index):
        index = int(index)
        data_dict = np.load(self.path[index % len(self.path)], allow_pickle=True )
        data_dict = data_dict.item()

        img = data_dict["img_LAP"]
        gt = data_dict["img_gt"]

        img = np.expand_dims(img, axis=2)
        gt = np.expand_dims(gt, axis=2)

        # normalization
        img = img.astype(np.float32) / 255. / 255.
        gt = gt.astype(np.float32) / 255. / 255.
     
        if self.opt['name'] == 'TrainSet':
            scale = self.opt['scale']
            gt_size = self.opt['gt_size']
            gt, img = paired_random_crop(gt, img, gt_size, scale, None) 

        img = img2tensor(img)
        gt = img2tensor(gt)

        return {'lq': img, 'gt': gt,"lq_path": self.path[index % len(self.path)]}

    def __len__(self):
        return len(self.path)


class LAPDataSetWarp(data.Dataset):
    def __init__(self, opt):
        super(LAPDataSetWarp, self).__init__()
        self.opt = opt
        print(opt) 
        self.path = sorted(glob.glob(os.path.join(Path(opt['dataroot_gt'])) + "/*.*"))
        self.transform = transforms.Compose( [transforms.ToTensor(),])
    
    def __getitem__(self, index):
        index = int(index)
        data_dict = np.load(self.path[index % len(self.path)], allow_pickle=True )
        data_dict = data_dict.item()

        img = data_dict["img_LAP"]
        gt = data_dict["img_gt"]
        flow = data_dict["jit_information_noise"]
        flow_gt = data_dict["jit_information"]
        
        img = warp(img,flow)

        img = np.expand_dims(img, axis=2)
        gt = np.expand_dims(gt, axis=2)

        # normalization
        img = img.astype(np.float32) / 255. / 255.
        gt = gt.astype(np.float32) / 255. / 255.
     
        if self.opt['name'] == 'TrainSet':
            scale = self.opt['scale']
            gt_size = self.opt['gt_size']
            # gt, img = paired_random_crop(gt, img, gt_size, scale, None) 
            gt, img, flow, flow_gt = four_random_crop(gt, img, flow, flow_gt, gt_size, scale, None) 

        img = img2tensor(img)
        gt = img2tensor(gt)
        flow = img2tensor(flow)
        flow_gt = img2tensor(flow_gt)

        return {'lq': img, 'gt': gt, 'flow': flow,"flow_gt": flow_gt,"lq_path": self.path[index % len(self.path)]}

    def __len__(self):
        return len(self.path)