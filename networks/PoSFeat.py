'''
WSFModel without global header
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

from abc import ABC, abstractmethod
from path import Path
import os

import networks

__all__=['PoSFeat']

class PoSFeat(ABC):
    def __init__(self, configs, device, no_cuda=None):
        self.config = configs
        self.device = device
        self.no_cuda = no_cuda
        self.align_local_grad = self.config['align_local_grad']
        self.local_input_elements = self.config['local_input_elements']
        self.local_with_img = self.config['local_with_img']
        self.parameters = []

        self.backbone = networks.ResUNet(**self.config['backbone_config']).to(self.device)
        self.parameters += list(self.backbone.parameters())

        self.localheader = networks.SpHeader_HR_Deep_IN(**self.config['localheader_config']).to(self.device)
        self.parameters += list(self.localheader.parameters())
        self.modules = ['localheader', 'backbone']

    def set_parallel(self, local_rank):
        self.backbone = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.backbone)
        self.backbone = torch.nn.parallel.DistributedDataParallel(self.backbone,
            find_unused_parameters=True,device_ids=[local_rank],output_device=local_rank)

        self.localheader = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.localheader)
        self.localheader = torch.nn.parallel.DistributedDataParallel(self.localheader,
            find_unused_parameters=True,device_ids=[local_rank],output_device=local_rank)

    def load_checkpoint(self, load_path):
        load_root = Path(load_path)
        model_list = ['backbone', 'localheader']
        for name in model_list:
            model_path = load_root/'{}.pth'.format(name)
            if os.path.exists(model_path):
                print('load {} from checkpoint'.format(name))
            else:
                print('{} does not exist, skipping load'.format(name))
                continue
            model = getattr(self, name)
            model_param = torch.load(model_path)
            model.load_state_dict(model_param)

    def save_checkpoint(self, save_path):
        save_root = Path(save_path)
        model_list = ['backbone', 'localheader']
        for name in model_list:
            model_path = save_root/'{}.pth'.format(name)
            model = getattr(self, name)
            model_param = model.state_dict()
            torch.save(model_param, model_path)

    def set_train(self):
        self.backbone.train()
        self.localheader.train()

    def set_eval(self):
        self.backbone.eval()
        self.localheader.eval()

    def extract(self, img_tensor, postfix=""):
        feat_maps = self.backbone(img_tensor)
        b, c, h, w = feat_maps['global_map'].shape
        g_map = torch.ones(b,1, h, w).type_as(feat_maps['local_map']).to(feat_maps['local_map'].device)
        local_list = []
        for name in self.local_input_elements:
            local_list.append(feat_maps[name])
        local_input = torch.cat(local_list, dim=1)
        if not self.align_local_grad:
            local_input = local_input.detach()
        if self.local_with_img:
            local_input = [local_input, img_tensor]
        l_map = self.localheader(local_input)

        if l_map.shape[1] == 1:
            local_thr = torch.zeros_like(l_map)
        elif l_map.shape[1] == 2:
            local_thr = l_map[:,1:,:,:]
            l_map = l_map[:,:1,:,:]

        g_desc = g_map*feat_maps['global_map']
        # g_desc = g_desc.sum([2,3])
        g_desc = F.normalize(g_desc, p=2, dim=1).mean([2,3])

        outputs = {
            'local_map': feat_maps['local_map'],
            'global_map': feat_maps['global_map'],
            'global_feat': g_desc,
            'local_point': l_map,
            'local_thr': local_thr,
            'global_point': g_map
        }
        return outputs

    def forward(self, inputs):
        for key, val in inputs.items():
            if key in self.no_cuda:
                continue
            inputs[key] = val.to(self.device)

        preds1 = self.extract(inputs['im1'],1)
        preds2 = self.extract(inputs['im2'],2)

        return {'preds1':preds1, 'preds2':preds2}

