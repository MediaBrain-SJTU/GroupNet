from .preprocessor import preprocess
from .ethucy_split import get_ethucy_split
from utils.utils import print_log
from torch.utils.data import Dataset
import torch

from IPython import embed
import glob
import pandas as pd
import pickle
import os
import torch
from torch import nn
from torch.utils import data
import random
import numpy as np


def seq_collate(data):
    # batch_abs, batch_norm, shift_value, seq_list, nei_list, nei_num, batch_pednum = inputs

    (pre_motion_3D, fut_motion_3D,pre_motion_mask,fut_motion_mask) = zip(*data)

    pre_motion_3D = torch.cat(pre_motion_3D,dim=0)
    fut_motion_3D = torch.cat(fut_motion_3D,dim=0)
    fut_motion_mask = torch.cat(fut_motion_mask,dim=0)
    pre_motion_mask = torch.cat(pre_motion_mask,dim=0)

    data = {
        'pre_motion_3D': pre_motion_3D,
        'fut_motion_3D': fut_motion_3D,
        'fut_motion_mask': fut_motion_mask,
        'pre_motion_mask': pre_motion_mask,
        'traj_scale': 1,
        'pred_mask': None,
        'seq': 'sdd',
    }

    return data
class SDDdataset(data.Dataset):
    def __init__(self, set_name="train", id=False, scale = 1.0):
        if set_name == 'train':
            load_name = '/DATA5_DB8/data/cxxu/traj_forecast/PECNet/social_pool_data/train_all_512_0_100.pickle'
        else:
            load_name = '/DATA5_DB8/data/cxxu/traj_forecast/PECNet/social_pool_data/test_all_4096_0_100.pickle'
        # if set_name == 'train':
        #     load_name = '/DATA5_DB8/data/cxxu/traj_forecast/PECNet/social_pool_data/train_all_512_0_100.pickle'
        # else:
        #     load_name = '/DATA5_DB8/data/cxxu/traj_forecast/NMMP/PMP_NMMP/data/sdd/test_all_512_0_80.pickle'

        print(load_name)
        with open(load_name, 'rb') as f:
            data = pickle.load(f)

        traj, masks = data
        traj_new = []

        if id==False:
            for t in traj:
                t = np.array(t)
                t = t[:,:,2:]
                traj_new.append(t)
                if set_name=="train":
                    #augment training set with reversed tracklets...
                    reverse_t = np.flip(t, axis=1).copy()
                    traj_new.append(reverse_t)
        else:
            for t in traj:
                t = np.array(t)
                traj_new.append(t)

                if set_name=="train":
                    #augment training set with reversed tracklets...
                    reverse_t = np.flip(t, axis=1).copy()
                    traj_new.append(reverse_t)

        masks_new = []
        for m in masks:
            masks_new.append(m)

            if set_name=="train":
                #add second time for the reversed tracklets...
                masks_new.append(m)
        
        seq_start_end_list = []
        for m in masks:
            total_num = m.shape[0]
            scene_start_idx = 0
            num_list = []
            for i in range(total_num):
                if i < scene_start_idx:
                    continue
                scene_actor_num = np.sum(m[i])
                scene_start_idx += scene_actor_num 
                num_list.append(scene_actor_num)
            cum_start_idx = [0] + np.cumsum(np.array(num_list)).tolist()
            seq_start_end = [(start, end) for start, end in zip(cum_start_idx, cum_start_idx[1:])]
            seq_start_end_list.append(seq_start_end)
            if set_name=="train":
                #add second time for the reversed tracklets...
                seq_start_end_list.append(seq_start_end)

        traj_new = np.array(traj_new)
        masks_new = np.array(masks_new)

        self.trajectory_batches = traj_new.copy()
        self.mask_batches = masks_new.copy()
        # self.initial_pos_batches = np.array(initial_pos(self.trajectory_batches)) #for relative positioning
        self.seq_start_end_batches = seq_start_end_list
        self.trajs_abs = []
        self.obs_len = 8
        self.pred_len = 12
        for idx,t in enumerate(self.trajectory_batches):
            for seq_start_end in self.seq_start_end_batches[idx]:
                start,end = seq_start_end
                self.trajs_abs.append(t[start:end]/scale)

    def __len__(self):
        return len(self.trajs_abs)

    def __getitem__(self, index):
        # print(self.traj_abs.shape)
        pre_motion_3D = torch.from_numpy(self.trajs_abs[index][:, :self.obs_len, :]).type(torch.float)
        fut_motion_3D = torch.from_numpy(self.trajs_abs[index][:, self.obs_len:, :]).type(torch.float)
        pre_motion_mask = 1 - (pre_motion_3D[:,:,0] == 0).type(torch.float)
        fut_motion_mask = 1 - (fut_motion_3D[:,:,0] == 0).type(torch.float)
        out = [
            pre_motion_3D, fut_motion_3D,
            pre_motion_mask, fut_motion_mask
        ]
        return out


