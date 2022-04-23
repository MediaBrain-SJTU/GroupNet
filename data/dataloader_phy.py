from data.nuscenes_pred_split import get_nuscenes_pred_split
import os, random, numpy as np, copy

from .preprocessor import preprocess
from .ethucy_split import get_ethucy_split
from utils.utils import print_log
from torch.utils.data import Dataset
import torch


def seq_collate(data):
    # batch_abs, batch_norm, shift_value, seq_list, nei_list, nei_num, batch_pednum = inputs

    (pre_motion_3D, fut_motion_3D,pre_motion_mask,fut_motion_mask,motion_link) = zip(*data)

    pre_motion_3D = torch.stack(pre_motion_3D,dim=0)
    fut_motion_3D = torch.stack(fut_motion_3D,dim=0)
    fut_motion_mask = torch.stack(fut_motion_mask,dim=0)
    pre_motion_mask = torch.stack(pre_motion_mask,dim=0)

    # print(pre_motion_3D.shape)
    # print(fut_motion_3D.shape)
    # print(fut_motion_mask.shape)
    # print(pre_motion_mask.shape)
    # time.sleep(1000)
    # batch_abs = torch.cat(batch_abs_list,dim=0).permute(1,0,2)
    # # print(batch_abs.shape)
    # # .permute(1,0,2,3)
    # # batch_abs = batch_abs.view(batch_abs.shape[0],batch_abs.shape[1]*batch_abs.shape[2],batch_abs.shape[3])
    # batch_norm = torch.cat(batch_norm_list,dim=0).permute(1,0,2)
    # # batch_norm = batch_abs.view(batch_norm.shape[0],batch_norm.shape[1]*batch_norm.shape[2],batch_norm.shape[3])
    # shift_value = torch.cat(shift_value_list,dim=0).permute(1,0,2)
    # # shift_value = shift_value.view(shift_value.shape[0],shift_value.shape[1]*shift_value.shape[2],shift_value.shape[3])
    # seq_list = torch.ones(batch_abs.shape[0],batch_abs.shape[1])
    # batch_size = int(batch_abs.shape[1] / 11)
    # nei_list = torch.from_numpy(np.kron(np.diag([1]*batch_size),np.ones((11,11),dtype='float32'))-np.eye(batch_size*11)).repeat(batch_abs.shape[0],1,1)
    # nei_num = torch.ones(batch_abs.shape[0],batch_abs.shape[1]) * 10
    # batch_pednum = torch.from_numpy(np.array([11]*batch_size))

    data = {
        'pre_motion_3D': pre_motion_3D,
        'fut_motion_3D': fut_motion_3D,
        'fut_motion_mask': fut_motion_mask,
        'pre_motion_mask': pre_motion_mask,
        'traj_scale': 1,
        'pred_mask': None,
        'seq': 'phy',
        'link': motion_link,
    }
    # out = [
    #     batch_abs, batch_norm, shift_value, seq_list, nei_list, nei_num, batch_pednum 
    # ]

    return data

class PHYDataset(Dataset):
    def __init__(
        self, obs_len=5, pred_len=10, training=True
    ):        
        super(PHYDataset, self).__init__()
        # data_root = '/DATA5_DB8/data/cxxu/traj_forecast/PECNet/phy_simulate_data/traj_3type.npy'
        # link_root = '/DATA5_DB8/data/cxxu/traj_forecast/PECNet/phy_simulate_data/link_3type.npy'
        # print('3 type')
        # data_root = '/DATA5_DB8/data/cxxu/traj_forecast/PECNet/phy_simulate_data/traj_charge2.npy'
        # link_root = '/DATA5_DB8/data/cxxu/traj_forecast/PECNet/phy_simulate_data/factor_charge2.npy'
        # print('charge')
        data_root = '/GPFS/data/cxxu/trajectory_prediction/AgentFormer/datasets/phy/traj_springstaff6.npy'
        link_root = '/GPFS/data/cxxu/trajectory_prediction/AgentFormer/datasets/phy/link_springstaff6.npy'
        print('hyperedge')
        self.trajs = np.load(data_root) #(N,12,30,2)
        self.links = np.load(link_root,allow_pickle=True)
        if training:
            self.trajs = self.trajs[:10000]
            self.links = self.links[:10000]
        else:
            self.trajs = self.trajs[40000:]
            self.links = self.links[40000:]

        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = self.obs_len + self.pred_len

        self.trajs_norm = self.trajs[:,:,:20,:] #- self.trajs[:,:1,:,:]
        self.links = list(self.links)
        print(self.trajs_norm.shape)
        print(self.links[:5])
        self.ball_num = self.trajs_norm.shape[1]
        # print(self.links[0:5])
        # print(self.trajs.shape)
        self.batch_len = len(self.trajs_norm)
        # print(self.batch_len)

        self.traj_abs = torch.from_numpy(self.trajs_norm).type(torch.float)
        # self.traj_norm = torch.from_numpy(self.trajs-self.trajs[:,self.obs_len-1:self.obs_len]).type(torch.float)

        # self.traj_abs = self.traj_abs.permute(0,2,1,3)
        # self.traj_norm = self.traj_norm.permute(0,2,1,3)


    def __len__(self):
        return len(self.trajs_norm)
    
    def __getitem__(self, index):
        # print(self.traj_abs.shape)
        pre_motion_3D = self.traj_abs[index, :, :self.obs_len, :]
        fut_motion_3D = self.traj_abs[index, :, self.obs_len:, :]
        pre_motion_mask = torch.ones(self.ball_num,self.obs_len)
        fut_motion_mask = torch.ones(self.ball_num,self.pred_len)
        motion_link = self.links[index]
        out = [
            pre_motion_3D, fut_motion_3D,
            pre_motion_mask, fut_motion_mask, motion_link
        ]
        return out

class NBADataset(Dataset):
    """Dataloder for the Trajectory datasets"""
    def __init__(
        self, obs_len=5, pred_len=10, training=True
    ):
        """
        Args:
        - data_dir: Directory containing dataset files in the format
        <frame_id> <ped_id> <x> <y>
        - obs_len: Number of time-steps in input trajectories
        - pred_len: Number of time-steps in output trajectories
        - skip: Number of frames to skip while making the dataset
        - threshold: Minimum error to be considered for non linear traj
        when using a linear predictor
        - min_ped: Minimum number of pedestrians that should be in a seqeunce
        - delim: Delimiter in the dataset files
        """

        super(NBADataset, self).__init__()

        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = self.obs_len + self.pred_len
        # self.norm_lap_matr = norm_lap_matr

        if training:
            data_root = '/DATA7_DB7/data/cxxu/NBA-Player-Movements/data_subset/subset_new_new/train.npy'
        else:
            data_root = '/DATA7_DB7/data/cxxu/NBA-Player-Movements/data_subset/subset_new_new/test.npy'

        self.trajs = np.load(data_root) #(N,15,11,2)
        self.trajs /= (94/28) 
        if training:
            self.trajs = self.trajs[:32500]
        else:
            self.trajs = self.trajs[:12500]

        self.batch_len = len(self.trajs)
        print(self.batch_len)

        self.traj_abs = torch.from_numpy(self.trajs).type(torch.float)
        self.traj_norm = torch.from_numpy(self.trajs-self.trajs[:,self.obs_len-1:self.obs_len]).type(torch.float)

        self.traj_abs = self.traj_abs.permute(0,2,1,3)
        self.traj_norm = self.traj_norm.permute(0,2,1,3)
        # print(self.traj_abs.shape)

    def __len__(self):
        return self.batch_len

    def __getitem__(self, index):
        # print(self.traj_abs.shape)
        pre_motion_3D = self.traj_abs[index, :, :self.obs_len, :]
        fut_motion_3D = self.traj_abs[index, :, self.obs_len:, :]
        pre_motion_mask = torch.ones(11,self.obs_len)
        fut_motion_mask = torch.ones(11,self.pred_len)
        out = [
            pre_motion_3D, fut_motion_3D,
            pre_motion_mask, fut_motion_mask
        ]
        return out

class data_generator(object):
    def __init__(self, parser, log, split='train', phase='training'):
        self.past_frames = parser.past_frames
        self.min_past_frames = parser.min_past_frames
        self.frame_skip = parser.get('frame_skip', 1)
        self.phase = phase
        self.split = split
        assert phase in ['training', 'testing'], 'error'
        assert split in ['train', 'val', 'test'], 'error'

        if parser.dataset == 'nuscenes_pred':
            data_root = parser.data_root_nuscenes_pred           
            seq_train, seq_val, seq_test = get_nuscenes_pred_split(data_root)
            self.init_frame = 0
        elif parser.dataset in {'eth', 'hotel', 'univ', 'zara1', 'zara2'}:
            data_root = parser.data_root_ethucy            
            seq_train, seq_val, seq_test = get_ethucy_split(parser.dataset)
            self.init_frame = 0
        else:
            raise ValueError('Unknown dataset!')

        process_func = preprocess
        self.data_root = data_root

        print_log("\n-------------------------- loading %s data --------------------------" % split, log=log)
        if self.split == 'train':  self.sequence_to_load = seq_train
        elif self.split == 'val':  self.sequence_to_load = seq_val
        elif self.split == 'test': self.sequence_to_load = seq_test
        else:                      assert False, 'error'

        self.num_total_samples = 0
        self.num_sample_list = []
        self.sequence = []
        for seq_name in self.sequence_to_load:
            print_log("loading sequence {} ...".format(seq_name), log=log)
            preprocessor = process_func(data_root, seq_name, parser, log, self.split, self.phase)

            num_seq_samples = preprocessor.num_fr - (parser.min_past_frames - 1) * self.frame_skip - parser.min_future_frames * self.frame_skip + 1
            self.num_total_samples += num_seq_samples
            self.num_sample_list.append(num_seq_samples)
            self.sequence.append(preprocessor)
            
        self.sample_list = list(range(self.num_total_samples))
        self.index = 0
        print_log(f'total num samples: {self.num_total_samples}', log)
        print_log("------------------------------ done --------------------------------\n", log=log)

    def shuffle(self):
        random.shuffle(self.sample_list)
        
    def get_seq_and_frame(self, index):
        index_tmp = copy.copy(index)
        for seq_index in range(len(self.num_sample_list)):    # 0-indexed
            if index_tmp < self.num_sample_list[seq_index]:
                frame_index = index_tmp + (self.min_past_frames - 1) * self.frame_skip + self.sequence[seq_index].init_frame     # from 0-indexed list index to 1-indexed frame index (for mot)
                return seq_index, frame_index
            else:
                index_tmp -= self.num_sample_list[seq_index]

        assert False, 'index is %d, out of range' % (index)

    def is_epoch_end(self):
        if self.index >= self.num_total_samples:
            self.index = 0      # reset
            return True
        else:
            return False

    def next_sample(self):
        sample_index = self.sample_list[self.index]
        seq_index, frame = self.get_seq_and_frame(sample_index)
        seq = self.sequence[seq_index]
        self.index += 1
        
        data = seq(frame)
        return data      

    def __call__(self):
        return self.next_sample()
