import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from collections import defaultdict
from .common.mlp import MLP as MLP2
from .agentformer_loss_resdec_nba import loss_func
from .common.dist import *
from .agentformer_lib import AgentFormerEncoderLayer, AgentFormerDecoderLayer, AgentFormerDecoder, AgentFormerEncoder
from .map_encoder import MapEncoder
from utils.torch import *
from utils.utils import initialize_weights
from .MS_HGNN_batch import NmpNet,NmpNet_hyper_weight, MLP
import math



def generate_ar_mask(sz, agent_num, agent_mask):
    assert sz % agent_num == 0
    T = sz // agent_num
    mask = agent_mask.repeat(T, T)
    for t in range(T-1):
        i1 = t * agent_num
        i2 = (t+1) * agent_num
        mask[i1:i2, i2:] = float('-inf')
    return mask


def generate_mask(tgt_sz, src_sz, agent_num, agent_mask):
    assert tgt_sz % agent_num == 0 and src_sz % agent_num == 0
    mask = agent_mask.repeat(tgt_sz // agent_num, src_sz // agent_num)
    return mask

class DecomposeBlock(nn.Module):
    '''
    Balance between reconstruction task and prediction task.
    '''
    def __init__(self, past_len, future_len, input_dim):
        super(DecomposeBlock, self).__init__()
        # * HYPER PARAMETERS
        channel_in = 2
        channel_out = 32
        dim_kernel = 3
        dim_embedding_key = 96
        self.past_len = past_len
        self.future_len = future_len

        self.conv_past = nn.Conv1d(channel_in, channel_out, dim_kernel, stride=1, padding=1)
        self.encoder_past = nn.GRU(channel_out, dim_embedding_key, 1, batch_first=True)
        
        self.decoder_y = MLP(dim_embedding_key + input_dim, future_len * 2, hidden_size=(512, 256))
        self.decoder_x = MLP(dim_embedding_key + input_dim, past_len * 2, hidden_size=(512, 256))

        self.relu = nn.ReLU()

        # kaiming initialization
        self.init_parameters()

    def init_parameters(self):
        nn.init.kaiming_normal_(self.conv_past.weight)
        nn.init.kaiming_normal_(self.encoder_past.weight_ih_l0)
        nn.init.kaiming_normal_(self.encoder_past.weight_hh_l0)

        nn.init.zeros_(self.conv_past.bias)
        nn.init.zeros_(self.encoder_past.bias_ih_l0)
        nn.init.zeros_(self.encoder_past.bias_hh_l0)


    def forward(self, x_true, x_hat, f):
        '''
        >>> Input:
            x_true: N, T_p, 2
            x_hat: N, T_p, 2
            f: N, f (96)

        >>> Output:
            x_hat_after: N, T_p, 2
            y_hat: n, T_f, 2
        '''
        x_ = x_true - x_hat
        x_ = torch.transpose(x_, 1, 2)
        
        past_embed = self.relu(self.conv_past(x_))
        past_embed = torch.transpose(past_embed, 1, 2)
        # N, T, F

        _, state_past = self.encoder_past(past_embed)
        state_past = state_past.squeeze(0)
        # N, F2

        input_feat = torch.cat((f, state_past), dim=1)

        x_hat_after = self.decoder_x(input_feat).contiguous().view(-1, self.past_len, 2)
        y_hat = self.decoder_y(input_feat).contiguous().view(-1, self.future_len, 2)
        
        return x_hat_after, y_hat


""" Positional Encoding """
class PositionalAgentEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_t_len=200, max_a_len=200, concat=False, use_agent_enc=False, agent_enc_learn=False):
        super(PositionalAgentEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.concat = concat
        self.d_model = d_model
        self.use_agent_enc = use_agent_enc
        if concat:
            self.fc = nn.Linear((3 if use_agent_enc else 2) * d_model, d_model)

        pe = self.build_pos_enc(max_t_len)
        self.register_buffer('pe', pe)
        if use_agent_enc:
            if agent_enc_learn:
                self.ae = nn.Parameter(torch.randn(max_a_len, 1, d_model) * 0.1)
            else:
                ae = self.build_pos_enc(max_a_len)
                self.register_buffer('ae', ae)

    def build_pos_enc(self, max_len):
        pe = torch.zeros(max_len, self.d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-np.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # pe = pe.unsqueeze(0).transpose(0, 1)
        return pe
    
    def get_pos_enc(self, num_t, num_a, t_offset):
        pe = self.pe[t_offset: num_t + t_offset, :]
        pe = pe[None].repeat(num_a,1,1)
        # pe = pe.repeat_interleave(num_a, dim=0)
        return pe

    def get_agent_enc(self, num_t, num_a, a_offset, agent_enc_shuffle):
        if agent_enc_shuffle is None:
            ae = self.ae[a_offset: num_a + a_offset, :]
        else:
            ae = self.ae[agent_enc_shuffle]
        ae = ae.repeat(num_t, 1, 1)
        return ae

    def forward(self, x, num_a, agent_enc_shuffle=None, t_offset=0, a_offset=0):
        num_t = x.shape[1]
        pos_enc = self.get_pos_enc(num_t, num_a, t_offset) #(N,T,D)
        if self.use_agent_enc:
            agent_enc = self.get_agent_enc(num_t, num_a, a_offset, agent_enc_shuffle)
        if self.concat:
            feat = [x, pos_enc]
            if self.use_agent_enc:
                feat.append(agent_enc.repeat(1, x.size(1), 1))
            x = torch.cat(feat, dim=-1)
            x = self.fc(x)
        else:
            x += pos_enc
            if self.use_agent_enc:
                x += agent_enc
        return self.dropout(x) #(N,T,D)


""" Context (Past) Encoder """
class ContextEncoder(nn.Module):
    def __init__(self, cfg, ctx, **kwargs):
        super().__init__()
        self.cfg = cfg
        self.ctx = ctx
        self.motion_dim = ctx['motion_dim']
        # self.model_dim = ctx['tf_model_dim']
        self.model_dim = 16
        self.ff_dim = ctx['tf_ff_dim']
        self.nhead = ctx['tf_nhead']
        self.dropout = ctx['tf_dropout']
        self.nlayer = cfg.get('nlayer', 6)
        self.input_type = ctx['input_type']
        self.pooling = cfg.get('pooling', 'mean')
        self.agent_enc_shuffle = ctx['agent_enc_shuffle']
        self.vel_heading = ctx['vel_heading']
        ctx['context_dim'] = self.model_dim
        in_dim = self.motion_dim * len(self.input_type)
        if 'map' in self.input_type:
            in_dim += ctx['map_enc_dim'] - self.motion_dim
        self.input_fc = nn.Linear(in_dim, self.model_dim)
        self.input_fc2 = nn.Linear(self.model_dim*5,self.model_dim)
        self.input_fc3 = nn.Linear(self.model_dim+3,self.model_dim)

        self.interaction = NmpNet(
            embedding_dim=16,
            h_dim=self.model_dim,
            mlp_dim=64,
            bottleneck_dim=self.model_dim,
            batch_norm=0,
            nmp_layers=2,
            vis=False
        )

        self.interaction_hyper = NmpNet_hyper_weight(
            embedding_dim=16,
            h_dim=self.model_dim,
            mlp_dim=64,
            bottleneck_dim=self.model_dim,
            batch_norm=0,
            nmp_layers=1,
            scale=2,
            vis=False
        )

        self.interaction_hyper2 = NmpNet_hyper_weight(
            embedding_dim=16,
            h_dim=self.model_dim,
            mlp_dim=64,
            bottleneck_dim=self.model_dim,
            batch_norm=0,
            nmp_layers=1,
            scale=1,
            vis=False
        )

        self.interaction_hyper3 = NmpNet_hyper_weight(
            embedding_dim=16,
            h_dim=self.model_dim,
            mlp_dim=64,
            bottleneck_dim=self.model_dim,
            batch_norm=0,
            nmp_layers=1,
            scale=3,
            vis=False
        )

        # encoder_layers = AgentFormerEncoderLayer(ctx['tf_cfg'], self.model_dim, self.nhead, self.ff_dim, self.dropout)
        # self.tf_encoder = AgentFormerEncoder(encoder_layers, self.nlayer)
        self.pos_encoder = PositionalAgentEncoding(self.model_dim, self.dropout, concat=ctx['pos_concat'], max_a_len=ctx['max_agent_len'], use_agent_enc=ctx['use_agent_enc'], agent_enc_learn=ctx['agent_enc_learn'])
    
    def add_category(self,x):
        B = x.shape[0]
        N = x.shape[1]
        category = torch.zeros(N,3).type_as(x)
        category[0:5,0] = 1
        category[5:10,1] = 1
        category[10,2] = 1
        category = category.repeat(B,1,1)
        x = torch.cat((x,category),dim=-1)
        return x

    def forward(self, data):
        traj_in = []
        for key in self.input_type:
            if key == 'pos':
                traj_in.append(data['pre_motion'])
            elif key == 'vel':
                vel = data['pre_vel']
                if len(self.input_type) > 1:
                    vel = torch.cat([vel[:,[0]], vel], dim=1)
                if self.vel_heading:
                    vel = rotation_2d_torch(vel, -data['heading'])[0]
                traj_in.append(vel)
            elif key == 'norm':
                traj_in.append(data['pre_motion_norm'])
            elif key == 'scene_norm':
                traj_in.append(data['pre_motion_scene_norm'])
            elif key == 'heading':
                hv = data['heading_vec'].unsqueeze(0).repeat((data['pre_motion'].shape[0], 1, 1))
                traj_in.append(hv)
            elif key == 'map':
                map_enc = data['map_enc'].unsqueeze(0).repeat((data['pre_motion'].shape[0], 1, 1))
                traj_in.append(map_enc)
            else:
                raise ValueError('unknown input_type!')
        traj_in = torch.cat(traj_in, dim=-1)  #(B,N,T,2)
        batch_size = data['batch_size']
        length = data['pre_motion_scene_norm'].shape[1]
        actor_num = 11
        tf_in = self.input_fc(traj_in).view(batch_size*actor_num, length, self.model_dim)
        agent_enc_shuffle = data['agent_enc_shuffle'] if self.agent_enc_shuffle else None
        tf_in_pos = self.pos_encoder(tf_in, num_a=batch_size*actor_num, agent_enc_shuffle=agent_enc_shuffle)
        tf_in_pos = tf_in_pos.view(batch_size, actor_num, length, self.model_dim)
        
        # thres = 6
        # if data['agent_num'] > thres:
        #     ftraj_input_all = self.input_fc2(tf_in_pos.contiguous().view(data['agent_num'],-1))
        #     ind_all = data['pre_motion_scene_norm'][-1,:,0].argsort(0)
        #     out = []
        #     split_number = math.ceil(data['agent_num']/thres)
        #     for piece in range(split_number):
        #         if piece == split_number-1:
        #             ind = ind_all[piece*(math.ceil(data['agent_num'] / split_number)):]
        #         else:
        #             ind = ind_all[piece*(math.ceil(data['agent_num'] / split_number)):(piece+1)*(math.ceil(data['agent_num'] / split_number))]
        #         ftraj_input = ftraj_input_all[ind]
        #         query_input = F.normalize(ftraj_input,p=2,dim=1)
        #         feat_corr = torch.matmul(query_input,query_input.permute(1,0))
        #         ftraj_inter,_ = self.interaction(ftraj_input)
        #         ftraj_inter_hyper,_ = self.interaction_hyper(ftraj_input,feat_corr)
        #         ftraj_inter_hyper2,_ = self.interaction_hyper2(ftraj_input,feat_corr)
        #         cat_feat = torch.cat((ftraj_input,ftraj_inter,ftraj_inter_hyper,ftraj_inter_hyper2),dim=-1)
        #         out.append(cat_feat)
        #     cxt_out_temp = torch.cat(out,dim=0)
        #     cxt_out= torch.zeros_like(cxt_out_temp)
        #     for i,idx in enumerate(ind_all):
        #         cxt_out[idx] = cxt_out_temp[i]
        #         cxt_out[idx] = cxt_out_temp[i]
        #     data['agent_context'] = cxt_out
        # else:
        ftraj_input = self.input_fc2(tf_in_pos.contiguous().view(batch_size, actor_num, length*self.model_dim))
        ftraj_input = self.input_fc3(self.add_category(ftraj_input))
        query_input = F.normalize(ftraj_input,p=2,dim=2)
        feat_corr = torch.matmul(query_input,query_input.permute(0,2,1))
        ftraj_inter,_ = self.interaction(ftraj_input)
        ftraj_inter_hyper,_ = self.interaction_hyper(ftraj_input,feat_corr)
        ftraj_inter_hyper2,_ = self.interaction_hyper2(ftraj_input,feat_corr)
        # ftraj_inter_hyper3,_ = self.interaction_hyper3(ftraj_input,feat_corr)

        # final_feature = ftraj_input
        # final_feature = torch.cat((ftraj_input,ftraj_inter),dim=-1)
        # final_feature = torch.cat((ftraj_input,ftraj_inter,ftraj_inter_hyper),dim=-1)
        final_feature = torch.cat((ftraj_input,ftraj_inter,ftraj_inter_hyper,ftraj_inter_hyper2),dim=-1)
        # final_feature = torch.cat((ftraj_input,ftraj_inter,ftraj_inter_hyper,ftraj_inter_hyper2,ftraj_inter_hyper3),dim=-1)

        data['agent_context'] = final_feature.view(batch_size*actor_num,-1)
        # data['agent_context'] = torch.cat((ftraj_input,ftraj_input,ftraj_input,ftraj_input),dim=-1)

        # src_agent_mask = data['agent_mask'].clone()
        # src_mask = generate_mask(tf_in.shape[0], tf_in.shape[0], data['agent_num'], src_agent_mask).to(tf_in.device)
        
        # data['context_enc'] = self.tf_encoder(tf_in_pos, mask=src_mask, num_agent=data['agent_num'])
        
        # context_rs = data['context_enc'].view(-1, data['agent_num'], self.model_dim)
        # # compute per agent context
        # if self.pooling == 'mean':
        #     data['agent_context'] = torch.mean(context_rs, dim=0)
        # else:
        #     data['agent_context'] = torch.max(context_rs, dim=0)[0]


""" Future Encoder """
class FutureEncoder(nn.Module):
    def __init__(self, cfg, ctx, **kwargs):
        super().__init__()
        self.cfg = cfg
        self.context_dim = context_dim = ctx['context_dim']
        self.forecast_dim = forecast_dim = ctx['forecast_dim']
        self.nz = ctx['nz']
        self.z_type = ctx['z_type']
        self.z_tau_annealer = ctx.get('z_tau_annealer', None)
        # self.model_dim = ctx['tf_model_dim']
        self.model_dim = 16
        self.ff_dim = ctx['tf_ff_dim']
        self.nhead = ctx['tf_nhead']
        self.dropout = ctx['tf_dropout']
        self.nlayer = cfg.get('nlayer', 6)
        # self.out_mlp_dim = cfg.get('out_mlp_dim', None)
        self.out_mlp_dim = [128]
        self.input_type = ctx['fut_input_type']
        self.pooling = cfg.get('pooling', 'mean')
        self.agent_enc_shuffle = ctx['agent_enc_shuffle']
        self.vel_heading = ctx['vel_heading']
        # networks
        in_dim = forecast_dim * len(self.input_type)
        if 'map' in self.input_type:
            in_dim += ctx['map_enc_dim'] - forecast_dim
        self.input_fc = nn.Linear(in_dim, self.model_dim)
        scale_num = 4
        self.input_fc2 = nn.Linear(self.model_dim*10, self.model_dim)
        self.input_fc3 = nn.Linear(self.model_dim+3, self.model_dim)

        # decoder_layers = AgentFormerDecoderLayer(ctx['tf_cfg'], self.model_dim, self.nhead, self.ff_dim, self.dropout)
        # self.tf_decoder = AgentFormerDecoder(decoder_layers, self.nlayer)
        self.interaction = NmpNet(
            embedding_dim=16,
            h_dim=self.model_dim,
            mlp_dim=64,
            bottleneck_dim=self.model_dim,
            batch_norm=0,
            nmp_layers=2,
            vis=False
        )

        self.interaction_hyper = NmpNet_hyper_weight(
            embedding_dim=16,
            h_dim=self.model_dim,
            mlp_dim=64,
            bottleneck_dim=self.model_dim,
            batch_norm=0,
            nmp_layers=1,
            scale=2,
            vis=False
        )

        self.interaction_hyper2 = NmpNet_hyper_weight(
            embedding_dim=16,
            h_dim=self.model_dim,
            mlp_dim=64,
            bottleneck_dim=self.model_dim,
            batch_norm=0,
            nmp_layers=1,
            scale=1,
            vis=False
        )

        self.interaction_hyper3 = NmpNet_hyper_weight(
            embedding_dim=16,
            h_dim=self.model_dim,
            mlp_dim=64,
            bottleneck_dim=self.model_dim,
            batch_norm=0,
            nmp_layers=1,
            scale=3,
            vis=False
        )

        self.pos_encoder = PositionalAgentEncoding(self.model_dim, self.dropout, concat=ctx['pos_concat'], max_a_len=ctx['max_agent_len'], use_agent_enc=ctx['use_agent_enc'], agent_enc_learn=ctx['agent_enc_learn'])
        num_dist_params = 2 * self.nz if self.z_type == 'gaussian' else self.nz     # either gaussian or discrete
        # if self.out_mlp_dim is None:
        #     self.q_z_net = nn.Linear(self.model_dim, num_dist_params)
        # else:
        self.out_mlp = MLP2(scale_num*2*self.model_dim, self.out_mlp_dim, 'relu')
        self.q_z_net = nn.Linear(self.out_mlp.out_dim, num_dist_params)
        # initialize
        initialize_weights(self.q_z_net.modules())

    def add_category(self,x):
        B = x.shape[0]
        N = x.shape[1]
        category = torch.zeros(N,3).type_as(x)
        category[0:5,0] = 1
        category[5:10,1] = 1
        category[10,2] = 1
        category = category.repeat(B,1,1)
        x = torch.cat((x,category),dim=-1)
        return x

    def forward(self, data, reparam=True):
        traj_in = []
        for key in self.input_type:
            if key == 'pos':
                traj_in.append(data['fut_motion'])
            elif key == 'vel':
                vel = data['fut_vel']
                if self.vel_heading:
                    vel = rotation_2d_torch(vel, -data['heading'])[0]
                traj_in.append(vel)
            elif key == 'norm':
                traj_in.append(data['fut_motion_norm'])
            elif key == 'scene_norm':
                traj_in.append(data['fut_motion_scene_norm'])
            elif key == 'heading':
                hv = data['heading_vec'].unsqueeze(0).repeat((data['fut_motion'].shape[0], 1, 1))
                traj_in.append(hv)
            elif key == 'map':
                map_enc = data['map_enc'].unsqueeze(0).repeat((data['fut_motion'].shape[0], 1, 1))
                traj_in.append(map_enc)
            else:
                raise ValueError('unknown input_type!')

        traj_in = torch.cat(traj_in, dim=-1)  #(B,N,T,2)
        batch_size = data['batch_size']
        length = data['fut_motion_scene_norm'].shape[1]
        actor_num = 11
        tf_in = self.input_fc(traj_in).view(batch_size*actor_num, length, self.model_dim)
        agent_enc_shuffle = data['agent_enc_shuffle'] if self.agent_enc_shuffle else None
        tf_in_pos = self.pos_encoder(tf_in, num_a=batch_size*actor_num, agent_enc_shuffle=agent_enc_shuffle)
        tf_in_pos = tf_in_pos.view(batch_size, actor_num, length, self.model_dim)

        # thres = 6
        # if data['agent_num'] > thres:
        #     ftraj_input_all = self.input_fc2(tf_in_pos.contiguous().view(data['agent_num'],-1))
        #     ind_all = data['pre_motion_scene_norm'][-1,:,0].argsort(0)
        #     out = []
        #     split_number = math.ceil(data['agent_num']/thres)
        #     for piece in range(split_number):
        #         if piece == split_number-1:
        #             ind = ind_all[piece*(math.ceil(data['agent_num'] / split_number)):]
        #         else:
        #             ind = ind_all[piece*(math.ceil(data['agent_num'] / split_number)):(piece+1)*(math.ceil(data['agent_num'] / split_number))]
        #         ftraj_input = ftraj_input_all[ind]
        #         query_input = F.normalize(ftraj_input,p=2,dim=1)
        #         feat_corr = torch.matmul(query_input,query_input.permute(1,0))
        #         ftraj_inter,_ = self.interaction(ftraj_input)
        #         ftraj_inter_hyper,_ = self.interaction_hyper(ftraj_input,feat_corr)
        #         ftraj_inter_hyper2,_ = self.interaction_hyper2(ftraj_input,feat_corr)
        #         cat_feat = torch.cat((ftraj_input,ftraj_inter,ftraj_inter_hyper,ftraj_inter_hyper2),dim=-1)
        #         out.append(cat_feat)
        #     tf_out_temp = torch.cat(out,dim=0)
        #     tf_out= torch.zeros_like(tf_out_temp)
        #     for i,idx in enumerate(ind_all):
        #         tf_out[idx] = tf_out_temp[i]
        #         tf_out[idx] = tf_out_temp[i]
        # else:

        ftraj_input = self.input_fc2(tf_in_pos.contiguous().view(batch_size, actor_num, -1))
        ftraj_input = self.input_fc3(self.add_category(ftraj_input))
        query_input = F.normalize(ftraj_input,p=2,dim=2)
        feat_corr = torch.matmul(query_input,query_input.permute(0,2,1))
        ftraj_inter,_ = self.interaction(ftraj_input)
        ftraj_inter_hyper,_ = self.interaction_hyper(ftraj_input,feat_corr)
        ftraj_inter_hyper2,_ = self.interaction_hyper2(ftraj_input,feat_corr)
        # ftraj_inter_hyper3,_ = self.interaction_hyper3(ftraj_input,feat_corr)

        # tf_out = ftraj_input
        # tf_out = torch.cat((ftraj_input,ftraj_inter),dim=-1)
        # tf_out = torch.cat((ftraj_input,ftraj_inter,ftraj_inter_hyper),dim=-1)
        tf_out = torch.cat((ftraj_input,ftraj_inter,ftraj_inter_hyper,ftraj_inter_hyper2),dim=-1)
        # tf_out = torch.cat((ftraj_input,ftraj_inter,ftraj_inter_hyper,ftraj_inter_hyper2,ftraj_inter_hyper3),dim=-1)
        tf_out = tf_out.view(batch_size*actor_num,-1)

        h = torch.cat((data['agent_context'],tf_out),dim=-1)
        # if self.out_mlp_dim is not None:
        h = self.out_mlp(h)
        q_z_params = self.q_z_net(h)
        if self.z_type == 'gaussian':
            data['q_z_dist'] = Normal(params=q_z_params)
        else:
            data['q_z_dist'] = Categorical(logits=q_z_params, temp=self.z_tau_annealer.val())
        data['q_z_samp'] = data['q_z_dist'].rsample()


""" Future Decoder """
class FutureDecoder(nn.Module):
    def __init__(self, cfg, ctx, **kwargs):
        super().__init__()
        self.cfg = cfg
        self.ar_detach = ctx['ar_detach']
        self.context_dim = context_dim = ctx['context_dim']
        self.forecast_dim = forecast_dim = ctx['forecast_dim']
        self.pred_scale = cfg.get('pred_scale', 1.0)
        self.pred_type = ctx['pred_type']
        self.sn_out_type = ctx['sn_out_type']
        self.sn_out_heading = ctx['sn_out_heading']
        self.input_type = ctx['dec_input_type']
        self.future_frames = ctx['future_frames']
        self.past_frames = ctx['past_frames']
        self.nz = ctx['nz']
        self.z_type = ctx['z_type']
        # self.model_dim = ctx['tf_model_dim']
        self.model_dim = 16
        self.ff_dim = ctx['tf_ff_dim']
        self.nhead = ctx['tf_nhead']
        self.dropout = ctx['tf_dropout']
        self.nlayer = cfg.get('nlayer', 6)
        self.out_mlp_dim = cfg.get('out_mlp_dim', None)
        self.out_cat = cfg.get('out_cat', [])
        self.pos_offset = cfg.get('pos_offset', False)
        self.agent_enc_shuffle = ctx['agent_enc_shuffle']
        self.learn_prior = ctx['learn_prior']
        # networks
        in_dim = forecast_dim + len(self.input_type) * forecast_dim + self.nz
        if 'map' in self.input_type:
            in_dim += ctx['map_enc_dim'] - forecast_dim
        self.input_fc = nn.Linear(in_dim, self.model_dim)
        self.decode_way = 'RES'
        # print(self.decode_way)
        scale_num = 4
        if self.decode_way == 'MLP':
            self.out_MLP = MLP2(scale_num*self.model_dim+self.nz,(256,64),'relu')
            self.out_FC = nn.Linear(64, self.future_frames*forecast_dim)
        if self.decode_way == 'RES':
            self.num_decompose = 2
            input_dim = scale_num*self.model_dim+self.nz
            self.decompose = nn.ModuleList([DecomposeBlock(self.past_frames, self.future_frames, input_dim) for _ in range(self.num_decompose)])

        # decoder_layers = AgentFormerDecoderLayer(ctx['tf_cfg'], self.model_dim, self.nhead, self.ff_dim, self.dropout)
        # self.tf_decoder = AgentFormerDecoder(decoder_layers, self.nlayer)

        self.pos_encoder = PositionalAgentEncoding(self.model_dim, self.dropout, concat=ctx['pos_concat'], max_a_len=ctx['max_agent_len'], use_agent_enc=ctx['use_agent_enc'], agent_enc_learn=ctx['agent_enc_learn'])
        if self.out_mlp_dim is None:
            self.out_fc = nn.Linear(5*self.model_dim, forecast_dim)
        else:
            in_dim = 5*self.model_dim
            if 'map' in self.out_cat:
                in_dim += ctx['map_enc_dim']
            if 'z' in self.out_cat:
                in_dim += self.nz
            self.out_mlp = MLP2(in_dim, self.out_mlp_dim, 'relu')
            self.out_fc = nn.Linear(self.out_mlp.out_dim, forecast_dim)
        initialize_weights(self.out_fc.modules())
        if self.learn_prior:
            num_dist_params = 2 * self.nz if self.z_type == 'gaussian' else self.nz     # either gaussian or discrete
            self.p_z_net = nn.Linear(scale_num*self.model_dim, num_dist_params)
            initialize_weights(self.p_z_net.modules())

    def decode_traj_ar(self, data, mode, context, pre_motion, pre_vel, pre_motion_scene_norm, z, sample_num, need_weights=False):
        agent_num = data['batch_size'] * 11
        # if self.pred_type == 'vel':
        #     dec_in = pre_vel[[-1]]
        # elif self.pred_type == 'pos':
        #     dec_in = pre_motion[[-1]]
        # elif self.pred_type == 'scene_norm':
        #     dec_in = pre_motion_scene_norm[[-1]]
        # else:
        #     dec_in = torch.zeros_like(pre_motion[[-1]])
        # dec_in = dec_in.view(-1, sample_num, dec_in.shape[-1])
        z_in = z.view(-1, sample_num, z.shape[-1])
        # in_arr = [dec_in, z_in]
        # for key in self.input_type:
        #     if key == 'heading':
        #         heading = data['heading_vec'].unsqueeze(1).repeat((1, sample_num, 1))
        #         in_arr.append(heading)
        #     elif key == 'map':
        #         map_enc = data['map_enc'].unsqueeze(1).repeat((1, sample_num, 1))
        #         in_arr.append(map_enc)
        #     else:
        #         raise ValueError('wrong decode input type!')
        # dec_in_z = torch.cat(in_arr, dim=-1)

        # mem_agent_mask = data['agent_mask'].clone()
        # tgt_agent_mask = data['agent_mask'].clone()

        if self.decode_way == 'MLP':
            hidden = torch.cat((context,z_in),dim=-1)
            norm_motion = self.out_FC(self.out_MLP(hidden))
            norm_motion = norm_motion.view(agent_num,sample_num,self.future_frames,2)
            norm_motion = norm_motion.permute(2,0,1,3).view(self.future_frames, agent_num * sample_num,2)
            seq_out = norm_motion + pre_motion_scene_norm.transpose(0,1)[[-1]]
            seq_out = seq_out.view(-1, agent_num * sample_num, seq_out.shape[-1])
            data[f'{mode}_seq_out'] = seq_out

            # if self.pred_type == 'vel':
            #     dec_motion = torch.cumsum(seq_out, dim=0)
            #     dec_motion += pre_motion[[-1]]
            # elif self.pred_type == 'pos':
            #     dec_motion = seq_out.clone()
            # elif self.pred_type == 'scene_norm':
            #     dec_motion = seq_out + data['scene_orig']
            # else:
            #     dec_motion = seq_out + pre_motion[[-1]]

            dec_motion = seq_out
            dec_motion = dec_motion.transpose(0, 1).contiguous()       # M x frames x 7
            # print(dec_motion.shape)
            # print(data['scene_orig'].shape)
            # print(pre_motion_scene_norm[[-1]].shape)
            # time.sleep(1000)
            if mode == 'infer':
                dec_motion = dec_motion.view(-1, sample_num, *dec_motion.shape[1:])        # M x Samples x frames x 3
            data[f'{mode}_dec_motion'] = dec_motion

        if self.decode_way == "RES":
            hidden = torch.cat((context,z_in),dim=-1)
            hidden = hidden.view(agent_num*sample_num,-1)
            x_true = pre_motion_scene_norm.clone() #torch.transpose(pre_motion_scene_norm, 0, 1)

            x_hat = torch.zeros_like(x_true)
            batch_size = x_true.size(0)
            prediction = torch.zeros((batch_size, self.future_frames, 2)).cuda()
            reconstruction = torch.zeros((batch_size, self.past_frames, 2)).cuda()
            # print(hidden.shape)
            # print(pre_motion_scene_norm.shape)
            # print(prediction.shape)
            # print(x_true.shape)
            # print(hidden.shape)
            # time.sleep(1000)
            for i in range(self.num_decompose):
                x_hat, y_hat = self.decompose[i](x_true, x_hat, hidden)
                prediction += y_hat
                reconstruction += x_hat
            norm_motion = prediction.view(agent_num,sample_num,self.future_frames,2)
            recover_pre_motion = reconstruction.view(agent_num,sample_num,self.past_frames,2)

            norm_motion = norm_motion.permute(2,0,1,3).view(self.future_frames, agent_num * sample_num,2)
            recover_pre_motion = recover_pre_motion.view(agent_num * sample_num,self.past_frames,2)
            # print(norm_motion.shape)
            # print(pre_motion_scene_norm.shape)
            # time.sleep(1000)
            seq_out = norm_motion + pre_motion_scene_norm.transpose(0,1)[[-1]]
            seq_out = seq_out.view(-1, agent_num * sample_num, seq_out.shape[-1])
            data[f'{mode}_seq_out'] = seq_out

            # if self.pred_type == 'vel':
            #     dec_motion = torch.cumsum(seq_out, dim=0)
            #     dec_motion += pre_motion[[-1]]
            # elif self.pred_type == 'pos':
            #     dec_motion = seq_out.clone()
            # elif self.pred_type == 'scene_norm':
            #     dec_motion = seq_out + data['scene_orig']
            # else:
            #     dec_motion = seq_out + pre_motion[[-1]]
            dec_motion = seq_out
            dec_motion = dec_motion.transpose(0, 1).contiguous()       # M x frames x 7
            # recover_pre_motion = recover_pre_motion.transpose(0, 1).contiguous() 
            # print(dec_motion.shape)
            # print(data['scene_orig'].shape)
            # print(pre_motion_scene_norm[[-1]].shape)
            # time.sleep(1000)
            if mode == 'infer':
                dec_motion = dec_motion.view(-1, sample_num, *dec_motion.shape[1:])        # M x Samples x frames x 3
            data[f'{mode}_dec_motion'] = dec_motion
            data[f'{mode}_recover_motion'] = recover_pre_motion

        # hidden = torch.cat((context,z_in),dim=-1)
        # for i in range(self.future_frames):
        #     seq_out, hidden, res_ = self.step_forward(dec_in, hidden)
        #     if self.pred_type == 'scene_norm' and self.sn_out_type in {'vel', 'norm'}:
        #         norm_motion = seq_out.view(-1, agent_num * sample_num, seq_out.shape[-1])
        #         if self.sn_out_type == 'vel':
        #             norm_motion = torch.cumsum(norm_motion, dim=0)
        #         if self.sn_out_heading:
        #             angles = data['heading'].repeat_interleave(sample_num)
        #             norm_motion = rotation_2d_torch(norm_motion, angles)[0]
        #         seq_out = norm_motion + pre_motion_scene_norm[[-1]]
        #         seq_out = seq_out.view(tf_out.shape[0], -1, seq_out.shape[-1])
                


        # for i in range(self.future_frames):
        #     tf_in = self.input_fc(dec_in_z.view(-1, dec_in_z.shape[-1])).view(dec_in_z.shape[0], -1, self.model_dim)
        #     agent_enc_shuffle = data['agent_enc_shuffle'] if self.agent_enc_shuffle else None
        #     tf_in_pos = self.pos_encoder(tf_in, num_a=agent_num, agent_enc_shuffle=agent_enc_shuffle, t_offset=self.past_frames-1 if self.pos_offset else 0)
        #     # tf_in_pos = tf_in
        #     # mem_mask = generate_mask(tf_in.shape[0], context.shape[0], data['agent_num'], mem_agent_mask).to(tf_in.device)
        #     # tgt_mask = generate_ar_mask(tf_in_pos.shape[0], agent_num, tgt_agent_mask).to(tf_in.device)

        #     # tf_out, attn_weights = self.tf_decoder(tf_in_pos, context, memory_mask=mem_mask, tgt_mask=tgt_mask, num_agent=data['agent_num'], need_weights=need_weights)
        #     print(context.shape)
        #     print(tf_in_pos.shape)
        #     tf_out = torch.cat((context,tf_in_pos),dim=-1)
        #     out_tmp = tf_out.view(-1, tf_out.shape[-1])
        #     if self.out_mlp_dim is not None:
        #         cat_arr = [out_tmp]
        #         if 'map' in self.out_cat:
        #             map_tmp = data['map_enc'].unsqueeze(0).repeat((i + 1, sample_num, 1)).view(-1, data['map_enc'].shape[-1])
        #             cat_arr.append(map_tmp)
        #         if 'z' in self.out_cat:
        #             z_tmp = z_in.repeat((i + 1, 1, 1)).view(-1, self.nz)
        #             cat_arr.append(z_tmp)
        #         out_tmp = torch.cat(cat_arr, dim=-1)
        #         out_tmp = self.out_mlp(out_tmp)
        #     seq_out = self.out_fc(out_tmp).view(tf_out.shape[0], -1, self.forecast_dim)
        #     if self.pred_type == 'scene_norm' and self.sn_out_type in {'vel', 'norm'}:
        #         norm_motion = seq_out.view(-1, agent_num * sample_num, seq_out.shape[-1])
        #         if self.sn_out_type == 'vel':
        #             norm_motion = torch.cumsum(norm_motion, dim=0)
        #         if self.sn_out_heading:
        #             angles = data['heading'].repeat_interleave(sample_num)
        #             norm_motion = rotation_2d_torch(norm_motion, angles)[0]
        #         seq_out = norm_motion + pre_motion_scene_norm[[-1]]
        #         seq_out = seq_out.view(tf_out.shape[0], -1, seq_out.shape[-1])
        #     if self.ar_detach:
        #         out_in = seq_out[-agent_num:].clone().detach()
        #     else:
        #         out_in = seq_out[-agent_num:]
        #     # create dec_in_z
        #     in_arr = [out_in, z_in]
        #     for key in self.input_type:
        #         if key == 'heading':
        #             in_arr.append(heading)
        #         elif key == 'map':
        #             in_arr.append(map_enc)
        #         else:
        #             raise ValueError('wrong decoder input type!')
        #     out_in_z = torch.cat(in_arr, dim=-1)
        #     dec_in_z = torch.cat([dec_in_z, out_in_z], dim=0)

        # seq_out = seq_out.view(-1, agent_num * sample_num, seq_out.shape[-1])
        # data[f'{mode}_seq_out'] = seq_out

        # if self.pred_type == 'vel':
        #     dec_motion = torch.cumsum(seq_out, dim=0)
        #     dec_motion += pre_motion[[-1]]
        # elif self.pred_type == 'pos':
        #     dec_motion = seq_out.clone()
        # elif self.pred_type == 'scene_norm':
        #     dec_motion = seq_out + data['scene_orig']
        # else:
        #     dec_motion = seq_out + pre_motion[[-1]]

        # dec_motion = dec_motion.transpose(0, 1).contiguous()       # M x frames x 7
        # if mode == 'infer':
        #     dec_motion = dec_motion.view(-1, sample_num, *dec_motion.shape[1:])        # M x Samples x frames x 3
        # data[f'{mode}_dec_motion'] = dec_motion
        # if need_weights:
        #     data['attn_weights'] = attn_weights

    def decode_traj_batch(self, data, mode, context, pre_motion, pre_vel, pre_motion_scene_norm, z, sample_num):
        raise NotImplementedError

    def forward(self, data, mode, sample_num=1, autoregress=True, z=None, need_weights=False):
        context = data['agent_context'][:,None,:].repeat_interleave(sample_num, dim=1)       # 80 x 64
        pre_motion = data['pre_motion'].repeat_interleave(sample_num, dim=0)             # 10 x 80 x 2
        pre_vel = data['pre_vel'].repeat_interleave(sample_num, dim=0) if self.pred_type == 'vel' else None
        pre_motion_scene_norm = data['pre_motion_scene_norm'].repeat_interleave(sample_num, dim=0)
        
        # p(z)
        prior_key = 'p_z_dist' + ('_infer' if mode == 'infer' else '')
        if self.learn_prior:
            h = data['agent_context'].repeat_interleave(sample_num, dim=0)
            p_z_params = self.p_z_net(h)
            if self.z_type == 'gaussian':
                data[prior_key] = Normal(params=p_z_params)
            else:
                data[prior_key] = Categorical(params=p_z_params)
        else:
            if self.z_type == 'gaussian':
                data[prior_key] = Normal(mu=torch.zeros(pre_motion.shape[0], self.nz).to(pre_motion.device), logvar=torch.zeros(pre_motion.shape[0], self.nz).to(pre_motion.device))
            else:
                data[prior_key] = Categorical(logits=torch.zeros(pre_motion.shape[0], self.nz).to(pre_motion.device))

        if z is None:
            if mode in {'train', 'recon'}:
                z = data['q_z_samp'] if mode == 'train' else data['q_z_dist'].mode()
            elif mode == 'infer':
                z = data['p_z_dist_infer'].sample()
            else:
                raise ValueError('Unknown Mode!')

        if autoregress:
            self.decode_traj_ar(data, mode, context, pre_motion, pre_vel, pre_motion_scene_norm, z, sample_num, need_weights=need_weights)
        else:
            self.decode_traj_batch(data, mode, context, pre_motion, pre_vel, pre_motion_scene_norm, z, sample_num)
        


""" AgentFormer """
class AgentFormer(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.device = torch.device('cpu')
        self.cfg = cfg

        input_type = cfg.get('input_type', 'pos')
        pred_type = cfg.get('pred_type', input_type)
        if type(input_type) == str:
            input_type = [input_type]
        fut_input_type = cfg.get('fut_input_type', input_type)
        dec_input_type = cfg.get('dec_input_type', [])
        self.ctx = {
            'tf_cfg': cfg.get('tf_cfg', {}),
            'nz': cfg.nz,
            'z_type': cfg.get('z_type', 'gaussian'),
            'future_frames': cfg.future_frames,
            'past_frames': cfg.past_frames,
            'motion_dim': cfg.motion_dim,
            'forecast_dim': cfg.forecast_dim,
            'input_type': input_type,
            'fut_input_type': fut_input_type,
            'dec_input_type': dec_input_type,
            'pred_type': pred_type,
            'tf_nhead': cfg.tf_nhead,
            'tf_model_dim': cfg.tf_model_dim,
            'tf_ff_dim': cfg.tf_ff_dim,
            'tf_dropout': cfg.tf_dropout,
            'pos_concat': cfg.get('pos_concat', False),
            'ar_detach': cfg.get('ar_detach', True),
            'max_agent_len': cfg.get('max_agent_len', 128),
            'use_agent_enc': cfg.get('use_agent_enc', False),
            'agent_enc_learn': cfg.get('agent_enc_learn', False),
            'agent_enc_shuffle': cfg.get('agent_enc_shuffle', False),
            'sn_out_type': cfg.get('sn_out_type', 'scene_norm'),
            'sn_out_heading': cfg.get('sn_out_heading', False),
            'vel_heading': cfg.get('vel_heading', False),
            'learn_prior': cfg.get('learn_prior', False),
            'use_map': cfg.get('use_map', False)
        }
        self.use_map = self.ctx['use_map']
        self.rand_rot_scene = cfg.get('rand_rot_scene', False)
        self.discrete_rot = cfg.get('discrete_rot', False)
        self.map_global_rot = cfg.get('map_global_rot', False)
        self.ar_train = cfg.get('ar_train', True)
        self.max_train_agent = cfg.get('max_train_agent', 100)
        self.loss_cfg = self.cfg.loss_cfg
        self.loss_names = list(self.loss_cfg.keys())
        self.compute_sample = 'sample' in self.loss_names
        self.param_annealers = nn.ModuleList()
        if self.ctx['z_type'] == 'discrete':
            self.ctx['z_tau_annealer'] = z_tau_annealer = ExpParamAnnealer(cfg.z_tau.start, cfg.z_tau.finish, cfg.z_tau.decay)
            self.param_annealers.append(z_tau_annealer)

        # save all computed variables
        self.data = None
        
        # map encoder
        if self.use_map:
            self.map_encoder = MapEncoder(cfg.map_encoder)
            self.ctx['map_enc_dim'] = self.map_encoder.out_dim

        # models
        self.context_encoder = ContextEncoder(cfg.context_encoder, self.ctx)
        self.future_encoder = FutureEncoder(cfg.future_encoder, self.ctx)
        self.future_decoder = FutureDecoder(cfg.future_decoder, self.ctx)
        
    def set_device(self, device):
        self.device = device
        self.to(device)

    def set_data(self, data):
        device = self.device
        in_data = data

        self.data = defaultdict(lambda: None)
        self.data['batch_size'] = in_data['pre_motion_3D'].shape[0]
        self.data['agent_num'] = 11
        self.data['pre_motion'] = in_data['pre_motion_3D'].view(self.data['batch_size']*self.data['agent_num'],self.ctx['past_frames'],2).to(device).contiguous()
        self.data['fut_motion'] = in_data['fut_motion_3D'].view(self.data['batch_size']*self.data['agent_num'],self.ctx['future_frames'],2).to(device).contiguous()
        self.data['fut_motion_orig'] = in_data['fut_motion_3D'].view(self.data['batch_size']*self.data['agent_num'],self.ctx['future_frames'],2).to(device)   # future motion without transpose
        self.data['fut_mask'] = in_data['fut_motion_mask'].view(self.data['batch_size']*self.data['agent_num'],self.ctx['future_frames']).to(device)
        self.data['pre_mask'] = in_data['pre_motion_mask'].view(self.data['batch_size']*self.data['agent_num'],self.ctx['past_frames']).to(device)
        # self.data['scene_orig'] = torch.cat([self.data['pre_motion'], self.data['fut_motion']]).view(-1, 2).mean(dim=0)
        # print(self.data['batch_size'])  # N
        # print(self.data['agent_num']) # N
        # print(self.data['pre_motion'].shape) # (T,N,2)
        # print(self.data['fut_motion'].shape) # (T,N,2)
        # print(self.data['fut_mask'].shape)# (N,T)
        # print(self.data['pre_mask'].shape)# (N,T)
        # print(self.data['scene_orig'].shape) # (2)
        # rotate the scene
        # if self.rand_rot_scene and self.training:
        #     if self.discrete_rot:
        #         theta = torch.randint(high=24, size=(1,)).to(device) * (np.pi / 12)
        #     else:
        #         theta = torch.rand(1).to(device) * np.pi * 2
        #     for key in ['pre_motion', 'fut_motion', 'fut_motion_orig']:
        #         self.data[f'{key}'], self.data[f'{key}_scene_norm'] = rotation_2d_torch(self.data[key], theta, self.data['scene_orig'])
        #     if in_data['heading'] is not None:
        #         self.data['heading'] += theta
        # else:
        #     theta = torch.zeros(1).to(device)
        #     for key in ['pre_motion', 'fut_motion', 'fut_motion_orig']:
        #         self.data[f'{key}_scene_norm'] = self.data[key] - self.data['scene_orig']   # normalize per scene
        for key in ['pre_motion', 'fut_motion', 'fut_motion_orig']:
            self.data[f'{key}_scene_norm'] = self.data[key]

        self.data['pre_vel'] = self.data['pre_motion'][:,1:] - self.data['pre_motion'][:,:-1, :]
        self.data['fut_vel'] = self.data['fut_motion'] - torch.cat([self.data['pre_motion'][:,[-1]], self.data['fut_motion'][:,:-1, :]],dim=1)
        self.data['cur_motion'] = self.data['pre_motion'][:,[-1]]
        # self.data['pre_motion_norm'] = self.data['pre_motion'][:-1] - self.data['cur_motion']   # normalize pos per agent
        # self.data['fut_motion_norm'] = self.data['fut_motion'] - self.data['cur_motion']
        # if in_data['heading'] is not None:
        #     self.data['heading_vec'] = torch.stack([torch.cos(self.data['heading']), torch.sin(self.data['heading'])], dim=-1)

    def step_annealer(self):
        for anl in self.param_annealers:
            anl.step()

    def forward(self):
        self.context_encoder(self.data)
        self.future_encoder(self.data)
        self.future_decoder(self.data, mode='train', autoregress=self.ar_train)
        if self.compute_sample:
            self.inference(sample_num=self.loss_cfg['sample']['k'])
        return self.data

    def inference(self, mode='infer', sample_num=20, need_weights=False):
        if self.use_map and self.data['map_enc'] is None:
            self.data['map_enc'] = self.map_encoder(self.data['agent_maps'])
        if self.data['context_enc'] is None:
            self.context_encoder(self.data)
        if mode == 'recon':
            sample_num = 1
            self.future_encoder(self.data)
        self.future_decoder(self.data, mode=mode, sample_num=sample_num, autoregress=True, need_weights=need_weights)
        return self.data[f'{mode}_dec_motion'], self.data

    def compute_loss(self):
        total_loss = 0
        loss_dict = {}
        loss_unweighted_dict = {}
        for loss_name in self.loss_names:
            loss, loss_unweighted = loss_func[loss_name](self.data, self.loss_cfg[loss_name])
            total_loss += loss
            loss_dict[loss_name] = loss.item()
            loss_unweighted_dict[loss_name] = loss_unweighted.item()
        return total_loss, loss_dict, loss_unweighted_dict
