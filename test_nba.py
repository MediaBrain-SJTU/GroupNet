import numpy as np
import argparse
import os
import sys
import subprocess
import shutil

sys.path.append(os.getcwd())
from data.dataloader_nba import NBADataset, seq_collate
from utils.torch import *
from utils.config import Config
from model.model_lib_hyper_nba import model_dict
from utils.utils import prepare_seed, print_log, mkdir_if_missing
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.lines as mlines

def get_model_prediction(data, sample_k):
    model.set_data(data)
    recon_motion_3D, _ = model.inference(mode='recon', sample_num=sample_k)
    import time 
    start = time.time()
    sample_motion_3D, data = model.inference(mode='infer', sample_num=sample_k, need_weights=False)
    end = time.time()
    print(end-start)
    sample_motion_3D = sample_motion_3D.transpose(0, 1).contiguous()
    return recon_motion_3D, sample_motion_3D

def get_model_attvalue(data, sample_k):
    model.set_data(data)
    value_list = model.calc_att(mode='infer', sample_num=sample_k, need_weights=False)
    return value_list

def save_prediction(pred, data, suffix, save_dir):
    pred_num = 0
    pred_arr = []
    fut_data, seq_name, frame, valid_id, pred_mask = data['fut_data'], data['seq'], data['frame'], data['valid_id'], data['pred_mask']

    for i in range(len(valid_id)):    # number of agents
        identity = valid_id[i]
        if pred_mask is not None and pred_mask[i] != 1.0:
            continue

        """future frames"""
        for j in range(cfg.future_frames):
            cur_data = fut_data[j]
            if len(cur_data) > 0 and identity in cur_data[:, 1]:
                data = cur_data[cur_data[:, 1] == identity].squeeze()
            else:
                data = most_recent_data.copy()
                data[0] = frame + j + 1
            data[[13, 15]] = pred[i, j].cpu().numpy()   # [13, 15] corresponds to 2D pos
            most_recent_data = data.copy()
            pred_arr.append(data)
        pred_num += 1

    if len(pred_arr) > 0:
        pred_arr = np.vstack(pred_arr)
        indices = [0, 1, 13, 15]            # frame, ID, x, z (remove y which is the height)
        pred_arr = pred_arr[:, indices]
        # save results
        fname = f'{save_dir}/{seq_name}/frame_{int(frame):06d}{suffix}.txt'
        mkdir_if_missing(fname)
        np.savetxt(fname, pred_arr, fmt="%.3f")
    return pred_num

class Constant:
    """A class for handling constants"""
    NORMALIZATION_COEF = 7
    PLAYER_CIRCLE_SIZE = 12 / NORMALIZATION_COEF
    INTERVAL = 10
    DIFF = 6
    X_MIN = 0
    X_MAX = 100
    Y_MIN = 0
    Y_MAX = 50
    COL_WIDTH = 0.3
    SCALE = 1.65
    FONTSIZE = 6
    X_CENTER = X_MAX / 2 - DIFF / 1.5 + 0.10
    Y_CENTER = Y_MAX - DIFF / 1.5 - 0.35
    MESSAGE = 'You can rerun the script and choose any event from 0 to '

def update_radius2(i, player_circles, ball_circle, annotations, player_traj):
	stamp = player_traj[:-1,i,:]
	all_traj = player_traj[:,i,:]
	for j, circle in enumerate(player_circles):
		circle.center = stamp[j,0], stamp[j,1]
		annotations[j].set_position(circle.center)
		# edges[j] = plt.plot(x, y,color='red')
	ball_circle.center = player_traj[-1,i,0], player_traj[-1,i,1]
	ball_circle.radius = 1
	return player_circles, ball_circle

def draw_gif(traj,idx,mode='pre'):
    # traj (N,T,2)
    plt.clf()
    traj = traj*94/28
    actor_num = traj.shape[0]
    length = traj.shape[1]


    ax = plt.axes(xlim=(Constant.X_MIN,
                        Constant.X_MAX),
                    ylim=(Constant.Y_MIN,
                        Constant.Y_MAX))
    ax.axis('off')
    fig = plt.gcf()
    ax.grid(False)  # Remove grid

    annotations = [ax.annotate(i, xy=[0, 0], color='w',
                                horizontalalignment='center',
                                verticalalignment='center', fontweight='bold')
                    for i in range(10)]

    player_circles = []
    for i in range(actor_num-1):
        if i < 5:
            player_circles.append(plt.Circle((traj[i,5,0], traj[i,5,1]), Constant.PLAYER_CIRCLE_SIZE, color='orangered'))
        else:
            player_circles.append(plt.Circle((traj[i,5,0], traj[i,5,1]), Constant.PLAYER_CIRCLE_SIZE, color='royalblue'))


    ball_circle = plt.Circle((0, 0), Constant.PLAYER_CIRCLE_SIZE,color='lime')
    for circle in player_circles:
        ax.add_patch(circle)
    ax.add_patch(ball_circle)


    anim = animation.FuncAnimation(
                        fig, update_radius2,
                        fargs=(player_circles, ball_circle, annotations, traj),
                        frames=15, interval=400)

    court = plt.imread("/DATA7_DB7/data/cxxu/NBA-Player-Movements/court.png")
    plt.imshow(court, zorder=0, extent=[Constant.X_MIN, Constant.X_MAX - Constant.DIFF,
                                        Constant.Y_MAX, Constant.Y_MIN])
    # anim.save('test_img/interaction/'+str(idx)+'.gif',writer='imagemagick')
    if mode == 'gt':
        anim.save('vis/gif/'+str(idx)+'_gt.gif',writer='imagemagick')
    else:
        anim.save('vis/gif/'+str(idx)+'_pre.gif',writer='imagemagick')
    print('ok')
    return

def draw_gif2(traj,idx,mode='pre'):
    # traj (N,T,2)
    plt.clf()
    traj = traj*94/28
    actor_num = traj.shape[0]
    length = traj.shape[1]


    ax = plt.axes(xlim=(Constant.X_MIN,
                        Constant.X_MAX),
                    ylim=(Constant.Y_MIN,
                        Constant.Y_MAX))
    ax.axis('off')
    fig = plt.gcf()
    ax.grid(False)  # Remove grid

    annotations = [ax.annotate(i, xy=[0, 0], color='w',
                                horizontalalignment='center',
                                verticalalignment='center', fontweight='bold')
                    for i in range(10)]

    player_circles = []
    for i in range(actor_num-1):
        if i < 5:
            player_circles.append(plt.Circle((traj[i,5,0], traj[i,5,1]), Constant.PLAYER_CIRCLE_SIZE, color='orangered'))
        else:
            player_circles.append(plt.Circle((traj[i,5,0], traj[i,5,1]), Constant.PLAYER_CIRCLE_SIZE, color='royalblue'))


    ball_circle = plt.Circle((0, 0), Constant.PLAYER_CIRCLE_SIZE,color='lime')
    for circle in player_circles:
        ax.add_patch(circle)
    ax.add_patch(ball_circle)


    anim = animation.FuncAnimation(
                        fig, update_radius2,
                        fargs=(player_circles, ball_circle, annotations, traj),
                        frames=15, interval=400)

    court = plt.imread("/DATA7_DB7/data/cxxu/NBA-Player-Movements/court.png")
    plt.imshow(court, zorder=0, extent=[Constant.X_MIN, Constant.X_MAX - Constant.DIFF,
                                        Constant.Y_MAX, Constant.Y_MIN])
    # anim.save('test_img/interaction/'+str(idx)+'.gif',writer='imagemagick')
    if mode == 'gt':
        anim.save('vis/gif/'+str(idx)+'_gt.gif',writer='imagemagick')
    else:
        anim.save('vis/gif/'+str(idx)+'_pre.gif',writer='imagemagick')
    print('ok')
    return

def draw_result(future,past,mode='pre'):
    # b n t 2
    print('drawing...')
    trajs = np.concatenate((past,future), axis = 2)
    batch = trajs.shape[0]
    for idx in range(500):
        plt.clf()
        traj = trajs[idx]
        traj = traj*94/28
        actor_num = traj.shape[0]
        length = traj.shape[1]
        
        ax = plt.axes(xlim=(Constant.X_MIN,
                            Constant.X_MAX),
                        ylim=(Constant.Y_MIN,
                            Constant.Y_MAX))
        ax.axis('off')
        # ax.spines['right'].set_visible(False)
        # ax.spines['top'].set_visible(False)
        # ax.spines['left'].set_visible(False)
        # ax.spines['bottom'].set_visible(False)
        fig = plt.gcf()
        ax.grid(False)  # Remove grid

        colorteam1 = 'dodgerblue'
        colorteam2 = 'orangered'
        colorball = 'limegreen'
        colorteam1_pre = 'skyblue'
        colorteam2_pre = 'lightsalmon'
        colorball_pre = 'mediumspringgreen'

        # for i in range(actor_num):
        # 	if i < 5:
        # 		ax.add_patch(plt.Circle((traj[i,5,0], traj[i,5,1]), Constant.PLAYER_CIRCLE_SIZE/2, color=colorteam1))
        # 	elif i < 10:
        # 		ax.add_patch(plt.Circle((traj[i,5,0], traj[i,5,1]), Constant.PLAYER_CIRCLE_SIZE/2, color=colorteam2))
        # 	else:
        # 		ax.add_patch(plt.Circle((traj[i,5,0], traj[i,5,1]), 1, color=colorball))
		
        for j in range(actor_num):
            if j < 5:
                color = colorteam1
                color_pre = colorteam1_pre
            elif j < 10:
                color = colorteam2
                color_pre = colorteam2_pre
            else:
                color_pre = colorball_pre
                color = colorball
            for i in range(length):
                points = [(traj[j,i,0],traj[j,i,1])]
                (x, y) = zip(*points)
                # plt.scatter(x, y, color=color,s=20,alpha=0.3+i*((1-0.3)/length))
                if i < 5:
                    plt.scatter(x, y, color=color_pre,s=20,alpha=1)
                else:
                    plt.scatter(x, y, color=color,s=20,alpha=1)

            for i in range(length-1):
                points = [(traj[j,i,0],traj[j,i,1]),(traj[j,i+1,0],traj[j,i+1,1])]
                (x, y) = zip(*points)
                # plt.plot(x, y, color=color,alpha=0.3+i*((1-0.3)/length),linewidth=2)
                if i < 4:
                    plt.plot(x, y, color=color_pre,alpha=0.5,linewidth=2)
                else:
                    plt.plot(x, y, color=color,alpha=1,linewidth=2)

                # # ax.arrow(traj[node_1,i,0], traj[node_1,i,1], traj[node_1,i+1,0]-traj[node_1,i,0], traj[node_1,i+1,1]-traj[node_1,i,1], head_width=0.02, head_length=0.1, shape="full",fc='red',ec='red',alpha=0.9, overhang=0.5)
                # # ax.arrow(traj[node_2,i,0], traj[node_2,i,1], traj[node_2,i+1,0]-traj[node_2,i,0], traj[node_2,i+1,1]-traj[node_2,i,1], head_width=0.02, head_length=0.1, shape="full",fc='red',ec='red',alpha=0.9, overhang=0.5)
                # if node_1 != 10 and node_2 != 10:
                # 	# ax.arrow(traj[10,i,0], traj[10,i,1], traj[10,i+1,0]-traj[10,i,0], traj[10,i+1,1]-traj[10,i,1], head_width=0.02, head_length=0.1, shape="full",fc='red',ec='red',alpha=0.9, overhang=0.5)
                # 	points = [(traj[10,i,0],traj[10,i,1]),(traj[10,i+1,0],traj[10,i+1,1])]
                # 	(x, y) = zip(*points)
                # 	plt.plot(x, y, color='orange')
                # 	plt.scatter(x, y, color='orange',s=8)

        court = plt.imread("/DATA7_DB7/data/cxxu/NBA-Player-Movements/court.png")
        plt.imshow(court, zorder=0, extent=[Constant.X_MIN, Constant.X_MAX - Constant.DIFF,
                                            Constant.Y_MAX, Constant.Y_MIN],alpha=0.5)
        if mode == 'pre':
            plt.savefig('vis/nba_result2/'+str(idx)+'pre.png')
        else:
            plt.savefig('vis/nba_result2/'+str(idx)+'gt.png')
    print('ok')
    return 


def print_att(test_loader, cfg):
    total_num_pred = 0
    all_num = 0
    value_list_all = np.zeros(1)
    for data in test_loader:
        gt_motion_3D = np.array(data['fut_motion_3D']) # B,N,T,2
        previous_3D = np.array(data['pre_motion_3D'])
        with torch.no_grad():
            value_list= get_model_attvalue(data, cfg.sample_k)
        batch = gt_motion_3D.shape[0]
        value_list_all += (value_list*batch)
        all_num += batch
    value_list_all /= all_num
    print(value_list_all)
    import time
    time.sleep(1000)

def vis_result_gif(test_loader, cfg):
    total_num_pred = 0
    all_num = 0

    for data in test_loader:
        gt_motion_3D = np.array(data['fut_motion_3D']) # B,N,T,2
        previous_3D = np.array(data['pre_motion_3D'])
        with torch.no_grad():
            recon_motion_3D, sample_motion_3D = get_model_prediction(data, cfg.sample_k)
        recon_motion_3D, sample_motion_3D = recon_motion_3D * cfg.traj_scale, sample_motion_3D * cfg.traj_scale
        sample_motion_3D = np.array(sample_motion_3D.cpu()) #(20,BN,T,2)
        batch = gt_motion_3D.shape[0]
        actor_num = gt_motion_3D.shape[1]
        y = np.reshape(gt_motion_3D,(batch*actor_num,cfg.future_frames, 2))
        y = y[None].repeat(20,axis=0)
        error = np.linalg.norm(y[:,:,-1,:] - sample_motion_3D[:,:,-1,:],axis=2)
        # error = np.mean(np.linalg.norm(y- sample_motion_3D,axis=3),axis=2)
        indices = np.argmin(error, axis = 0)
        best_guess = sample_motion_3D[indices,np.arange(batch*actor_num)]
        best_guess = np.reshape(best_guess, (batch,actor_num, cfg.future_frames, 2))
        gt = np.reshape(gt_motion_3D,(batch,actor_num,cfg.future_frames, 2))
        previous_3D = np.reshape(previous_3D,(batch,actor_num,cfg.past_frames, 2))

        best_guess = np.concatenate((previous_3D,best_guess),axis=2)
        gt = np.concatenate((previous_3D,gt),axis=2)
        items = [110,132,62,240,246,354]
        for l in items:
            print(best_guess[l].shape)
            draw_gif(best_guess[l],l)
            draw_gif(gt[l],l,mode='gt')
        import time
        time.sleep(1000)

def vis_result(test_loader, cfg):
    total_num_pred = 0
    all_num = 0

    for data in test_loader:
        gt_motion_3D = np.array(data['fut_motion_3D']) # B,N,T,2
        previous_3D = np.array(data['pre_motion_3D'])
        with torch.no_grad():
            recon_motion_3D, sample_motion_3D = get_model_prediction(data, cfg.sample_k)
        recon_motion_3D, sample_motion_3D = recon_motion_3D * cfg.traj_scale, sample_motion_3D * cfg.traj_scale
        sample_motion_3D = np.array(sample_motion_3D.cpu()) #(20,BN,T,2)
        batch = gt_motion_3D.shape[0]
        actor_num = gt_motion_3D.shape[1]
        y = np.reshape(gt_motion_3D,(batch*actor_num,cfg.future_frames, 2))
        y = y[None].repeat(20,axis=0)
        # error = np.linalg.norm(y[:,:,-1,:] - sample_motion_3D[:,:,-1,:],axis=2)
        error = np.mean(np.linalg.norm(y- sample_motion_3D,axis=3),axis=2)
        indices = np.argmin(error, axis = 0)
        best_guess = sample_motion_3D[indices,np.arange(batch*actor_num)]
        best_guess = np.reshape(best_guess, (batch,actor_num, cfg.future_frames, 2))
        gt = np.reshape(gt_motion_3D,(batch,actor_num,cfg.future_frames, 2))
        previous_3D = np.reshape(previous_3D,(batch,actor_num,cfg.past_frames, 2))

        draw_result(best_guess,previous_3D)
        draw_result(gt,previous_3D,mode='gt')
        import time
        time.sleep(100000)
        # sample_motion_3D = sample_motion_3D.transpose(1,0,2,3)
        # print(y.shape)
        # print(sample_motion_3D.shape)
        # time.sleep(1000)

def test_model(test_loader, cfg):
    total_num_pred = 0
    all_num = 0
    l2error_overall = 0
    l2error_dest = 0
    l2error_avg_04s = 0
    l2error_dest_04s = 0
    l2error_avg_08s = 0
    l2error_dest_08s = 0
    l2error_avg_12s = 0
    l2error_dest_12s = 0
    l2error_avg_16s = 0
    l2error_dest_16s = 0
    l2error_avg_20s = 0
    l2error_dest_20s = 0
    l2error_avg_24s = 0
    l2error_dest_24s = 0
    l2error_avg_28s = 0
    l2error_dest_28s = 0
    l2error_avg_32s = 0
    l2error_dest_32s = 0
    l2error_avg_36s = 0
    l2error_dest_36s = 0

    for data in test_loader:
        gt_motion_3D = np.array(data['fut_motion_3D']) * cfg.traj_scale # B,N,T,2
        with torch.no_grad():
            recon_motion_3D, sample_motion_3D = get_model_prediction(data, cfg.sample_k)
        recon_motion_3D, sample_motion_3D = recon_motion_3D * cfg.traj_scale, sample_motion_3D * cfg.traj_scale
        sample_motion_3D = np.array(sample_motion_3D.cpu()) #(BN,20,T,2)
        batch = gt_motion_3D.shape[0]
        actor_num = gt_motion_3D.shape[1]
        y = np.reshape(gt_motion_3D,(batch*actor_num,cfg.future_frames, 2))
        y = y[None].repeat(20,axis=0)
        # sample_motion_3D = sample_motion_3D.transpose(1,0,2,3)
        # print(y.shape)
        # print(sample_motion_3D.shape)
        # time.sleep(1000)
        error_dest = np.linalg.norm(y[:,:,-1,:] - sample_motion_3D[:,:,-1,:],axis=2)
        indices = np.argmin(error_dest, axis = 0)
        best_guess = sample_motion_3D[indices,np.arange(batch*actor_num)]
        y = y[0]
        l2error_avg_04s += np.mean(np.mean(np.linalg.norm(y[:,:1,:] - best_guess[:,:1,:], axis = 2),axis=1))*batch
        l2error_dest_04s += np.mean(np.mean(np.linalg.norm(y[:,0:1,:] - best_guess[:,0:1,:], axis = 2),axis=1))*batch
        l2error_avg_08s += np.mean(np.mean(np.linalg.norm(y[:,:2,:] - best_guess[:,:2,:], axis = 2),axis=1))*batch
        l2error_dest_08s += np.mean(np.mean(np.linalg.norm(y[:,1:2,:] - best_guess[:,1:2,:], axis = 2),axis=1))*batch
        l2error_avg_12s += np.mean(np.mean(np.linalg.norm(y[:,:3,:] - best_guess[:,:3,:], axis = 2),axis=1))*batch
        l2error_dest_12s += np.mean(np.mean(np.linalg.norm(y[:,2:3,:] - best_guess[:,2:3,:], axis = 2),axis=1))*batch
        l2error_avg_16s += np.mean(np.mean(np.linalg.norm(y[:,:4,:] - best_guess[:,:4,:], axis = 2),axis=1))*batch
        l2error_dest_16s += np.mean(np.mean(np.linalg.norm(y[:,3:4,:] - best_guess[:,3:4,:], axis = 2),axis=1))*batch
        l2error_avg_20s += np.mean(np.mean(np.linalg.norm(y[:,:5,:] - best_guess[:,:5,:], axis = 2),axis=1))*batch
        l2error_dest_20s += np.mean(np.mean(np.linalg.norm(y[:,4:5,:] - best_guess[:,4:5,:], axis = 2),axis=1))*batch
        l2error_avg_24s += np.mean(np.mean(np.linalg.norm(y[:,:6,:] - best_guess[:,:6,:], axis = 2),axis=1))*batch
        l2error_dest_24s += np.mean(np.mean(np.linalg.norm(y[:,5:6,:] - best_guess[:,5:6,:], axis = 2),axis=1))*batch
        l2error_avg_28s += np.mean(np.mean(np.linalg.norm(y[:,:7,:] - best_guess[:,:7,:], axis = 2),axis=1))*batch
        l2error_dest_28s += np.mean(np.mean(np.linalg.norm(y[:,6:7,:] - best_guess[:,6:7,:], axis = 2),axis=1))*batch
        l2error_avg_32s += np.mean(np.mean(np.linalg.norm(y[:,:8,:] - best_guess[:,:8,:], axis = 2),axis=1))*batch
        l2error_dest_32s += np.mean(np.mean(np.linalg.norm(y[:,7:8,:] - best_guess[:,7:8,:], axis = 2),axis=1))*batch
        l2error_avg_36s += np.mean(np.mean(np.linalg.norm(y[:,:9,:] - best_guess[:,:9,:], axis = 2),axis=1))*batch
        l2error_dest_36s += np.mean(np.mean(np.linalg.norm(y[:,8:9,:] - best_guess[:,8:9,:], axis = 2),axis=1))*batch
        l2error_overall += np.mean(np.mean(np.linalg.norm(y[:,:10,:] - best_guess[:,:10,:], axis = 2),axis=1))*batch
        l2error_dest += np.mean(np.mean(np.linalg.norm(y[:,9:10,:] - best_guess[:,9:10,:], axis = 2),axis=1))*batch
        all_num += batch

    l2error_overall /= all_num
    l2error_dest /= all_num

    l2error_avg_04s /= all_num
    l2error_dest_04s /= all_num
    l2error_avg_08s /= all_num
    l2error_dest_08s /= all_num
    l2error_avg_12s /= all_num
    l2error_dest_12s /= all_num
    l2error_avg_16s /= all_num
    l2error_dest_16s /= all_num
    l2error_avg_20s /= all_num
    l2error_dest_20s /= all_num
    l2error_avg_24s /= all_num
    l2error_dest_24s /= all_num
    l2error_avg_28s /= all_num
    l2error_dest_28s /= all_num
    l2error_avg_32s /= all_num
    l2error_dest_32s /= all_num
    l2error_avg_36s /= all_num
    l2error_dest_36s /= all_num
    print('##################')
    print('ADE 0.4s:',l2error_avg_04s)
    print('ADE 0.8s:',l2error_avg_08s)
    print('ADE 1.2s:',l2error_avg_12s)
    print('ADE 1.6s:',l2error_avg_16s)
    print('ADE 2.0s:',l2error_avg_20s)
    print('ADE 2.4s:',l2error_avg_24s)
    print('ADE 2.8s:',l2error_avg_28s)
    print('ADE 3.2s:',l2error_avg_32s)
    print('ADE 3.6s:',l2error_avg_36s)
    print('ADE 4.0s:',l2error_overall)

    print('FDE 0.4s:',l2error_dest_04s)
    print('FDE 0.8s:',l2error_dest_08s)
    print('FDE 1.2s:',l2error_dest_12s)
    print('FDE 1.6s:',l2error_dest_16s)
    print('FDE 2.0s:',l2error_dest_20s)
    print('FDE 2.4s:',l2error_dest_24s)
    print('FDE 2.8s:',l2error_dest_28s)
    print('FDE 3.2s:',l2error_dest_32s)
    print('FDE 3.6s:',l2error_dest_36s)
    print('FDE 4.0s:',l2error_dest)
    print('##################')
    import time
    time.sleep(10000)

def test_model_all(test_loader, cfg):
    total_num_pred = 0
    all_num = 0
    l2error_overall = 0
    l2error_dest = 0
    l2error_avg_04s = 0
    l2error_dest_04s = 0
    l2error_avg_08s = 0
    l2error_dest_08s = 0
    l2error_avg_12s = 0
    l2error_dest_12s = 0
    l2error_avg_16s = 0
    l2error_dest_16s = 0
    l2error_avg_20s = 0
    l2error_dest_20s = 0
    l2error_avg_24s = 0
    l2error_dest_24s = 0
    l2error_avg_28s = 0
    l2error_dest_28s = 0
    l2error_avg_32s = 0
    l2error_dest_32s = 0
    l2error_avg_36s = 0
    l2error_dest_36s = 0

    for data in test_loader:
        gt_motion_3D = np.array(data['fut_motion_3D']) * cfg.traj_scale # B,N,T,2
        with torch.no_grad():
            recon_motion_3D, sample_motion_3D = get_model_prediction(data, cfg.sample_k)
        recon_motion_3D, sample_motion_3D = recon_motion_3D * cfg.traj_scale, sample_motion_3D * cfg.traj_scale
        sample_motion_3D = np.array(sample_motion_3D.cpu()) #(BN,20,T,2)
        batch = gt_motion_3D.shape[0]
        actor_num = gt_motion_3D.shape[1]
        y = np.reshape(gt_motion_3D,(batch*actor_num,cfg.future_frames, 2))
        y = y[None].repeat(20,axis=0)
        # sample_motion_3D = sample_motion_3D.transpose(1,0,2,3)
        # print(y.shape)
        # print(sample_motion_3D.shape)
        # time.sleep(1000)
        l2error_avg_04s += np.mean(np.min(np.mean(np.linalg.norm(y[:,:,:1,:] - sample_motion_3D[:,:,:1,:], axis = 3),axis=2),axis=0))*batch
        l2error_dest_04s += np.mean(np.min(np.mean(np.linalg.norm(y[:,:,0:1,:] - sample_motion_3D[:,:,0:1,:], axis = 3),axis=2),axis=0))*batch
        l2error_avg_08s += np.mean(np.min(np.mean(np.linalg.norm(y[:,:,:2,:] - sample_motion_3D[:,:,:2,:], axis = 3),axis=2),axis=0))*batch
        l2error_dest_08s += np.mean(np.min(np.mean(np.linalg.norm(y[:,:,1:2,:] - sample_motion_3D[:,:,1:2,:], axis = 3),axis=2),axis=0))*batch
        l2error_avg_12s += np.mean(np.min(np.mean(np.linalg.norm(y[:,:,:3,:] - sample_motion_3D[:,:,:3,:], axis = 3),axis=2),axis=0))*batch
        l2error_dest_12s += np.mean(np.min(np.mean(np.linalg.norm(y[:,:,2:3,:] - sample_motion_3D[:,:,2:3,:], axis = 3),axis=2),axis=0))*batch
        l2error_avg_16s += np.mean(np.min(np.mean(np.linalg.norm(y[:,:,:4,:] - sample_motion_3D[:,:,:4,:], axis = 3),axis=2),axis=0))*batch
        l2error_dest_16s += np.mean(np.min(np.mean(np.linalg.norm(y[:,:,3:4,:] - sample_motion_3D[:,:,3:4,:], axis = 3),axis=2),axis=0))*batch
        l2error_avg_20s += np.mean(np.min(np.mean(np.linalg.norm(y[:,:,:5,:] - sample_motion_3D[:,:,:5,:], axis = 3),axis=2),axis=0))*batch
        l2error_dest_20s += np.mean(np.min(np.mean(np.linalg.norm(y[:,:,4:5,:] - sample_motion_3D[:,:,4:5,:], axis = 3),axis=2),axis=0))*batch
        l2error_avg_24s += np.mean(np.min(np.mean(np.linalg.norm(y[:,:,:6,:] - sample_motion_3D[:,:,:6,:], axis = 3),axis=2),axis=0))*batch
        l2error_dest_24s += np.mean(np.min(np.mean(np.linalg.norm(y[:,:,5:6,:] - sample_motion_3D[:,:,5:6,:], axis = 3),axis=2),axis=0))*batch
        l2error_avg_28s += np.mean(np.min(np.mean(np.linalg.norm(y[:,:,:7,:] - sample_motion_3D[:,:,:7,:], axis = 3),axis=2),axis=0))*batch
        l2error_dest_28s += np.mean(np.min(np.mean(np.linalg.norm(y[:,:,6:7,:] - sample_motion_3D[:,:,6:7,:], axis = 3),axis=2),axis=0))*batch
        l2error_avg_32s += np.mean(np.min(np.mean(np.linalg.norm(y[:,:,:8,:] - sample_motion_3D[:,:,:8,:], axis = 3),axis=2),axis=0))*batch
        l2error_dest_32s += np.mean(np.min(np.mean(np.linalg.norm(y[:,:,7:8,:] - sample_motion_3D[:,:,7:8,:], axis = 3),axis=2),axis=0))*batch
        l2error_avg_36s += np.mean(np.min(np.mean(np.linalg.norm(y[:,:,:9,:] - sample_motion_3D[:,:,:9,:], axis = 3),axis=2),axis=0))*batch
        l2error_dest_36s += np.mean(np.min(np.mean(np.linalg.norm(y[:,:,8:9,:] - sample_motion_3D[:,:,8:9,:], axis = 3),axis=2),axis=0))*batch
        l2error_overall += np.mean(np.min(np.mean(np.linalg.norm(y[:,:,:10,:] - sample_motion_3D[:,:,:10,:], axis = 3),axis=2),axis=0))*batch
        l2error_dest += np.mean(np.min(np.mean(np.linalg.norm(y[:,:,9:10,:] - sample_motion_3D[:,:,9:10,:], axis = 3),axis=2),axis=0))*batch
        all_num += batch
    print(all_num)
    l2error_overall /= all_num
    l2error_dest /= all_num

    l2error_avg_04s /= all_num
    l2error_dest_04s /= all_num
    l2error_avg_08s /= all_num
    l2error_dest_08s /= all_num
    l2error_avg_12s /= all_num
    l2error_dest_12s /= all_num
    l2error_avg_16s /= all_num
    l2error_dest_16s /= all_num
    l2error_avg_20s /= all_num
    l2error_dest_20s /= all_num
    l2error_avg_24s /= all_num
    l2error_dest_24s /= all_num
    l2error_avg_28s /= all_num
    l2error_dest_28s /= all_num
    l2error_avg_32s /= all_num
    l2error_dest_32s /= all_num
    l2error_avg_36s /= all_num
    l2error_dest_36s /= all_num
    print('##################')
    # print('ADE 0.4s:',l2error_avg_04s)
    # print('ADE 0.8s:',l2error_avg_08s)
    # print('ADE 1.2s:',l2error_avg_12s)
    # print('ADE 1.6s:',l2error_avg_16s)
    # print('ADE 2.0s:',l2error_avg_20s)
    # print('ADE 2.4s:',l2error_avg_24s)
    # print('ADE 2.8s:',l2error_avg_28s)
    # print('ADE 3.2s:',l2error_avg_32s)
    # print('ADE 3.6s:',l2error_avg_36s)
    # print('ADE 4.0s:',l2error_overall)

    # print('FDE 0.4s:',l2error_dest_04s)
    # print('FDE 0.8s:',l2error_dest_08s)
    # print('FDE 1.2s:',l2error_dest_12s)
    # print('FDE 1.6s:',l2error_dest_16s)
    # print('FDE 2.0s:',l2error_dest_20s)
    # print('FDE 2.4s:',l2error_dest_24s)
    # print('FDE 2.8s:',l2error_dest_28s)
    # print('FDE 3.2s:',l2error_dest_32s)
    # print('FDE 3.6s:',l2error_dest_36s)
    # print('FDE 4.0s:',l2error_dest)
    print('ADE 1.0s:',(l2error_avg_08s+l2error_avg_12s)/2)
    print('ADE 2.0s:',l2error_avg_20s)
    print('ADE 3.0s:',(l2error_avg_32s+l2error_avg_28s)/2)
    print('ADE 4.0s:',l2error_overall)

    print('FDE 1.0s:',(l2error_dest_08s+l2error_dest_12s)/2)
    print('FDE 2.0s:',l2error_dest_20s)
    print('FDE 3.0s:',(l2error_dest_28s+l2error_dest_32s)/2)
    print('FDE 4.0s:',l2error_dest)
    print('##################')
    import time
    time.sleep(10000)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default=None)
    parser.add_argument('--data_eval', default='test')
    parser.add_argument('--epochs', default=None)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--cached', action='store_true', default=False)
    parser.add_argument('--cleanup', action='store_true', default=False)
    args = parser.parse_args()

    """ setup """
    cfg = Config(args.cfg)
    if args.epochs is None:
        epochs = [cfg.get_last_epoch()]
    else:
        epochs = [int(x) for x in args.epochs.split(',')]

    torch.set_default_dtype(torch.float32)
    device = torch.device('cuda', index=args.gpu) if args.gpu >= 0 and torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available(): torch.cuda.set_device(args.gpu)
    torch.set_grad_enabled(False)
    log = open(os.path.join(cfg.log_dir, 'log_test.txt'), 'w')

    test_dset = NBADataset(
        obs_len=cfg.past_frames,
        pred_len=cfg.future_frames,
        training=False)

    test_loader = DataLoader(
        test_dset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        collate_fn=seq_collate,
        pin_memory=True)

    for epoch in epochs:
        prepare_seed(cfg.seed)
        """ model """
        if not args.cached:
            model_id = cfg.get('model_id', 'gnnv1')
            model = model_dict[model_id](cfg)
            model.set_device(device)
            model.eval()
            if epoch > 0:
                cp_path = cfg.model_path % epoch
                print_log(f'loading model model from checkpoint: {cp_path}', log, display=True)
                model_cp = torch.load(cp_path, map_location='cpu')
                model.load_state_dict(model_cp['model_dict'], strict=False)
            vis = False
            calc_att = False
            vis_gif = False
            if vis:
                vis_result(test_loader, cfg)
            if calc_att:
                print_att(test_loader, cfg)
            if vis_gif:
                vis_result_gif(test_loader, cfg)
            test_model_all(test_loader, cfg)

        # """ save results and compute metrics """
        # data_splits = [args.data_eval]

        # for split in data_splits:  
        #     save_dir = f'{cfg.result_dir}/epoch_{epoch:04d}/{split}'; mkdir_if_missing(save_dir)
        #     eval_dir = f'{save_dir}/samples'
        #     if not args.cached:
        #         test_model(generator, save_dir, cfg)

        #     log_file = os.path.join(cfg.log_dir, 'log_eval.txt')
        #     cmd = f"python eval.py --dataset {cfg.dataset} --results_dir {eval_dir} --data {split} --log {log_file}"
        #     subprocess.run(cmd.split(' '))

        #     # remove eval folder to save disk space
        #     if args.cleanup:
        #         shutil.rmtree(save_dir)


