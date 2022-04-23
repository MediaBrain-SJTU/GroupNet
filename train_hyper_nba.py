import os
import sys
import argparse
import time
import numpy as np
import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.getcwd())
from data.dataloader_nba import NBADataset, seq_collate
from model.model_lib_hyper_nba import model_dict
from utils.torch import *
from utils.config import Config
from utils.utils import prepare_seed, print_log, AverageMeter, convert_secs2time, get_timestring
import math
from torch.utils.data import DataLoader


torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


def logging(cfg, epoch, total_epoch, iter, total_iter, ep, losses_str, log):
	print_log('{} | Epo: {:02d}/{:02d}, '
		'It: {:04d}/{:04d}, '
		'EP: {:s}, {}'
        .format(cfg, epoch, total_epoch, iter, total_iter, \
		convert_secs2time(ep), losses_str), log)


def train(epoch):
    global tb_ind
    since_train = time.time()
    train_loss_meter = {x: AverageMeter() for x in cfg.loss_cfg.keys()}
    train_loss_meter['total_loss'] = AverageMeter()
    last_generator_index = 0
    total_iter_num = len(train_loader)
    iter_num = 0
    for data in train_loader:
        model.set_data(data)
        model_data = model()
        total_loss, loss_dict, loss_unweighted_dict = model.compute_loss()
        """ optimize """
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        train_loss_meter['total_loss'].update(total_loss.item())
        for key in loss_unweighted_dict.keys():
            train_loss_meter[key].update(loss_unweighted_dict[key])

        if iter_num % cfg.print_freq == 0:
            ep = time.time() - since_train
            losses_str = ' '.join([f'{x}: {y.avg:.3f} ({y.val:.3f})' for x, y in train_loss_meter.items()])
            logging(args.cfg, epoch, cfg.num_epochs, iter_num, total_iter_num, ep, losses_str, log)
            for name, meter in train_loss_meter.items():
                tb_logger.add_scalar('model_' + name, meter.avg, tb_ind)
            tb_ind += 1
        iter_num += 1

    scheduler.step()
    model.step_annealer()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='k10_res')
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--tmp', action='store_true', default=False)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    """ setup """
    cfg = Config(args.cfg, args.tmp, create_dirs=True)
    prepare_seed(cfg.seed)
    torch.set_default_dtype(torch.float32)
    device = torch.device('cuda', index=args.gpu) if torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available(): torch.cuda.set_device(args.gpu)

    time_str = get_timestring()
    log = open(os.path.join(cfg.log_dir, 'log.txt'), 'a+')
    print_log("time str: {}".format(time_str), log)
    print_log("python version : {}".format(sys.version.replace('\n', ' ')), log)
    print_log("torch version : {}".format(torch.__version__), log)
    print_log("cudnn version : {}".format(torch.backends.cudnn.version()), log)
    tb_logger = SummaryWriter(cfg.tb_dir)
    tb_ind = 0

    """ data """
    # generator = data_generator(cfg, log, split='train', phase='training')
    train_dset = NBADataset(
        obs_len=cfg.past_frames,
        pred_len=cfg.future_frames,
        training=True)

    train_loader = DataLoader(
        train_dset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=seq_collate,
        pin_memory=True)

    """ model """
    model_id = cfg.get('model_id', 'gnnv1')
    model = model_dict[model_id](cfg)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler_type = cfg.get('lr_scheduler', 'linear')
    if scheduler_type == 'linear':
        scheduler = get_scheduler(optimizer, policy='lambda', nepoch_fix=cfg.lr_fix_epochs, nepoch=cfg.num_epochs)
    elif scheduler_type == 'step':
        scheduler = get_scheduler(optimizer, policy='step', decay_step=cfg.decay_step, decay_gamma=cfg.decay_gamma)
    else:
        raise ValueError('unknown scheduler type!')

    if args.start_epoch > 0:
        cp_path = cfg.model_path % args.start_epoch
        print_log(f'loading model from checkpoint: {cp_path}', log)
        model_cp = torch.load(cp_path, map_location='cpu')
        model.load_state_dict(model_cp['model_dict'])
        if 'opt_dict' in model_cp:
            optimizer.load_state_dict(model_cp['opt_dict'])
        if 'scheduler_dict' in model_cp:
            scheduler.load_state_dict(model_cp['scheduler_dict'])

    """ start training """
    model.set_device(device)
    model.train()
    for i in range(args.start_epoch, cfg.num_epochs):
        train(i)
        """ save model """
        if cfg.model_save_freq > 0 and (i + 1) % cfg.model_save_freq == 0:
            cp_path = cfg.model_path % (i + 1)
            model_cp = {'model_dict': model.state_dict(), 'opt_dict': optimizer.state_dict(), 'scheduler_dict': scheduler.state_dict(), 'epoch': i + 1}
            torch.save(model_cp, cp_path)

