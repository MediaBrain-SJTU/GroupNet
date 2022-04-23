import os
import sys
import argparse
import time
import numpy as np
import torch
import random
from torch import optim
from torch.optim import lr_scheduler
sys.path.append(os.getcwd())
from torch.utils.data import DataLoader
from data.dataloader_nba import NBADataset, seq_collate
from model.GroupNet_nba import GroupNet
import math

torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--dataset', default='nba')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--past_length', type=int, default=5)
parser.add_argument('--future_length', type=int, default=10)
parser.add_argument('--traj_scale', type=int, default=1)
parser.add_argument('--learn_prior', action='store_true', default=False)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--sample_k', type=int, default=20)
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--decay_step', type=int, default=10)
parser.add_argument('--decay_gamma', type=float, default=0.5)
parser.add_argument('--iternum_print', type=int, default=100)

parser.add_argument('--ztype', default='gaussian')
parser.add_argument('--zdim', type=int, default=32)
parser.add_argument('--hidden_dim', type=int, default=32)
parser.add_argument('--hyper_scales', nargs='+', type=int,default=[5,11])
parser.add_argument('--num_decompose', type=int, default=2)
parser.add_argument('--min_clip', type=float, default=2.0)

parser.add_argument('--model_save_dir', default='saved_models/nba')
parser.add_argument('--model_save_epoch', type=int, default=5)

parser.add_argument('--epoch_continue', type=int, default=0)
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()

""" setup """
np.random.seed(args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.set_default_dtype(torch.float32)
device = torch.device('cuda', index=args.gpu) if torch.cuda.is_available() else torch.device('cpu')
if torch.cuda.is_available(): 
    torch.cuda.set_device(args.gpu)
print('device:',device)
print(args)

def train(train_loader,epoch):
    model.train()
    total_iter_num = len(train_loader)
    iter_num = 0
    for data in train_loader:
        total_loss,loss_pred,loss_recover,loss_kl,loss_diverse = model(data)
        """ optimize """
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if iter_num % args.iternum_print == 0:
            print('Epochs: {:02d}/{:02d}| It: {:04d}/{:04d} | Total loss: {:03f}| Loss_pred: {:03f}| Loss_recover: {:03f}| Loss_kl: {:03f}| Loss_diverse: {:03f}'
            .format(epoch,args.num_epochs,iter_num,total_iter_num,total_loss.item(),loss_pred,loss_recover,loss_kl,loss_diverse))
        iter_num += 1

    scheduler.step()
    model.step_annealer()


""" model & optimizer """
model = GroupNet(args,device)
optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = lr_scheduler.StepLR(optimizer, step_size=args.decay_step, gamma=args.decay_gamma)

""" dataloader """
train_set = NBADataset(
    obs_len=args.past_length,
    pred_len=args.future_length,
    training=True)

train_loader = DataLoader(
    train_set,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=4,
    collate_fn=seq_collate,
    pin_memory=True)

""" Loading if needed """
if args.epoch_continue > 0:
    checkpoint_path = os.path.join(args.model_save_dir,str(args.epoch_continue)+'.p')
    print('load model from: {checkpoint_path}')
    model_load = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(model_load['model_dict'])
    if 'optimizer' in model_load:
        optimizer.load_state_dict(model_load['optimizer'])
    if 'scheduler' in model_load:
        scheduler.load_state_dict(model_load['scheduler'])

""" start training """
model.set_device(device)
for epoch in range(args.epoch_continue, args.num_epochs):
    train(train_loader,epoch)
    """ save model """
    if  (epoch + 1) % args.model_save_epoch == 0:
        model_saved = {'model_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'epoch': epoch + 1,'model_cfg': args}
        saved_path = os.path.join(args.model_save_dir,str(epoch+1)+'.p')
        torch.save(model_saved, saved_path)



