import argparse
import os
from random import shuffle
import sys
import time

import cv2
from sympy import deg
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import timm
from timm.data.dataset import ImageDataset
from timm.data.loader import create_loader
from transformers import AutoModelForImageClassification
import wandb

from dataloader import DHF1KDataset, HollywoodDataset
from loss import *
from utils import *

# TMFI-Net
import tmfi_net as TMFINet
# EEAA (ViNet-A)
from vinet_a import *
# add ViNet-S
from vinet_s import *

import pdb

parser = argparse.ArgumentParser()

parser.add_argument('--train_data_path',default='/mnt/Shared-Storage/sid/datasets/UCF/training',type=str)
parser.add_argument('--val_data_path',default='/mnt/Shared-Storage/sid/datasets/UCF/testing',type=str)

parser.add_argument('--batch_size',default=4,type=int)
parser.add_argument('--no_workers',default=4,type=int)
parser.add_argument('--teacher_name',type=str,required=True)
parser.add_argument('--student_name',type=str,required=True)

parser.add_argument('--kldiv',default=True, type=bool)
parser.add_argument('--cc',default=True, type=bool)
parser.add_argument('--nss',default=False, type=bool)
parser.add_argument('--sim',default=False, type=bool)
parser.add_argument('--l1',default=False, type=bool)

parser.add_argument('--kldiv_coeff',default=1.0, type=float)
parser.add_argument('--cc_coeff',default=-1.0, type=float)
parser.add_argument('--nss_coeff',default=0.0, type=float)
parser.add_argument('--sim_coeff',default=0.0, type=float)
parser.add_argument('--l1_coeff',default=0.0, type=float)

parser.add_argument('--t_lr',default=1e-6,type=float)
parser.add_argument('--s_lr',default=1e-6,type=float)
parser.add_argument('--momentum',default=0.9,type=float)
parser.add_argument('--weight_decay',default=1e-5,type=float)
parser.add_argument('--no_epochs',default=100,type=int)
parser.add_argument('--log_interval',default=5,type=int)

parser.add_argument('--t_model_val_path',default='best_sal_bi_kd_t.pt',type=str)
parser.add_argument('--s_model_val_path',default='best_sal_bi_kd_s.pt',type=str)
parser.add_argument('--load_t_path',default='',type=str)
parser.add_argument('--load_s_path',default='',type=str)

parser.add_argument('--dataset',default='UCF',type=str)
parser.add_argument('--clip_size',default=64,type=int)

args = parser.parse_args()

# wandb.login(key='76a9ff26e9b1fc6f397721a26e2d565f91815ec3')

# run = wandb.init(
# 	project='Bi-KD',
# 	config={
# 		'teacher': args.teacher_name,
# 		'student': args.student_name,
# 		'batch_size': args.batch_size,
# 		't_lr': args.t_lr,
# 		's_lr': args.s_lr
# 	}
# )

print(args)

# TMFI-Net
teacher_model = TMFINet.VideoSaliencyModel('default')

# TMFI-Net with efficient decoder
# teacher_model = VideoSaliencyModel('efficient', use_neck=True, decoder_groups=32)

student_model = EEAA(use_skip=True, use_channel_shuffle=True, decoder_groups=32)

if args.dataset == "DHF1K":
	teacher_model.load_state_dict(torch.load('/mnt/Shared-Storage/sid/checkpoints/best_VSTNet_DHF1K.pth'))
	student_model.load_state_dict(torch.load('/mnt/Shared-Storage/sid/checkpoints/DHF1K_baseline_32g_0.85294.pt'))

elif args.dataset == "UCF":
	teacher_model.load_state_dict(torch.load('/mnt/Shared-Storage/sid/checkpoints/best_VSTNet_UCF.pth'))
	student_model.load_state_dict(torch.load('/mnt/Shared-Storage/sid/checkpoints/EEAA-B_UCF_action_detection_64csz.pt'))

else:
	teacher_model.load_state_dict(torch.load('/mnt/Shared-Storage/sid/checkpoints/best_VSTNet_Hollywood2.pth'))
	student_model.load_state_dict(torch.load('/mnt/Shared-Storage/sid/checkpoints/Hollywood2_baseline_-0.17169.pt'))

if args.load_t_path != '':
	teacher_model.load_state_dict(torch.load(args.load_t_path)['model_state_dict'])
	
if args.load_s_path != '':
	student_model.load_state_dict(torch.load(args.load_s_path)['model_state_dict'])

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
print("Using seed: ", seed)

for name, param in teacher_model.named_parameters():
	if param.requires_grad:
		print(name, param.size())

model_params_and_size(teacher_model)

for name, param in student_model.named_parameters():
	if param.requires_grad:
		print(name, param.size())

model_params_and_size(student_model)

if args.dataset == "DHF1K":
	train_dataset = DHF1KDataset(args.train_data_path, args.clip_size, mode="train")
	val_dataset = DHF1KDataset(args.val_data_path, args.clip_size, mode="val")
elif args.dataset == "UCF":
	train_dataset = UCFDataset(args.train_data_path, args.clip_size, mode="train")
	val_dataset = UCFDataset(args.val_data_path, args.clip_size, mode="val")
else:
	train_dataset = HollywoodDataset(args.train_data_path, args.clip_size, mode="train")
	val_dataset = HollywoodDataset(args.val_data_path, args.clip_size, mode="val")

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.no_workers)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.no_workers)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
teacher_model.to(device)
student_model.to(device)

print(device)

teacher_params = list(filter(lambda p: p.requires_grad, teacher_model.parameters()))
student_params = list(filter(lambda p: p.requires_grad, student_model.parameters()))

teacher_optimizer = torch.optim.Adam(teacher_params, lr=args.t_lr, weight_decay=args.weight_decay)
student_optimizer = torch.optim.Adam(student_params, lr=args.s_lr, weight_decay=args.weight_decay)

if args.load_t_path != '':
	teacher_optimizer.load_state_dict(torch.load(args.load_t_path)['optimizer_state_dict'])

if args.load_s_path != '':
	student_optimizer.load_state_dict(torch.load(args.load_s_path)['optimizer_state_dict'])
	 
def train(teacher_model, student_model, teacher_optimizer, student_optimizer, train_loader, epoch, device, args):
	teacher_model.train()
	student_model.train()
	
	total_loss = AverageMeter()
	curr_loss = AverageMeter()
	total_t_loss = AverageMeter()
	total_s_loss = AverageMeter()
	curr_t_loss = AverageMeter()
	curr_s_loss = AverageMeter()
	total_t_cc = AverageMeter()
	total_s_cc = AverageMeter()
	total_t_kldiv = AverageMeter()
	total_s_kldiv = AverageMeter()
	total_t_sim = AverageMeter()
	total_s_sim = AverageMeter()
	tic = time.time()

	for idx, sample in enumerate(train_loader):
		t_img_clips, s_img_clips, gt_sal = sample[0], sample[1], sample[2]

		t_img_clips = t_img_clips.to(device)
		t_img_clips = t_img_clips.permute((0,2,1,3,4))
		s_img_clips = s_img_clips.to(device)
		s_img_clips = s_img_clips.permute((0,2,1,3,4))
		gt_sal = gt_sal.to(device)

		teacher_optimizer.zero_grad()
		student_optimizer.zero_grad()

		t_pred_sal = teacher_model(t_img_clips)
		s_pred_sal = student_model(s_img_clips)

		t_pred_sal = F.interpolate(
			t_pred_sal.unsqueeze(1),
			size=(s_pred_sal.size(1), s_pred_sal.size(2)),
			mode="bilinear",
			align_corners=False
		).squeeze(1)

		# same gt size for both models
		assert t_pred_sal.size() == gt_sal.size()
		assert s_pred_sal.size() == gt_sal.size()

		t_loss = loss_func(t_pred_sal, gt_sal, args)
		s_loss = loss_func(s_pred_sal, gt_sal, args)

		# loss wise selection of models
		# mask_t = (t_loss < s_loss).float()
		# mask_s = (t_loss >= s_loss).float()

		# pixel wise confidence selection of models
		# mask_t = (t_pred_sal > s_pred_sal).float()
		# mask_s = (t_pred_sal <= s_pred_sal).float()

		# loss wise KL divergence between models
		kl_t = kldiv(s_pred_sal, t_pred_sal)
		kl_s = kldiv(t_pred_sal, s_pred_sal)
		# bce_t = F.binary_cross_entropy(s_pred_sal, t_pred_sal)
		# bce_s = F.binary_cross_entropy(t_pred_sal, s_pred_sal)
		
		# pixel wise KL divergence between models
		# kl_t = F.kl_div(s_pred_sal.log(), t_pred_sal, reduction='none')
		# kl_s = F.kl_div(t_pred_sal.log(), s_pred_sal, reduction='none')
		
		# loss wise model selection
		# loss = (t_loss + s_loss + mask_t * kl_t + mask_s * kl_s).mean()
		loss = (t_loss + s_loss + kl_t + kl_s).mean()
		# loss = (t_loss + s_loss + mask_t * bce_t + mask_s * bce_s).mean()

		# pixel wise confidence model selection
		# loss = t_loss + s_loss + (mask_t * kl_t + mask_s * kl_s).mean()

		# loss wise model selection
		t_cc = cc(t_pred_sal, gt_sal).mean()
		s_cc = cc(s_pred_sal, gt_sal).mean()
		t_kldiv = kldiv(t_pred_sal, gt_sal).mean()
		s_kldiv = kldiv(s_pred_sal, gt_sal).mean()

		# pixel wise confidence based model selection
		# t_cc = cc(t_pred_sal, gt_sal)
		# s_cc = cc(s_pred_sal, gt_sal)
		# t_kldiv = kldiv(t_pred_sal, gt_sal)
		# s_kldiv = kldiv(s_pred_sal, gt_sal)

		t_sim = similarity(t_pred_sal, gt_sal)
		s_sim = similarity(s_pred_sal, gt_sal)

		loss.backward()
		teacher_optimizer.step()
		student_optimizer.step()

		curr_loss.update(loss.item())
		total_loss.update(loss.item())
		# loss wise model selection
		curr_t_loss.update(t_loss.mean().item())
		curr_s_loss.update(s_loss.mean().item())
		total_t_loss.update(t_loss.mean().item())
		total_s_loss.update(s_loss.mean().item())

		# pixel wise confidence model selection
		# curr_t_loss.update(t_loss.item())
		# curr_s_loss.update(s_loss.item())
		# total_t_loss.update(t_loss.item())
		# total_s_loss.update(s_loss.item())

		total_t_cc.update(t_cc.item())
		total_s_cc.update(s_cc.item())
		total_t_kldiv.update(t_kldiv.item())
		total_s_kldiv.update(s_kldiv.item())
		total_t_sim.update(t_sim.item())
		total_s_sim.update(s_sim.item())

		if idx%args.log_interval==(args.log_interval-1):
			print('[{:2d}, {:5d}/{:5d}] avg_loss : {:.3f}, t_loss : {:.3f}, s_loss : {:.3f}, time : {:.3f} minutes'.format(epoch+1, idx+1, len(train_loader), curr_loss.avg, curr_t_loss.avg, curr_s_loss.avg, (time.time()-tic)/60))
			curr_loss.reset()
			curr_t_loss.reset()
			curr_s_loss.reset()
			sys.stdout.flush()
				  
	loss_dict = {
		'train_avg_loss': total_loss.avg,
		'train_t_avg_loss': total_t_loss.avg,
		'train_s_avg_loss': total_s_loss.avg,
		'train_t_cc': total_t_cc.avg,
		'train_s_cc': total_s_cc.avg,
		'train_t_kldiv': total_t_kldiv.avg,
		'train_s_kldiv': total_s_kldiv.avg,
		'train_t_sim': total_t_sim.avg,
		'train_s_sim': total_s_sim.avg
	}
	wandb.log(loss_dict, step=epoch+1)
	time_taken = (time.time()-tic)/60
	print('[{:2d}, train] avg_loss : {:.3f}, t_avg_loss : {:.3f}, s_avg_loss : {:.3f}, t_cc : {:.3f}, s_cc : {:.3f}, t_kldiv : {:.3f}, s_kldiv : {:.3f}, t_sim : {:.3f}, s_sim : {:.3f}, time : {:.3f}'.format(
		epoch+1, total_loss.avg, total_t_loss.avg, total_s_loss.avg, total_t_cc.avg, total_s_cc.avg, total_t_kldiv.avg, total_s_kldiv.avg, total_t_sim.avg, total_s_sim.avg, time_taken
	))
	sys.stdout.flush()

	return total_loss.avg

def validate(teacher_model, student_model, val_loader, epoch, device, args):
	teacher_model.eval()
	student_model.eval()
	total_t_loss = AverageMeter()
	total_s_loss = AverageMeter()
	total_t_cc = AverageMeter()
	total_s_cc = AverageMeter()
	total_t_kldiv = AverageMeter()
	total_s_kldiv = AverageMeter()
	total_t_sim = AverageMeter()
	total_s_sim = AverageMeter()
	tic = time.time()

	for sample in val_loader:
		t_img_clips, s_img_clips, gt_sal = sample[0], sample[1], sample[2]

		t_img_clips = t_img_clips.to(device)
		t_img_clips = t_img_clips.permute((0,2,1,3,4))
		s_img_clips = s_img_clips.to(device)
		s_img_clips = s_img_clips.permute((0,2,1,3,4))

		t_pred_sal = teacher_model(t_img_clips)
		s_pred_sal = student_model(s_img_clips)

		gt_sal = gt_sal.squeeze(0).numpy()

		t_pred_sal = t_pred_sal.cpu().squeeze(0).numpy()
		t_pred_sal = cv2.resize(t_pred_sal, (gt_sal.shape[1], gt_sal.shape[0]))
		t_pred_sal = blur(t_pred_sal).unsqueeze(0).cuda()

		s_pred_sal = s_pred_sal.cpu().squeeze(0).numpy()
		s_pred_sal = cv2.resize(s_pred_sal, (gt_sal.shape[1], gt_sal.shape[0]))
		s_pred_sal = blur(s_pred_sal).unsqueeze(0).cuda()

		gt_sal = torch.FloatTensor(gt_sal).unsqueeze(0).cuda()

		assert t_pred_sal.size() == gt_sal.size()
		assert s_pred_sal.size() == gt_sal.size()

		# loss wise model selection
		t_loss = loss_func(t_pred_sal, gt_sal, args).mean()
		s_loss = loss_func(s_pred_sal, gt_sal, args).mean()
		t_cc = cc(t_pred_sal, gt_sal).mean()
		s_cc = cc(s_pred_sal, gt_sal).mean()
		t_kldiv = kldiv(t_pred_sal, gt_sal).mean()
		s_kldiv = kldiv(s_pred_sal, gt_sal).mean()

		# pixel wise confidence model selection
		# t_loss = loss_func(t_pred_sal, gt_sal, args)
		# s_loss = loss_func(s_pred_sal, gt_sal, args)
		# t_cc = cc(t_pred_sal, gt_sal)
		# s_cc = cc(s_pred_sal, gt_sal)
		# t_kldiv = kldiv(t_pred_sal, gt_sal)
		# s_kldiv = kldiv(s_pred_sal, gt_sal)

		t_sim = similarity(t_pred_sal, gt_sal)
		s_sim = similarity(s_pred_sal, gt_sal)
		total_t_loss.update(t_loss.item())
		total_s_loss.update(s_loss.item())
		total_t_cc.update(t_cc.item())
		total_s_cc.update(s_cc.item())
		total_t_kldiv.update(t_kldiv.item())
		total_s_kldiv.update(s_kldiv.item())
		total_t_sim.update(t_sim.item())
		total_s_sim.update(s_sim.item())

	loss_dict = {
		'val_t_loss': total_t_loss.avg,
		'val_s_loss': total_s_loss.avg,
		'val_t_cc': total_t_cc.avg,
		'val_s_cc': total_s_cc.avg,
		'val_t_kldiv': total_t_kldiv.avg,
		'val_s_kldiv': total_s_kldiv.avg,
		'val_t_sim': total_t_sim.avg,
		'val_s_sim': total_s_sim.avg
	}
	wandb.log(loss_dict, step=epoch+1)
	time_taken = (time.time()-tic)/60
	print("[{:2d}, val] t_loss : {:.3f}, s_loss : {:.3f}, t_cc : {:.3f}, s_cc : {:.3f}, t_kldiv : {:.3f}, s_kldiv : {:.3f}, t_sim : {:.3f}, s_sim : {:.3f}, time : {:.3f}".format(
		epoch+1, total_t_loss.avg, total_s_loss.avg, total_t_cc.avg, total_s_cc.avg, total_t_kldiv.avg, total_s_kldiv.avg, total_t_sim.avg, total_s_sim.avg, time_taken
	))
	# pdb.set_trace()
	sys.stdout.flush()

	return total_t_loss.avg, total_s_loss.avg

best_teacher = None
best_student = None
for epoch in range(args.no_epochs):
	train_loss = train(teacher_model, student_model, teacher_optimizer, student_optimizer, train_loader, epoch, device, args)
	with torch.no_grad():
		t_val_loss, s_val_loss = validate(teacher_model, student_model, val_loader, epoch, device, args)
		if epoch == 0:
			if args.load_t_path != "":
				best_t_loss = torch.load(args.load_t_path)['loss']
			else:
				best_t_loss = np.inf
			if args.load_s_path != "":
				best_s_loss = torch.load(args.load_s_path)['loss']
			else:
				best_s_loss = np.inf
		if t_val_loss <= best_t_loss:
			best_t_loss = t_val_loss
			best_teacher = teacher_model
			best_t_checkpoint = {
				'model_state_dict': best_teacher.state_dict(),
				'optimizer_state_dict': teacher_optimizer.state_dict(),
				'loss': best_t_loss
			}
			print("[{}, save teacher, {}]".format(epoch+1, args.t_model_val_path))
			torch.save(best_t_checkpoint, args.t_model_val_path)
		if s_val_loss <= best_s_loss:
			best_s_loss = s_val_loss
			best_student = student_model
			best_s_checkpoint = {
				'model_state_dict': best_student.state_dict(),
				'optimizer_state_dict': student_optimizer.state_dict(),
				'loss': best_s_loss
			}
			print("[{}, save student, {}]".format(epoch+1, args.s_model_val_path))
			torch.save(best_s_checkpoint, args.s_model_val_path)
	print()

wandb.finish()
