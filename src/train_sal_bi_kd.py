import argparse
import sys
import time

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from dataloader import DHF1KDataset, HollywoodDataset
from models import create_sal_model
from loss import *
from utils import *


parser = argparse.ArgumentParser(description='Saliency Bi-KD Training')

# Data paths
parser.add_argument('--train_data_path', default='/mnt/SSD/DHF1K/annotation', type=str, help='path to training data')
parser.add_argument('--val_data_path', default='/mnt/SSD/DHF1K/val', type=str, help='path to validation data')

# Training hyperparameters
parser.add_argument('--batch_size', default=4, type=int, help='batch size')
parser.add_argument('--no_workers', default=4, type=int, help='number of data loading workers')
parser.add_argument('--seed', default=0, type=int, help='fix seed')
parser.add_argument('--no_epochs', default=100, type=int, help='number of epochs')
parser.add_argument('--weight_decay', default=1e-5, type=float, help='weight decay')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')

# Model configuration
parser.add_argument('--teacher_name', type=str, required=True, help='teacher model name')
parser.add_argument('--student_name', type=str, required=True, help='student model name')
parser.add_argument('--t_lr', default=1e-6, type=float, help='learning rate for teacher')
parser.add_argument('--s_lr', default=1e-6, type=float, help='learning rate for student')

# Loss coefficients
parser.add_argument('--kldiv', default=True, type=bool, help='use KL divergence loss')
parser.add_argument('--cc', default=True, type=bool, help='use correlation coefficient loss')
parser.add_argument('--nss', default=False, type=bool, help='use NSS loss')
parser.add_argument('--sim', default=False, type=bool, help='use similarity loss')
parser.add_argument('--l1', default=False, type=bool, help='use L1 loss')

parser.add_argument('--kldiv_coeff', default=1.0, type=float, help='KL divergence coefficient')
parser.add_argument('--cc_coeff', default=-1.0, type=float, help='correlation coefficient')
parser.add_argument('--nss_coeff', default=0.0, type=float, help='NSS coefficient')
parser.add_argument('--sim_coeff', default=0.0, type=float, help='similarity coefficient')
parser.add_argument('--l1_coeff', default=0.0, type=float, help='L1 loss coefficient')

# Checkpointing
parser.add_argument('--t_model_val_path', default='best_sal_bi_kd_t.pt', type=str, help='path to save best teacher model')
parser.add_argument('--s_model_val_path', default='best_sal_bi_kd_s.pt', type=str, help='path to save best student model')
parser.add_argument('--load_t_path', default='', type=str, help='path to load teacher model')
parser.add_argument('--load_s_path', default='', type=str, help='path to load student model')

# Dataset configuration
parser.add_argument('--dataset', default='DHF1K', choices=['DHF1K', 'Hollywood2'], type=str, help='dataset name')
parser.add_argument('--clip_size', default=64, type=int, help='clip size for video')

# ViNet-A arguments
parser.add_argument('--use_skip', default=True, type=bool, help='use skip connections')
parser.add_argument('--use_channel_shuffle', default=True, type=bool, help='use channel shuffle')
parser.add_argument('--decoder_groups', default=32, type=int, help='decoder groups')

# Logging
parser.add_argument('--log_interval', default=5, type=int, help='log interval')
parser.add_argument('--log_wandb', default=False, action='store_true', help='log training details to wandb')

args = parser.parse_args()

# Initialize wandb logging if enabled
if args.log_wandb:
	wandb.login()
	run = wandb.init(
		project='Bi-KD',
		config={
			'teacher': args.teacher_name,
			'student': args.student_name,
			'dataset': args.dataset,
			'batch_size': args.batch_size,
			't_lr': args.t_lr,
			's_lr': args.s_lr
		}
	)

print(args)
set_seed(args.seed)

# Create model arguments
model_args = {
	'use_skip': args.use_skip,
	'use_channel_shuffle': args.use_channel_shuffle,
	'decoder_groups': args.decoder_groups
}

# Create models
teacher_model = create_sal_model(args.teacher_name, model_args, pretrained=True, dataset=args.dataset)
student_model = create_sal_model(args.student_name, model_args, pretrained=True, dataset=args.dataset)

# Print model information
print_model_info(teacher_model, "Teacher")
print_model_info(student_model, "Student")

# Create datasets and data loaders
if args.dataset == "DHF1K":
	train_dataset = DHF1KDataset(args.train_data_path, args.clip_size, mode="train", reverse=args.student_name=="TMFI-Net")
	val_dataset = DHF1KDataset(args.val_data_path, args.clip_size, mode="val", reverse=args.student_name=="TMFI-Net")
elif args.dataset == "Hollywood2":
	train_dataset = HollywoodDataset(args.train_data_path, args.clip_size, mode="train")
	val_dataset = HollywoodDataset(args.val_data_path, args.clip_size, mode="val")

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.no_workers)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.no_workers)

# Setup device and move models to GPU(s)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.device_count() > 1:
	print(f"Using {torch.cuda.device_count()} GPUs")
	teacher_model = nn.DataParallel(teacher_model)
	student_model = nn.DataParallel(student_model)

teacher_model.to(device)
student_model.to(device)
print(device)

# Setup optimizers
teacher_params = list(filter(lambda p: p.requires_grad, teacher_model.parameters()))
student_params = list(filter(lambda p: p.requires_grad, student_model.parameters()))

teacher_optimizer = torch.optim.Adam(teacher_params, lr=args.t_lr, weight_decay=args.weight_decay)
student_optimizer = torch.optim.Adam(student_params, lr=args.s_lr, weight_decay=args.weight_decay)

# Load checkpoints if provided (including optimizer states and best loss)
best_t_loss = load_model_checkpoint(teacher_model, teacher_optimizer, args.load_t_path, metric_type='loss')
best_s_loss = load_model_checkpoint(student_model, student_optimizer, args.load_s_path, metric_type='loss')
	 
def train(teacher_model, student_model, teacher_optimizer, student_optimizer, train_loader, epoch, device, args):
	"""Train both teacher and student models using bidirectional knowledge distillation."""
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

		t_img_clips = t_img_clips.to(device).permute((0,2,1,3,4))
		s_img_clips = s_img_clips.to(device).permute((0,2,1,3,4))
		gt_sal = gt_sal.to(device)

		teacher_optimizer.zero_grad()
		student_optimizer.zero_grad()

		t_pred_sal = teacher_model(t_img_clips)
		s_pred_sal = student_model(s_img_clips)

		# Resize teacher predictions to match student predictions
		t_pred_sal = F.interpolate(
			t_pred_sal.unsqueeze(1),
			size=(s_pred_sal.size(1), s_pred_sal.size(2)),
			mode="bilinear",
			align_corners=False
		).squeeze(1)

		assert t_pred_sal.size() == gt_sal.size()
		assert s_pred_sal.size() == gt_sal.size()

		t_loss = loss_func(t_pred_sal, gt_sal, args)
		s_loss = loss_func(s_pred_sal, gt_sal, args)

		# Loss-wise selection of models for bidirectional distillation
		mask_t = (t_loss < s_loss).float()
		mask_s = (t_loss >= s_loss).float()

		# Bidirectional KL divergence
		kl_t = kldiv(s_pred_sal, t_pred_sal)
		kl_s = kldiv(t_pred_sal, s_pred_sal)
		
		loss = (t_loss + s_loss + mask_t * kl_t + mask_s * kl_s).mean()

		# Calculate metrics
		t_cc = cc(t_pred_sal, gt_sal).mean()
		s_cc = cc(s_pred_sal, gt_sal).mean()
		t_kldiv = kldiv(t_pred_sal, gt_sal).mean()
		s_kldiv = kldiv(s_pred_sal, gt_sal).mean()
		t_sim = similarity(t_pred_sal, gt_sal)
		s_sim = similarity(s_pred_sal, gt_sal)

		loss.backward()
		teacher_optimizer.step()
		student_optimizer.step()

		# Update metrics
		curr_loss.update(loss.item())
		total_loss.update(loss.item())
		curr_t_loss.update(t_loss.mean().item())
		curr_s_loss.update(s_loss.mean().item())
		total_t_loss.update(t_loss.mean().item())
		total_s_loss.update(s_loss.mean().item())
		total_t_cc.update(t_cc.item())
		total_s_cc.update(s_cc.item())
		total_t_kldiv.update(t_kldiv.item())
		total_s_kldiv.update(s_kldiv.item())
		total_t_sim.update(t_sim.item())
		total_s_sim.update(s_sim.item())

		# Log training progress at intervals
		if idx % args.log_interval == (args.log_interval - 1):
			print('[{:2d}, {:5d}/{:5d}] avg_loss : {:.3f}, t_loss : {:.3f}, s_loss : {:.3f}, time : {:.3f} minutes'.format(
				epoch+1, idx+1, len(train_loader), curr_loss.avg, curr_t_loss.avg, curr_s_loss.avg, (time.time()-tic)/60))
			curr_loss.reset()
			curr_t_loss.reset()
			curr_s_loss.reset()
			sys.stdout.flush()

	time_taken = (time.time()-tic)/60
	
	# Log to wandb if enabled
	if args.log_wandb:
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
	
	print('[{:2d}, train] avg_loss : {:.3f}, t_avg_loss : {:.3f}, s_avg_loss : {:.3f}, t_cc : {:.3f}, s_cc : {:.3f}, t_kldiv : {:.3f}, s_kldiv : {:.3f}, t_sim : {:.3f}, s_sim : {:.3f}, time : {:.3f}'.format(
		epoch+1, total_loss.avg, total_t_loss.avg, total_s_loss.avg, total_t_cc.avg, total_s_cc.avg, total_t_kldiv.avg, total_s_kldiv.avg, total_t_sim.avg, total_s_sim.avg, time_taken
	))
	sys.stdout.flush()

	return total_loss.avg

def validate(teacher_model, student_model, val_loader, epoch, device, args):
	"""Validate both models on the validation set."""
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

		t_img_clips = t_img_clips.to(device).permute((0,2,1,3,4))
		s_img_clips = s_img_clips.to(device).permute((0,2,1,3,4))

		t_pred_sal = teacher_model(t_img_clips)
		s_pred_sal = student_model(s_img_clips)

		gt_sal = gt_sal.squeeze(0).numpy()

		# Resize predictions to match ground truth
		t_pred_sal = t_pred_sal.cpu().squeeze(0).numpy()
		t_pred_sal = cv2.resize(t_pred_sal, (gt_sal.shape[1], gt_sal.shape[0]))
		t_pred_sal = blur(t_pred_sal).unsqueeze(0).cuda()

		s_pred_sal = s_pred_sal.cpu().squeeze(0).numpy()
		s_pred_sal = cv2.resize(s_pred_sal, (gt_sal.shape[1], gt_sal.shape[0]))
		s_pred_sal = blur(s_pred_sal).unsqueeze(0).cuda()

		gt_sal = torch.FloatTensor(gt_sal).unsqueeze(0).cuda()

		assert t_pred_sal.size() == gt_sal.size()
		assert s_pred_sal.size() == gt_sal.size()

		# Calculate losses and metrics
		t_loss = loss_func(t_pred_sal, gt_sal, args).mean()
		s_loss = loss_func(s_pred_sal, gt_sal, args).mean()
		t_cc = cc(t_pred_sal, gt_sal).mean()
		s_cc = cc(s_pred_sal, gt_sal).mean()
		t_kldiv = kldiv(t_pred_sal, gt_sal).mean()
		s_kldiv = kldiv(s_pred_sal, gt_sal).mean()
		t_sim = similarity(t_pred_sal, gt_sal)
		s_sim = similarity(s_pred_sal, gt_sal)

		# Update metrics
		total_t_loss.update(t_loss.item())
		total_s_loss.update(s_loss.item())
		total_t_cc.update(t_cc.item())
		total_s_cc.update(s_cc.item())
		total_t_kldiv.update(t_kldiv.item())
		total_s_kldiv.update(s_kldiv.item())
		total_t_sim.update(t_sim.item())
		total_s_sim.update(s_sim.item())

	time_taken = (time.time()-tic)/60
	
	# Log to wandb if enabled
	if args.log_wandb:
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
	
	print("[{:2d}, val] t_loss : {:.3f}, s_loss : {:.3f}, t_cc : {:.3f}, s_cc : {:.3f}, t_kldiv : {:.3f}, s_kldiv : {:.3f}, t_sim : {:.3f}, s_sim : {:.3f}, time : {:.3f}".format(
		epoch+1, total_t_loss.avg, total_s_loss.avg, total_t_cc.avg, total_s_cc.avg, total_t_kldiv.avg, total_s_kldiv.avg, total_t_sim.avg, total_s_sim.avg, time_taken
	))
	sys.stdout.flush()

	return total_t_loss.avg, total_s_loss.avg

# Training loop
for epoch in range(args.no_epochs):
	train_loss = train(teacher_model, student_model, teacher_optimizer, student_optimizer, train_loader, epoch, device, args)
	
	with torch.no_grad():
		t_val_loss, s_val_loss = validate(teacher_model, student_model, val_loader, epoch, device, args)
		
		# Save teacher checkpoint if improved
		if t_val_loss <= best_t_loss:
			best_t_loss = t_val_loss
			save_checkpoint(teacher_model, teacher_optimizer, best_t_loss, args.t_model_val_path, "teacher", epoch, metric_type='loss')
		
		# Save student checkpoint if improved
		if s_val_loss <= best_s_loss:
			best_s_loss = s_val_loss
			save_checkpoint(student_model, student_optimizer, best_s_loss, args.s_model_val_path, "student", epoch, metric_type='loss')
	print()

if args.log_wandb:
	wandb.finish()
