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


parser = argparse.ArgumentParser(description='Saliency Tri-KD Training')

# Data paths
parser.add_argument('--train_data_path', default='/mnt/Shared-Storage/sid/datasets/UCF/training', type=str, help='path to training data')
parser.add_argument('--val_data_path', default='/mnt/Shared-Storage/sid/datasets/UCF/testing', type=str, help='path to validation data')

# Training hyperparameters
parser.add_argument('--batch_size', default=4, type=int, help='batch size')
parser.add_argument('--no_workers', default=4, type=int, help='number of data loading workers')
parser.add_argument('--seed', default=0, type=int, help='fix seed')
parser.add_argument('--no_epochs', default=100, type=int, help='number of epochs')
parser.add_argument('--weight_decay', default=1e-5, type=float, help='weight decay')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')

# Model configuration
parser.add_argument('--model_1', type=str, required=True, help='first model name')
parser.add_argument('--model_2', type=str, required=True, help='second model name')
parser.add_argument('--model_3', type=str, required=True, help='third model name')
parser.add_argument('--lr_1', default=1e-6, type=float, help='learning rate for model 1')
parser.add_argument('--lr_2', default=1e-6, type=float, help='learning rate for model 2')
parser.add_argument('--lr_3', default=1e-6, type=float, help='learning rate for model 3')

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
parser.add_argument('--model_1_val_path', default='best_sal_tri_kd_1.pt', type=str, help='path to save best model 1')
parser.add_argument('--model_2_val_path', default='best_sal_tri_kd_2.pt', type=str, help='path to save best model 2')
parser.add_argument('--model_3_val_path', default='best_sal_tri_kd_3.pt', type=str, help='path to save best model 3')
parser.add_argument('--load_1_path', default='', type=str, help='path to load model 1')
parser.add_argument('--load_2_path', default='', type=str, help='path to load model 2')
parser.add_argument('--load_3_path', default='', type=str, help='path to load model 3')

# Dataset configuration
parser.add_argument('--dataset', default='DHF1K', choices=['DHF1K', 'Hollywood2'], type=str, help='dataset name')
parser.add_argument('--clip_size', default=64, type=int, help='clip size for video')

# ViNet-S arguments
parser.add_argument('--decoder_upsample', default=1, type=int, help='use upsampling in decoder')
parser.add_argument('--num_hier', default=3, type=int, help='number of hierarchical levels')
parser.add_argument('--grouped_conv', default=True, type=bool, help='use grouped convolutions')
parser.add_argument('--root_grouping', default=True, type=bool, help='use root grouping')
parser.add_argument('--depth_grouping', default=False, type=bool, help='use depth grouping')
parser.add_argument('--efficientnet', default=False, type=bool, help='use EfficientNet backbone')
parser.add_argument('--use_trilinear_upsampling', default=False, type=bool, help='use trilinear upsampling')

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
			'model_1': args.model_1,
			'model_2': args.model_2,
			'model_3': args.model_3,
			'dataset': args.dataset,
			'batch_size': args.batch_size,
			'lr_1': args.lr_1,
			'lr_2': args.lr_2,
			'lr_3': args.lr_3
		}
	)

print(args)
set_seed(args.seed)

# Create model arguments for ViNet-S
vinet_s_args = {
	'decoder_upsample': args.decoder_upsample,
	'num_hier': args.num_hier,
	'grouped_conv': args.grouped_conv,
	'root_grouping': args.root_grouping,
	'depth_grouping': args.depth_grouping,
	'efficientnet': args.efficientnet,
	'use_trilinear_upsampling': args.use_trilinear_upsampling
}

# Create model arguments for ViNet-A
vinet_a_args = {
	'use_skip': args.use_skip,
	'use_channel_shuffle': args.use_channel_shuffle,
	'decoder_groups': args.decoder_groups
}

# Select appropriate model args based on model names
model_1_args = vinet_s_args if args.model_1 == 'ViNet-S' else vinet_a_args if args.model_1 == 'ViNet-A' else {}
model_2_args = vinet_s_args if args.model_2 == 'ViNet-S' else vinet_a_args if args.model_2 == 'ViNet-A' else {}
model_3_args = vinet_s_args if args.model_3 == 'ViNet-S' else vinet_a_args if args.model_3 == 'ViNet-A' else {}

# Create models
model_1 = create_sal_model(args.model_1, model_1_args, pretrained=True, dataset=args.dataset)
model_2 = create_sal_model(args.model_2, model_2_args, pretrained=True, dataset=args.dataset)
model_3 = create_sal_model(args.model_3, model_3_args, pretrained=True, dataset=args.dataset)

# Print model information
print_model_info(model_1, "Model 1")
print_model_info(model_2, "Model 2")
print_model_info(model_3, "Model 3")

# Create datasets and data loaders
if args.dataset == "DHF1K":
	train_dataset = DHF1KDataset(args.train_data_path, args.clip_size, mode="train")
	val_dataset = DHF1KDataset(args.val_data_path, args.clip_size, mode="val")
elif args.dataset == "Hollywood2":
	train_dataset = HollywoodDataset(args.train_data_path, args.clip_size, mode="train")
	val_dataset = HollywoodDataset(args.val_data_path, args.clip_size, mode="val")

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.no_workers)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.no_workers)

# Setup device and move models to GPU(s)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.device_count() > 1:
	print(f"Using {torch.cuda.device_count()} GPUs")
	model_1 = nn.DataParallel(model_1)
	model_2 = nn.DataParallel(model_2)
	model_3 = nn.DataParallel(model_3)

model_1.to(device)
model_2.to(device)
model_3.to(device)
print(device)

# Setup optimizers
model_1_params = list(filter(lambda p: p.requires_grad, model_1.parameters()))
model_2_params = list(filter(lambda p: p.requires_grad, model_2.parameters()))
model_3_params = list(filter(lambda p: p.requires_grad, model_3.parameters()))

optimizer_1 = torch.optim.Adam(model_1_params, lr=args.lr_1, weight_decay=args.weight_decay)
optimizer_2 = torch.optim.Adam(model_2_params, lr=args.lr_2, weight_decay=args.weight_decay)
optimizer_3 = torch.optim.Adam(model_3_params, lr=args.lr_3, weight_decay=args.weight_decay)

# Load checkpoints if provided (including optimizer states and best loss)
best_1_loss = load_model_checkpoint(model_1, optimizer_1, args.load_1_path, metric_type='loss')
best_2_loss = load_model_checkpoint(model_2, optimizer_2, args.load_2_path, metric_type='loss')
best_3_loss = load_model_checkpoint(model_3, optimizer_3, args.load_3_path, metric_type='loss')


def train(model_1, model_2, model_3, optimizer_1, optimizer_2, optimizer_3, train_loader, epoch, device, args):
	"""Train all three models using tri-directional knowledge distillation.
	
	Uses loss-wise confidence to select which model acts as teacher for others.
	Only the model with the lowest loss for a sample teaches the other two models.
	"""
	model_1.train()
	model_2.train()
	model_3.train()
	
	total_loss = AverageMeter()
	curr_loss = AverageMeter()
	total_1_loss = AverageMeter()
	total_2_loss = AverageMeter()
	total_3_loss = AverageMeter()
	curr_1_loss = AverageMeter()
	curr_2_loss = AverageMeter()
	curr_3_loss = AverageMeter()
	total_1_cc = AverageMeter()
	total_2_cc = AverageMeter()
	total_3_cc = AverageMeter()
	total_1_kldiv = AverageMeter()
	total_2_kldiv = AverageMeter()
	total_3_kldiv = AverageMeter()
	total_1_sim = AverageMeter()
	total_2_sim = AverageMeter()
	total_3_sim = AverageMeter()
	tic = time.time()

	for idx, sample in enumerate(train_loader):
		img_1_2_clips, img_3_clips, gt_sal = sample[0], sample[1], sample[2]

		img_1_2_clips = img_1_2_clips.to(device).permute((0,2,1,3,4))
		img_3_clips = img_3_clips.to(device).permute((0,2,1,3,4))
		gt_sal = gt_sal.to(device)

		optimizer_1.zero_grad()
		optimizer_2.zero_grad()
		optimizer_3.zero_grad()

		pred_sal_1 = model_1(img_1_2_clips)
		pred_sal_2 = model_2(img_1_2_clips)
		pred_sal_3 = model_3(img_3_clips)

		# Resize predictions to match ground truth size
		pred_sal_1 = F.interpolate(
			pred_sal_1.unsqueeze(1),
			size=(pred_sal_3.size(1), pred_sal_3.size(2)),
			mode="bilinear",
			align_corners=False
		).squeeze(1)

		pred_sal_2 = F.interpolate(
			pred_sal_2.unsqueeze(1),
			size=(pred_sal_3.size(1), pred_sal_3.size(2)),
			mode="bilinear",
			align_corners=False
		).squeeze(1)

		assert pred_sal_1.size() == gt_sal.size()
		assert pred_sal_2.size() == gt_sal.size()
		assert pred_sal_3.size() == gt_sal.size()

		# Calculate individual losses
		loss_1 = loss_func(pred_sal_1, gt_sal, args)
		loss_2 = loss_func(pred_sal_2, gt_sal, args)
		loss_3 = loss_func(pred_sal_3, gt_sal, args)

		# Find the best performing model (lowest loss) for each sample
		loss_stack = torch.stack((loss_1, loss_2, loss_3), dim=1)

		min_loss, min_indices = torch.min(loss_stack, dim=1)
		min_count = torch.sum(loss_stack == min_loss.unsqueeze(1), dim=1)
		valid_mask = (min_count == 1)

		# Create masks for loss-wise selection of teacher models
		mask_1 = (valid_mask & (min_indices == 0)).float()  # Model 1 is best
		mask_2 = (valid_mask & (min_indices == 1)).float()  # Model 2 is best
		mask_3 = (valid_mask & (min_indices == 2)).float()  # Model 3 is best

		# Tri-directional KL divergence: best model teaches others
		kl_1 = kldiv(pred_sal_2, pred_sal_1) + kldiv(pred_sal_3, pred_sal_1)
		kl_2 = kldiv(pred_sal_1, pred_sal_2) + kldiv(pred_sal_3, pred_sal_2)
		kl_3 = kldiv(pred_sal_1, pred_sal_3) + kldiv(pred_sal_2, pred_sal_3)
		
		# Combined loss: ground truth loss + distillation loss
		loss = (loss_1 + loss_2 + loss_3 + mask_1 * kl_1 + mask_2 * kl_2 + mask_3 * kl_3).mean()

		# Calculate metrics
		cc_1 = cc(pred_sal_1, gt_sal).mean()
		cc_2 = cc(pred_sal_2, gt_sal).mean()
		cc_3 = cc(pred_sal_3, gt_sal).mean()
		kldiv_1 = kldiv(pred_sal_1, gt_sal).mean()
		kldiv_2 = kldiv(pred_sal_2, gt_sal).mean()
		kldiv_3 = kldiv(pred_sal_3, gt_sal).mean()
		sim_1 = similarity(pred_sal_1, gt_sal)
		sim_2 = similarity(pred_sal_2, gt_sal)
		sim_3 = similarity(pred_sal_3, gt_sal)

		loss.backward()
		optimizer_1.step()
		optimizer_2.step()
		optimizer_3.step()

		# Update metrics
		curr_loss.update(loss.item())
		total_loss.update(loss.item())
		curr_1_loss.update(loss_1.mean().item())
		curr_2_loss.update(loss_2.mean().item())
		curr_3_loss.update(loss_3.mean().item())
		total_1_loss.update(loss_1.mean().item())
		total_2_loss.update(loss_2.mean().item())
		total_3_loss.update(loss_3.mean().item())
		total_1_cc.update(cc_1.item())
		total_2_cc.update(cc_2.item())
		total_3_cc.update(cc_3.item())
		total_1_kldiv.update(kldiv_1.item())
		total_2_kldiv.update(kldiv_2.item())
		total_3_kldiv.update(kldiv_3.item())
		total_1_sim.update(sim_1.item())
		total_2_sim.update(sim_2.item())
		total_3_sim.update(sim_3.item())

		# Log training progress at intervals
		if idx % args.log_interval == (args.log_interval - 1):
			print('[{:2d}, {:5d}/{:5d}] avg_loss : {:.3f}, 1_loss : {:.3f}, 2_loss : {:.3f}, 3_loss : {:.3f}, time : {:.3f} minutes'.format(
				epoch+1, idx+1, len(train_loader), curr_loss.avg, curr_1_loss.avg, curr_2_loss.avg, curr_3_loss.avg, (time.time()-tic)/60))
			curr_loss.reset()
			curr_1_loss.reset()
			curr_2_loss.reset()
			curr_3_loss.reset()
			sys.stdout.flush()

	time_taken = (time.time()-tic)/60
	
	# Log to wandb if enabled
	if args.log_wandb:
		loss_dict = {
			'train_avg_loss': total_loss.avg,
			'train_1_avg_loss': total_1_loss.avg,
			'train_2_avg_loss': total_2_loss.avg,
			'train_3_avg_loss': total_3_loss.avg,	
			'train_1_cc': total_1_cc.avg,
			'train_2_cc': total_2_cc.avg,
			'train_3_cc': total_3_cc.avg,
			'train_1_kldiv': total_1_kldiv.avg,
			'train_2_kldiv': total_2_kldiv.avg,
			'train_3_kldiv': total_3_kldiv.avg,
			'train_1_sim': total_1_sim.avg,
			'train_2_sim': total_2_sim.avg,
			'train_3_sim': total_3_sim.avg
		}
		wandb.log(loss_dict, step=epoch+1)
	
	print('[{:2d}, train] avg_loss : {:.3f}, 1_avg_loss : {:.3f}, 2_avg_loss : {:.3f}, 3_avg_loss : {:.3f}, 1_cc : {:.3f}, 2_cc : {:.3f}, 3_cc : {:.3f}, '
	'1_kldiv : {:.3f}, 2_kldiv : {:.3f}, 3_kldiv : {:.3f}, 1_sim : {:.3f}, 2_sim : {:.3f}, 3_sim : {:.3f}, time : {:.3f}'.format(
		epoch+1, total_loss.avg, total_1_loss.avg, total_2_loss.avg, total_3_loss.avg, total_1_cc.avg, total_2_cc.avg, total_3_cc.avg, 
		total_1_kldiv.avg, total_2_kldiv.avg, total_3_kldiv.avg, total_1_sim.avg, total_2_sim.avg, total_3_sim.avg, time_taken
	))
	sys.stdout.flush()

	return total_loss.avg



def validate(model_1, model_2, model_3, val_loader, epoch, device, args):
	"""Validate all three models on the validation set."""
	model_1.eval()
	model_2.eval()
	model_3.eval()
	
	total_1_loss = AverageMeter()
	total_2_loss = AverageMeter()
	total_3_loss = AverageMeter()
	total_1_cc = AverageMeter()
	total_2_cc = AverageMeter()
	total_3_cc = AverageMeter()
	total_1_kldiv = AverageMeter()
	total_2_kldiv = AverageMeter()
	total_3_kldiv = AverageMeter()
	total_1_sim = AverageMeter()
	total_2_sim = AverageMeter()
	total_3_sim = AverageMeter()
	tic = time.time()

	for sample in val_loader:
		img_1_2_clips, img_3_clips, gt_sal = sample[0], sample[1], sample[2]

		img_1_2_clips = img_1_2_clips.to(device).permute((0,2,1,3,4))
		img_3_clips = img_3_clips.to(device).permute((0,2,1,3,4))

		pred_sal_1 = model_1(img_1_2_clips)
		pred_sal_2 = model_2(img_1_2_clips)
		pred_sal_3 = model_3(img_3_clips)

		gt_sal = gt_sal.squeeze(0).numpy()

		# Resize predictions to match ground truth and apply blur
		pred_sal_1 = pred_sal_1.cpu().squeeze(0).numpy()
		pred_sal_1 = cv2.resize(pred_sal_1, (gt_sal.shape[1], gt_sal.shape[0]))
		pred_sal_1 = blur(pred_sal_1).unsqueeze(0).cuda()

		pred_sal_2 = pred_sal_2.cpu().squeeze(0).numpy()
		pred_sal_2 = cv2.resize(pred_sal_2, (gt_sal.shape[1], gt_sal.shape[0]))
		pred_sal_2 = blur(pred_sal_2).unsqueeze(0).cuda()

		pred_sal_3 = pred_sal_3.cpu().squeeze(0).numpy()
		pred_sal_3 = cv2.resize(pred_sal_3, (gt_sal.shape[1], gt_sal.shape[0]))
		pred_sal_3 = blur(pred_sal_3).unsqueeze(0).cuda()

		gt_sal = torch.FloatTensor(gt_sal).unsqueeze(0).cuda()

		assert pred_sal_1.size() == gt_sal.size()
		assert pred_sal_2.size() == gt_sal.size()
		assert pred_sal_3.size() == gt_sal.size()

		# Calculate losses and metrics
		loss_1 = loss_func(pred_sal_1, gt_sal, args).mean()
		loss_2 = loss_func(pred_sal_2, gt_sal, args).mean()
		loss_3 = loss_func(pred_sal_3, gt_sal, args).mean()
		cc_1 = cc(pred_sal_1, gt_sal).mean()
		cc_2 = cc(pred_sal_2, gt_sal).mean()
		cc_3 = cc(pred_sal_3, gt_sal).mean()
		kldiv_1 = kldiv(pred_sal_1, gt_sal).mean()
		kldiv_2 = kldiv(pred_sal_2, gt_sal).mean()
		kldiv_3 = kldiv(pred_sal_3, gt_sal).mean()
		sim_1 = similarity(pred_sal_1, gt_sal)
		sim_2 = similarity(pred_sal_2, gt_sal)
		sim_3 = similarity(pred_sal_3, gt_sal)

		# Update metrics
		total_1_loss.update(loss_1.item())
		total_2_loss.update(loss_2.item())
		total_3_loss.update(loss_3.item())
		total_1_cc.update(cc_1.item())
		total_2_cc.update(cc_2.item())
		total_3_cc.update(cc_3.item())
		total_1_kldiv.update(kldiv_1.item())
		total_2_kldiv.update(kldiv_2.item())
		total_3_kldiv.update(kldiv_3.item())
		total_1_sim.update(sim_1.item())
		total_2_sim.update(sim_2.item())
		total_3_sim.update(sim_3.item())

	time_taken = (time.time()-tic)/60
	
	# Log to wandb if enabled
	if args.log_wandb:
		loss_dict = {
			'val_1_loss': total_1_loss.avg,
			'val_2_loss': total_2_loss.avg,
			'val_3_loss': total_3_loss.avg,
			'val_1_cc': total_1_cc.avg,
			'val_2_cc': total_2_cc.avg,
			'val_3_cc': total_3_cc.avg,
			'val_1_kldiv': total_1_kldiv.avg,
			'val_2_kldiv': total_2_kldiv.avg,
			'val_3_kldiv': total_3_kldiv.avg,
			'val_1_sim': total_1_sim.avg,
			'val_2_sim': total_2_sim.avg,
			'val_3_sim': total_3_sim.avg
		}
		wandb.log(loss_dict, step=epoch+1)
	
	print("[{:2d}, val] 1_loss : {:.3f}, 2_loss : {:.3f}, 3_loss : {:.3f}, 1_cc : {:.3f}, 2_cc : {:.3f}, 3_cc : {:.3f}, "
	"1_kldiv : {:.3f}, 2_kldiv : {:.3f}, 3_kldiv : {:.3f}, 1_sim : {:.3f}, 2_sim : {:.3f}, 3_sim : {:.3f}, time : {:.3f}".format(
		epoch+1, total_1_loss.avg, total_2_loss.avg, total_3_loss.avg, total_1_cc.avg, total_2_cc.avg, total_3_cc.avg, 
		total_1_kldiv.avg, total_2_kldiv.avg, total_3_kldiv.avg, total_1_sim.avg, total_2_sim.avg, total_3_sim.avg, time_taken
	))
	sys.stdout.flush()

	return total_1_loss.avg, total_2_loss.avg, total_3_loss.avg



# Training loop
for epoch in range(args.no_epochs):
	train_loss = train(model_1, model_2, model_3, optimizer_1, optimizer_2, optimizer_3, train_loader, epoch, device, args)
	
	with torch.no_grad():
		val_loss_1, val_loss_2, val_loss_3 = validate(model_1, model_2, model_3, val_loader, epoch, device, args)
		
		# Save model 1 checkpoint if improved
		if val_loss_1 <= best_1_loss:
			best_1_loss = val_loss_1
			save_checkpoint(model_1, optimizer_1, best_1_loss, args.model_1_val_path, "model_1", epoch, metric_type='loss')
		
		# Save model 2 checkpoint if improved
		if val_loss_2 <= best_2_loss:
			best_2_loss = val_loss_2
			save_checkpoint(model_2, optimizer_2, best_2_loss, args.model_2_val_path, "model_2", epoch, metric_type='loss')
		
		# Save model 3 checkpoint if improved
		if val_loss_3 <= best_3_loss:
			best_3_loss = val_loss_3
			save_checkpoint(model_3, optimizer_3, best_3_loss, args.model_3_val_path, "model_3", epoch, metric_type='loss')
	print()

if args.log_wandb:
	wandb.finish()
