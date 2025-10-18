import argparse
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.data.dataset import ImageDataset
from timm.data.loader import create_loader
import wandb

from models import create_model
from utils import *


parser = argparse.ArgumentParser(description='Tri-KD Training (3 models)')

# Data paths
parser.add_argument('--train_data_path',default='/mnt/SSD/ImageNet/train',type=str,help='path to training data')
parser.add_argument('--val_data_path',default='/mnt/SSD/ImageNet/val',type=str,help='path to validation data')

# Training hyperparameters
parser.add_argument('--batch_size',default=128,type=int,help='batch size')
parser.add_argument('--seed',default=0,type=int,help='fix seed')
parser.add_argument('--no_epochs',default=20,type=int,help='number of epochs')
parser.add_argument('--weight_decay',default=1e-5,type=float,help='weight decay')

# Model configuration
parser.add_argument('--model_1',type=str,required=True)
parser.add_argument('--model_2',type=str,required=True)
parser.add_argument('--model_3',type=str,required=True)
parser.add_argument('--lr_1',default=1e-6,type=float,help='learning rate for model 1')
parser.add_argument('--lr_2',default=1e-6,type=float,help='learning rate for model 2')
parser.add_argument('--lr_3',default=1e-6,type=float,help='learning rate for model 3')

# Checkpointing
parser.add_argument('--load_1_path',default='',type=str,help='path to load model 1')
parser.add_argument('--load_2_path',default='',type=str,help='path to load model 2')
parser.add_argument('--load_3_path',default='',type=str,help='path to load model 3')
parser.add_argument('--model_1_val_path',default='best_tri_kd_1.pt',type=str,help='path to save the best model 1')
parser.add_argument('--model_2_val_path',default='best_tri_kd_2.pt',type=str,help='path to save the best model 2')
parser.add_argument('--model_3_val_path',default='best_tri_kd_3.pt',type=str,help='path to save the best model 3')

# Logging
parser.add_argument('--log_interval',default=5,type=int,help='log interval')
parser.add_argument('--log_wandb', default=False, action='store_true',help='log training details to wandb')

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
			'batch_size': args.batch_size,
			'lr_1': args.lr_1,
			'lr_2': args.lr_2,
			'lr_3': args.lr_3
		}
	)

print(args)
set_seed(args.seed)

# Create models
model_1 = create_model(args.model_1, pretrained=True)
model_2 = create_model(args.model_2, pretrained=True)
model_3 = create_model(args.model_3, pretrained=True)

# Print model information
print_model_info(model_1, "Model 1")
print_model_info(model_2, "Model 2")
print_model_info(model_3, "Model 3")

# Create datasets and data loaders
# Print model information
print_model_info(model_1, "Model 1")
print_model_info(model_2, "Model 2")
print_model_info(model_3, "Model 3")

# Create datasets and data loaders
train_dataset = ImageDataset(args.train_data_path)
val_dataset = ImageDataset(args.val_data_path)

train_loader = create_loader(train_dataset, (3, 224, 224), batch_size=args.batch_size, is_training=True, use_prefetcher=False)
val_loader = create_loader(val_dataset, (3, 224, 224), batch_size=1, is_training=False, use_prefetcher=False)

# Setup device and move models to GPU(s)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.device_count() > 1:
	print("Using", torch.cuda.device_count(), "GPUs!")
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

model_1_optimizer = torch.optim.Adam(model_1_params, lr=args.lr_1, weight_decay=args.weight_decay)
model_2_optimizer = torch.optim.Adam(model_2_params, lr=args.lr_2, weight_decay=args.weight_decay)
model_3_optimizer = torch.optim.Adam(model_3_params, lr=args.lr_3, weight_decay=args.weight_decay)

# Load checkpoints if provided (including optimizer states and best accuracy)
best_1_acc = load_model_checkpoint(model_1, model_1_optimizer, args.load_1_path, metric_type='acc')
best_2_acc = load_model_checkpoint(model_2, model_2_optimizer, args.load_2_path, metric_type='acc')
best_3_acc = load_model_checkpoint(model_3, model_3_optimizer, args.load_3_path, metric_type='acc')

criterion = nn.CrossEntropyLoss()

def train(model_1, model_2, model_3, model_1_optimizer, model_2_optimizer, model_3_optimizer, train_loader, epoch, device, args):
	"""Train three models using tri-directional knowledge distillation."""
	model_1.train()
	model_2.train()
	model_3.train()
	
	correct_1, correct_2, correct_3, total = 0, 0, 0, 0
	curr_loss = AverageMeter()
	total_loss = AverageMeter()
	curr_ce_1_loss = AverageMeter()
	curr_ce_2_loss = AverageMeter()
	curr_ce_3_loss = AverageMeter()
	total_ce_1_loss = AverageMeter()
	total_ce_2_loss = AverageMeter()
	total_ce_3_loss = AverageMeter()
	tic = time.time()

	for idx, sample in enumerate(train_loader):
		imgs, labels = sample[0], sample[1]
		imgs = imgs.to(device)
		labels = labels.to(device)

		model_1_optimizer.zero_grad()
		model_2_optimizer.zero_grad()
		model_3_optimizer.zero_grad()

		# Get predictions from all three models
		pred_1 = get_model_output(model_1, imgs, args.model_1)
		pred_2 = get_model_output(model_2, imgs, args.model_2)
		pred_3 = get_model_output(model_3, imgs, args.model_3)

		# Calculate softmax predictions for ground truth labels
		gt_soft_1 = pred_1.softmax(dim=1)[torch.arange(args.batch_size), labels]
		gt_soft_2 = pred_2.softmax(dim=1)[torch.arange(args.batch_size), labels]
		gt_soft_3 = pred_3.softmax(dim=1)[torch.arange(args.batch_size), labels]
		
		# Stack softmax values and find which model is most confident
		gt_soft_stack = torch.stack((gt_soft_1, gt_soft_2, gt_soft_3), dim=1)
		max_gt_soft, max_indices = torch.max(gt_soft_stack, dim=1)
		
		# Only use samples where one model is clearly most confident (no ties)
		max_count = torch.sum(gt_soft_stack == max_gt_soft.unsqueeze(1), dim=1)
		valid_mask = (max_count == 1)

		# Create masks: each model teaches when it's most confident
		mask_1 = (valid_mask & (max_indices == 0)).float()
		mask_2 = (valid_mask & (max_indices == 1)).float()
		mask_3 = (valid_mask & (max_indices == 2)).float()
		
		# Compute softmax distributions
		soft_1 = F.softmax(pred_1, dim=1)
		soft_2 = F.softmax(pred_2, dim=1)
		soft_3 = F.softmax(pred_3, dim=1)

		# Cross-entropy losses with ground truth
		ce_1 = criterion(pred_1, labels)
		ce_2 = criterion(pred_2, labels)
		ce_3 = criterion(pred_3, labels)

		# KL divergence losses: when one model teaches, the other two learn from it
		kl_1 = F.kl_div(soft_2.log(), soft_1, reduction='none').sum(dim=1) + F.kl_div(soft_3.log(), soft_1, reduction='none').sum(dim=1)
		kl_2 = F.kl_div(soft_1.log(), soft_2, reduction='none').sum(dim=1) + F.kl_div(soft_3.log(), soft_2, reduction='none').sum(dim=1)
		kl_3 = F.kl_div(soft_1.log(), soft_3, reduction='none').sum(dim=1) + F.kl_div(soft_2.log(), soft_3, reduction='none').sum(dim=1)

		# Total loss: CE losses + masked tri-directional KL divergence
		loss = ce_1 + ce_2 + ce_3 + (mask_1 * kl_1 + mask_2 * kl_2 + mask_3 * kl_3).mean()

		loss.backward()
		model_1_optimizer.step()
		model_2_optimizer.step()
		model_3_optimizer.step()

		# Update metrics
		correct_1 += (soft_1.argmax(dim=1) == labels).sum().item()
		correct_2 += (soft_2.argmax(dim=1) == labels).sum().item()
		correct_3 += (soft_3.argmax(dim=1) == labels).sum().item()
		total += labels.size(0)

		curr_loss.update(loss.item())
		total_loss.update(loss.item())
		curr_ce_1_loss.update(ce_1.item())
		curr_ce_2_loss.update(ce_2.item())
		curr_ce_3_loss.update(ce_3.item())
		total_ce_1_loss.update(ce_1.item())
		total_ce_2_loss.update(ce_2.item())
		total_ce_3_loss.update(ce_3.item())

		# Log training progress at intervals
		if idx%args.log_interval==(args.log_interval-1):
			print('[{:2d}, {:5d}/{:5d}] avg_loss : {:.3f}, ce_1 : {:.3f}, ce_2 : {:.3f}, ce_3 : {:.3f}, 1_acc : {:.2f}, 2_acc : {:.2f}, 3_acc : {:.2f}, time: {:.3f} minutes'.format(epoch+1, idx+1, len(train_loader), curr_loss.avg, curr_ce_1_loss.avg, curr_ce_2_loss.avg, curr_ce_3_loss.avg, (100 * correct_1) / total, (100 * correct_2) / total, (100 * correct_3) / total, (time.time()-tic)/60))
			curr_loss.reset()
			curr_ce_1_loss.reset()
			curr_ce_2_loss.reset()
			curr_ce_3_loss.reset()
			sys.stdout.flush()
			
	time_taken = (time.time()-tic)/60
	
	# Log to wandb if enabled
	if args.log_wandb:
		wandb.log({'train_avg_loss': total_loss.avg, 'train_ce_1': total_ce_1_loss.avg, 'train_ce_2': total_ce_2_loss.avg, 'train_ce_3': total_ce_3_loss.avg, 'train_1_acc': ((100 * correct_1) / total), 'train_2_acc': ((100 * correct_2) / total), 'train_3_acc': ((100 * correct_3) / total)}, step=epoch+1)
	
	print('[{:2d}, train] avg_loss : {:.3f}, ce_1 : {:.3f}, ce_2 : {:.3f}, ce_3 : {:.3f}, 1_acc : {:.2f}, 2_acc : {:.2f}, 3_acc : {:.2f}, time : {:.3f} minutes'.format(epoch+1, total_loss.avg, total_ce_1_loss.avg, total_ce_2_loss.avg, total_ce_3_loss.avg, ((100 * correct_1) / total), ((100 * correct_2) / total), ((100 * correct_3) / total), time_taken))
	sys.stdout.flush()

	return total_loss.avg

def validate(model_1, model_2, model_3, val_loader, epoch, device):
	"""Validate all three models on the validation set."""
	model_1.eval()
	model_2.eval()
	model_3.eval()

	correct_1, correct_2, correct_3, total = 0, 0, 0, 0
	tic = time.time()

	for sample in val_loader:
		img, label = sample[0], sample[1]
		img = img.to(device)
		label = label.to(device)

		# Get predictions from all three models
		output_1 = get_model_output(model_1, img, args.model_1)
		output_2 = get_model_output(model_2, img, args.model_2)
		output_3 = get_model_output(model_3, img, args.model_3)

		correct_1 += topk_accuracy(output_1, label, k=1)
		correct_2 += topk_accuracy(output_2, label, k=1)
		correct_3 += topk_accuracy(output_3, label, k=1)

		total += label.size(0)

	# Calculate accuracies
	val_1_acc = (100 * correct_1) / total
	val_2_acc = (100 * correct_2) / total
	val_3_acc = (100 * correct_3) / total
	
	# Log to wandb if enabled
	if args.log_wandb:
		wandb.log({'val_1_acc': val_1_acc, 'val_2_acc': val_2_acc, 'val_3_acc': val_3_acc}, step=epoch+1)
	
	print("Accuracy on the ImageNet validation set: {:.2f}% (model 1), {:.2f}% (model 2), {:.2f}% (model 3), time : {:.3f} minutes".format(val_1_acc, val_2_acc, val_3_acc, (time.time() - tic) / 60))
	return val_1_acc, val_2_acc, val_3_acc

# Training loop
for epoch in range(args.no_epochs):
	train_loss = train(model_1, model_2, model_3, model_1_optimizer, model_2_optimizer, model_3_optimizer, train_loader, epoch, device, args)

	with torch.no_grad():
		val_1_acc, val_2_acc, val_3_acc = validate(model_1, model_2, model_3, val_loader, epoch, device)
		
		# Save model 1 checkpoint if improved
		if val_1_acc >= best_1_acc:
			best_1_acc = val_1_acc
			save_checkpoint(model_1, model_1_optimizer, best_1_acc, args.model_1_val_path, "model 1", epoch, metric_type='acc')

		# Save model 2 checkpoint if improved
		if val_2_acc >= best_2_acc:
			best_2_acc = val_2_acc
			save_checkpoint(model_2, model_2_optimizer, best_2_acc, args.model_2_val_path, "model 2", epoch, metric_type='acc')

		# Save model 3 checkpoint if improved
		if val_3_acc >= best_3_acc:
			best_3_acc = val_3_acc
			save_checkpoint(model_3, model_3_optimizer, best_3_acc, args.model_3_val_path, "model 3", epoch, metric_type='acc')
	print()

if args.log_wandb:
	wandb.finish()
