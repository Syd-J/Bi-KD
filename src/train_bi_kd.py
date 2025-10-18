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


parser = argparse.ArgumentParser(description='Bi-KD Training')

# Data paths
parser.add_argument('--train_data_path',default='/mnt/SSD/ImageNet/train',type=str,help='path to training data')
parser.add_argument('--val_data_path',default='/mnt/SSD/ImageNet/val',type=str,help='path to validation data')

# Training hyperparameters
parser.add_argument('--batch_size',default=128,type=int,help='batch size')
parser.add_argument('--seed',default=0,type=int,help='fix seed')
parser.add_argument('--no_epochs',default=20,type=int,help='number of epochs')
parser.add_argument('--weight_decay',default=1e-5,type=float,help='weight decay')

# Model configuration
parser.add_argument('--teacher_name',type=str,required=True)
parser.add_argument('--student_name',type=str,required=True)
parser.add_argument('--t_lr',default=1e-6,type=float,help='learning rate for teacher')
parser.add_argument('--s_lr',default=1e-6,type=float,help='learning rate for student')

# Checkpointing
parser.add_argument('--load_t_path',default='',type=str,help='path to load the teacher model')
parser.add_argument('--load_s_path',default='',type=str,help='path to load the student model')
parser.add_argument('--t_model_val_path',default='best_bi_kd_t.pt',type=str,help='path to save the best teacher model')
parser.add_argument('--s_model_val_path',default='best_bi_kd_s.pt',type=str,help='path to save the best student model')

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
			'teacher': args.teacher_name,
			'student': args.student_name,
			'batch_size': args.batch_size,
			't_lr': args.t_lr,
			's_lr': args.s_lr
		}
	)

print(args)
set_seed(args.seed)

# Create models
teacher_model = create_model(args.teacher_name, pretrained=True)
student_model = create_model(args.student_name, pretrained=True)

# Print model information
print_model_info(teacher_model, "Teacher")
print_model_info(student_model, "Student")

# Create datasets and data loaders
train_dataset = ImageDataset(args.train_data_path)
val_dataset = ImageDataset(args.val_data_path)

train_loader = create_loader(train_dataset, (3, 224, 224), batch_size=args.batch_size, is_training=True)
val_loader = create_loader(val_dataset, (3, 224, 224), batch_size=1, is_training=False)

# Setup device and move models to GPU(s)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.device_count() > 1:
	print("Using", torch.cuda.device_count(), "GPUs!")
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

# Load checkpoints if provided (including optimizer states and best accuracy)
best_t_acc = load_model_checkpoint(teacher_model, teacher_optimizer, args.load_t_path, metric_type='acc')
best_s_acc = load_model_checkpoint(student_model, student_optimizer, args.load_s_path, metric_type='acc')

criterion = nn.CrossEntropyLoss()

def train(teacher_model, student_model, teacher_optimizer, student_optimizer, train_loader, epoch, device, args):
	"""Train both teacher and student models using bidirectional knowledge distillation."""
	teacher_model.train()
	student_model.train()
	
	t_correct, s_correct, total = 0, 0, 0
	curr_loss = AverageMeter()
	total_loss = AverageMeter()
	curr_ce_t_loss = AverageMeter()
	curr_ce_s_loss = AverageMeter()
	total_ce_t_loss = AverageMeter()
	total_ce_s_loss = AverageMeter()
	tic = time.time()

	for idx, sample in enumerate(train_loader):
		imgs, labels = sample[0], sample[1]
		imgs = imgs.to(device)
		labels = labels.to(device)

		teacher_optimizer.zero_grad()
		student_optimizer.zero_grad()
		
		# Get predictions from both models
		pred_t = get_model_output(teacher_model, imgs, args.teacher_name)
		pred_s = get_model_output(student_model, imgs, args.student_name)

		# Calculate softmax predictions for ground truth labels
		gt_soft_t = pred_t.softmax(dim=1)[torch.arange(args.batch_size), labels]
		gt_soft_s = pred_s.softmax(dim=1)[torch.arange(args.batch_size), labels]
		
		# Create masks: teacher teaches when more confident, student teaches otherwise
		mask_t = (gt_soft_t > gt_soft_s).float()
		mask_s = (gt_soft_t <= gt_soft_s).float()
		
		# Compute softmax distributions
		soft_t = F.softmax(pred_t, dim=1)
		soft_s = F.softmax(pred_s, dim=1)
		
		# Cross-entropy losses with ground truth
		ce_t = criterion(pred_t, labels)
		ce_s = criterion(pred_s, labels)

		# KL divergence losses for knowledge distillation
		kl_t = F.kl_div(soft_s.log(), soft_t, reduction='none').sum(dim=1)
		kl_s = F.kl_div(soft_t.log(), soft_s, reduction='none').sum(dim=1)

		# Total loss: CE losses + masked bidirectional KL divergence
		loss = ce_t + ce_s + (mask_t * kl_t + mask_s * kl_s).mean()
		
		loss.backward()
		teacher_optimizer.step()
		student_optimizer.step()

		# Update metrics
		t_correct += (soft_t.argmax(dim=1) == labels).sum().item()
		s_correct += (soft_s.argmax(dim=1) == labels).sum().item()
		total += labels.size(0)

		curr_loss.update(loss.item())
		total_loss.update(loss.item())
		curr_ce_t_loss.update(ce_t.item())
		curr_ce_s_loss.update(ce_s.item())
		total_ce_t_loss.update(ce_t.item())
		total_ce_s_loss.update(ce_s.item())

		# Log training progress at intervals
		if idx%args.log_interval==(args.log_interval-1):
			print('[{:2d}, {:5d}/{:5d}] avg_loss : {:.3f}, ce_t : {:.3f}, ce_s : {:.3f}, t_acc : {:.2f}, s_acc : {:.2f}, time: {:.3f} minutes'.format(epoch+1, idx+1, len(train_loader), curr_loss.avg, curr_ce_t_loss.avg, curr_ce_s_loss.avg, (100 * t_correct) / total, (100 * s_correct) / total, (time.time()-tic)/60))
			curr_loss.reset()
			curr_ce_t_loss.reset()
			curr_ce_s_loss.reset()
			sys.stdout.flush()
			
	time_taken = (time.time()-tic)/60
	
	# Log to wandb if enabled
	if args.log_wandb:
		wandb.log({'train_avg_loss': total_loss.avg, 'train_ce_t': total_ce_t_loss.avg, 'train_ce_s': total_ce_s_loss.avg, 'train_t_acc': ((100 * t_correct) / total), 'train_s_acc': ((100 * s_correct) / total)}, step=epoch+1)
	
	print('[{:2d}, train] avg_loss : {:.3f}, ce_t : {:.3f}, ce_s : {:.3f}, t_acc : {:.2f}, s_acc : {:.2f}, time : {:.3f} minutes'.format(epoch+1, total_loss.avg, total_ce_t_loss.avg, total_ce_s_loss.avg, ((100 * t_correct) / total), ((100 * s_correct) / total), time_taken))
	sys.stdout.flush()

	return total_loss.avg

def validate(teacher_model, student_model, val_loader, epoch, device):
	"""Validate both models on the validation set."""
	teacher_model.eval()
	student_model.eval()
	
	t_correct, s_correct, total = 0, 0, 0
	tic = time.time()
	
	for sample in val_loader:
		img, label = sample[0], sample[1]
		img = img.to(device)
		label = label.to(device)

		# Get predictions from both models
		t_output = get_model_output(teacher_model, img, args.teacher_name)
		s_output = get_model_output(student_model, img, args.student_name)

		pred_t = t_output.argmax(dim=1)
		pred_s = s_output.argmax(dim=1)
		t_correct += (pred_t == label).sum().item()
		s_correct += (pred_s == label).sum().item()
		total += label.size(0)

	# Calculate accuracies
	t_val_acc = (100 * t_correct) / total
	s_val_acc = (100 * s_correct) / total
	
	# Log to wandb if enabled
	if args.log_wandb:
		wandb.log({'t_val_acc': t_val_acc, 's_val_acc': s_val_acc}, step=epoch+1)
	
	print("Accuracy on the ImageNet validation set: {:.2f}% (teacher), {:.2f}% (student), time : {:.3f} minutes".format(t_val_acc, s_val_acc, (time.time() - tic) / 60))
	return t_val_acc, s_val_acc

# Training loop
for epoch in range(args.no_epochs):
	train_loss = train(teacher_model, student_model, teacher_optimizer, student_optimizer, train_loader, epoch, device, args)

	with torch.no_grad():
		t_val_acc, s_val_acc = validate(teacher_model, student_model, val_loader, epoch, device)
		
		# Save teacher checkpoint if improved
		if t_val_acc >= best_t_acc:
			best_t_acc = t_val_acc
			save_checkpoint(teacher_model, teacher_optimizer, best_t_acc, args.t_model_val_path, "teacher", epoch, metric_type='acc')

		# Save student checkpoint if improved
		if s_val_acc >= best_s_acc:
			best_s_acc = s_val_acc
			save_checkpoint(student_model, student_optimizer, best_s_acc, args.s_model_val_path, "student", epoch, metric_type='acc')
	print()

if args.log_wandb:
	wandb.finish()
