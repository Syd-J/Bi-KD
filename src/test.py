import argparse
import time

import torch

from timm.data.dataset import ImageDataset
from timm.data.loader import create_loader
import wandb

from models import create_model
from utils import *


parser = argparse.ArgumentParser(description='Model Testing on ImageNet')

# Data paths
parser.add_argument('--val_data_path', default='/mnt/Shared-Storage/sid/datasets/ImageNet/val', type=str, help='path to validation data')

# Model configuration
parser.add_argument('--model_name', type=str, required=True, help='model name to test')
parser.add_argument('--load_model_path', default='', type=str, help='path to load model checkpoint')

# Testing configuration
parser.add_argument('--batch_size', default=1, type=int, help='batch size for testing')
parser.add_argument('--seed', default=0, type=int, help='random seed')

# Logging
parser.add_argument('--log_wandb', default=False, action='store_true', help='log testing details to wandb')

args = parser.parse_args()

# Initialize wandb logging if enabled
if args.log_wandb:
	wandb.login()
	run = wandb.init(
		project='Bi-KD',
		group='test',
		config={
			'model_name': args.model_name,
			'load_model_path': args.load_model_path,
		}
	)

print(args)
set_seed(args.seed)

# Create model
model = create_model(args.model_name, pretrained=True)

# Load checkpoint if provided
if args.load_model_path != '':
	print(f"Loading checkpoint: {args.load_model_path}")
	checkpoint = torch.load(args.load_model_path)
	model.load_state_dict(checkpoint['model_state_dict'])

# Print model information
print_model_info(model, args.model_name)

# Create validation dataset and data loader
val_dataset = ImageDataset(args.val_data_path)
val_loader = create_loader(val_dataset, (3, 224, 224), batch_size=args.batch_size, is_training=False, use_prefetcher=False)

# Setup device and move model to GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print(f"Using device: {device}")


def validate(model, loader, device, args):
	"""Evaluate model on validation set and report top-1, top-3, and top-5 accuracies."""
	model.eval()
	
	correct_top1, correct_top3, correct_top5, total = 0, 0, 0, 0
	tic = time.time()
	
	for idx, sample in enumerate(loader):
		img, label = sample[0], sample[1]
		img = img.to(device)
		label = label.to(device)

		# Get model predictions
		logits = get_model_output(model, img, args.model_name)

		# Calculate top-k accuracies
		correct_top1 += topk_accuracy(logits, label, k=1)
		correct_top3 += topk_accuracy(logits, label, k=3)
		correct_top5 += topk_accuracy(logits, label, k=5)

		total += label.size(0)
		
		# Log progress periodically
		if (idx + 1) % 100 == 0 or (idx + 1) == len(loader):
			top1_acc = (100 * correct_top1) / total
			top3_acc = (100 * correct_top3) / total
			top5_acc = (100 * correct_top5) / total
			elapsed = (time.time() - tic) / 60
			
			print(f"[{idx+1}/{len(loader)}] Top-1: {top1_acc:.2f}%, Top-3: {top3_acc:.2f}%, Top-5: {top5_acc:.2f}%, Time: {elapsed:.3f} min")
			
			# Log to wandb if enabled
			if args.log_wandb:
				wandb.log({
					'sample_idx': idx + 1,
					'pred_per_sample': topk_accuracy(logits, label, k=1)
				})
	
	# Calculate final accuracies
	final_top1 = (100 * correct_top1) / total
	final_top3 = (100 * correct_top3) / total
	final_top5 = (100 * correct_top5) / total
	
	print("\n" + "="*80)
	print(f"Final Results on ImageNet Validation Set:")
	print(f"  Top-1 Accuracy: {final_top1:.2f}%")
	print(f"  Top-3 Accuracy: {final_top3:.2f}%")
	print(f"  Top-5 Accuracy: {final_top5:.2f}%")
	print("="*80 + "\n")
	
	# Log final results to wandb if enabled
	if args.log_wandb:
		wandb.log({
			'final_top1_acc': final_top1,
			'final_top3_acc': final_top3,
			'final_top5_acc': final_top5
		})
	
	return final_top1, final_top3, final_top5


# Run validation
with torch.no_grad():
	validate(model, val_loader, device, args)

if args.log_wandb:
	wandb.finish()

