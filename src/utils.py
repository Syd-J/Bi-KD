import random

import numpy as np
import torch
from torch import nn

from loss import *


def set_seed(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	print("Using seed: ", seed)


def topk_accuracy(output, target, k=5):
	topk_pred = torch.topk(output, k=k, dim=1).indices

	target = target.view(-1, 1).expand_as(topk_pred)

	correct = topk_pred.eq(target).sum().item()

	return correct


def model_params_and_size(model):
	model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	print("The number of parameters in the model: ", model_params)

	param_size = 0
	buffer_size = 0
	for param in model.parameters():
		param_size += param.nelement() * param.element_size()

	for buffer in model.buffers():
		buffer_size += buffer.nelement() * buffer.element_size()
	
	size_all_mb = (param_size + buffer_size) / 1024**2
	print('Size: {:.3f} MB'.format(size_all_mb))


class AverageMeter(object):
	def __init__(self):
		self.reset()

	def reset(self):

		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n = 1):
		self.val = val
		self.sum += val*n
		self.count += n
		self.avg = self.sum / self.count


def get_model_output(model, imgs, model_name):
	"""Extract logits from model output, handling special cases like MambaVision."""
	output = model(imgs)
	return output['logits'] if model_name == "MambaVision-T2" else output


def load_model_checkpoint(model, optimizer, checkpoint_path, metric_type='acc'):
	"""Load model and optimizer state from checkpoint.
	
	Args:
		model: Model to load state into
		optimizer: Optimizer to load state into
		checkpoint_path: Path to checkpoint file
		metric_type: Type of metric to return ('acc' for accuracy, 'loss' for loss)
	
	Returns:
		Metric value from checkpoint (accuracy or loss), or default value if not found
	"""
	if checkpoint_path != '':
		checkpoint = torch.load(checkpoint_path)
		model.load_state_dict(checkpoint['model_state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		
		# Return the appropriate metric
		if metric_type == 'loss':
			return checkpoint.get('loss', np.inf)
		else:
			return checkpoint.get('acc', -np.inf)
	
	# Return default values
	return np.inf if metric_type == 'loss' else -np.inf


def print_model_info(model, model_name):
	"""Print trainable parameters and model size."""
	print(f"\n{model_name} trainable parameters:")
	for name, param in model.named_parameters():
		if param.requires_grad:
			print(name, param.size())
	model_params_and_size(model)


def save_checkpoint(model, optimizer, metric, save_path, model_type, epoch, metric_type='acc'):
	"""Save model checkpoint with optimizer state and metric (accuracy or loss).
	
	Args:
		model: Model to save
		optimizer: Optimizer to save
		metric: Metric value (accuracy or loss)
		save_path: Path to save checkpoint
		model_type: Description of model (e.g., 'teacher', 'student')
		epoch: Current epoch number
		metric_type: Type of metric ('acc' for accuracy, 'loss' for loss)
	"""
	checkpoint = {
		'model_state_dict': model.state_dict(),
		'optimizer_state_dict': optimizer.state_dict(),
	}
	
	# Store metric with appropriate key based on type
	if metric_type == 'loss':
		checkpoint['loss'] = metric
		checkpoint['acc'] = -metric  # Store negative loss as pseudo-accuracy for backward compatibility
	else:
		checkpoint['acc'] = metric
		checkpoint['loss'] = np.inf  # Store inf as pseudo-loss for backward compatibility
	
	print("[{}, save {}, {}]".format(epoch+1, model_type, save_path))
	torch.save(checkpoint, save_path)


def get_loss(pred_map, gt, args, fix_map=None):
	"""Calculate loss for saliency prediction."""
	loss = torch.FloatTensor(pred_map.size(0)*[0.0]).cuda()
	
	if args.kldiv:
		loss += args.kldiv_coeff * kldiv(pred_map, gt)
	if args.cc:
		loss += args.cc_coeff * cc(pred_map, gt)
	if args.nss:
		nss_loss = torch.FloatTensor([0.0]).cuda()
		for i in range(pred_map.shape[0]):
			pred_sal_resize = cv2.resize(pred_map[i].cpu().detach().numpy(), (fix_map[i].size(1), fix_map[i].size(0)))
			nss_loss += nss(torch.from_numpy(pred_sal_resize).unsqueeze(0), fix_map[i].unsqueeze(0))
		nss_loss /= pred_map.shape[0]
		loss += args.nss_coeff * nss_loss
	if args.l1:
		criterion = nn.L1Loss(reduction='none')
		loss += args.l1_coeff * criterion(pred_map, gt).mean(dim=(1,2))
	if args.sim:
		loss += args.sim_coeff * similarity(pred_map, gt)

	return loss

def loss_func(pred_map, gt, args, fix_map=None):
	"""Calculate loss function for saliency prediction with clips support."""
	loss = torch.FloatTensor(pred_map.size(0)*[0.0]).cuda()
	criterion = nn.L1Loss()
	assert pred_map.size() == gt.size()

	if len(pred_map.size()) == 4:
		# Clips: BxClXHxW
		assert pred_map.size(0) == args.batch_size
		pred_map = pred_map.permute((1,0,2,3))
		gt = gt.permute((1,0,2,3))
		if fix_map:
			fix_map = fix_map.permute((1,0,2,3))

		for i in range(pred_map.size(0)):
			if fix_map:
				loss += get_loss(pred_map[i], gt[i], args, fix_map[i])
			else:
				loss += get_loss(pred_map[i], gt[i], args)

		loss /= pred_map.size(0)
		return loss
	
	return get_loss(pred_map, gt, args, fix_map)

def blur(img):
	"""Apply Gaussian blur to image."""
	k_size = 11
	bl = cv2.GaussianBlur(img,(k_size,k_size),0)
	return torch.FloatTensor(bl)
