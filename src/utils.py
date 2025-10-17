import os
import sys
import time
import math
import random
import datetime
import subprocess
from collections import defaultdict, deque

from matplotlib.pylab import seed
import numpy as np
import torch
from torch import nn
import torch.distributed as dist
from PIL import ImageFilter, ImageOps
from einops import rearrange, reduce, repeat

from torch.nn import functional as F
# from attmask import AttMask
import random
# from sinkhorn_knopp import SinkhornKnopp
# for saliency prediction
# from loss import *


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


def load_model_checkpoint(model, optimizer, checkpoint_path):
	"""Load model and optimizer state from checkpoint."""
	if checkpoint_path != '':
		checkpoint = torch.load(checkpoint_path)
		model.load_state_dict(checkpoint['model_state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		return checkpoint.get('acc', -np.inf)
	return -np.inf


def print_model_info(model, model_name):
	"""Print trainable parameters and model size."""
	print(f"\n{model_name} trainable parameters:")
	for name, param in model.named_parameters():
		if param.requires_grad:
			print(name, param.size())
	model_params_and_size(model)


def save_checkpoint(model, optimizer, acc, save_path, model_type, epoch):
	"""Save model checkpoint with optimizer state and accuracy."""
	checkpoint = {
		'model_state_dict': model.state_dict(),
		'optimizer_state_dict': optimizer.state_dict(),
		'acc': acc
	}
	print("[{}, save {}, {}]".format(epoch+1, model_type, save_path))
	torch.save(checkpoint, save_path)
