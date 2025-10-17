import torch
import os
import h5py
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import copy
import cv2
import numpy as np
import json


class DHF1KDataset(Dataset):
	def __init__(self, path_data, len_snippet, mode="train", reverse=False): # reverse denotes wether TMFI-Net is the teacher (False) or student (True)
		self.path_data = path_data
		self.len_snippet = len_snippet
		self.mode = mode
		self.reverse = reverse

		if not self.reverse:
			self.teacher_img_transform = transforms.Compose([
				transforms.Resize((224, 384)),
				transforms.ToTensor(),
				transforms.Normalize(
					[0.485, 0.456, 0.406],
					[0.229, 0.224, 0.225]
				)
			])

			self.student_img_transform = transforms.Compose([
				transforms.Resize((256, 456)),
				transforms.ToTensor(),
				transforms.Normalize(
					# [0.485, 0.456, 0.406],
					# [0.229, 0.224, 0.225]
					[0.450, 0.450, 0.450],
					[0.225, 0.225, 0.225]
				)
			])

		else:
			self.student_img_transform = transforms.Compose([
				transforms.Resize((224, 384)),
				transforms.ToTensor(),
				transforms.Normalize(
					[0.485, 0.456, 0.406],
					[0.229, 0.224, 0.225]
				)
			])

			self.teacher_img_transform = transforms.Compose([
				transforms.Resize((256, 456)),
				transforms.ToTensor(),
				transforms.Normalize(
					# [0.485, 0.456, 0.406],
					# [0.229, 0.224, 0.225]
					[0.450, 0.450, 0.450],
					[0.225, 0.225, 0.225]
				)
			])

		if self.mode == "train":
			self.video_names = os.listdir(path_data)
			self.list_num_frame = [len(os.listdir(os.path.join(path_data, d, 'images'))) for d in self.video_names]
		elif self.mode == "val":
			self.list_num_frame = []
			for v in os.listdir(path_data):
				for i in range(0, len(os.listdir(os.path.join(path_data, v, 'images'))) - self.len_snippet, 32):
					self.list_num_frame.append((v, i))

	def __len__(self):
		return len(self.list_num_frame)
	
	def __getitem__(self, idx):
		if self.mode == "train":
			file_name = self.video_names[idx]
			gt_idx = np.random.randint(0, self.list_num_frame[idx])
		elif self.mode == "val":
			file_name, gt_idx = self.list_num_frame[idx]

		path_clip = os.path.join(self.path_data, file_name, 'images')
		path_annt = os.path.join(self.path_data, file_name, 'maps')

		# print("File name: ", file_name, len(path_clip))
		# print("GT index: ", gt_idx)

		teacher_clip_img = []

		student_clip_img = []

		# TMFINet_frames = []
		# ViNetA_frames = []
	
		for i in range(self.len_snippet):

			if not self.reverse:

				if i < self.len_snippet // 2:
					if gt_idx - (self.len_snippet // 2) + i + 2 <= 0:
						teacher_img = Image.open(os.path.join(path_clip, '0001.png')).convert('RGB')
						# TMFINet_frames.append(0)
					else:
						teacher_img = Image.open(os.path.join(path_clip, '%04d.png' % (gt_idx - (self.len_snippet // 2) + i + 2))).convert('RGB')
						# TMFINet_frames.append(gt_idx - (self.len_snippet // 2) + i + 2)

					teacher_clip_img.append(self.teacher_img_transform(teacher_img))

				if i % 2 == 0:
					if (gt_idx - (self.len_snippet // 2) + i + 1) <= 0:
						student_img = Image.open(os.path.join(path_clip, '0001.png')).convert('RGB')
						# ViNetA_frames.append(0)
					elif (gt_idx - (self.len_snippet // 2) + i + 1) >= len(os.listdir(path_clip)):
						student_img = Image.open(os.path.join(path_clip, '%04d.png' % (len(os.listdir(path_clip))))).convert('RGB')
						# ViNetA_frames.append(len(os.listdir(path_clip)))
					else:
						student_img = Image.open(os.path.join(path_clip, '%04d.png' % (gt_idx - (self.len_snippet // 2) + i + 1))).convert('RGB')
						# ViNetA_frames.append(gt_idx - (self.len_snippet // 2) + i + 1)

					student_clip_img.append(self.student_img_transform(student_img))

			else:

				if i < self.len_snippet // 2:
					if gt_idx - (self.len_snippet // 2) + i + 2 <= 0:
						student_img = Image.open(os.path.join(path_clip, '0001.png')).convert('RGB')
						# TMFINet_frames.append(0)
					else:
						student_img = Image.open(os.path.join(path_clip, '%04d.png' % (gt_idx - (self.len_snippet // 2) + i + 2))).convert('RGB')
						# TMFINet_frames.append(gt_idx - (self.len_snippet // 2) + i + 2)

					student_clip_img.append(self.student_img_transform(student_img))

				if i % 2 == 0:
					if (gt_idx - (self.len_snippet // 2) + i + 1) <= 0:
						teacher_img = Image.open(os.path.join(path_clip, '0001.png')).convert('RGB')
						# ViNetA_frames.append(0)
					elif (gt_idx - (self.len_snippet // 2) + i + 1) >= len(os.listdir(path_clip)):
						teacher_img = Image.open(os.path.join(path_clip, '%04d.png' % (len(os.listdir(path_clip))))).convert('RGB')
						# ViNetA_frames.append(len(os.listdir(path_clip)))
					else:
						teacher_img = Image.open(os.path.join(path_clip, '%04d.png' % (gt_idx - (self.len_snippet // 2) + i + 1))).convert('RGB')
						# ViNetA_frames.append(gt_idx - (self.len_snippet // 2) + i + 1)

					teacher_clip_img.append(self.teacher_img_transform(teacher_img))

		teacher_clip_img = torch.FloatTensor(torch.stack(teacher_clip_img, dim=0))
		student_clip_img = torch.FloatTensor(torch.stack(student_clip_img, dim=0))

		# print("TMFI-Net frames: ", TMFINet_frames, len(TMFINet_frames))
		# print("ViNet-A frames: ", ViNetA_frames, len(ViNetA_frames))

		gt = np.array(Image.open(os.path.join(path_annt, '%04d.png' % (gt_idx + 1))).convert('L'))
		gt = gt.astype('float')
		if self.mode == "train":
			if not self.reverse:
				gt = cv2.resize(gt, (456, 256))
			else:
				gt = cv2.resize(gt, (384, 224))
		if np.max(gt) > 1.0:
			gt = gt / 255.0

		return teacher_clip_img, student_clip_img, gt


class HollywoodDataset(Dataset):
	def __init__(self, path_data, len_snippet, mode="train"):
		self.path_data = path_data
		self.len_snippet = len_snippet
		self.mode = mode

		self.teacher_img_transform = transforms.Compose([
			transforms.Resize((224, 384)),
			transforms.ToTensor(),
			transforms.Normalize(
				[0.485, 0.456, 0.406],
				[0.229, 0.224, 0.225]
			)
		])

		self.student_img_transform = transforms.Compose([
			transforms.Resize((256, 456)),
			transforms.ToTensor(),
			transforms.Normalize(
				# [0.485, 0.456, 0.406],
				# [0.229, 0.224, 0.225]
				[0.450, 0.450, 0.450],
				[0.225, 0.225, 0.225]
			)
		])

		if self.mode == "train":
			self.video_clips = os.listdir(self.path_data)
			self.list_num_frame = [len(os.listdir(os.path.join(self.path_data, d, 'images'))) for d in self.video_clips]
			self.video_names = np.unique([video_name.split('_')[0] for video_name in self.video_clips])

		elif self.mode == "val":
			self.list_num_frame = []
			self.video_clips = os.listdir(self.path_data)
			self.video_names = np.unique([file_name.split('_')[0] for file_name in self.video_clips])

			for video_name in self.video_names:
				num_frames = sum([len(os.listdir(os.path.join(self.path_data, file_name, 'images'))) for file_name in self.video_clips if file_name.split('_')[0] == video_name])
				for gt_idx in range(0, num_frames - self.len_snippet, 2*self.len_snippet):
					self.list_num_frame.append((video_name, num_frames, gt_idx))

	def __len__(self):
		if self.mode == "train":
			return len(self.video_names)
		return len(self.list_num_frame)
	
	def __getitem__(self, idx):
		if self.mode == "train":
			video_name = self.video_names[idx]
			video_clips = [video_clip for video_clip in self.video_clips if video_clip.split('_')[0] == video_name]
			num_frames = sum([len(os.listdir(os.path.join(self.path_data, file_name, 'images'))) for file_name in video_clips])
			gt_idx = np.random.randint(0, num_frames - self.len_snippet)

		elif self.mode == "val":
			(video_name, num_frames, gt_idx) = self.list_num_frame[idx]
			video_clips = [video_clip for video_clip in self.video_clips if video_clip.split('_')[0] == video_name]

		list_clips, list_sal_clips = [], []
		for file_name in sorted(video_clips):

			path_clip = os.path.join(self.path_data, file_name, 'images')
			path_annt = os.path.join(self.path_data, file_name, 'maps')

			temp1 = sorted([os.path.join(path_clip, image_file) for image_file in os.listdir(path_clip)])
			temp2 = sorted([os.path.join(path_annt, image_file) for image_file in os.listdir(path_annt)])

			list_clips.extend(temp1)
			list_sal_clips.extend(temp2)

		# print("File name: ", video_name, len(list_clips))
		# print("GT index: ", gt_idx)

		teacher_clip_img = []
		student_clip_img = []

		# TMFINet_frames = []
		# ViNetA_frames = []		

		for i in range(self.len_snippet):

			if gt_idx - self.len_snippet + i + 1 <= 0:
				teacher_img = Image.open(os.path.join(path_clip, list_clips[0])).convert('RGB')
				# TMFINet_frames.append(0)
			else:
				teacher_img = Image.open(os.path.join(path_clip, list_clips[gt_idx - self.len_snippet + i + 1])).convert('RGB')
				# TMFINet_frames.append(gt_idx - self.len_snippet + i + 1)
			teacher_clip_img.append(self.teacher_img_transform(teacher_img))

			if gt_idx - (self.len_snippet // 2) + i <= 0:
				student_img = Image.open(os.path.join(path_clip, list_clips[0])).convert('RGB')
				# ViNetA_frames.append(0)
			elif gt_idx - (self.len_snippet // 2) + i >= len(list_clips):
				student_img = Image.open(os.path.join(path_clip, list_clips[-1])).convert('RGB')
				# ViNetA_frames.append(len(list_clips) - 1)
			else:
				student_img = Image.open(os.path.join(path_clip, list_clips[gt_idx - (self.len_snippet // 2) + i])).convert('RGB')
				# ViNetA_frames.append(gt_idx - (self.len_snippet // 2) + i)
			student_clip_img.append(self.student_img_transform(student_img))

		teacher_clip_img = torch.FloatTensor(torch.stack(teacher_clip_img, dim=0))
		student_clip_img = torch.FloatTensor(torch.stack(student_clip_img, dim=0))

		# print("TMFI-Net frames: ", TMFINet_frames, len(TMFINet_frames))
		# print("ViNet-A frames: ", ViNetA_frames, len(ViNetA_frames))

		gt = np.array(Image.open(os.path.join(path_annt, list_sal_clips[gt_idx])).convert('L'))
		gt = gt.astype('float')
		if self.mode == "train":
			gt = cv2.resize(gt, (456, 256))
		if np.max(gt) > 1.0:
			gt = gt / 255.0

		return teacher_clip_img, student_clip_img, gt
	