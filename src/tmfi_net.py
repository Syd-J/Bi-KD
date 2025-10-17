import torch
from torch import nn
import math
from swin_transformer import *
from collections import OrderedDict
from einops import rearrange


class Gate(nn.Module):
	def __init__(self, in_plane):
		super(Gate, self).__init__()
		self.gate = nn.Conv3d(in_plane, in_plane, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))

	def forward(self, rgb_fea):
		gate = torch.sigmoid(self.gate(rgb_fea))
		gate_fea = rgb_fea * gate + rgb_fea

		return gate_fea


class ShuffleBlock(nn.Module):
	def __init__(self, groups):
		super(ShuffleBlock, self).__init__()
		self.groups = groups
	def forward(self, x):
		'''Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]'''
		N,C,T,H,W = x.size()
		g = self.groups
		return x.view(N,g,C//g,T,H,W).permute(0,2,1,3,4,5).reshape(N,C,T,H,W)


class VideoSaliencyModel(nn.Module):
	def __init__(self, decoder, use_neck=False, decoder_groups=0, pretrain=None):
		super(VideoSaliencyModel, self).__init__()

		self.backbone = SwinTransformer3D(pretrained=pretrain)
		self.decoder_type = decoder
		self.use_neck = use_neck

		if decoder == "default":
			self.decoder = DecoderConvUp()
		elif decoder == "efficient":
			self.decoder = Decoder(decoder_groups=decoder_groups)
		
		if self.use_neck:
			self.neck = Neck()

	def forward(self, x):
		x, [y1, y2, y3, y4] = self.backbone(x)

		if self.use_neck:
			x, [y1, y2, y3, y4] = self.neck(x, [y1, y2, y3, y4])

		if self.decoder_type == "default":
			return self.decoder(x, y3, y2, y1)
		else:
			return self.decoder(x, [y1, y2, y3, y4])


class Neck(nn.Module):
	def __init__(self):
		super(Neck, self).__init__()
		
		self.conv_skip1 = nn.Sequential(
				nn.Conv3d(96, 48, kernel_size=(1, 3, 3), stride=(1, 1, 1),
			  		padding=(0, 1, 1), bias=False),
				nn.ReLU()
		)

		self.conv_skip2 = nn.Sequential(
				nn.Conv3d(192, 96, kernel_size=(1, 3, 3), stride=(1, 1, 1),
			  		padding=(0, 1, 1), bias=False),
				nn.ReLU()
		)

		self.conv_skip3 = nn.Sequential(
				nn.Conv3d(384, 192, kernel_size=(1, 3, 3), stride=(1, 1, 1),
			  		padding=(0, 1, 1), bias=False),
				nn.ReLU()
		)

		self.conv_skip4 = nn.Sequential(
				nn.Conv3d(768, 384, kernel_size=(1, 3, 3), stride=(1, 1, 1),
			  		padding=(0, 1, 1), bias=False),
				nn.ReLU()
		)


	def forward(self, latent_features, skip_connections):
		
			skip1, skip2, skip3, skip4 = skip_connections

			skip1 = self.conv_skip1(skip1)
			skip2 = self.conv_skip2(skip2)
			skip3 = self.conv_skip3(skip3)
			skip4 = self.conv_skip4(skip4)


			return latent_features, (skip1, skip2, skip3, skip4)


class DecoderConvUp(nn.Module):
	def __init__(self):
		super(DecoderConvUp, self).__init__()

		self.upsampling2 = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear')
		self.upsampling4 = nn.Upsample(scale_factor=(1, 4, 4), mode='trilinear')
		self.upsampling8 = nn.Upsample(scale_factor=(1, 8, 8), mode='trilinear')

		self.conv1 = nn.Conv3d(96, 192, kernel_size=(2, 1, 1), stride=(2, 1, 1))
		self.conv2 = nn.Conv3d(192, 192, kernel_size=(2, 1, 1), stride=(2, 1, 1))
		self.conv3 = nn.Conv3d(384, 192, kernel_size=(2, 1, 1), stride=(2, 1, 1))
		self.conv4 = nn.Conv3d(768, 192, kernel_size=(2, 1, 1), stride=(2, 1, 1))

		self.convs1 = nn.Conv3d(192, 192, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=False)
		self.convs2 = nn.Conv3d(192, 192, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=False)
		self.convs3 = nn.Conv3d(192, 192, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=False)

		self.convtsp1 = nn.Sequential(
			nn.Conv3d(192, 96, kernel_size=(2, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1), bias=False),
			nn.ReLU(),
			self.upsampling2,
			nn.Conv3d(96, 48, kernel_size=(2, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1), bias=False),
			nn.ReLU(),
			self.upsampling2,
			nn.Conv3d(48, 24, kernel_size=(2, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1), bias=False),
			nn.ReLU(),
			nn.Conv3d(24, 1, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=False),
			nn.Sigmoid()
		)
		self.convtsp2 = nn.Sequential(
			nn.Conv3d(192, 96, kernel_size=(2, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1), bias=False),
			nn.ReLU(),
			self.upsampling2,
			nn.Conv3d(96, 48, kernel_size=(2, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1), bias=False),
			nn.ReLU(),
			self.upsampling2,
			nn.Conv3d(48, 24, kernel_size=(2, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1), bias=False),
			nn.ReLU(),
			self.upsampling2,
			nn.Conv3d(24, 1, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=False),
			nn.Sigmoid()
		)
		self.convtsp3 = nn.Sequential(
			nn.Conv3d(192, 96, kernel_size=(2, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1), bias=False),
			nn.ReLU(),
			self.upsampling2,
			nn.Conv3d(96, 48, kernel_size=(2, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1), bias=False),
			nn.ReLU(),
			self.upsampling2,
			nn.Conv3d(48, 24, kernel_size=(2, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1), bias=False),
			nn.ReLU(),
			self.upsampling2,
			nn.Conv3d(24, 1, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=False),
			self.upsampling2,
			nn.Sigmoid()
		)
		self.convtsp4 = nn.Sequential(
			nn.Conv3d(192, 96, kernel_size=(2, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1), bias=False),
			nn.ReLU(),
			self.upsampling2,
			nn.Conv3d(96, 48, kernel_size=(2, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1), bias=False),
			nn.ReLU(),
			self.upsampling2,
			nn.Conv3d(48, 24, kernel_size=(2, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1), bias=False),
			nn.ReLU(),
			self.upsampling2,
			nn.Conv3d(24, 1, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=False),
			self.upsampling4,
			nn.Sigmoid()
		)

		self.convout = nn.Sequential(
			nn.Conv3d(4, 1, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),
			nn.Sigmoid()
		)

		self.gate1 = Gate(192)
		self.gate2 = Gate(192)
		self.gate3 = Gate(192)
		self.gate4 = Gate(192)

	def forward(self, y4, y3, y2, y1):
		y1 = self.conv1(y1)
		y2 = self.conv2(y2)
		y3 = self.conv3(y3)
		y4 = self.conv4(y4)

		t3 = self.upsampling2(y4) + y3
		y3 = self.convs3(t3)
		t2 = self.upsampling2(t3) + y2 + self.upsampling4(y4)
		y2 = self.convs2(t2)
		t1 = self.upsampling2(t2) + y1 + self.upsampling8(y4)
		y1 = self.convs1(t1)


		y1 = self.gate1(y1)
		y2 = self.gate2(y2)
		y3 = self.gate3(y3)
		y4 = self.gate4(y4)

		z1 = self.convtsp1(y1)

		z2 = self.convtsp2(y2)

		z3 = self.convtsp3(y3)

		z4 = self.convtsp4(y4)

		z0 = self.convout(torch.cat((z1, z2, z3, z4), 1))

		z0 = z0.view(z0.size(0), z0.size(3), z0.size(4))
		return z0


class Decoder(nn.Module):
	def __init__(self,verbose=False,use_skip=True,decoder_groups=64,use_channel_shuffle=True):
		super(Decoder, self).__init__()
		self.verbose = verbose
		self.use_skip = use_skip
		self.decoder_groups = decoder_groups
		self.use_channel_shuffle = use_channel_shuffle

		print("DECODER GROUPS USED IS : ",self.decoder_groups)

		if self.use_channel_shuffle:
			if max(1,self.decoder_groups) >= 8:
				self.shuffle1 = ShuffleBlock(max(1,self.decoder_groups))
				print("SHUFFLE1 USED")
			if max(1,self.decoder_groups // 2) >= 8:
				self.shuffle2 = ShuffleBlock(max(1,self.decoder_groups // 2))
				print("SHUFFLE2 USED")
			if max(1,self.decoder_groups // 4) >= 8:
				self.shuffle3 = ShuffleBlock(max(1,self.decoder_groups // 4))
				print("SHUFFLE3 USED")
			if max(1,self.decoder_groups // 8) >= 8:
				self.shuffle4 = ShuffleBlock(max(1,self.decoder_groups // 8))
				print("SHUFFLE4 USED")

		# self.convtsp1 = nn.Sequential(
		# 	nn.Conv3d(1536, 640, kernel_size=(3, 3, 3),
		# 			  stride=(1,1,1), padding=(1, 1, 1), bias=False, groups=self.decoder_groups),
		# 	nn.ReLU(),
		# )

		# added
		self.convtsp1 = nn.Sequential(
			nn.Conv3d(768, 384, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=False, groups=self.decoder_groups),
			nn.ReLU()
		)

		# self.convtsp2 = nn.Sequential(
		# 	nn.Conv3d(1280, 320, kernel_size=(3, 3, 3), stride=(1,1,1), padding=(1, 1, 1), bias=False, groups=max(1,self.decoder_groups // 2)),
		# 	nn.ReLU(),
		# 	nn.Upsample(
		# 		scale_factor=(1, 32/16, 57/29), mode='trilinear')
		# )

		# added
		self.convtsp2 = nn.Sequential(
			nn.Conv3d(768, 384, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=False, groups=max(1, self.decoder_groups // 2)),
			nn.ReLU(),
			nn.Upsample(
				scale_factor=(1, 2, 2), mode='trilinear'
			)
		)

		# self.convtsp3 = nn.Sequential(
		# 	nn.Conv3d(640, 160, kernel_size=(3, 3, 3), stride=(1,1,1), padding=(1, 1, 1), bias=False, groups=max(1,self.decoder_groups // 4)),
		# 	nn.ReLU(),
		# 	nn.Upsample(
		# 		scale_factor=(1, 2, 2), mode='trilinear')
		# )

		# added
		self.convtsp3 = nn.Sequential(
			nn.Conv3d(576, 288, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=False, groups=max(1, self.decoder_groups // 4)),
			nn.ReLU(),
			nn.Upsample(
				scale_factor=(1, 2, 2), mode='trilinear'
			)
		)

		# self.convtsp4 = nn.Sequential(
		# 	nn.Conv3d(320, 40, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False, groups=max(1,self.decoder_groups // 4)),
		# 	nn.ReLU()
		# )

		# added
		self.convtsp4 = nn.Sequential(
			nn.Conv3d(384, 96, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=False, groups=max(1, self.decoder_groups // 4)),
			nn.ReLU(),
			nn.Upsample(
				scale_factor=(1, 2, 2), mode='trilinear'
			)
		)

		# self.convtsp5 = nn.Sequential(
		# 	nn.Conv3d(80, 64, kernel_size=(2, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1), bias=False, groups=max(1,self.decoder_groups // 8)),
		# 	nn.ReLU(),
		# 	nn.Upsample(
		# 		scale_factor=(1, 2, 2), mode='trilinear')
		# )

		# added
		self.convtsp5 = nn.Sequential(
			nn.Conv3d(144, 72, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=False, groups=max(1, self.decoder_groups // 8)),
			nn.ReLU(),
			nn.Upsample(
				scale_factor=(1, 2, 2), mode='trilinear'
			)
		)

		# self.convtsp6 = nn.Sequential(
		# 	nn.Conv3d(64, 32, kernel_size=(2, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1), bias=False, groups=max(1,self.decoder_groups // 16)),
		# 	nn.ReLU(),
		# 	nn.Upsample(
		# 		scale_factor=(1, 2, 2), mode='trilinear'),

		# 	nn.Conv3d(32, 16, kernel_size=(2, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=False, groups=max(1,self.decoder_groups // 32)),
		# 	nn.ReLU(),

		# 	# 4 time dimension
		# 	nn.Conv3d(16, 16, kernel_size=(1, 1, 1),
		# 			  stride=(1, 1, 1), bias=False),
		# 	nn.ReLU(),
		# 	nn.Conv3d(16, 1, kernel_size=(1, 1, 1),
		# 			  stride=1, bias=True),
		# 	nn.Sigmoid(),
		# )
		
		# added
		self.convtsp6 = nn.Sequential(
			nn.Conv3d(72, 36, kernel_size=(2, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1), bias=False, groups=max(1, self.decoder_groups // 16)),
			nn.ReLU(),
			nn.Upsample(
				scale_factor=(1, 2, 2), mode='trilinear'
			),

			nn.Conv3d(36, 18, kernel_size=(2, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1), bias=False, groups=max(1, self.decoder_groups // 32)),
			nn.ReLU(),

			nn.Conv3d(18, 9, kernel_size=(2, 1, 1),
			 		stride=(2, 1, 1), bias=False),
			nn.ReLU(),
			nn.Conv3d(9, 1, kernel_size=(2, 1, 1),
			 		stride=(2, 1, 1), bias=True),
			nn.Sigmoid()
		)

	def forward(self, latent_feats,skip_features=None):

		if skip_features:
			skip1, skip2, skip3, skip4 = skip_features
		
		z = self.convtsp1(latent_feats)
		if self.verbose:
			print('convtsp1', z.shape)

		if skip_features:
			z = torch.cat((z, skip4), 1)
		if self.verbose:
			print('cat_convtsp2', z.shape)

		if self.use_channel_shuffle and max(1,self.decoder_groups) >= 8:
			z = self.shuffle1(z)

		z = self.convtsp2(z)
		if self.verbose:
			print('convtsp2', z.shape)

		if skip_features:
			z = torch.cat((z, skip3), 1)
		if self.verbose:
			print("cat_convtsp3", z.shape)

		if self.use_channel_shuffle and max(1,self.decoder_groups // 2) >= 8:
			z = self.shuffle2(z)

		z = self.convtsp3(z)
		if self.verbose:
			print('convtsp3', z.shape)

		if skip_features:
			z = torch.cat((z, skip2), 1)
		if self.verbose:
			print("cat_convtsp4", z.shape)

		if self.use_channel_shuffle and max(1,self.decoder_groups // 4) >= 8:
			z = self.shuffle3(z)

		z = self.convtsp4(z)
		if self.verbose:
			print('convtsp4', z.shape)

		if skip_features:
			z = torch.cat((z, skip1), 1)
		if self.verbose:
			print("cat_convtsp5", z.shape)

		if self.use_channel_shuffle and max(1,self.decoder_groups // 8) >= 8:
			z = self.shuffle4(z)

		z = self.convtsp5(z)
		if self.verbose:
			print('convtsp5', z.shape)

		z = self.convtsp6(z)
		if self.verbose:
			print('convtsp6', z.shape)

		z = z.view(z.size(0), z.size(3), z.size(4))
		if self.verbose:
			print('output', z.shape)

		return z
