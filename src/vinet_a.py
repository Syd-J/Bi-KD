import torch
from torch import nn
from einops import rearrange


BN = nn.BatchNorm3d

class Bottleneck(nn.Module):
	expansion = 4

	def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1, head_conv=1):
		super(Bottleneck, self).__init__()
		if head_conv == 1:
			self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
			self.bn1 = BN(planes)
		elif head_conv == 3:
			self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=(3, 1, 1), bias=False, padding=(1, 0, 0))
			self.bn1 = BN(planes)
		else:
			raise ValueError("Unsupported head_conv!")
		self.conv2 = nn.Conv3d(
			planes, planes, kernel_size=(1, 3, 3), stride=(1, stride, stride), 
			padding=(0, dilation, dilation), dilation=(1, dilation, dilation), bias=False)
		self.bn2 = BN(planes)
		self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
		self.bn3 = BN(planes * 4)
		self.relu = nn.ReLU(inplace=True)
		self.downsample = downsample
		if downsample is not None:
			self.downsample_bn = BN(planes * 4)
		self.stride = stride

	def forward(self, x):
		res = x
		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)
		out = self.relu(out)

		out = self.conv3(out)
		out = self.bn3(out)

		if self.downsample is not None:
			res = self.downsample(x)
			res = self.downsample_bn(res)

		out = out + res
		out = self.relu(out)
		
		return out


class SlowFast(nn.Module):
	def __init__(self, block, layers, alpha=4, beta=0.125, fuse_only_conv=True, fuse_kernel_size=5, slow_full_span=False,use_skip=True):
		super(SlowFast, self).__init__()
		self.alpha = alpha
		self.beta = beta
		self.slow_full_span = slow_full_span
		self.use_skip = use_skip

		'''Fast Network'''
		self.fast_inplanes = int(64 * beta)
		self.fast_conv1 = nn.Conv3d(3, self.fast_inplanes, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False)
		self.fast_bn1 = BN(self.fast_inplanes)
		self.fast_relu = nn.ReLU(inplace=True)
		self.fast_maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

		self.fast_res1 = self._make_layer_fast(block, int(64 * beta), layers[0], head_conv=3)
		self.fast_res2 = self._make_layer_fast(block, int(128 * beta), layers[1], stride=2, head_conv=3)
		self.fast_res3 = self._make_layer_fast(block, int(256 * beta), layers[2], stride=2, head_conv=3)
		self.fast_res4 = self._make_layer_fast(block, int(512 * beta), layers[3], head_conv=3, dilation=2)

		'''Slow Network'''
		self.slow_inplanes = 64
		self.slow_conv1 = nn.Conv3d(3, self.slow_inplanes, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3), bias=False)
		self.slow_bn1 = BN(self.slow_inplanes)
		self.slow_relu = nn.ReLU(inplace=True)
		self.slow_maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

		self.slow_res1 = self._make_layer_slow(block, 64, layers[0], head_conv=1)
		self.slow_res2 = self._make_layer_slow(block, 128, layers[1], stride=2, head_conv=1)
		self.slow_res3 = self._make_layer_slow(block, 256, layers[2], stride=2, head_conv=3)
		self.slow_res4 = self._make_layer_slow(block, 512, layers[3], head_conv=3, dilation=2)

		'''Lateral Connections'''
		fuse_padding = fuse_kernel_size // 2
		fuse_kwargs = {'kernel_size': (fuse_kernel_size, 1, 1), 'stride': (alpha, 1, 1), 'padding': (fuse_padding, 0, 0), 'bias': False}
		if fuse_only_conv:
			def fuse_func(in_channels, out_channels):
				return nn.Conv3d(in_channels, out_channels, **fuse_kwargs)
		else:
			def fuse_func(in_channels, out_channels):
				return nn.Sequential(
					nn.Conv3d(in_channels, out_channels, **fuse_kwargs),
					BN(out_channels),
					nn.ReLU(inplace=True)
				)
		self.Tconv1 = fuse_func(int(64 * beta), int(128 * beta))
		self.Tconv2 = fuse_func(int(256 * beta), int(512 * beta))
		self.Tconv3 = fuse_func(int(512 * beta), int(1024 * beta))
		self.Tconv4 = fuse_func(int(1024 * beta), int(2048 * beta))
		

	def forward(self, input):

		fast, Tc = self.FastPath(input)
		if self.slow_full_span:
			slow_input = torch.index_select(
				input,
				2,
				torch.linspace(
					0,
					input.shape[2] - 1,
					input.shape[2] // self.alpha,
				).long().cuda(),
			)
		else:
			slow_input = input[:, :, ::self.alpha, :, :]
			
		if self.use_skip:
			slow,skip = self.SlowPath(slow_input, Tc)
			return [slow, fast, *skip] 
		slow = self.SlowPath(slow_input, Tc)
		return [slow, fast]

	def SlowPath(self, input, Tc):
		# verbose = False
		x = self.slow_conv1(input)
		x = self.slow_bn1(x)
		x = self.slow_relu(x)
		x = self.slow_maxpool(x)
		x = torch.cat([x, Tc[0]], dim=1)
		if self.use_skip:
			skip1 = x
		# if verbose:
		# 	print("skip1 shape: ",skip1.shape)

		x = self.slow_res1(x)
		x = torch.cat([x, Tc[1]], dim=1)
		if self.use_skip:
			skip2 = x
		# if verbose:
		# 	print("skip2 shape: ",skip2.shape)
		x = self.slow_res2(x)
		x = torch.cat([x, Tc[2]], dim=1)
		if self.use_skip:
			skip3 = x
		# if verbose:
		# 	print("skip3 shape: ",skip3.shape)
		x = self.slow_res3(x)
		x = torch.cat([x, Tc[3]], dim=1)
		if self.use_skip:
			skip4 = x
		# if verbose:
		# 	print("skip4 shape: ",skip4.shape)
		x = self.slow_res4(x)
		if self.use_skip:
			return x,[skip1,skip2,skip3,skip4]
		return x

	def FastPath(self, input):
		x = self.fast_conv1(input)
		x = self.fast_bn1(x)
		x = self.fast_relu(x)
		x = self.fast_maxpool(x)
		Tc1 = self.Tconv1(x)
		x = self.fast_res1(x)
		Tc2 = self.Tconv2(x)
		x = self.fast_res2(x)
		Tc3 = self.Tconv3(x)
		x = self.fast_res3(x)
		Tc4 = self.Tconv4(x)
		x = self.fast_res4(x)
		return x, [Tc1, Tc2, Tc3, Tc4]

	def _make_layer_fast(self, block, planes, blocks, stride=1, head_conv=1, dilation=1):
		downsample = None
		if stride != 1 or self.fast_inplanes != planes * block.expansion:
			downsample = nn.Sequential(
				nn.Conv3d(
					self.fast_inplanes,
					planes * block.expansion,
					kernel_size=1,
					stride=(1, stride, stride),
					bias=False
				)
			)

		layers = []
		layers.append(block(self.fast_inplanes, planes, stride, downsample, dilation=dilation, head_conv=head_conv))
		self.fast_inplanes = planes * block.expansion
		for i in range(1, blocks):
			layers.append(block(self.fast_inplanes, planes, dilation=dilation, head_conv=head_conv))

		return nn.Sequential(*layers)

	def _make_layer_slow(self, block, planes, blocks, stride=1, head_conv=1, dilation=1):
		downsample = None
		fused_inplanes = self.slow_inplanes + int(self.slow_inplanes * self.beta) * 2
		if stride != 1 or fused_inplanes != planes * block.expansion:
			downsample = nn.Sequential(
				nn.Conv3d(
					fused_inplanes,
					planes * block.expansion,
					kernel_size=1,
					stride=(1, stride, stride),
					bias=False
				)
			)

		layers = []
		layers.append(block(fused_inplanes, planes, stride, downsample, dilation=dilation, head_conv=head_conv))
		self.slow_inplanes = planes * block.expansion
		for i in range(1, blocks):
			layers.append(block(self.slow_inplanes, planes, dilation=dilation, head_conv=head_conv))

		return nn.Sequential(*layers)

	
def slowfast50(use_skip, **kwargs):
	"""Constructs a SlowFast-50 model.
	"""
	model = SlowFast(Bottleneck, [3, 4, 6, 3], use_skip=use_skip, **kwargs)
	return model


class Neck(nn.Module):
	def __init__(self):
		super(Neck, self).__init__()
		
		self.conv_slow= nn.Sequential(
				nn.Conv3d(2048, 1024, kernel_size=(1, 1, 1),
						stride=(1,1,1), padding=(0, 0, 0), bias=False),
				nn.ReLU()
				)
		self.adaptive_maxpool = nn.AdaptiveMaxPool3d((8, 16, 29))

		self.conv_skip1= nn.Sequential(
				nn.Conv3d(80, 40, kernel_size=(1, 1, 1),
						stride=(1,1,1), padding=(0, 0, 0), bias=False),
				nn.ReLU()
				)
		self.conv_skip2= nn.Sequential(
				nn.Conv3d(320, 160, kernel_size=(1, 1, 1),
						stride=(1,1,1), padding=(0, 0, 0), bias=False),
				nn.ReLU()
				)
		self.conv_skip3= nn.Sequential(
				nn.Conv3d(640, 320, kernel_size=(1, 1, 1),
						stride=(1,1,1), padding=(0, 0, 0), bias=False),
				nn.ReLU()
				)
		self.conv_skip4= nn.Sequential(
				nn.Conv3d(1280, 640, kernel_size=(1, 1, 1),
						stride=(1,1,1), padding=(0, 0, 0), bias=False),
				nn.ReLU()
				)	


	def forward(self, slow_features,fast_features,skip_connections):
		
			fast_features = rearrange(fast_features, 'b c t h w -> b (c t) h w')
			fast_features = rearrange(fast_features, 'b (c t) h w -> b c t h w',
							c=256*2, t=int(32/2))

			slow_features = self.conv_slow(slow_features)

			fast_features = self.adaptive_maxpool(fast_features)

			slow_fast_features = torch.cat((slow_features, fast_features), 1)

			skip1,skip2,skip3,skip4 = skip_connections

			skip1 = self.conv_skip1(skip1)
			skip2 = self.conv_skip2(skip2)
			skip3 = self.conv_skip3(skip3)
			skip4 = self.conv_skip4(skip4)


			return slow_fast_features,(skip1,skip2,skip3,skip4)


class ShuffleBlock(nn.Module):
    def __init__(self, groups):
        super(ShuffleBlock, self).__init__()
        self.groups = groups
    def forward(self, x):
        '''Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]'''
        N,C,T,H,W = x.size()
        g = self.groups
        return x.view(N,g,C//g,T,H,W).permute(0,2,1,3,4,5).reshape(N,C,T,H,W)

class Decoder(nn.Module):
	def __init__(self,verbose=False,use_skip=True,decoder_groups=64,use_attention=True,use_channel_shuffle=True):
		super(Decoder, self).__init__()
		self.verbose = verbose
		self.use_skip = use_skip
		self.decoder_groups = decoder_groups
		self.use_attention = use_attention
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

		self.convtsp1 = nn.Sequential(
			nn.Conv3d(1536, 640, kernel_size=(3, 3, 3),
					  stride=(1,1,1), padding=(1, 1, 1), bias=False, groups=self.decoder_groups),
			nn.ReLU(),
		)
		self.convtsp2 = nn.Sequential(
			nn.Conv3d(1280, 320, kernel_size=(3, 3, 3), stride=(1,1,1), padding=(1, 1, 1), bias=False, groups=max(1,self.decoder_groups // 2)),
			nn.ReLU(),
			nn.Upsample(
				scale_factor=(1, 32/16, 57/29), mode='trilinear')
		)
		self.convtsp3 = nn.Sequential(
			nn.Conv3d(640, 160, kernel_size=(3, 3, 3), stride=(1,1,1), padding=(1, 1, 1), bias=False, groups=max(1,self.decoder_groups // 4)),
			nn.ReLU(),
			nn.Upsample(
				scale_factor=(1, 2, 2), mode='trilinear')
		)
		self.convtsp4 = nn.Sequential(
			nn.Conv3d(320, 40, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False, groups=max(1,self.decoder_groups // 4)),
			nn.ReLU()
		)
		self.convtsp5 = nn.Sequential(
			nn.Conv3d(80, 64, kernel_size=(2, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1), bias=False, groups=max(1,self.decoder_groups // 8)),
			nn.ReLU(),
			nn.Upsample(
				scale_factor=(1, 2, 2), mode='trilinear')
		)
		self.convtsp6 = nn.Sequential(
			nn.Conv3d(64, 32, kernel_size=(2, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1), bias=False, groups=max(1,self.decoder_groups // 16)),
			nn.ReLU(),
			nn.Upsample(
				scale_factor=(1, 2, 2), mode='trilinear'),

			nn.Conv3d(32, 16, kernel_size=(2, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=False, groups=max(1,self.decoder_groups // 32)),
			nn.ReLU(),

			# 4 time dimension
			nn.Conv3d(16, 16, kernel_size=(1, 1, 1),
					  stride=(1, 1, 1), bias=False),
			nn.ReLU(),
			nn.Conv3d(16, 1, kernel_size=(1, 1, 1),
					  stride=1, bias=True),
			nn.Sigmoid()
		)
		

	def forward(self, high_order_feats,skip_features=None):

		if skip_features:
			skip1, skip2, skip3, skip4 = skip_features
		
		z = self.convtsp1(high_order_feats)
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


class ViNetA(nn.Module):
	def __init__(self,
				use_skip = True,
				use_channel_shuffle=True,
				decoder_groups=32
			):
		super(ViNetA, self).__init__()

		self.use_skip = use_skip
		self.use_channel_shuffle = use_channel_shuffle

		self.backbone = slowfast50(self.use_skip)

		self.decoder_groups = decoder_groups

		self.neck = Neck()

		self.decoder = Decoder(decoder_groups=self.decoder_groups, use_channel_shuffle=self.use_channel_shuffle)


	def forward(self, x):

		y = self.backbone(x)
		slow_features,fast_features = y[:2]
		if self.use_skip:
			skip_connections = y[2:]
			slow_fast_features, skip_connections = self.neck(slow_features, fast_features, skip_connections)
			return self.decoder(slow_fast_features, skip_connections)

		return self.decoder(slow_features, fast_features)
	