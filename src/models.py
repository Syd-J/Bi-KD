import timm
import torch
from transformers import AutoModelForImageClassification

import mae
import tmfi_net as TMFINet
from vinet_a import ViNetA
import vinet_s as ViNetS


_timm_models = {
    'SeNet154': 'senet154.gluon_in1k',
    'SWSL-ResNext101': 'resnext101_32x8d.fb_swsl_ig1b_ft_in1k',
    'PiT-B': 'pit_b_224.in1k',
    'ResMLP-36': 'resmlp_36_224.fb_in1k',
    'ResMLP-24-dist': 'resmlp_24_224.fb_distilled_in1k',
    'CoaT-lite-mini': 'coat_lite_mini.in1k',
    'PiT-XS': 'pit_xs_224.in1k'
}

def create_model(model_name, pretrained=True):
    if model_name in _timm_models:
        model = timm.create_model(_timm_models[model_name], pretrained=pretrained)
        return model
    
    elif model_name == 'MAE':
        model = mae.__dict__['vit_base_patch16'](
            num_classes=1000,
            drop_path_rate=0.1,
            global_pool=True
        )
        model.load_state_dict(torch.load('pretrained/mae_finetuned_vit_base.pth')['model'])
        return model
    
    elif model_name == 'MambaVision-T2':
        model = AutoModelForImageClassification.from_pretrained("nvidia/MambaVision-T2-1K", trust_remote_code=True)
        return model
    
    elif model_name == 'DINOv2':
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_lc')
        return model
    
    else:
        raise NotImplementedError(f'Model {model_name} not implemented.')

def create_sal_model(model_name, args, pretrained=True, dataset='DHF1K'):
    if model_name == 'TMFI-Net':
        model = TMFINet.VideoSaliencyModel()
        if pretrained:
            model.load_state_dict(torch.load(f'pretrained/best_VSTNet_{dataset}.pth'))
        return model
    elif model_name == 'ViNet-A':
        model = ViNetA(use_skip=args['use_skip'], use_channel_shuffle=args['use_channel_shuffle'], decoder_groups=args['decoder_groups'])
        if pretrained:
            model.load_state_dict(torch.load(f'pretrained/ViNetA_{dataset}_32g.pt'))
        return model
    elif model_name == 'ViNet-S':
        model = ViNetS.VideoSaliencyModel(
		use_upsample=bool(args['decoder_upsample']),
        num_hier=args['num_hier'],
        num_clips=32,
        grouped_conv=args['grouped_conv'],
        root_grouping=args['root_grouping'],
        depth=args['depth_grouping'],
        efficientnet=args['efficientnet'],
        BiCubic = not args['use_trilinear_upsampling'],
	)
        if pretrained:
            model.load_state_dict(torch.load(f'pretrained/vinet_rootgrouped_32_{dataset}.pt'))
        return model
    else:
        raise NotImplementedError(f'Saliency model {model_name} not implemented.')
