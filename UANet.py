import torch
import torch.nn as nn
import torch.nn.functional as F
from .pvtv2 import *
from .Res2Net_v1b import *
from .resnet import resnet50
import numpy as np
from torchvision import models
import warnings
warnings.filterwarnings('ignore')

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x



class Uncertainty_Rank_Algorithm(nn.Module):
    def __init__(self):
        super(Uncertainty_Rank_Algorithm,self).__init__()
        self.prob = nn.Sigmoid()
    
    def forward(self, map):
        prob_map = self.prob(map)
        fore_uncertainty_map = prob_map-0.5
        back_uncertainty_map = 0.5 - prob_map

        fore_rank_map = torch.zeros_like(map)
        back_rank_map = torch.zeros_like(map)

        fore_rank_map[fore_uncertainty_map>0.] = 5
        fore_rank_map[fore_uncertainty_map>0.1] = 4
        fore_rank_map[fore_uncertainty_map>0.2] = 3
        fore_rank_map[fore_uncertainty_map>0.3] = 2
        fore_rank_map[fore_uncertainty_map>0.4] = 1

        back_rank_map[back_uncertainty_map>0.] = 5
        back_rank_map[back_uncertainty_map>0.1] = 4
        back_rank_map[back_uncertainty_map>0.2] = 3
        back_rank_map[back_uncertainty_map>0.3] = 2
        back_rank_map[back_uncertainty_map>0.4] = 1


        return fore_rank_map.detach(), back_rank_map.detach()



class Uncertainty_Aware_Fusion_Module(nn.Module):
    def __init__(self,high_channel,low_channel,out_channel,num_classes):
        super(Uncertainty_Aware_Fusion_Module, self).__init__()
        self.rank = Uncertainty_Rank_Algorithm()
        self.high_channel = high_channel
        self.low_channel = low_channel
        self.out_channel = out_channel
        self.conv_high = BasicConv2d(2*self.high_channel,self.out_channel,3,1,1)
        self.conv_low = BasicConv2d(2*self.low_channel,self.out_channel,3,1,1)
        self.conv_fusion = nn.Conv2d(2*self.out_channel,self.out_channel,3,1,1)

        self.seg_out = nn.Conv2d(self.out_channel,num_classes,1)


    def forward(self, feature_low, feature_high, map):
        map = map[:,1,:,:].unsqueeze(1)
        uncertainty_fore_map_high, uncertainty_back_map_high = self.rank(map)
        uncertainty_feature_high = torch.cat((uncertainty_fore_map_high * feature_high, uncertainty_back_map_high * feature_high),dim=1)
        uncertainty_high_up = F.interpolate(self.conv_high(uncertainty_feature_high), feature_low.size()[2:], mode='bilinear', align_corners=True)

        low_map = F.interpolate(map, feature_low.size()[2:], mode='bilinear', align_corners=True)
        uncertainty_fore_map_low, uncertainty_back_map_low = self.rank(low_map)
        uncertainty_feature_low = torch.cat((uncertainty_fore_map_low * feature_low, uncertainty_back_map_low * feature_low),dim=1)
        uncertainty_low = self.conv_low(uncertainty_feature_low)

        seg_fusion = torch.cat((uncertainty_high_up, uncertainty_low), dim=1)

        seg_fusion = self.conv_fusion(seg_fusion)

        seg = self.seg_out(seg_fusion)

        return seg_fusion, seg

class FPN(nn.Module):
    def __init__(self, in_channels,out_channels=256,num_outs=4,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 extra_convs_on_inputs=False,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 upsample_cfg=dict(mode='nearest')):
        super(FPN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # True
            if extra_convs_on_inputs:
                # For compatibility with previous release
                # TODO: deprecate `extra_convs_on_inputs`
                self.add_extra_convs = 'on_input'
            else:
                self.add_extra_convs = 'on_output'

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = nn.Conv2d(
                in_channels[i],
                out_channels,
                1)
            fpn_conv = nn.Conv2d(
                out_channels,
                out_channels,
                3,
                padding=1)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = nn.Conv2d(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1)
                self.fpn_convs.append(extra_fpn_conv)

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform(m)

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [lateral_conv(inputs[i + self.start_level])for i, lateral_conv in enumerate(self.lateral_convs)]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            if 'scale_factor' in self.upsample_cfg:
                laterals[i - 1] += F.interpolate(laterals[i],
                                                 **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] += F.interpolate(
                    laterals[i], size=prev_shape, **self.upsample_cfg)

        # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)

class SemanticFPNDecoder(nn.Module):
    def __init__(self,channel, feature_strides, num_classes):
        super(SemanticFPNDecoder, self).__init__()
        self.in_channels = [channel, channel, channel, channel]
        self.feature_strides = feature_strides
        self.scale_heads = nn.ModuleList()
        self.channels = channel
        BN_relu = nn.Sequential(nn.SyncBatchNorm(self.channels),nn.ReLU())
        for i in range(len(feature_strides)):
            head_length = max(
                1,
                int(np.log2(feature_strides[i]) - np.log2(feature_strides[0])))
            scale_head = []
            for k in range(head_length):
                scale_head.append(
                    nn.Sequential(nn.Conv2d(
                        32 if k == 0 else self.channels,
                        self.channels,
                        kernel_size=3,
                        padding=1), nn.BatchNorm2d(self.channels), nn.ReLU(inplace=True)))
                if feature_strides[i] != feature_strides[0]:
                    scale_head.append(
                        nn.Upsample(
                            scale_factor=2,
                            mode='bilinear',
                            align_corners=False))
            self.scale_heads.append(nn.Sequential(*scale_head))

        self.cls_seg = nn.Conv2d(self.channels, num_classes, kernel_size=1)

    def forward(self, x):
        output = self.scale_heads[0](x[0])
        for i in range(1, len(self.feature_strides)):
            output = output + nn.functional.interpolate(
                self.scale_heads[i](x[i]),
                size=output.shape[2:],
                mode='bilinear',
                align_corners=False)

        output = self.cls_seg(output)
        return output
    
class CGM(nn.Module):
    def __init__(self):
        super(CGM, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.prob = nn.Sigmoid()

    def forward(self, feature, map):
        map = map[:,1,:,:].unsqueeze(1)
        m_batchsize, C, height, width = feature.size()
        proj_query = feature.view(m_batchsize, C, -1)
        proj_key = map.view(m_batchsize, 1, -1).permute(0, 2, 1)
        attention = torch.bmm(proj_query, proj_key)
        attention = attention.unsqueeze(2)
        attention = self.prob(attention)
        out = attention * feature
        out = self.gamma * out + feature
        
        return out

class PSM(nn.Module):
    def __init__(self):
        super(PSM, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, feature, map):
        map = map[:,1,:,:].unsqueeze(1)
        m_batchsize, C, height, width = feature.size()
        feature_enhance = []
        for i in range(0,C):
            feature_channel = feature[:, i, :, :].unsqueeze(1)
            proj_query = feature_channel.view(m_batchsize, -1, width * height).permute(0, 2, 1)
            proj_key = map.view(m_batchsize, -1, width * height)
            energy = torch.bmm(proj_query, proj_key)
            attention = self.softmax(energy)
            proj_value = feature_channel.view(m_batchsize, -1, width * height)
            out = torch.bmm(proj_value, attention.permute(0, 2, 1))
            out = out.view(m_batchsize, 1, height, width)
            feature_enhance.append(out)
        feature_enhance = torch.cat(feature_enhance,dim=1)
        final_feature = self.gamma * feature_enhance + feature
        return final_feature


class MBDC(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(MBDC, self).__init__()
        self.relu = nn.ReLU(True)
        out_channel_sum = out_channel * 3

        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1, 1, 0, 1)
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1, 1, 0, 1),
            BasicConv2d(out_channel, out_channel, 3, 1, 4, 4),
            BasicConv2d(out_channel, out_channel, 3, 1, 8, 8),
            BasicConv2d(out_channel, out_channel, 3, 1, 16, 16)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1, 1, 0, 1),
            BasicConv2d(out_channel, out_channel, 3, 1, 2, 2),
            BasicConv2d(out_channel, out_channel, 3, 1, 4, 4),
            BasicConv2d(out_channel, out_channel, 3, 1, 8, 8)
        )
        self.conv_cat =BasicConv2d(out_channel_sum, out_channel, 3, 1, 1, 1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1, 1, 0, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        x_cat = self.conv_cat(torch.cat((x0, x1, x2), dim=1))

        x = self.relu(x_cat + self.conv_res(x))
        return x


class UANet_VGG(nn.Module):
    def __init__(self, channel, num_classes):
        super(UANet_VGG, self).__init__()
        vgg16_bn = models.vgg16_bn(pretrained=True)
        self.inc = vgg16_bn.features[:5]  # 64
        self.down1 = vgg16_bn.features[5:12]  # 64
        self.down2 = vgg16_bn.features[12:22]  # 64
        self.down3 = vgg16_bn.features[22:32]  # 64
        self.down4 = vgg16_bn.features[32:42]  # 64

        self.conv_1 = BasicConv2d(64,channel,3,1,1)
        self.conv_2 = nn.Sequential(MBDC(128,channel))
        self.conv_3 = nn.Sequential(MBDC(256,channel))
        self.conv_4 = nn.Sequential(MBDC(512,channel))
        self.conv_5 = nn.Sequential(MBDC(512,channel))

        self.neck = FPN(in_channels=[channel, channel, channel, channel], out_channels=channel)

        self.decoder = SemanticFPNDecoder(channel = channel,feature_strides=[4, 8, 16, 32],num_classes=num_classes)

        self.cgm = CGM()
        self.psm = PSM()

        self.ufm_layer4 = Uncertainty_Aware_Fusion_Module(high_channel = channel,low_channel = channel, out_channel = channel, num_classes=num_classes)
        self.ufm_layer3 = Uncertainty_Aware_Fusion_Module(high_channel = channel,low_channel = channel, out_channel = channel, num_classes=num_classes)
        self.ufm_layer2 = Uncertainty_Aware_Fusion_Module(high_channel = channel,low_channel = channel, out_channel = channel,num_classes=num_classes)
        self.ufm_layer1 = Uncertainty_Aware_Fusion_Module(high_channel = channel,low_channel = channel, out_channel = channel, num_classes=num_classes)



    def forward(self, x):
        size = x.size()[2:]
        layer1 = self.inc(x)
        layer2 = self.down1(layer1)
        layer3 = self.down2(layer2)
        layer4 = self.down3(layer3)
        layer5 = self.down4(layer4)


        layer5 = self.conv_5(layer5)
        layer4 = self.conv_4(layer4)
        layer3 = self.conv_3(layer3)
        layer2 = self.conv_2(layer2)
        layer1 = self.conv_1(layer1)

        # print(layer1.size(),layer2.size())

        predict_5 = self.decoder(self.neck([layer2,layer3,layer4,layer5]))

        predict_5_down = F.interpolate(predict_5, layer5.size()[2:], mode='bilinear', align_corners=True)

        layer5 = self.psm(layer5,predict_5_down)
        layer5 = self.cgm(layer5,predict_5_down)

        fusion, predict_4 = self.ufm_layer4(layer4,layer5,predict_5_down)
        fusion, predict_3 = self.ufm_layer3(layer3,fusion,predict_4)
        fusion, predict_2 = self.ufm_layer2(layer2,fusion,predict_3)
        fusion, predict_1 = self.ufm_layer1(layer1,fusion,predict_2)

        # return F.interpolate(predict_1, size, mode='bilinear', align_corners=True)

        return F.interpolate(predict_5, size, mode='bilinear', align_corners=True),\
        F.interpolate(predict_4, size, mode='bilinear', align_corners=True),F.interpolate(predict_3, size, mode='bilinear', align_corners=True),\
               F.interpolate(predict_2, size, mode='bilinear', align_corners=True),F.interpolate(predict_1, size, mode='bilinear', align_corners=True)



class UANet_Res50(nn.Module):
    def __init__(self, channel, num_classes):
        super(UANet_Res50, self).__init__()
        self.backbone = resnet50(backbone_path='./pretrained_model/resnet50-19c8e357.pth') 

        self.conv_1 = BasicConv2d(64,channel,3,1,1)
        self.conv_2 = nn.Sequential(MBDC(256,channel))
        self.conv_3 = nn.Sequential(MBDC(512,channel))
        self.conv_4 = nn.Sequential(MBDC(1024,channel))
        self.conv_5 = nn.Sequential(MBDC(2048,channel))

        self.neck = FPN(in_channels=[channel, channel, channel, channel], out_channels=channel)

        self.decoder = SemanticFPNDecoder(channel = channel,feature_strides=[4, 8, 16, 32],num_classes=num_classes)

        self.cgm = CGM()
        self.psm = PSM()

        self.ufm_layer4 = Uncertainty_Aware_Fusion_Module(high_channel = channel,low_channel = channel, out_channel = channel, num_classes=num_classes)
        self.ufm_layer3 = Uncertainty_Aware_Fusion_Module(high_channel = channel,low_channel = channel, out_channel = channel, num_classes=num_classes)
        self.ufm_layer2 = Uncertainty_Aware_Fusion_Module(high_channel = channel,low_channel = channel, out_channel = channel,num_classes=num_classes)
        self.ufm_layer1 = Uncertainty_Aware_Fusion_Module(high_channel = channel,low_channel = channel, out_channel = channel, num_classes=num_classes)



    def forward(self, x):
        size = x.size()[2:]
        layer1,layer2,layer3,layer4,layer5 = self.backbone(x)

        layer5 = self.conv_5(layer5)
        layer4 = self.conv_4(layer4)
        layer3 = self.conv_3(layer3)
        layer2 = self.conv_2(layer2)
        layer1 = self.conv_1(layer1)

        predict_5 = self.decoder(self.neck([layer2,layer3,layer4,layer5]))

        predict_5_down = F.interpolate(predict_5, layer5.size()[2:], mode='bilinear', align_corners=True)

        layer5 = self.psm(layer5,predict_5_down)
        layer5 = self.cgm(layer5,predict_5_down)

        fusion, predict_4 = self.ufm_layer4(layer4,layer5,predict_5_down)
        fusion, predict_3 = self.ufm_layer3(layer3,fusion,predict_4)
        fusion, predict_2 = self.ufm_layer2(layer2,fusion,predict_3)
        fusion, predict_1 = self.ufm_layer1(layer1,fusion,predict_2)

        return F.interpolate(predict_5, size, mode='bilinear', align_corners=True),\
        F.interpolate(predict_4, size, mode='bilinear', align_corners=True),F.interpolate(predict_3, size, mode='bilinear', align_corners=True),\
               F.interpolate(predict_2, size, mode='bilinear', align_corners=True),F.interpolate(predict_1, size, mode='bilinear', align_corners=True)


class UANet_Res250(nn.Module):
    def __init__(self, channel, num_classes):
        super(UANet_Res250, self).__init__()
        self.resnet = res2net50_v1b_26w_4s(pretrained=True)

        self.conv_1 = BasicConv2d(64,channel,3,1,1)
        self.conv_2 = nn.Sequential(MBDC(256,channel))
        self.conv_3 = nn.Sequential(MBDC(512,channel))
        self.conv_4 = nn.Sequential(MBDC(1024,channel))
        self.conv_5 = nn.Sequential(MBDC(2048,channel))

        self.neck = FPN(in_channels=[channel, channel, channel, channel], out_channels=channel)

        self.decoder = SemanticFPNDecoder(channel = channel,feature_strides=[4, 8, 16, 32],num_classes=num_classes)

        self.cgm = CGM()
        self.psm = PSM()

        self.ufm_layer4 = Uncertainty_Aware_Fusion_Module(high_channel = channel,low_channel = channel, out_channel = channel, num_classes=num_classes)
        self.ufm_layer3 = Uncertainty_Aware_Fusion_Module(high_channel = channel,low_channel = channel, out_channel = channel, num_classes=num_classes)
        self.ufm_layer2 = Uncertainty_Aware_Fusion_Module(high_channel = channel,low_channel = channel, out_channel = channel,num_classes=num_classes)
        self.ufm_layer1 = Uncertainty_Aware_Fusion_Module(high_channel = channel,low_channel = channel, out_channel = channel, num_classes=num_classes)



    def forward(self, x):
        size = x.size()[2:]
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        layer1 = self.resnet.relu(x)
        x = self.resnet.maxpool(layer1)  
        layer2 = self.resnet.layer1(x)  
        
        layer3 = self.resnet.layer2(layer2)  
        
        layer4 = self.resnet.layer3(layer3)  
        
        layer5 = self.resnet.layer4(layer4)
        

        layer5 = self.conv_5(layer5)
        layer4 = self.conv_4(layer4)
        layer3 = self.conv_3(layer3)
        layer2 = self.conv_2(layer2)
        layer1 = self.conv_1(layer1)

        # print(layer1.size(),layer2.size())

        predict_5 = self.decoder(self.neck([layer2,layer3,layer4,layer5]))

        predict_5_down = F.interpolate(predict_5, layer5.size()[2:], mode='bilinear', align_corners=True)

        layer5 = self.psm(layer5,predict_5_down)
        layer5 = self.cgm(layer5,predict_5_down)

        fusion, predict_4 = self.ufm_layer4(layer4,layer5,predict_5_down)
        fusion, predict_3 = self.ufm_layer3(layer3,fusion,predict_4)
        fusion, predict_2 = self.ufm_layer2(layer2,fusion,predict_3)
        fusion, predict_1 = self.ufm_layer1(layer1,fusion,predict_2)

        # return F.interpolate(predict_1, size, mode='bilinear', align_corners=True)

        return F.interpolate(predict_5, size, mode='bilinear', align_corners=True),\
        F.interpolate(predict_4, size, mode='bilinear', align_corners=True),F.interpolate(predict_3, size, mode='bilinear', align_corners=True),\
               F.interpolate(predict_2, size, mode='bilinear', align_corners=True),F.interpolate(predict_1, size, mode='bilinear', align_corners=True)


class UANet_pvt(nn.Module):
    def __init__(self, channel, num_classes):
        super(UANet_pvt, self).__init__()
        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        path = './pretrain_model/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        self.conv_2 = nn.Sequential(MBDC(64,channel))
        self.conv_3 = nn.Sequential(MBDC(128,channel))
        self.conv_4 = nn.Sequential(MBDC(320,channel))
        self.conv_5 = nn.Sequential(MBDC(512,channel))

        self.neck = FPN(in_channels=[channel, channel, channel, channel], out_channels=channel)

        self.decoder = SemanticFPNDecoder(channel = channel,feature_strides=[4, 8, 16, 32],num_classes=num_classes)

        self.cgm = CGM()
        self.psm = PSM()

        self.ufm_layer4 = Uncertainty_Aware_Fusion_Module(high_channel = channel,low_channel = channel, out_channel = channel, num_classes=num_classes)
        self.ufm_layer3 = Uncertainty_Aware_Fusion_Module(high_channel = channel,low_channel = channel, out_channel = channel, num_classes=num_classes)
        self.ufm_layer2 = Uncertainty_Aware_Fusion_Module(high_channel = channel,low_channel = channel, out_channel = channel,num_classes=num_classes)
        self.ufm_layer1 = Uncertainty_Aware_Fusion_Module(high_channel = channel,low_channel = channel, out_channel = channel, num_classes=num_classes)



    def forward(self, x):
        size = x.size()[2:]
        layer2,layer3,layer4,layer5 = self.backbone(x)
        

        layer5 = self.conv_5(layer5)
        layer4 = self.conv_4(layer4)
        layer3 = self.conv_3(layer3)
        layer2 = self.conv_2(layer2)

        predict_5 = self.decoder(self.neck([layer2,layer3,layer4,layer5]))

        predict_5_down = F.interpolate(predict_5, layer5.size()[2:], mode='bilinear', align_corners=True)

        layer5 = self.psm(layer5,predict_5_down)
        layer5 = self.cgm(layer5,predict_5_down)

        fusion, predict_4 = self.ufm_layer4(layer4,layer5,predict_5_down)
        fusion, predict_3 = self.ufm_layer3(layer3,fusion,predict_4)
        fusion, predict_2 = self.ufm_layer2(layer2,fusion,predict_3)

        return F.interpolate(predict_5, size, mode='bilinear', align_corners=True),\
        F.interpolate(predict_4, size, mode='bilinear', align_corners=True),F.interpolate(predict_3, size, mode='bilinear', align_corners=True),\
               F.interpolate(predict_2, size, mode='bilinear', align_corners=True)