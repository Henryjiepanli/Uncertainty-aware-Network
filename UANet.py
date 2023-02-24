import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import math

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

class Uncertainty_Rank(nn.Module):
    def __init__(self):
        super(Uncertainty_Rank,self).__init__()
        self.prob = nn.Sigmoid()
    
    def forward(self, map):
        rank_map_1 = torch.zeros_like(map)
        rank_map_2 = torch.zeros_like(map)
        rank_map_3 = torch.zeros_like(map)
        rank_map_4 = torch.zeros_like(map)
        rank_map_5 = torch.zeros_like(map)
        prob_map = self.prob(map)
        abs_map = torch.abs(prob_map-0.5)
        rank_map_1[abs_map <= 0.5] = 1
        rank_map_2[abs_map <= 0.4] = 0
        rank_map_3[abs_map <= 0.3] = 0.5
        rank_map_4[abs_map <= 0.2] = 1
        rank_map_5[abs_map <= 0.1] = 1

        rank_map = rank_map_1 + rank_map_2 + rank_map_3 +rank_map_4 + rank_map_5

        return rank_map.detach()


class Uncertainty_Fusion_Module(nn.Module):
    def __init__(self,high_channel,low_channel,num_classes):
        super(Uncertainty_Fusion_Module,self).__init__()
        self.rank = Uncertainty_Rank()
        self.prob = nn.Sigmoid()
        self.high_channel = high_channel
        self.low_channel = low_channel
        self.up_high = nn.ConvTranspose2d(2*self.high_channel, self.low_channel, kernel_size = 3, stride=2, padding=1, output_padding=1)
        self.align_conv = nn.ConvTranspose2d(1, 1, kernel_size = 3, stride=2, padding=1, output_padding=1)

        self.seg_out = nn.Sequential(nn.Conv2d(3*self.low_channel,self.low_channel,3,1,1),nn.Conv2d(self.low_channel,num_classes,1))


    def forward(self, feature_low, feature_high, map):
        map = map[:,1,:,:].unsqueeze(1)
        uncertainty_map = self.rank(map)
        seg_feature_high = self.prob(map) * feature_high
        edge_feature_high = uncertainty_map * feature_high

        fusion_high = torch.cat((seg_feature_high, edge_feature_high), dim=1)
        fusion_high_up = self.up_high(fusion_high)

        low_map = self.align_conv(map)
        uncertainty_map_low = self.rank(low_map)

        seg_feature_low = self.prob(low_map) * feature_low
        edge_feature_low = uncertainty_map_low * feature_low

        fusion_low = torch.cat((seg_feature_low,edge_feature_low), dim=1)

        seg_fusion = torch.cat((fusion_low, fusion_high_up), dim=1)


        seg = self.seg_out(seg_fusion)

        return seg

class PAM_Module(nn.Module):
    """ Position attention module"""

    # Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
            解释  :
                bmm : 实现batch的叉乘
                Parameter：绑定在层里，所以是可以更新的
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out

class CAM_Module(nn.Module):
    """ Channel attention module"""

    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out

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

class FEM(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(FEM, self).__init__()
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

class UANet(nn.Module):
    def __init__(self, channel, num_classes):
        super(UANet, self).__init__()
        vgg16_bn = models.vgg16_bn(pretrained=True)
        self.inc = vgg16_bn.features[:5]  # 64
        self.down1 = vgg16_bn.features[5:12]  # 64
        self.down2 = vgg16_bn.features[12:22]  # 64
        self.down3 = vgg16_bn.features[22:32]  # 64
        self.down4 = vgg16_bn.features[32:42]  # 64

        self.conv_1 = nn.Sequential(BasicConv2d(64, channel, 3, 1, 1))
        self.conv_2 = nn.Sequential(BasicConv2d(128, channel, 3, 1, 1),FEM(channel,channel))
        self.conv_3 = nn.Sequential(BasicConv2d(256, channel, 3, 1, 1),FEM(channel,channel))
        self.conv_4 = nn.Sequential(BasicConv2d(512, channel, 3, 1, 1),FEM(channel,channel))
        self.conv_5 = nn.Sequential(BasicConv2d(512, channel, 3, 1, 1),FEM(channel,channel),PAM_Module(channel), CAM_Module(channel))

        self.up_5 = nn.ConvTranspose2d(channel, channel, kernel_size = 3, stride=2, padding=1, output_padding=1)
        self.up_4 = nn.ConvTranspose2d(channel, channel, kernel_size = 3, stride=2, padding=1, output_padding=1)
        self.up_3 = nn.ConvTranspose2d(channel, channel, kernel_size = 3, stride=2, padding=1, output_padding=1)
        self.up_2 = nn.ConvTranspose2d(channel, channel, kernel_size = 3, stride=2, padding=1, output_padding=1)


        self.seg_out_1 = nn.Sequential(BasicConv2d(channel, channel, 3, 1, 1),BasicConv2d(channel, num_classes, 3, 1, 1))
        self.cgm = CGM()
        self.psm = PSM()
        self.ufm_layer4 = Uncertainty_Fusion_Module(high_channel = channel,low_channel = channel, num_classes=num_classes)
        self.ufm_layer3 = Uncertainty_Fusion_Module(high_channel = channel,low_channel = channel, num_classes=num_classes)
        self.ufm_layer2 = Uncertainty_Fusion_Module(high_channel = channel,low_channel = channel, num_classes=num_classes)
        self.ufm_layer1 = Uncertainty_Fusion_Module(high_channel = channel,low_channel = channel, num_classes=num_classes)

    def forward(self,x):
        size = x.size()[2:]
        layer1 = self.inc(x)
        layer2 = self.down1(layer1)
        layer3 = self.down2(layer2)
        layer4 = self.down3(layer3)
        layer5 = self.down4(layer4)

        layer1 = self.conv_1(layer1)
        layer2 = self.conv_2(layer2)
        layer3 = self.conv_3(layer3)
        layer4 = self.conv_4(layer4)
        layer5 = self.conv_5(layer5)

        fusion = self.up_5(layer5) + layer4
        fusion = self.up_4(fusion) + layer3
        fusion = self.up_3(fusion) + layer2
        fusion = self.up_2(fusion) + layer1

        seg_1 = self.seg_out_1(fusion)
        seg_1_down = F.interpolate(seg_1, layer5.size()[2:], mode='bilinear', align_corners=True)

        layer5 = self.psm(layer5,seg_1_down)
        layer5 = self.cgm(layer5,seg_1_down)

        seg_2 = self.ufm_layer4(layer4,layer5,seg_1_down)
        seg_3 = self.ufm_layer3(layer3,layer4,seg_2)
        seg_4 = self.ufm_layer2(layer2,layer3,seg_3)
        seg_5 = self.ufm_layer1(layer1,layer2,seg_4)


        return F.interpolate(seg_1, size, mode='bilinear', align_corners=True),F.interpolate(seg_2, size, mode='bilinear', align_corners=True),\
               F.interpolate(seg_3, size, mode='bilinear', align_corners=True),F.interpolate(seg_4, size, mode='bilinear', align_corners=True),\
               F.interpolate(seg_5, size, mode='bilinear', align_corners=True)









