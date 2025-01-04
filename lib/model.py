import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torchvision.transforms.functional import rgb_to_grayscale

from lib.Res2Net_v1b import res2net50_v1b_26w_4s
from lib.pvt_v2 import pvt_v2_b3

from lib.Res2Net_v1b import res2net101_v1b_26w_4s
from lib.modules import FLM, make_laplace_pyramid, CBAM


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
        return x
###################################################################
class ConvBR(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, dilation=1):
        super(ConvBR, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class conv_upsample(nn.Module):
    def __init__(self, channel = 64):
        super(conv_upsample, self).__init__()
        self.conv = BasicConv2d(channel, channel, 1)

    def forward(self, x, target):
        if x.size()[2:] != target.size()[2:]:
            x = self.conv(F.upsample(x, size=target.size()[2:], mode='bilinear', align_corners=True))
        return x

###################################################################
class Conv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)
###################################################################
class Up(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Sequential(
            Conv(in_channels, in_channels // 4),
            Conv(in_channels // 4, out_channels)

        )
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x1, x2):
        x = torch.cat([x2, x1], dim=1)
        out = self.conv(x)
        return self.up(out)

###################################################################
class Out(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(Out, self).__init__()
        self.conv1 = Conv(in_channels, in_channels // 4, kernel_size=kernel_size,
                               stride=stride, padding=padding)

        self.conv2 = nn.Conv2d(in_channels // 4, out_channels, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x
###################################################################
###################################################################
class MFAM0(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(MFAM0, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        self.conv_1_1 = BasicConv2d(in_channels, out_channels, 1)
        self.conv_1_2 = BasicConv2d(in_channels, out_channels, 1)
        self.conv_1_3 = BasicConv2d(in_channels, out_channels, 1)
        self.conv_1_4 = BasicConv2d(in_channels, out_channels, 1)
        self.conv_1_5 = BasicConv2d(out_channels, out_channels, 3, stride=1, padding=1)

        self.conv_3_1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv_3_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.conv_5_1 = nn.Conv2d(out_channels, out_channels, kernel_size=5, stride=1, padding=2)
        self.conv_5_2 = nn.Conv2d(out_channels, out_channels, kernel_size=5, stride=1, padding=2)

    def forward(self, x):
        ###+
        x1 = x  # self.conv_1_1(x)
        x2 = x  # self.conv_1_2(x)
        x3 = x  # self.conv_1_3(x)

        x_3_1 = self.relu(self.conv_3_1(x2))  ## (BS, 32, ***, ***)
        x_5_1 = self.relu(self.conv_5_1(x3))  ## (BS, 32, ***, ***)

        x_3_2 = self.relu(self.conv_3_2(x_3_1 + x_5_1))  ## (BS, 64, ***, ***)
        x_5_2 = self.relu(self.conv_5_2(x_5_1 + x_3_1))  ## (BS, 64, ***, ***)

        x_mul = torch.mul(x_3_2, x_5_2)
        out = self.relu(x1 + self.conv_1_5(x_mul + x_3_1 + x_5_1))

        return out


class MFAM(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(MFAM, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        self.conv_1_1 = BasicConv2d(in_channels, out_channels, 1)
        self.conv_1_2 = BasicConv2d(in_channels, out_channels, 1)
        self.conv_1_3 = BasicConv2d(in_channels, out_channels, 1)
        self.conv_1_4 = BasicConv2d(in_channels, out_channels, 1)
        self.conv_1_5 = BasicConv2d(out_channels, out_channels, 3, stride=1, padding=1)

        self.conv_3_1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv_3_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.conv_5_1 = nn.Conv2d(out_channels, out_channels, kernel_size=5, stride=1, padding=2)
        self.conv_5_2 = nn.Conv2d(out_channels, out_channels, kernel_size=5, stride=1, padding=2)

    def forward(self, x):
        ###+
        x1 = self.conv_1_1(x)
        x2 = self.conv_1_2(x)
        x3 = self.conv_1_3(x)

        x_3_1 = self.relu(self.conv_3_1(x2))  ## (BS, 32, ***, ***)
        x_5_1 = self.relu(self.conv_5_1(x3))  ## (BS, 32, ***, ***)

        x_3_2 = self.relu(self.conv_3_2(x_3_1 + x_5_1))  ## (BS, 64, ***, ***)
        x_5_2 = self.relu(self.conv_5_2(x_5_1 + x_3_1))  ## (BS, 64, ***, ***)

        x_mul = torch.mul(x_3_2, x_5_2)

        out = self.relu(x1 + self.conv_1_5(x_mul + x_3_1 + x_5_1))

        return out


        ###################################################################
class BLM(nn.Module):
    def __init__(self, channel=64):
        super(BLM, self).__init__()

        self.upconv2 = conv_upsample()

        self.conv1 = nn.Sequential(ConvBR(channel, channel, kernel_size=3, stride=1, padding=1))
        self.conv2 = nn.Sequential(ConvBR(channel, channel, kernel_size=3, stride=1, padding=1))
        self.conv3 = nn.Sequential(ConvBR(channel, channel, kernel_size=3, stride=1, padding=1))
        self.conv4 = nn.Sequential(ConvBR(channel, channel, kernel_size=3, stride=1, padding=1))
        # self.Bconv1 = nn.Sequential(ConvBR(channel, channel,kernel_size=3, stride=1,padding=1))
        # self.ms = MS_CAM()
        self.fuse1 = nn.Sequential(ConvBR(channel, channel, kernel_size=3, stride=1, padding=1))
        self.fuse2 = nn.Sequential(ConvBR(channel, channel, kernel_size=3, stride=1, padding=1))

        self.final_fuse = nn.Sequential(
            ConvBR(channel * 2, channel * 2, kernel_size=3, stride=1, padding=1),
            BasicConv2d(channel * 2, channel, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, sen_f, edge_f, edge_previous):  # x guide y
        s1 = F.upsample(sen_f, size=edge_f.size()[2:], mode='bilinear', align_corners=True)
        s1 = self.conv1(s1)  # upsample
        s2 = self.conv2(sen_f)
        e1 = self.conv3(edge_f)
        e2 = self.conv4(edge_previous)

        e1 = self.fuse1(e1 * s1) + e1
        e2 = self.fuse2(e2 * s2) + e2

        e2 = self.upconv2(e2, e1)  # upsample

        out = self.final_fuse(torch.cat((e1, e2), 1))

        return out


###################################################################
class FeaFusion(nn.Module):
    def __init__(self, channels):
        self.init__ = super(FeaFusion, self).__init__()
        
        self.relu     = nn.ReLU()
        self.layer1   = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        
        self.layer2_1 = nn.Conv2d(channels, channels //4, kernel_size=3, stride=1, padding=1)
        self.layer2_2 = nn.Conv2d(channels, channels //4, kernel_size=3, stride=1, padding=1)
        
        self.layer_fu = nn.Conv2d(channels//4, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x1, x2):
        
        ###
        wweight    = nn.Sigmoid()(self.layer1(x1+x2))
        
        ###
        xw_resid_1 = x1+ x1.mul(wweight)
        xw_resid_2 = x2+ x2.mul(wweight)
        
        ###
        x1_2       = self.layer2_1(xw_resid_1)
        x2_2       = self.layer2_2(xw_resid_2)
        
        out        = self.relu(self.layer_fu(x1_2 + x2_2))
        
        return out
    
###################################################################  
class FeaProp(nn.Module):
    def __init__(self, in_planes):
        self.init__ = super(FeaProp, self).__init__()
        

        act_fn = nn.ReLU(inplace=True)
        
        self.layer_1  = nn.Sequential(nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(in_planes),act_fn)
        self.layer_2  = nn.Sequential(nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(in_planes),act_fn)
        
        self.gate_1   = nn.Conv2d(in_planes*2, 1, kernel_size=1, bias=True)
        self.gate_2   = nn.Conv2d(in_planes*2, 1, kernel_size=1, bias=True)

        self.softmax  = nn.Softmax(dim=1)
        

    def forward(self, x10, x20):
        
        ###
        x1 = self.layer_1(x10)
        x2 = self.layer_2(x20)
        
        cat_fea = torch.cat([x1,x2], dim=1)
        
        ###
        att_vec_1  = self.gate_1(cat_fea)
        att_vec_2  = self.gate_2(cat_fea)

        att_vec_cat  = torch.cat([att_vec_1, att_vec_2], dim=1)
        att_vec_soft = self.softmax(att_vec_cat)
        
        att_soft_1, att_soft_2 = att_vec_soft[:, 0:1, :, :], att_vec_soft[:, 1:2, :, :]
        x_fusion = x1 * att_soft_1 + x2 * att_soft_2
        
        return x_fusion
###################################################################
class ConvBNR(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, bias=False):
        super(ConvBNR, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size, stride=stride, padding=dilation, dilation=dilation, bias=bias),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class Conv1x1(nn.Module):
    def __init__(self, inplanes, planes):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(inplanes, planes, 1)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class MBE(nn.Module):
    def __init__(self):
        super(MBE, self).__init__()
        self.reduce1 = Conv1x1(64, 64)
        self.reduce4 = Conv1x1(64, 256)
        self.block = nn.Sequential(
            ConvBNR(256 + 64, 256, 3),
            ConvBNR(256, 256, 3),
            nn.Conv2d(256, 64, 1))

    def forward(self, x4, x1):
        size = x1.size()[2:]
        x1 = self.reduce1(x1) #(16,256,104,104)->(16,64,104,104)
        x4 = self.reduce4(x4) #(16,2048,13,13)->(16,256,13,13)
        x4 = F.interpolate(x4, size, mode='bilinear', align_corners=False) #(16,256,13,13)->(16,256,104,104)
        out = torch.cat((x4, x1), dim=1)
        out = self.block(out) #(16,320,104,104)->(16,1,104,104)
        return out

########################Local Refinement Module##############################
class LRM(nn.Module):
    """
    Local Refinement Module: including channel resampling and spatial gating.
    Params:
        c_en: encoder channels
        c_de: decoder channels
    """
    def __init__(self, c_en, c_de):
        super(LRM, self).__init__()
        self.c_en = c_en
        self.c_de = c_de

    def forward(self, input_en: torch.Tensor, input_de: torch.Tensor, gate_map):
        b, c, h, w = input_de.size()
        input_en = input_en.view(b, self.c_en, -1)

        # Channel Resampling
        energy = input_de.view(b, self.c_de, -1).matmul(input_en.transpose(-1, -2))
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(
            energy) - energy  # Prevent loss divergence during training
        channel_attention_map = torch.softmax(energy_new, dim=-1)
        input_en = channel_attention_map.matmul(input_en).view(b, -1, h, w)  # channel_attention_feat

        # Spatial Gating
        gate_map = torch.sigmoid(gate_map)
        input_en = input_en.mul(gate_map)

        return input_en

###################################################################      

class MBILNet(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channel=32, opt=None):
        super(MBILNet, self).__init__()
        
        
        act_fn           = nn.ReLU(inplace=True)
        self.nf          = channel

        self.resnet      = res2net50_v1b_26w_4s(pretrained=True)
        #self.resnet = res2net101_v1b_26w_4s(pretrained=True)

        self.resnet      = pvt_v2_b3(pretrained=True)

        self.downSample  = nn.MaxPool2d(2, stride=2)
        
        ##  
        self.rf1         = MFAM0(64,  self.nf)
        self.rf2         = MFAM(256,  self.nf)
        self.rf3         = MFAM(512,  self.nf)
        self.rf4         = MFAM(1024, self.nf)
        self.rf5         = MFAM(2048, self.nf)
        
        
        ##
        self.cfusion2    = FeaFusion(self.nf)
        self.cfusion3    = FeaFusion(self.nf)
        self.cfusion4    = FeaFusion(self.nf)
        self.cfusion5    = FeaFusion(self.nf)
        
        ##
        self.cgate5      = FeaProp(self.nf)
        self.cgate4      = FeaProp(self.nf)
        self.cgate3      = FeaProp(self.nf)
        self.cgate2      = FeaProp(self.nf)
        
        
        self.de_5        = nn.Sequential(nn.Conv2d(self.nf, self.nf, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(self.nf),act_fn)
        self.de_4        = nn.Sequential(nn.Conv2d(self.nf, self.nf, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(self.nf),act_fn)
        self.de_3        = nn.Sequential(nn.Conv2d(self.nf, self.nf, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(self.nf),act_fn)
        self.de_2        = nn.Sequential(nn.Conv2d(self.nf, self.nf, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(self.nf),act_fn)

        ##
        self.edge_conv0 = nn.Sequential(nn.Conv2d(64,       self.nf, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(self.nf),act_fn) 
        self.edge_conv1 = nn.Sequential(nn.Conv2d(256,      self.nf, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(self.nf),act_fn) 
        self.edge_conv2 = nn.Sequential(nn.Conv2d(self.nf,  self.nf, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(self.nf),act_fn) 
        self.edge_conv3 = BasicConv2d(self.nf,   1,  kernel_size=3, padding=1)
        
        
        self.fu_5        = nn.Sequential(nn.Conv2d(self.nf, self.nf, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(self.nf),act_fn)
        self.fu_4        = nn.Sequential(nn.Conv2d(self.nf, self.nf, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(self.nf),act_fn)
        self.fu_3        = nn.Sequential(nn.Conv2d(self.nf, self.nf, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(self.nf),act_fn)
        self.fu_2        = nn.Sequential(nn.Conv2d(self.nf, self.nf, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(self.nf),act_fn)

        self.flm1 = FLM(64)
        self.flm2 = FLM(64)
        self.flm3 = FLM(128)
        self.flm4 = FLM(256)

        self.blm1 = BLM()
        self.blm2 = BLM()
        self.blm3 = BLM()

        self.out5 = Out(512, 1)
        self.out4 = Out(256, 1)
        self.out3 = Out(128, 1)
        self.out2 = Out(64, 1)
        self.out1 = Out(64, 1)

        self.out6 = Out(64, 1)
        self.out7 = Out(64, 1)
        self.out8 = Out(64, 1)

        self.mbe1 = MBE()
        self.mbe2 = MBE()
        self.mbe3 = MBE()
        self.mbe4 = MBE()

        ##
        self.up_2        = nn.Upsample(scale_factor=2,  mode='bilinear', align_corners=True)
        self.up_4        = nn.Upsample(scale_factor=4,  mode='bilinear', align_corners=True)
        self.up_8        = nn.Upsample(scale_factor=8,  mode='bilinear', align_corners=True)
        self.up_16       = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
        self.up5 = nn.Sequential(
            Conv(64, 512),
        )
        self.up4 = Up(768, 256)
        self.up3 = Up(384, 128)
        self.up2 = Up(192, 64)
        self.up1 = Up(128, 64)

        self.trans1=nn.Sequential(
            Conv(64, 256),
        )
        self.trans2 = nn.Sequential(
            Conv(64, 128),
        )
        self.trans3 = nn.Sequential(
            Conv(64, 256),
        )
        self.trans4 = nn.Sequential(
            Conv(256, 64),
        )
        self.trans5 = nn.Sequential(
            Conv(64, 128),
        )
        self.trans6 = nn.Sequential(
            Conv(128, 64),
        )
        self.con1_1=nn.Sequential(Conv1x1(64, 64), )
        self.con1_2=nn.Sequential(Conv1x1(64, 256),)
        self.conv1_3=nn.Sequential(Conv1x1(128, 512),)
        self.conv1_4=nn.Sequential(Conv1x1(320, 1024),)
        self.conv1_5=nn.Sequential(Conv1x1(320, 2048),)
    def forward(self, xx):

        
        # ---- feature abstraction -----
        # x   = self.resnet.conv1(xx)
        # x   = self.resnet.bn1(x)
        # x   = self.resnet.relu(x)
        #
        # # - low-level features
        # x1  = self.resnet.maxpool(x)       # (BS, 64, 88, 88)
        # x2  = self.resnet.layer1(x1)       # (BS, 256, 88, 88)
        # x3  = self.resnet.layer2(x2)       # (BS, 512, 44, 44)
        # x4  = self.resnet.layer3(x3)     # (BS, 1024, 22, 22)
        # x5  = self.resnet.layer4(x4)     # (BS, 2048, 11, 11)
        #
        x=self.resnet.forward_features(xx)
        ## -------------------------------------- ##
        # 假设 x 是原始的特征列表，包含三个张量
        # x1, x2, x3 分别对应第一个特征图的三个尺度
        x1 = x[0]  # 维度 (8, 64, 80, 80)
        x2 = x[1]  # 维度 (8, 128, 40, 40)
        x3 = x[2]  # 维度 (8, 320, 20, 20)

        _,c,h,w=x[0].shape
        # 目标特征维度
        target_dims = [
            (2, 64, h, w),  # x1
            (2, 256, h, w),  # x2
            (2, 512, h//2, w//2),  # x3
            (2, 1024, h//4, w//4),  # x4
            (2, 2048, h//8, w//8)  # x5
        ]

        # 利用插值或卷积调整特征图
        def adjust_features(feature, target_dim):
            _, c, h, w = target_dim
            # 调整空间尺寸
            feature = F.interpolate(feature, size=(h, w), mode='bilinear', align_corners=False)
            return feature

        # 调整特征图列表
        adjusted_features = [
            adjust_features(x1, target_dims[0]),
            adjust_features(x1, target_dims[1]),
            adjust_features(x2, target_dims[2]),
            adjust_features(x3, target_dims[3]),
            adjust_features(x3, target_dims[4]),
        ]
        adjusted_features[0]=self.con1_1(adjusted_features[0])
        adjusted_features[1]=self.con1_2(adjusted_features[1])
        adjusted_features[2]=self.conv1_3(adjusted_features[2])
        adjusted_features[3]=self.conv1_4(adjusted_features[3])
        adjusted_features[4]=self.conv1_5(adjusted_features[4])


        xf1 = self.rf1(adjusted_features[0])
        xf2 = self.rf2(adjusted_features[1])
        xf3 = self.rf3(adjusted_features[2])
        xf4 = self.rf4(adjusted_features[3])
        xf5 = self.rf5(adjusted_features[4])

        grayscale_img = rgb_to_grayscale(xx)
        edge_feature = make_laplace_pyramid(grayscale_img, 5, 64)
        e_f1 = edge_feature[1]
        e_f2 = edge_feature[2]
        e_f3 = edge_feature[3]
        e_f4 = edge_feature[4]

        e_f1 = self.mbe1(xf3, e_f1) #用局部细化后的高级语义信息指导边界学习
        e_f2 = self.mbe2(xf3, e_f2)
        e_f3 = self.mbe3(xf3, e_f3)
        e_f4 = self.mbe4(xf3, e_f4)

        #----------------特征融合模块------------
        en_fusion5   = self.cfusion5(self.up_2(xf5), xf4)

        en_fusion4   = self.cfusion4(self.up_2(xf4), xf3)
        en_fusion4 = F.interpolate(self.trans1(en_fusion4), scale_factor=1 / 2, mode='bilinear')

        en_fusion3   = self.cfusion3(self.up_2(xf3), xf2)
        en_fusion3 = F.interpolate(self.trans2(en_fusion3), scale_factor=1 / 2, mode='bilinear')

        en_fusion2   = self.cfusion2(self.up_2(xf2), self.up_2(xf1))
        en_fusion2 = F.interpolate(en_fusion2, scale_factor=1 / 2, mode='bilinear')

        #----------------边界交互学习模块----------
        d5 = self.up5(en_fusion5)
        out5 = self.out5(d5)
        edge_feature_1 = self.blm1(en_fusion5, e_f3, e_f4)
        flm4 = self.flm4(self.trans3(edge_feature_1), en_fusion4, out5)

        d4 = self.up4(d5, flm4)
        out4 = self.out4(d4)
        edge_feature_2 = self.blm2(self.trans4(self.up_2(flm4)), e_f2, edge_feature_1)
        flm3 = self.flm3(self.trans5(edge_feature_2), en_fusion3, out4)

        d3 = self.up3(d4, flm3)
        out3 = self.out3(d3)
        edge_feature_3 = self.blm3(self.trans6(self.up_2(flm3)), e_f1, edge_feature_2)
        flm2 = self.flm2(edge_feature_3, en_fusion2, out3)

        d2 = self.up2(d3, flm2)
        out2 = self.out2(d2)

        out5 = self.up_16(out5)
        out4 = self.up_8(out4)
        out3 = self.up_4(out3)
        out2 = self.up_2(out2)
        edge_feature_1 = self.up_8(self.out6(edge_feature_1))
        edge_feature_2 = self.up_4(self.out7(edge_feature_2))
        edge_feature_3 = self.up_2(self.out8(edge_feature_3))
        # ---- output ----
        return out2, out3, out4, out5, edge_feature_1, edge_feature_2, edge_feature_3
    
     