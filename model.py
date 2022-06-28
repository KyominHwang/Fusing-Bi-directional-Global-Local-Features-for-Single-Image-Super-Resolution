import torch
import torch.nn as nn
from utils import *
import torch.nn.functional as F
from functools import partial
from einops.layers.torch import Rearrange
from einops import repeat


class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.conv1 = nn.Conv2d(channel, channel, 3, 1, 1)
        self.conv2 = nn.Conv2d(channel, channel, 3, 1, 1)
        self.relu = nn.ReLU()

    def forward(self, x, com=None):
        if com is not None:
            connection = com
        else:
            connection = x
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x) + connection


class PSA_p(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=1, stride=1):
        super(PSA_p, self).__init__()

        self.inplanes = inplanes
        self.inter_planes = planes // 2
        self.planes = planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = (kernel_size - 1) // 2

        self.conv_q_right = nn.Conv2d(self.inplanes, 1, kernel_size=1, stride=stride, padding=0, bias=False)
        self.conv_v_right = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0,
                                      bias=False)
        self.conv_up = nn.Conv2d(self.inter_planes, self.planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.softmax_right = nn.Softmax(dim=2)
        self.sigmoid = nn.Sigmoid()

        self.conv_q_left = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0,
                                     bias=False)  # g
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_v_left = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0,
                                     bias=False)  # theta
        self.softmax_left = nn.Softmax(dim=2)

        self.conv1 = nn.Conv2d(inplanes, planes, 3, 1, 1)
        self.conv2 = nn.Conv2d(inplanes, planes, 3, 1, 1)
        self.relu = nn.GELU()

    def spatial_pool(self, x):
        input_x = self.conv_v_right(x)

        batch, channel, height, width = input_x.size()

        # [N, IC, H*W]
        input_x = input_x.view(batch, channel, height * width)

        # [N, 1, H, W]
        context_mask = self.conv_q_right(x)

        # [N, 1, H*W]
        context_mask = context_mask.view(batch, 1, height * width)

        # [N, 1, H*W]
        context_mask = self.softmax_right(context_mask)

        # [N, IC, 1]
        # context = torch.einsum('ndw,new->nde', input_x, context_mask)
        context = torch.matmul(input_x, context_mask.transpose(1, 2))
        # [N, IC, 1, 1]
        context = context.unsqueeze(-1)

        # [N, OC, 1, 1]
        context = self.conv_up(context)

        # [N, OC, 1, 1]
        mask_ch = self.sigmoid(context)

        out = x * mask_ch

        return out

    def channel_pool(self, x):
        # [N, IC, H, W]
        g_x = self.conv_q_left(x)

        batch, channel, height, width = g_x.size()

        # [N, IC, 1, 1]
        avg_x = self.avg_pool(g_x)

        batch, channel, avg_x_h, avg_x_w = avg_x.size()

        # [N, 1, IC]
        avg_x = avg_x.view(batch, channel, avg_x_h * avg_x_w).permute(0, 2, 1)

        # [N, IC, H*W]
        theta_x = self.conv_v_left(x).view(batch, self.inter_planes, height * width)

        # [N, 1, H*W]
        # context = torch.einsum('nde,new->ndw', avg_x, theta_x)
        context = torch.matmul(avg_x, theta_x)
        # [N, 1, H*W]
        context = self.softmax_left(context)

        # [N, 1, H, W]
        context = context.view(batch, 1, height, width)

        # [N, 1, H, W]
        mask_sp = self.sigmoid(context)

        out = x * mask_sp

        return out

    def forward(self, x, com=None):
        if com is not None:
            connection = com
        else:
            connection = x
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        # [N, C, H, W]
        context_channel = self.spatial_pool(x)
        # [N, C, H, W]
        context_spatial = self.channel_pool(context_channel)
        # [N, C, H, W]
        out = context_spatial + connection
        return out


class FeatureExtractor(nn.Module):
    def __init__(self, in_channels=3, img_range=1.0, n_feats=32, n_res_blocks=8, down_sample=2):
        super(FeatureExtractor, self).__init__()
        self.img_range = img_range
        self.down_sample = down_sample
        self.in_channels = in_channels
        self.conv = nn.Conv2d(in_channels, n_feats, 3, 1, 1)
        self.pool = nn.AvgPool2d(2, 2)

        self.conv2 = nn.Sequential(
            Upsampler(2, n_feats),
            nn.Conv2d(n_feats, in_channels, 3, 1, 1)
        )

        cnn_body = [
            PSA_p(n_feats, n_feats) for _ in range(n_res_blocks)
        ]
        self.body = nn.Sequential(*cnn_body)
        weights1 = torch.tensor([[0., -1., 0.],
                                 [-1., 4., -1.],
                                 [0., -1., 0.]])
        self.weights1 = weights1.view(1, 1, 3, 3).repeat(1, in_channels, 1, 1).cuda()
        self.weights1.requires_grad = False
        if in_channels == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.down_sample - h % self.down_sample) % self.down_sample
        mod_pad_w = (self.down_sample - w % self.down_sample) % self.down_sample
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward(self, x, train=False):
        _, _, H, W = x.size()
        original = x
        x = self.check_image_size(x)
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range
        x = self.conv(x)
        x = self.pool(x)
        residual = x
        x = self.body(x) + residual
        x = self.conv2(x)
        x = x / self.img_range + self.mean
        if train:
            loss = torch.mean(torch.pow(F.conv2d(x[:, :, :H, :W], self.weights1), 2))
            return x[:, :, :H, :W], loss
        return x[:, :, :H, :W]


class gMLP(nn.Module):
    def __init__(self, channel, norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer=nn.GELU):
        super().__init__()
        self.bn1 = norm_layer(4)
        self.act = act_layer()
        self.dense1 = nn.Linear(4, 4)
        self.bn2 = norm_layer(4)
        self.dense2 = nn.Linear(4, 4)
        self.proj = nn.Linear(4, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.reshape(B, 4, C, H // 2, W // 2).permute(0, 3, 4, 2, 1).reshape(B, H // 2 * W // 2, C, 4)
        x = self.bn1(x)
        x = self.dense1(x)
        x = self.act(x)  # *B, D, S, C

        split = x
        connection = split
        split = self.bn2(split)
        split = repeat(self.proj(split).squeeze(dim=3), 'b s c -> b s c d', d=4)  # (B, H//2*W//2,C,4)
        x = split * connection
        x = self.dense2(x).permute(0, 3, 2, 1).reshape(B, C, H, W)
        return x


class Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mlp = gMLP(dim)

    def forward(self, x, com=None):
        if com is not None:
            x = com + self.mlp(x)
        else:
            x = x + self.mlp(x)
        return x


class HLBlock(nn.Module):
    def __init__(self, n_feats, n_res_blocks):
        super(HLBlock, self).__init__()
        res = [
            eca_layer(n_feats) for _ in range(n_res_blocks)
        ]
        self.res = nn.Sequential(*res)
        trans = [
            Block(n_feats) for _ in range(1)
        ]
        self.trans = nn.Sequential(*trans)
        self.conv_lf = nn.Conv2d(n_feats * 2, n_feats, 3, 1, 1)
        self.conv_hf = nn.Conv2d(n_feats * 2, n_feats, 3, 1, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, hf, lf):
        original_hf = hf
        original_lf = lf
        hf = self.res(hf)
        lf = self.trans(lf)
        # original_lf = lf
        # original_hf = hf

        x = torch.cat([hf, lf], dim = 1)
        conv_lf = self.sigmoid(self.conv_lf(x))
        conv_hf = self.sigmoid(self.conv_hf(x))

        lf = conv_lf * lf + original_lf
        hf = conv_hf * hf + original_hf
        return hf, lf


class HLGroup(nn.Module):
    def __init__(self, n_blocks, n_feats, n_res_blocks):
        super(HLGroup, self).__init__()
        for i in range(n_blocks):
            self.add_module('deep_feature_' + str(i),
                            HLBlock(n_feats, n_res_blocks)
                            )
        self.num_blocks = n_blocks

    def forward(self, hf, lf):
        for i in range(self.num_blocks):
            hf, lf = eval('self.deep_feature_' + str(i))(hf, lf)
        return hf, lf


class SRModel(nn.Module):
    def __init__(self, args):  # 8,5
        super(SRModel, self).__init__()
        ###############################################
        ######################params###################
        ###############################################
        self.in_channels = args.in_channels
        self.n_feats = args.n_feats
        self.down_sample = args.down_sample
        self.scale_factor = args.scale_factor
        self.img_range = args.img_range
        self.n_res_blocks = args.n_res_blocks
        self.n_blocks = args.n_blocks
        m = torch.load('./OctaveConv_TVConv_BEST_SELF_resi_x4.pth')
        self.feature_extractor = FeatureExtractor().cuda()
        self.feature_extractor.load_state_dict(m["model_state_dict"])
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        ################################################
        ##################scale shift###################
        ################################################
        if self.in_channels == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)

        ################################################
        ############shallow feature extractor###########
        ################################################
        self.shallow_feature_extractor_lf = nn.Conv2d(self.in_channels, self.n_feats, 3, 1, 1)
        self.shallow_feature_extractor_hf = nn.Conv2d(self.in_channels, self.n_feats, 3, 1, 1)

        self.global_residual = nn.Conv2d(self.in_channels, self.n_feats, 3, 1, 1)

        ################################################
        ##############deep feature extractor############
        ################################################
        self.HL = HLGroup(self.n_blocks, self.n_feats, self.n_res_blocks)

        #################################################
        ###################upsample######################
        #################################################

        self.upsample = nn.Sequential(
            Upsampler(self.scale_factor, self.n_feats * 2),
            nn.Conv2d(self.n_feats * 2, self.in_channels, 3, 1, 1)
        )

        self.shuffle = nn.Conv2d(self.n_feats * 2, self.n_feats * 2, 3, 1, 1)

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.scale_factor - h % self.scale_factor) % self.scale_factor
        mod_pad_w = (self.scale_factor - w % self.scale_factor) % self.scale_factor
        pad_h = (self.down_sample - (h + mod_pad_h) % self.down_sample) % self.down_sample
        pad_w = (self.down_sample - (w + mod_pad_w) % self.down_sample) % self.down_sample
        x = F.pad(x, (0, mod_pad_w + pad_w, 0, mod_pad_h + pad_h), 'reflect')
        return x

    def forward(self, x):
        lf = torch.clamp(self.feature_extractor(x), 0, 1)
        hf = x - lf
        B, _, H, W = x.size()
        lf = self.check_image_size(lf)
        hf = self.check_image_size(hf)
        x = self.check_image_size(x)
        _, _, CH, CW = x.size()
        ##################################################
        ######################data scaling################
        ##################################################
        self.mean = self.mean.type_as(x)
        lf = (lf - self.mean) * self.img_range
        hf = (hf - self.mean) * self.img_range
        x = (x - self.mean) * self.img_range

        ##################################################
        #############shallow feature extractor############
        ##################################################
        global_residual = self.global_residual(x)

        lf = self.shallow_feature_extractor_lf(lf)
        connection_lf = lf

        hf = self.shallow_feature_extractor_hf(hf)
        connection_hf = hf

        ###################################################
        ###############deep feature extractor##############
        ###################################################
        hf, lf = self.HL(hf, lf)
        ###################################################
        ################frequency concate##################
        #################### ###############################
        lf = lf + connection_lf
        hf = hf + connection_hf
        x = lf + hf
        x = self.shuffle(torch.cat([x, global_residual], dim=1))
        ###################################################
        ####################upsampling#####################
        ###################################################
        x = self.upsample(x)

        ###################################################
        #####################scaling data##################
        ###################################################
        x = x / self.img_range + self.mean
        return x[:, :, :H * self.scale_factor, :W * self.scale_factor]