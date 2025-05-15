import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import math
import numpy as np
from functools import partial
from typing import Optional, Callable, Optional, Dict, Union
from einops import rearrange
from collections import OrderedDict
from ..modules.conv import Conv, DWConv
from pytorch_wavelets import DWTInverse, DWTForward
from torch import Tensor
__all__ = ['PatchEmbed','SimFusion_3in', 'SimFusion_4in', 'IFM', 'InjectionMultiSum_Auto_pool', 'PyramidPoolAgg', 'AdvPoolFusion', 'TopBasicLayer',
           'PatchEmbed','SimFusion_3in_DW','AdvPoolFusion_DW',
           'InjectionMultiSum_Auto_pool_dw'
           ]

class PatchEmbed(nn.Module):

    def __init__(self, patch_size, patch_stride, in_chans, embed_dim, norm_layer):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_stride, bias=False)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(self.proj(x))
        return x



class PatchMerging(nn.Module):

    def __init__(self, patch_size2, patch_stride2, dim, norm_layer):
        super().__init__()
        self.reduction = nn.Conv2d(dim, 2 * dim, kernel_size=patch_size2, stride=patch_stride2, bias=False)
        if norm_layer is not None:
            self.norm = norm_layer(2 * dim)
        else:
            self.norm = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(self.reduction(x))
        return x

class Partial_conv3(nn.Module):

    def __init__(self, dim, n_div, forward):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)

        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self, x: Tensor) -> Tensor:
        # only for inference
        x = x.clone()  # !!! Keep the original input intact for the residual connection later
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])

        return x

class MLPBlock(nn.Module):

    def __init__(self,
                 dim,
                 n_div,
                 mlp_ratio,
                 drop_path,
                 layer_scale_init_value,
                 act_layer,
                 norm_layer,
                 pconv_fw_type
                 ):

        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.n_div = n_div

        mlp_hidden_dim = int(dim * mlp_ratio)

        mlp_layer: List[nn.Module] = [
            nn.Conv2d(dim, mlp_hidden_dim, 1, bias=False),
            norm_layer(mlp_hidden_dim),
            act_layer(),
            nn.Conv2d(mlp_hidden_dim, dim, 1, bias=False)
        ]

        self.mlp = nn.Sequential(*mlp_layer)

        self.spatial_mixing = Partial_conv3(
            dim,
            n_div,
            pconv_fw_type
        )

        if layer_scale_init_value > 0:
            self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.forward = self.forward_layer_scale
        else:
            self.forward = self.forward

    def forward(self, x: Tensor) -> Tensor:
        shortcut = x
        x = self.spatial_mixing(x)
        x = shortcut + self.drop_path(self.mlp(x))
        return x

    def forward_layer_scale(self, x: Tensor) -> Tensor:
        shortcut = x
        x = self.spatial_mixing(x)
        x = shortcut + self.drop_path(
            self.layer_scale.unsqueeze(-1).unsqueeze(-1) * self.mlp(x))
        return x


class BasicStage(nn.Module):

    def __init__(self,
                 dim,
                 depth,
                 n_div,
                 mlp_ratio,
                 drop_path,
                 layer_scale_init_value,
                 norm_layer,
                 act_layer,
                 pconv_fw_type
                 ):
        super().__init__()

        blocks_list = [
            MLPBlock(
                dim=dim,
                n_div=n_div,
                mlp_ratio=mlp_ratio,
                drop_path=drop_path[i],
                layer_scale_init_value=layer_scale_init_value,
                norm_layer=norm_layer,
                act_layer=act_layer,
                pconv_fw_type=pconv_fw_type
            )
            for i in range(depth)
        ]

        self.blocks = nn.Sequential(*blocks_list)

    def forward(self, x: Tensor) -> Tensor:
        x = self.blocks(x)
        return x


####  gold 

def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1, bias=False):
    '''Basic cell for rep-style block, including conv and bn'''
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                        kernel_size=kernel_size, stride=stride, padding=padding, groups=groups,
                                        bias=bias))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    return result

class RepVGGBlock(nn.Module):
    '''RepVGGBlock is a basic rep-style block, including training and deploy status
    This code is based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    '''
    
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, dilation=1, groups=1, padding_mode='zeros', deploy=False, use_se=False):
        super(RepVGGBlock, self).__init__()
        """ Initialization of the class.
        Args:
            in_channels (int): Number of channels in the input image
            out_channels (int): Number of channels produced by the convolution
            kernel_size (int or tuple): Size of the convolving kernel
            stride (int or tuple, optional): Stride of the convolution. Default: 1
            padding (int or tuple, optional): Zero-padding added to both sides of
                the input. Default: 1
            dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
            groups (int, optional): Number of blocked connections from input
                channels to output channels. Default: 1
            padding_mode (string, optional): Default: 'zeros'
            deploy: Whether to be deploy status or training status. Default: False
            use_se: Whether to use se. Default: False
        """
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        assert kernel_size == 3
        assert padding == 1
        
        padding_11 = padding - kernel_size // 2
        
        self.nonlinearity = nn.ReLU()
        
        if use_se:
            raise NotImplementedError("se block not supported yet")
        else:
            self.se = nn.Identity()
        
        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding, dilation=dilation, groups=groups, bias=True,
                                         padding_mode=padding_mode)
        
        else:
            self.rbr_identity = nn.BatchNorm2d(
                    num_features=in_channels) if out_channels == in_channels and stride == 1 else None
            self.rbr_dense = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                     stride=stride, padding=padding, groups=groups)
            self.rbr_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride,
                                   padding=padding_11, groups=groups)
    
    def forward(self, inputs):
        '''Forward process'''
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)))
        
        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)
        
        return self.nonlinearity(self.se(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out))
    
    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid
    
    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])
    
    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std
    
    def switch_to_deploy(self):
        if hasattr(self, 'rbr_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(in_channels=self.rbr_dense.conv.in_channels,
                                     out_channels=self.rbr_dense.conv.out_channels,
                                     kernel_size=self.rbr_dense.conv.kernel_size, stride=self.rbr_dense.conv.stride,
                                     padding=self.rbr_dense.conv.padding, dilation=self.rbr_dense.conv.dilation,
                                     groups=self.rbr_dense.conv.groups, bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        self.deploy = True

from pytorch_wavelets import DWTForward

class Down_wt(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down_wt, self).__init__()
        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        self.conv_bn_relu = nn.Sequential(
                                    nn.Conv2d(in_ch*4, out_ch, kernel_size=1, stride=1),
                                    nn.BatchNorm2d(out_ch),
                                    nn.ReLU(inplace=True),
                                    )
    def forward(self, x):
        yL, yH = self.wt(x)
        y_HL = yH[0][:,:,0,::]
        y_LH = yH[0][:,:,1,::]
        y_HH = yH[0][:,:,2,::]
        x = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)
        x = self.conv_bn_relu(x)

        return x

def onnx_AdaptiveAvgPool2d(x, output_size):
    stride_size = np.floor(np.array(x.shape[-2:]) / output_size).astype(np.int32)
    kernel_size = np.array(x.shape[-2:]) - (output_size - 1) * stride_size
    avg = nn.AvgPool2d(kernel_size=list(kernel_size), stride=list(stride_size))
    x = avg(x)
    return x

def get_avg_pool():
    if torch.onnx.is_in_onnx_export():
        avg_pool = onnx_AdaptiveAvgPool2d
    else:
        avg_pool = nn.functional.adaptive_avg_pool2d
    return avg_pool

class DW_down(nn.Module):
    def __init__(self,in_channel,out_channel):
        super().__init__()
        self.dw = DWTForward(J=1, mode='zero', wave='haar')
        self.cv1 = DWConv(in_channel, in_channel*4, 3, 2,1)
        self.cv2 = Conv(in_channel*4, out_channel, 1,act=nn.ReLU())
    def forward(self,x):
        yL, yH = self.dw(x)
        y_HL = yH[0][:, :, 0, ::]
        y_LH = yH[0][:, :, 1, ::]
        y_HH = yH[0][:, :, 2, ::]
        x1 = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)
        x2 = self.cv1(x)
        return self.cv2(x1+x2)
class SimFusion_3in_DW(nn.Module):
    def __init__(self, in_channel_list, out_channels):
        super().__init__()
        # self.cv1 = Conv(in_channel_list[0], out_channels, act=nn.ReLU()) if in_channel_list[
        #                                                                         0] != out_channels else nn.Identity()
        self.cv2 = Conv(in_channel_list[1], out_channels, act=nn.ReLU()) if in_channel_list[
                                                                                1] != out_channels else nn.Identity()
        self.cv3 = Conv(in_channel_list[2], out_channels, act=nn.ReLU()) if in_channel_list[
                                                                                2] != out_channels else nn.Identity()
        self.cv_fuse = Conv(out_channels * 3, out_channels, act=nn.ReLU())
        #self.downsample = nn.functional.adaptive_avg_pool2d
        self.down = DW_down(in_channel_list[0],out_channels)

    def forward(self, x):
        N, C, H, W = x[1].shape
        output_size = (H, W)

        if torch.onnx.is_in_onnx_export():
            self.downsample = onnx_AdaptiveAvgPool2d
            output_size = np.array([H, W])

        x0 = self.down(x[0])
        x1 = self.cv2(x[1])
        x2 = self.cv3(F.interpolate(x[2], size=(H, W), mode='bilinear', align_corners=False))

        return self.cv_fuse(torch.cat((x0, x1, x2), dim=1))


class SimFusion_3in(nn.Module):
    def __init__(self, in_channel_list, out_channels):
        super().__init__()
        self.cv1 = Conv(in_channel_list[0], out_channels, act=nn.ReLU()) if in_channel_list[0] != out_channels else nn.Identity()
        self.cv2 = Conv(in_channel_list[1], out_channels, act=nn.ReLU()) if in_channel_list[1] != out_channels else nn.Identity()
        self.cv3 = Conv(in_channel_list[2], out_channels, act=nn.ReLU()) if in_channel_list[2] != out_channels else nn.Identity()
        self.cv_fuse = Conv(out_channels * 3, out_channels, act=nn.ReLU())
        self.downsample = nn.functional.adaptive_avg_pool2d
    
    def forward(self, x):
        N, C, H, W = x[1].shape
        output_size = (H, W)
        
        if torch.onnx.is_in_onnx_export():
            self.downsample = onnx_AdaptiveAvgPool2d
            output_size = np.array([H, W])
        
        x0 = self.cv1(self.downsample(x[0], output_size))
        x1 = self.cv2(x[1])
        x2 = self.cv3(F.interpolate(x[2], size=(H, W), mode='bilinear', align_corners=False))

        return self.cv_fuse(torch.cat((x0, x1, x2), dim=1))

class SimFusion_4in(nn.Module):
    def __init__(self):
        super().__init__()
        self.avg_pool = nn.functional.adaptive_avg_pool2d
    
    def forward(self, x):
        #print('block 2905',len(x))
        #breakpoint()
        x_l, x_m, x_s, x_n = x
        B, C, H, W = x_s.shape
        output_size = np.array([H, W])
        
        if torch.onnx.is_in_onnx_export():
            self.avg_pool = onnx_AdaptiveAvgPool2d
        
        x_l = self.avg_pool(x_l, output_size)
        x_m = self.avg_pool(x_m, output_size)
        x_n = F.interpolate(x_n, size=(H, W), mode='bilinear', align_corners=False)
        
        out = torch.cat([x_l, x_m, x_s, x_n], 1)

        return out

class IFM(nn.Module):
    def __init__(self, inc, ouc, embed_dim_p=96, fuse_block_num=3) -> None:
        super().__init__()
        
        self.conv = nn.Sequential(
            Conv(inc, embed_dim_p),
            *[RepVGGBlock(embed_dim_p, embed_dim_p) for _ in range(fuse_block_num)],
            Conv(embed_dim_p, sum(ouc))
        )
    
    def forward(self, x):
        return self.conv(x)

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)
    
    def forward(self, x):
        return self.relu(x + 3) / 6

class InjectionMultiSum_Auto_pool(nn.Module):
    def __init__(
            self,
            inp: int,
            oup: int,
            global_inp: list,
            flag: int
    ) -> None:
        super().__init__()
        self.global_inp = global_inp
        self.flag = flag
        self.local_embedding = Conv(inp, oup, 1, act=False)
        self.global_embedding = Conv(global_inp[self.flag], oup, 1, act=False)
        self.global_act = Conv(global_inp[self.flag], oup, 1, act=False)
        self.act = h_sigmoid()
    
    def forward(self, x):
        '''
        x_g: global features
        x_l: local features
        '''
        x_l, x_g = x #; print(x_l.shape, x_g.shape);breakpoint() # fusion_3in, IFM
        B, C, H, W = x_l.shape
        g_B, g_C, g_H, g_W = x_g.shape
        use_pool = H < g_H
        
        gloabl_info = x_g.split(self.global_inp, dim=1)[self.flag]  # split to 64 and 32
        
        local_feat = self.local_embedding(x_l) # ([2, 512, 40, 40])
        
        global_act = self.global_act(gloabl_info)  # ([2, 512, 40, 40])
        global_feat = self.global_embedding(gloabl_info) # ([2, 512, 40, 40])
        
        if use_pool:
            avg_pool = get_avg_pool()
            output_size = np.array([H, W])
            
            sig_act = avg_pool(global_act, output_size)
            global_feat = avg_pool(global_feat, output_size)
        
        else:
            sig_act = F.interpolate(self.act(global_act), size=(H, W), mode='bilinear', align_corners=False)
            global_feat = F.interpolate(global_feat, size=(H, W), mode='bilinear', align_corners=False)
        
        out = local_feat * sig_act + global_feat
        return out


class InjectionMultiSum_Auto_pool_dw(nn.Module):
    def __init__(
            self,
            inp: int,
            oup: int,
            global_inp: list,
            flag: int,
            flag1: int
    ) -> None:
        super().__init__()
        self.global_inp = global_inp
        self.flag = flag
        self.local_embedding = Conv(inp, oup, 1, act=False)
        self.global_embedding = Conv(global_inp[self.flag]//2, oup, 1, act=False)
        self.global_act = Conv(global_inp[self.flag]//2, oup, 1, act=False)
        self.act = h_sigmoid()
        self.flag1 = flag1

    def forward(self, x):
        '''
        x_g: global features
        x_l: local features
        '''
        x_l, x_g = x  # ; print(x_l.shape, x_g.shape);breakpoint() # fusion_3in, IFM
        B, C, H, W = x_l.shape
        g_B, g_C, g_H, g_W = x_g.shape
        use_pool = H < g_H
        x_g=x_g.split(self.global_inp, dim=1)[self.flag]
        gloabl_info = x_g.split(self.global_inp[self.flag]//2, dim=1)[self.flag1]  # split to 64 and 32

        local_feat = self.local_embedding(x_l)  # ([2, 512, 40, 40])

        global_act = self.global_act(gloabl_info)  # ([2, 512, 40, 40])
        global_feat = self.global_embedding(gloabl_info)  # ([2, 512, 40, 40])

        if use_pool:
            avg_pool = get_avg_pool()
            output_size = np.array([H, W])

            sig_act = avg_pool(global_act, output_size)
            global_feat = avg_pool(global_feat, output_size)

        else:
            sig_act = F.interpolate(self.act(global_act), size=(H, W), mode='bilinear', align_corners=False)
            global_feat = F.interpolate(global_feat, size=(H, W), mode='bilinear', align_corners=False)

        out = local_feat * sig_act + global_feat
        return out
def get_shape(tensor):
    shape = tensor.shape
    if torch.onnx.is_in_onnx_export():
        shape = [i.cpu().numpy() for i in shape]
    return shape

class PyramidPoolAgg(nn.Module):
    def __init__(self, inc, ouc, stride, pool_mode='torch'):
        super().__init__()
        self.stride = stride
        if pool_mode == 'torch':
            self.pool = nn.functional.adaptive_avg_pool2d
        elif pool_mode == 'onnx':
            self.pool = onnx_AdaptiveAvgPool2d
        self.conv = Conv(inc, ouc)
    
    def forward(self, inputs):
        B, C, H, W = get_shape(inputs[-1])
        H = (H - 1) // self.stride + 1
        W = (W - 1) // self.stride + 1
        
        output_size = np.array([H, W])
        
        if not hasattr(self, 'pool'):
            self.pool = nn.functional.adaptive_avg_pool2d
        
        if torch.onnx.is_in_onnx_export():
            self.pool = onnx_AdaptiveAvgPool2d
        
        out = [self.pool(inp, output_size) for inp in inputs]

        return self.conv(torch.cat(out, dim=1))

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = Conv(in_features, hidden_features, act=False)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, bias=True, groups=hidden_features)
        self.act = nn.ReLU6()
        self.fc2 = Conv(hidden_features, out_features, act=False)
        self.drop = nn.Dropout(drop)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class ReAttention(nn.Module):
    """
    It is observed that similarity along same batch of data is extremely large.
    Thus can reduce the bs dimension when calculating the attention map.
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., expansion_ratio=3,
                 apply_transform=True, transform_scale=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.apply_transform = apply_transform

        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        if apply_transform:
            self.reatten_matrix = nn.Conv2d(self.num_heads, self.num_heads, 1, 1)
            self.var_norm = nn.BatchNorm2d(self.num_heads)
            self.qkv = nn.Linear(dim, dim * expansion_ratio, bias=qkv_bias)
            self.reatten_scale = self.scale if transform_scale else 1.0
        else:
            self.qkv = nn.Linear(dim, dim * expansion_ratio, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, atten=None):
        B, N, C = x.shape
        # x = self.fc(x)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        if self.apply_transform:
            attn = self.var_norm(self.reatten_matrix(attn)) * self.reatten_scale
        attn_next = attn
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class GOLDYOLO_Attention(torch.nn.Module):
    def __init__(self, dim, key_dim, num_heads, attn_ratio=4):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads  # num_head key_dim
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio
        
        self.to_q = Conv(dim, nh_kd, 1, act=False)
        self.to_k = Conv(dim, nh_kd, 1, act=False)
        self.to_v = Conv(dim, self.dh, 1, act=False)
        
        self.proj = torch.nn.Sequential(nn.ReLU6(), Conv(self.dh, dim, act=False))
    
    def forward(self, x):  # x (B,N,C)
        B, C, H, W = get_shape(x)
        
        qq = self.to_q(x).reshape(B, self.num_heads, self.key_dim, H * W).permute(0, 1, 3, 2)
        kk = self.to_k(x).reshape(B, self.num_heads, self.key_dim, H * W)
        vv = self.to_v(x).reshape(B, self.num_heads, self.d, H * W).permute(0, 1, 3, 2)
        
        attn = torch.matmul(qq, kk)
        attn = attn.softmax(dim=-1)  # dim = k
        
        xx = torch.matmul(attn, vv)
        
        xx = xx.permute(0, 1, 3, 2).reshape(B, self.dh, H, W)
        xx = self.proj(xx)
        return xx
from memory_efficient_attention_pytorch import Attention
class top_Block(nn.Module):
    
    def __init__(self, dim, key_dim, num_heads, mlp_ratio=4., attn_ratio=2., drop=0.,
                 drop_path=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        
        self.attn = GOLDYOLO_Attention(dim, key_dim=key_dim, num_heads=num_heads, attn_ratio=attn_ratio)
        #self.attn = Attention(dim=dim, heads=num_heads)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)
    
    def forward(self, x1):
        x1 = x1 + self.drop_path(self.attn(x1))
        x1 = x1 + self.drop_path(self.mlp(x1))
        return x1

class TopBasicLayer(nn.Module):
    def __init__(self, embedding_dim, ouc_list, block_num=2, key_dim=8, num_heads=4,
                 mlp_ratio=4., attn_ratio=2., drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        self.block_num = block_num
        
        self.transformer_blocks = nn.ModuleList()
        for i in range(self.block_num):
            self.transformer_blocks.append(top_Block(
                    embedding_dim, key_dim=key_dim, num_heads=num_heads,
                    mlp_ratio=mlp_ratio, attn_ratio=attn_ratio,
                    drop=drop, drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path))
        self.conv = nn.Conv2d(embedding_dim, sum(ouc_list), 1)
        
    def forward(self, x):
        # token * N 
        for i in range(self.block_num):
            x = self.transformer_blocks[i](x)
        return self.conv(x)

class AdvPoolFusion(nn.Module):
    def forward(self, x):
        x1, x2 = x
        if torch.onnx.is_in_onnx_export():
            self.pool = onnx_AdaptiveAvgPool2d
        else:
            self.pool = nn.functional.adaptive_avg_pool2d
        
        N, C, H, W = x2.shape
        output_size = np.array([H, W])
        x1 = self.pool(x1, output_size)
        
        return torch.cat([x1, x2], 1)

class DW_up(nn.Module):
    def __init__(self):
        super().__init__()
        self.idw = DWTInverse(wave='haar',mode='zero')

    def forward(self,x2):

        x2h = []
        x2 = x2.reshape((x2.shape[0], x2.shape[1]//4, 4, x2.shape[-2], x2.shape[-1]))
        x2l = x2[:, :, 0].contiguous()
        x2h.append(x2[:, :, 1:].contiguous())
        y = self.idw((x2l, x2h))
        return y

class AdvPoolFusion_DW(nn.Module):
    def __init__(self,in_channel,out_channel):
        super().__init__()
        self.dw_up = DW_up()
        self.dwconv = DWConv(in_channel[-1],in_channel[-1]//4,k=3)
    def forward(self, x):
        x1, x2 = x

        if torch.onnx.is_in_onnx_export():
            self.pool = onnx_AdaptiveAvgPool2d
        else:
            self.pool = nn.functional.adaptive_avg_pool2d

        B, C, H, W = x1.shape
        output_size = np.array([H, W])
        #x1 = self.pool(x1, output_size)
        x2_1 = self.dw_up(x2)
        x2_2 = self.dwconv(self.pool(x2, output_size))
        return torch.cat([x1, x2_1,x2_2], 1)

######################################## GOLD-YOLO end ###