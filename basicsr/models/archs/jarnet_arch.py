from einops import rearrange
import math

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn import init as init


class LayerNormFunction(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None


class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

"""Coordinate Attention, Start"""
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction) # reduction needs set

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0) # inp=128
        self.bn1 = nn.BatchNorm2d(mip) # from batchnorm to layernorm
        self.act = h_swish()
        
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        identity = x
        
        n,c,h,w = x.size()
        #print(h,w)
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        #print(x_h.shape,x_w.shape)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) 
        #print(y.shape)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out
    
"""Coordinate Attention, End"""

"""Frequency Branch, Start"""
def window_partitions(x, window_size):
    """
    Args:
        x: (B, C, H, W)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, C, window_size, window_size)
    """
    if isinstance(window_size, int):
        window_size = [window_size, window_size]
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size[0], window_size[0], W // window_size[1], window_size[1])
    windows = x.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, C, window_size[0], window_size[1])
    return windows

def window_reverses(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, C, window_size, window_size)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, C, H, W)
    """
    # B = int(windows.shape[0] / (H * W / window_size / window_size))
    # print('B: ', B)
    # print(H // window_size)
    # print(W // window_size)
    if isinstance(window_size, int):
        window_size = [window_size, window_size]
    C = windows.shape[1]
    # print('C: ', C)
    x = windows.view(-1, H // window_size[0], W // window_size[1], C, window_size[0], window_size[1])
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous().view(-1, C, H, W)
    return x

def window_partitionx(x, window_size):
    _, _, H, W = x.shape
    h, w = window_size * (H // window_size), window_size * (W // window_size)
    x_main = window_partitions(x[:, :, :h, :w], window_size)
    b_main = x_main.shape[0]
    if h == H and w == W:
        return x_main, [b_main]
    if h != H and w != W:
        x_r = window_partitions(x[:, :, :h, -window_size:], window_size)
        b_r = x_r.shape[0] + b_main
        x_d = window_partitions(x[:, :, -window_size:, :w], window_size)
        b_d = x_d.shape[0] + b_r
        x_dd = x[:, :, -window_size:, -window_size:]
        b_dd = x_dd.shape[0] + b_d
        # batch_list = [b_main, b_r, b_d, b_dd]
        return torch.cat([x_main, x_r, x_d, x_dd], dim=0), [b_main, b_r, b_d, b_dd]
    if h == H and w != W:
        x_r = window_partitions(x[:, :, :h, -window_size:], window_size)
        b_r = x_r.shape[0] + b_main
        return torch.cat([x_main, x_r], dim=0), [b_main, b_r]
    if h != H and w == W:
        x_d = window_partitions(x[:, :, -window_size:, :w], window_size)
        b_d = x_d.shape[0] + b_main
        return torch.cat([x_main, x_d], dim=0), [b_main, b_d]
    
def window_reversex(windows, window_size, H, W, batch_list):
    h, w = window_size * (H // window_size), window_size * (W // window_size)
    # print(windows[:batch_list[0], ...].shape)
    x_main = window_reverses(windows[:batch_list[0], ...], window_size, h, w)
    B, C, _, _ = x_main.shape
    # print('windows: ', windows.shape)
    # print('batch_list: ', batch_list)
    if torch.is_complex(windows):
        res = torch.complex(torch.zeros([B, C, H, W]), torch.zeros([B, C, H, W]))
        res = res.to(windows.device)
    else:
        res = torch.zeros([B, C, H, W], device=windows.device)

    res[:, :, :h, :w] = x_main
    if h == H and w == W:
        return res
    if h != H and w != W and len(batch_list) == 4:
        x_dd = window_reverses(windows[batch_list[2]:, ...], window_size, window_size, window_size)
        res[:, :, h:, w:] = x_dd[:, :, h - H:, w - W:]
        x_r = window_reverses(windows[batch_list[0]:batch_list[1], ...], window_size, h, window_size)
        res[:, :, :h, w:] = x_r[:, :, :, w - W:]
        x_d = window_reverses(windows[batch_list[1]:batch_list[2], ...], window_size, window_size, w)
        res[:, :, h:, :w] = x_d[:, :, h - H:, :]
        return res
    if w != W and len(batch_list) == 2:
        x_r = window_reverses(windows[batch_list[0]:batch_list[1], ...], window_size, h, window_size)
        res[:, :, :h, w:] = x_r[:, :, :, w - W:]
    if h != H and len(batch_list) == 2:
        x_d = window_reverses(windows[batch_list[0]:batch_list[1], ...], window_size, window_size, w)
        res[:, :, h:, :w] = x_d[:, :, h - H:, :]
    return res


class fft_bench_complex_mlp(nn.Module):
    def __init__(self, dim, dw=1, norm='backward', act_method=nn.ReLU, window_size=None, bias=False):
        super(fft_bench_complex_mlp, self).__init__()
        self.act_fft = act_method()
        self.window_size = window_size

        hid_dim = dim * dw

        self.complex_weight1_real = nn.Parameter(torch.Tensor(dim, hid_dim))
        self.complex_weight1_imag = nn.Parameter(torch.Tensor(dim, hid_dim))
        self.complex_weight2_real = nn.Parameter(torch.Tensor(hid_dim, dim))
        self.complex_weight2_imag = nn.Parameter(torch.Tensor(hid_dim, dim))
        init.kaiming_uniform_(self.complex_weight1_real, a=math.sqrt(16))
        init.kaiming_uniform_(self.complex_weight1_imag, a=math.sqrt(16))
        init.kaiming_uniform_(self.complex_weight2_real, a=math.sqrt(16))
        init.kaiming_uniform_(self.complex_weight2_imag, a=math.sqrt(16))
        if bias:
            self.b1_real = nn.Parameter(torch.zeros((1, 1, 1, hid_dim)), requires_grad=True)
            self.b1_imag = nn.Parameter(torch.zeros((1, 1, 1, hid_dim)), requires_grad=True)
            self.b2_real = nn.Parameter(torch.zeros((1, 1, 1, dim)), requires_grad=True)
            self.b2_imag = nn.Parameter(torch.zeros((1, 1, 1, dim)), requires_grad=True)
        self.bias = bias
        self.norm = norm

    def forward(self, x):
        _, _, H, W = x.shape
        if self.window_size is not None and (H != self.window_size or W != self.window_size):
            x, batch_list = window_partitionx(x, self.window_size)
        y = torch.fft.rfft2(x, norm=self.norm)
        dim = 1
        weight1 = torch.complex(self.complex_weight1_real, self.complex_weight1_imag)
        weight2 = torch.complex(self.complex_weight2_real, self.complex_weight2_imag)
        if self.bias:
            b1 = torch.complex(self.b1_real, self.b1_imag)
            b2 = torch.complex(self.b2_real, self.b2_imag)
        y = rearrange(y, 'b c h w -> b h w c')
        y = y @ weight1
        if self.bias:
            y = y + b1
        y = torch.cat([y.real, y.imag], dim=dim)

        y = self.act_fft(y)
        y_real, y_imag = torch.chunk(y, 2, dim=dim)
        y = torch.complex(y_real, y_imag)
        y = y @ weight2
        if self.bias:
            y = y + b2
        y = rearrange(y, 'b h w c -> b c h w')

        if self.window_size is not None and (H != self.window_size or W != self.window_size):
            y = torch.fft.irfft2(y, s=(self.window_size, self.window_size), norm=self.norm)
            y = window_reversex(y, self.window_size, H, W, batch_list)
        else:
            y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        return y
"""Frequency Branch, End"""

class SFResBlock(nn.Module):
    def __init__(self, c, num_heads=1, window_size=8, window_size_fft=-1, shift_size=-1, DW_Expand=2, FFN_Expand=2,
                 drop_out_rate=0., sin=True):
        super().__init__()
        dw_channel = c * DW_Expand
        self.sin = sin
        # print(sin)
        self.window_size = window_size
        self.window_size_fft = window_size_fft
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Coordinate Attention (CoA) Block
        self.coa = CoordAtt(inp=dw_channel // 2, oup=dw_channel // 2)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.sg = SimpleGate()

        if window_size_fft is None or window_size_fft >= 0:
            self.fft_block1 = fft_bench_complex_mlp(c, DW_Expand, window_size=window_size_fft, bias=True, act_method=nn.GELU) # , act_method=nn.GELU
            self.fft_block2 = fft_bench_complex_mlp(c, DW_Expand, window_size=window_size_fft, bias=True, act_method=nn.GELU)
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()


    def forward(self, inp):
        x = inp

        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)

        x = self.coa(x)

        x = self.conv3(x)
        if self.window_size_fft is None or self.window_size_fft >= 0:
            if self.sin:
                x = x + torch.sin(self.fft_block1(inp))
            else:
                x = x + self.fft_block1(inp)
        x = self.dropout1(x)
        y = inp + x * self.beta

        if self.window_size_fft is None or self.window_size_fft >= 0:
            x = self.norm2(y)
            if self.sin:
                x = torch.sin(self.fft_block2(y)) + self.conv5(self.sg(self.conv4(x)))
            else:
                x = self.fft_block2(y) + self.conv5(self.sg(self.conv4(x)))
        else:
            x = self.conv4(self.norm2(y))
            x = self.sg(x)
            x = self.conv5(x)

        x = self.dropout2(x)
        x = y + x * self.gamma
        return x


def flow_warp(x, flow, interp_mode='bilinear', padding_mode='zeros', align_corners=True):
    assert x.size()[-2:] == flow.size()[1:3]
    _, _, h, w = x.size()

    # create mesh grid
    grid_y, grid_x = torch.meshgrid(torch.arange(0, h).type_as(x), torch.arange(0, w).type_as(x))
    grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
    grid.requires_grad = False

    vgrid = grid + flow

    # scale grid to [-1,1]
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(w - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(h - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
    output = F.grid_sample(x, vgrid_scaled, mode=interp_mode, padding_mode=padding_mode, align_corners=align_corners)

    return output


class OFCBlock(nn.Module):
    def __init__(self):
        n_feat = 16
        bias = True
        self.flowin = nn.Sequential(nn.Conv2d(2, n_feat, kernel_size=3, padding=1, bias=bias),
                        nn.Conv2d(n_feat, n_feat*2, kernel_size=7, padding=3, bias=bias),
                        nn.Tanh(),
                        nn.Conv2d(n_feat*2, n_feat*4, kernel_size=7, padding=3, bias=bias),
                        nn.Conv2d(n_feat*4, n_feat*8, kernel_size=7, padding=3, bias=bias))
        self.sca = nn.Sequential(
                        nn.AdaptiveAvgPool2d(1),
                        nn.Conv2d(in_channels=n_feat*8, out_channels=n_feat*8, kernel_size=1, padding=0, stride=1,
                                groups=1, bias=True),
        )
        self.flowout = nn.Sequential(nn.Conv2d(n_feat*8, n_feat*4, kernel_size=7, padding=3, bias=bias),
                        nn.Conv2d(n_feat*4, n_feat*2, kernel_size=7, padding=3, bias=bias),
                        nn.Tanh(),
                        nn.Conv2d(n_feat*2, n_feat, kernel_size=7, padding=3, bias=bias),
                        nn.Conv2d(n_feat, 2, kernel_size=3, padding=1,bias=bias),)

    def forward(self, flow_inp):
        flow = self.flowin(flow_inp)
        flow = flow * self.sca(flow)
        flow = self.flowout(flow)
        flow = flow + flow_inp
        return flow


class JARNet(nn.Module):

    def __init__(self, img_channel=3, width=32, middle_blk_num=1, enc_blk_nums=[1, 1, 1, 28], dec_blk_nums=[1, 1, 1, 1],
                 window_size_e=[64,32,16,8], window_size_m=[8], window_size_e_fft=[64, 32, 16, -1], window_size_m_fft=[-1],
                 window_sizex_e=[8,8,8,8], window_sizex_m=[8], num_heads_e=[1, 2, 4, 8], num_heads_m=[16]):
        super().__init__()

        num_heads_d = num_heads_e[::-1]

        print(num_heads_e, window_size_e, window_size_e_fft, window_sizex_e)

        window_size_d = window_size_e[::-1]
        window_size_d_fft = window_size_e_fft[::-1]
        window_sizex_d = window_sizex_e[::-1]

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for i in range(len(enc_blk_nums)):
            self.encoders.append(
                nn.Sequential(
                    *[SFResBlock(chan, num_heads_e[i], window_size_e[i], window_size_e_fft[i], window_sizex_e[i]) for _ in range(enc_blk_nums[i])]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[SFResBlock(chan, num_heads_m[0], window_size_m[0], window_size_m_fft[0], window_sizex_m[0]) for _ in range(middle_blk_num)]
            )

        for j in range(len(dec_blk_nums)):
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2

            self.decoders.append(
                nn.Sequential(
                    *[SFResBlock(chan, num_heads_d[j], window_size_d[j], window_size_d_fft[j], window_sizex_d[j]) for _ in range(dec_blk_nums[j])]
                )
            )

        self.padder_size = 2 ** len(self.encoders)

        # Optical Flow Correction (OFC) Block 
        self.ofc = OFCBlock()
        
        self.feature_map = None # Only for visualizations

    def forward(self, inp, flow):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)
        flow_inp = self.check_image_size(flow)
        
        flow = self.ofc(flow_inp)
        
        inp = flow_warp(inp, -flow.permute(0,2,3,1).contiguous())

        if not self.training:
            self.feature_map = inp # Only for visualization

        x = self.intro(inp)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        x = x + inp

        return x[:, :, :H, :W], flow[:, :, :H, :W].permute(0,2,3,1).contiguous()

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x

def prepare_input(resolution):
    x1 = torch.FloatTensor(1, *resolution)
    x2 = torch.FloatTensor(1, 2, 256, 256)
    return dict(inp=x1, flow=x2)

if __name__ == '__main__':
    img_channel = 1
    width = 32

    enc_blks = [4, 4, 4, 4]
    middle_blk_num = 4
    dec_blks = [4, 4, 4, 4]
    
    net = JARNet(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num,
                      enc_blk_nums=enc_blks, dec_blk_nums=dec_blks, window_size_e_fft=[64, -1, -1, -1])


    inp_shape = (1, 256, 256)

    from ptflops import get_model_complexity_info
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    macs, params = get_model_complexity_info(net, inp_shape, input_constructor=prepare_input, verbose=False, print_per_layer_stat=False)

    params = float(params[:-1])
    macs = float(macs[:-4])

    print(macs, params)
