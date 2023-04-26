import torch.nn as nn
import torch
import math
from model import common
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers
from einops import rearrange
from torch.nn.parameter import Parameter
from torch.autograd import Variable
#from IPython import embed

def make_model(args, parent=False):
    return MODEL(args)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = Conv2d_CG(in_features, hidden_features, kernel_size=1, padding=0)
        self.act = act_layer()
        self.fc2 = Conv2d_CG(hidden_features, out_features, kernel_size=3, padding=1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Conv2d_CG(nn.Conv2d):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=1, padding=0, stride=1, dilation=1, groups=1,
                 bias=True):
        super(Conv2d_CG, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

        self.weight_conv = Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.001, requires_grad=True)
        self.bias_conv = Parameter(torch.Tensor(out_channels))
        nn.init.kaiming_normal_(self.weight_conv)

        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        if kernel_size == 0:
            self.ind = True
        else:
            self.ind = False
            self.oc = out_channels
            self.ks = kernel_size

            # target spatial size of the pooling layer
            ws = kernel_size
            self.avg_pool = nn.AdaptiveMaxPool2d((ws, ws))

            # the dimension of latent representation
            self.num_lat = int((kernel_size * kernel_size) / 2 + 1)

            # the context encoding module
            self.ce = nn.Linear(ws * ws, self.num_lat, False)

            self.act = nn.ReLU()

            # the number of groups in the channel interaction module
            if in_channels // 8:
                self.g = 8
            else:
                self.g = in_channels

            # the channel interacting module
            self.ci = nn.Linear(self.g, out_channels // (in_channels // self.g), bias=False)

            # the gate decoding module (spatial interaction)
            self.gd = nn.Linear(self.num_lat, kernel_size * kernel_size, False)
            self.gd2 = nn.Linear(self.num_lat, kernel_size * kernel_size, False)

            # used to prepare the input feature map to patches
            self.unfold = nn.Unfold(kernel_size, dilation, padding, stride)

            # sigmoid function
            self.sig = nn.Sigmoid()

    def forward(self, x):
        if self.ind:
            return F.conv2d(x, self.weight_conv, self.bias_conv, self.stride, self.padding, self.dilation, self.groups)
        else:
            b, c, h, w = x.size()  # x: batch x n_feat(=64) x h_patch x w_patch
            weight = self.weight_conv

            # allocate global information
            gl = self.avg_pool(x).view(b, c, -1)  # gl: batch x n_feat x 3 x 3 -> batch x n_feat x 9

            # context-encoding module
            out = self.ce(gl)  # out: batch x n_feat x 5

            # use different bn for following two branches
            ce2 = out  # ce2: batch x n_feat x 5
            out = self.act(out)  # out: batch x n_feat x 5 (just batch normalization)

            # gate decoding branch 1 (spatial interaction)
            out = self.gd(out)  # out: batch x n_feat x 9 (5 --> 9 = 3x3)

            # channel interacting module
            if self.g > 3:
                oc = self.ci(self.act(ce2.view(b, c // self.g, self.g, -1).transpose(2, 3))).transpose(2,3).contiguous()
            else:
                oc = self.ci(self.act(ce2.transpose(2, 1))).transpose(2, 1).contiguous()
            oc = oc.view(b, self.oc, -1)
            oc = self.act(oc)  # oc: batch x n_feat x 5 (after grouped linear layer)

            # gate decoding branch 2 (spatial interaction)
            oc = self.gd2(oc)  # oc: batch x n_feat x 9 (5 --> 9 = 3x3)

            # produce gate (equation (4) in the CRAN paper)
            out = self.sig(out.view(b, 1, c, self.ks, self.ks) + oc.view(b, self.oc, 1, self.ks, self.ks))  
            # out: batch x out_channel x in_channel x kernel_size x kernel_size (same dimension as conv2d weight)
            
            # unfolding input feature map to patches
            x_un = self.unfold(x)
            b, _, l = x_un.size()
            out = (out * weight.unsqueeze(0))#.to(device)
            out = out.view(b, self.oc, -1)

            # currently only handle square input and output
            return torch.matmul(out, x_un).view(b, self.oc, h, w)

class ConvAttention(nn.Module):
    def __init__(self, dim, num_heads=8, kernel_size=5, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.weight = nn.Parameter(torch.randn(num_heads, dim//num_heads, dim//num_heads) * 0.001, requires_grad=True)
        self.to_qkv = Conv2d_CG(dim, dim*3)

    def forward(self, x, k1=None, v1=None, return_x=False):
        weight = self.weight
        b,c,h,w = x.shape

        qkv = self.to_qkv(x)
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        if k1 is None:
            k = k
            v = v
        else:
            k = k1 + k
            v = v1 + v
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * weight
        attn = attn.softmax(dim=-1)
        x = (attn @ v)
        x = rearrange(x, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        if return_x:
            return x
        else:
            return x, k, v


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super(LayerNorm, self).__init__()
        self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU):
        super().__init__()
        self.norm1 = LayerNorm(dim)
        kernel_size1 = 1
        padding1 = 0
        kernel_size2 = 3
        padding2 = 1
        self.attn = ConvAttention(dim, num_heads, kernel_size1, padding1)
        self.attn1 = ConvAttention(dim, num_heads, kernel_size2, padding2)

        self.norm2 = LayerNorm(dim)
        self.norm3 = LayerNorm(dim)
        mlp_hidden_dim = int(dim*1)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim)

    def forward(self, x):
        res = x
        x, k1, v1 = self.attn(x)
        x = res + self.norm1(x)
        x = x + self.norm2(self.attn1(x, k1, v1, return_x=True))
        x = x + self.norm3(self.mlp(x))
        return x


class Scale(nn.Module):
    def __init__(self, init_value=1e-3):
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale


def activation(act_type, inplace=False, neg_slope=0.05, n_prelu=1):
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU()
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(neg_slope)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
    return layer
      
      
class eca_layer(nn.Module):
    def __init__(self, channel, k_size):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.k_size = k_size
        self.conv = nn.Conv1d(channel, channel, kernel_size=k_size, bias=False, groups=channel)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)
        y = nn.functional.unfold(y.transpose(-1, -3), kernel_size=(1, self.k_size), padding=(0, (self.k_size - 1) // 2))
        y = self.conv(y.transpose(-1, -2)).unsqueeze(-1)
        y = self.sigmoid(y)
        x = x * y.expand_as(x)
        return x           
      
       
class MaskPredictor(nn.Module):
    def __init__(self,in_channels, wn=lambda x: torch.nn.utils.weight_norm(x)):
        super(MaskPredictor,self).__init__()
        self.spatial_mask=nn.Conv2d(in_channels=in_channels,out_channels=3,kernel_size=1,bias=False)
        
    def forward(self,x):
        spa_mask=self.spatial_mask(x)
        spa_mask=F.gumbel_softmax(spa_mask,tau=1,hard=True,dim=1)
        return spa_mask 


class RB(nn.Module):
    def __init__(self, n_feats, wn=lambda x: torch.nn.utils.weight_norm(x)):
        super(RB, self).__init__()
        self.CA = eca_layer(n_feats, k_size=3)
        self.MaskPredictor = MaskPredictor(n_feats*8//8)      
        
        self.k = nn.Sequential(wn(nn.Conv2d(n_feats*8//8, n_feats*8//8, kernel_size=3, padding=1, stride=1, groups=1)),
                               nn.LeakyReLU(0.05),
                               )
                               
        self.k1 = nn.Sequential(wn(nn.Conv2d(n_feats*8//8, n_feats*8//8, kernel_size=3, padding=1, stride=1, groups=1)),
                                nn.LeakyReLU(0.05),
                               )
                                
        self.res_scale = Scale(1)
        self.x_scale = Scale(1)

    def forward(self, x): 
        res = x
        x = self.k(x)  
    
        MaskPredictor = self.MaskPredictor(x)
        mask = (MaskPredictor[:,1,...]).unsqueeze(1)
        x = x * (mask.expand_as(x))
    
        x1 = self.k1(x)
        x2  = self.CA(x1)
        out = self.x_scale(x2) + self.res_scale(res) 
        
        return out      
   
        
class SCConv(nn.Module):
    def __init__(self, n_feats, wn=lambda x: torch.nn.utils.weight_norm(x)):
        super(SCConv, self).__init__()
        pooling_r = 2
        med_feats = n_feats // 1                         
        self.k1 = nn.Sequential(nn.ConvTranspose2d(n_feats, n_feats*3//2, kernel_size=pooling_r, stride=pooling_r, padding=0, groups=1, bias=True),
                                nn.LeakyReLU(0.05),
                                nn.Conv2d(n_feats*3//2, n_feats, kernel_size=1, stride=2, padding=0, groups=1),
                                ) 
                                
        self.sig = nn.Sigmoid()                                        
                    
        self.k3 = RB(n_feats)
        
        self.k4 = RB(n_feats)
        
        self.k5 = RB(n_feats)
                    
        self.res_scale = Scale(1)
        self.x_scale = Scale(1)
        
    def forward(self, x):
        identity = x
        _, _, H, W = identity.shape
        x1_1 = self.k3(x)
        x1 = self.k4(x1_1)

        
        x1_s = self.sig(self.k1(x) + x)
        x1 = self.k5(x1_s * x1)
        
        out = self.res_scale(x1) + self.x_scale(identity)

        return out


class FCUUp(nn.Module):
    def __init__(self, inplanes, outplanes, up_stride, act_layer=nn.ReLU,
                 norm_layer=nn.BatchNorm2d, wn=lambda x: torch.nn.utils.weight_norm(x)):
        super(FCUUp, self).__init__()
        self.up_stride = up_stride
        self.conv_project = wn(nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0))
        self.act = act_layer()

    def forward(self, x_t):
        x_r = self.act(self.conv_project(x_t))

        return x_r
        
class FCUDown(nn.Module):
    def __init__(self, inplanes, outplanes, dw_stride, act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, wn=lambda x: torch.nn.utils.weight_norm(x)):
        super(FCUDown, self).__init__()
        self.conv_project = wn(nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0))

    def forward(self, x):
        x = self.conv_project(x)

        return x


class ConvBlock(nn.Module):
    def __init__(self, inplanes, outplanes, stride=1, res_conv=False, act_layer=nn.ReLU, groups=1, norm_layer=nn.BatchNorm2d, drop_block=None, drop_path=None):
        super(ConvBlock, self).__init__()

        expansion = 1
        med_planes = outplanes // expansion
        embed_dim = 144
        num_heads = 8
        mlp_ratio = 1.0
        
        self.rb_search1 = SCConv(med_planes)
        self.rb_search2 = SCConv(med_planes)
        self.rb_search3 = SCConv(med_planes)
        self.rb_search4 = SCConv(med_planes)
        
        self.trans_block = Block(
            dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=False, qk_scale=None,
            drop=0., attn_drop=0., drop_path=0.)
            
        self.trans_block1 = Block(
            dim=embed_dim, num_heads=num_heads*3//2, mlp_ratio=mlp_ratio, qkv_bias=False, qk_scale=None,
            drop=0., attn_drop=0., drop_path=0.)
            
        self.trans_block2 = Block(
            dim=embed_dim, num_heads=num_heads*2, mlp_ratio=mlp_ratio, qkv_bias=False, qk_scale=None,
            drop=0., attn_drop=0., drop_path=0.)
            
        self.trans_block3 = Block(
            dim=embed_dim, num_heads=num_heads*3//2, mlp_ratio=mlp_ratio, qkv_bias=False, qk_scale=None,
            drop=0., attn_drop=0., drop_path=0.)
            
        self.trans_block4 = Block(
            dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=False, qk_scale=None,
            drop=0., attn_drop=0., drop_path=0.)
            
        self.trans_block5 = Block(
            dim=embed_dim, num_heads=num_heads*3//2, mlp_ratio=mlp_ratio, qkv_bias=False, qk_scale=None,
            drop=0., attn_drop=0., drop_path=0.)
            
        self.trans_block6 = Block(
            dim=embed_dim, num_heads=num_heads*2, mlp_ratio=mlp_ratio, qkv_bias=False, qk_scale=None,
            drop=0., attn_drop=0., drop_path=0.)
            
        self.trans_block7 = Block(
            dim=embed_dim, num_heads=num_heads*3//2, mlp_ratio=mlp_ratio, qkv_bias=False, qk_scale=None,
            drop=0., attn_drop=0., drop_path=0.)
                   
        self.expand_block = FCUUp(inplanes=med_planes, outplanes=embed_dim, up_stride=1)
        self.squeeze_block = FCUDown(inplanes=embed_dim, outplanes=med_planes, dw_stride=1)
        self.expand_block1 = FCUUp(inplanes=med_planes, outplanes=embed_dim, up_stride=1)
        self.squeeze_block1 = FCUDown(inplanes=embed_dim, outplanes=med_planes, dw_stride=1)
        self.expand_block2 = FCUUp(inplanes=med_planes, outplanes=embed_dim, up_stride=1)
        self.squeeze_block2 = FCUDown(inplanes=embed_dim, outplanes=med_planes, dw_stride=1)
        self.expand_block3 = FCUUp(inplanes=med_planes, outplanes=embed_dim, up_stride=1)
        self.squeeze_block3 = FCUDown(inplanes=embed_dim, outplanes=med_planes, dw_stride=1)
        
        self.res_scale = Scale(1)
        self.x_scale = Scale(1)
        self.num_rbs = 1

        self.res_conv = res_conv
        self.drop_block = drop_block
        self.drop_path = drop_path

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x):
        residual = x

        x = self.squeeze_block(self.trans_block(self.expand_block(self.rb_search1(x)))) + x
        
        x = self.squeeze_block(self.trans_block1(self.expand_block(self.rb_search1(x)))) + x
        
        x = self.squeeze_block1(self.trans_block2(self.expand_block1(self.rb_search2(x)))) + x
        
        x = self.squeeze_block1(self.trans_block3(self.expand_block1(self.rb_search2(x)))) + x
        
        x = self.squeeze_block2(self.trans_block4(self.expand_block2(self.rb_search3(x)))) + x
        
        x = self.squeeze_block2(self.trans_block5(self.expand_block2(self.rb_search3(x)))) + x
        
        x = self.squeeze_block3(self.trans_block6(self.expand_block3(self.rb_search4(x)))) + x
        
        x = self.squeeze_block3(self.trans_block7(self.expand_block3(self.rb_search4(x)))) + x

        x = self.x_scale(x) + self.res_scale(residual)
      
        return x


class ConvTransBlock(nn.Module):

    def __init__(self, inplanes, outplanes, res_conv, stride, dw_stride, embed_dim, num_heads, mlp_ratio,
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 last_fusion=False, num_med_block=0, groups=1):
        super(ConvTransBlock, self).__init__()
        expansion = 1
        self.cnn_block = ConvBlock(inplanes=inplanes, outplanes=outplanes, res_conv=res_conv, stride=1, groups=groups)

        self.dw_stride = dw_stride
        self.embed_dim = embed_dim
        self.num_med_block = num_med_block
        self.last_fusion = last_fusion
        self.res_scale = Scale(1)
        self.x_scale = Scale(1)

    def forward(self, x):
        x = self.cnn_block(x)

        return x


class MODEL(nn.Module):
    def __init__(self, args, norm_layer=nn.LayerNorm, patch_size=1, window_size=8, num_heads=8, mlp_ratio=1.,
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., num_med_block=0, drop_path_rate=0.,
                 patch_norm=True):
        super(MODEL, self).__init__()
        scale = args.scale
        n_feats = 48
        n_colors = 3
        embed_dim = 64

        self.patch_norm = patch_norm
        self.num_features = embed_dim
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(255, rgb_mean, rgb_std)
        self.add_mean = common.MeanShift(255, rgb_mean, rgb_std, 1)
        #self.conv_first_trans = nn.Conv2d(n_colors, embed_dim, 3, 1, 1)
        self.conv_first_cnn = nn.Conv2d(n_colors, n_feats, 3, 1, 1)
        
        self.trans_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, 8)]  # stochastic depth decay rule

        # 2~final Stage
        init_stage = 2
        fin_stage = 3
        stage_1_channel = n_feats
        trans_dw_stride = patch_size
        for i in range(init_stage, fin_stage):
            if i%2==0:
                m = i
            else:
                m = i-1
            self.add_module('conv_trans_' + str(m),
                            ConvTransBlock(
                                stage_1_channel, stage_1_channel, res_conv=True, stride=1, dw_stride=trans_dw_stride,
                                embed_dim=embed_dim,
                                num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                                drop_path_rate=self.trans_dpr[i - 1],
                                num_med_block=num_med_block
                            )
                            )

        self.fin_stage = fin_stage
        self.dw_stride = trans_dw_stride

        self.conv_after_body = nn.Conv2d(n_feats, n_feats, 3, 1, 1)
        
        m = []
        m.append(nn.Conv2d(n_feats, (scale[0] ** 2) * n_colors, 3, 1, 1))
        m.append(nn.PixelShuffle(scale[0]))
        self.UP1 = nn.Sequential(*m)
        
        self.conv_stright = nn.Conv2d(n_colors, n_feats, 3, 1, 1)
        up_body = []
        up_body.append(nn.Conv2d(n_feats, (scale[0] ** 2) * n_colors, 3, 1, 1))
        up_body.append(nn.PixelShuffle(scale[0]))
        self.UP2 = nn.Sequential(*up_body)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        (H, W) = (x.shape[2], x.shape[3])
        residual = x
        x = self.sub_mean(x)
        x = self.conv_first_cnn(x)
    
        for i in range(2, self.fin_stage):
            if i%2==0:
                m = i
            else:
                m = i-1
            x = eval('self.conv_trans_' + str(m))(x)

        x = self.conv_after_body(x)
        y1 = self.UP1(x)
        y2 = self.UP2(self.conv_stright(residual))
        output = self.add_mean(y1 + y2)

        return output