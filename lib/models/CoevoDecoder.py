import numpy as np
import torch
import os.path as osp
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.vision_transformer import _cfg, Mlp

from core.config import cfg
from graph_utils import build_verts_joints_relation
from models.backbones.mesh import Mesh


BASE_DATA_DIR = cfg.DATASET.BASE_DATA_DIR

class AdaLayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-6):
        super(AdaLayerNorm, self).__init__()
        self.mlp_gamma = nn.Linear(2048, num_features)
        self.mlp_beta = nn.Linear(2048, num_features)
        self.eps = eps

    def forward(self, x, img_feat):
        size = x.size()
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        gamma = self.mlp_gamma(img_feat).view(size[0], 1, -1).expand(size)
        beta = self.mlp_beta(img_feat).view(size[0], 1, -1).expand(size)
        return gamma * (x - mean) / (std + self.eps) + beta

class CrossAttention(nn.Module):
    def __init__(self, dim, v_dim, kv_num, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.kv_num = kv_num
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(v_dim, v_dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(v_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, xq, xk, xv):

        B, N, C = xq.shape
        v_dim = xv.shape[-1]
        q = self.wq(xq).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # [B,N1,C] -> [B,N1,H,(C/H)] -> [B,H,N1,(C/H)]
        k = self.wk(xk).reshape(B, self.kv_num, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # [B,N2,C] -> [B,N2,H,(C/H)] -> [B,H,N2,(C/H)]
        v = self.wv(xv).reshape(B, self.kv_num, self.num_heads, v_dim // self.num_heads).permute(0, 2, 1, 3)  # [B,N2,C] -> [B,N2,H,(C/H)] -> [B,H,N2,(C/H)]

        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B,H,N1,(C/H)] @ [B,H,(C/H),N2] -> [B,H,N1,N2]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, v_dim)   # [B,H,N1,N2] @ [B,H,N2,(C/H)] -> [B,H,N1,(C/H)] -> [B,N1,H,(C/H)] -> [B,N1,C]
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class CrossAttentionBlock(nn.Module):
    def __init__(self, q_dim, k_dim, v_dim, kv_num, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0.2, 
                 attn_drop=0.2, drop_path=0.2, act_layer=nn.GELU, norm_layer=AdaLayerNorm, has_mlp=True):
        super().__init__()
        self.normq = norm_layer(q_dim)
        self.normk = norm_layer(k_dim)
        self.normv = norm_layer(v_dim)
        self.kv_num = kv_num
        self.attn = CrossAttention(q_dim, v_dim, kv_num = kv_num, num_heads=num_heads, qkv_bias=qkv_bias, 
                                   qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.has_mlp = has_mlp
        if has_mlp:
            self.norm2 = norm_layer(q_dim)
            mlp_hidden_dim = int(q_dim * mlp_ratio)
            self.mlp = Mlp(in_features=q_dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, xq, xk, xv, img_feat):
        xq = xq + self.drop_path(self.attn(self.normq(xq, img_feat), self.normk(xk, img_feat), self.normv(xv, img_feat)))
        if self.has_mlp:
            xq = xq + self.drop_path(self.mlp(self.norm2(xq, img_feat)))

        return xq
    
class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=AdaLayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, img_feat):
        x = x + self.drop_path(self.attn(self.norm1(x, img_feat)))
        x = x + self.drop_path(self.mlp(self.norm2(x, img_feat)))
        return x
    
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class CoevoBlock(nn.Module):
    def __init__(self, num_joint, num_vertx, joint_dim=64, vertx_dim=64):
        super(CoevoBlock, self).__init__()

        self.num_joint = num_joint
        self.num_vertx = num_vertx
        joint_num_heads = 8
        vertx_num_heads = 2
        mlp_ratio = 4.
        drop = 0.
        attn_drop = 0.
        drop_path = 0.2
        qkv_bias = True
        qk_scale = None

        self.joint_proj = nn.Linear(3, joint_dim)
        self.vertx_proj = nn.Linear(3, vertx_dim)

        self.joint_pos_embed = nn.Parameter(torch.randn(1, self.num_joint, joint_dim))
        self.vertx_pos_embed = nn.Parameter(torch.randn(1, self.num_vertx, vertx_dim))

        self.j_Q_embed = nn.Parameter(torch.randn(1, self.num_joint, joint_dim))
        self.v_Q_embed = nn.Parameter(torch.randn(1, self.num_vertx, vertx_dim))

        self.proj_v2j_dim = nn.Linear(vertx_dim, joint_dim)
        self.proj_j2v_dim = nn.Linear(joint_dim, vertx_dim)
        self.v2j_K_embed = nn.Parameter(torch.randn(1, self.num_vertx, joint_dim))
        self.j2v_K_embed = nn.Parameter(torch.randn(1, self.num_joint, vertx_dim))

        self.joint_SA_FFN = Block(dim=joint_dim, num_heads=joint_num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, 
                                  drop=drop, attn_drop=attn_drop, drop_path=drop_path)
        self.vertx_SA_FFN = Block(dim=vertx_dim, num_heads=vertx_num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, 
                                  drop=drop, attn_drop=attn_drop, drop_path=drop_path)

        self.joint_CA_FFN = CrossAttentionBlock(q_dim=joint_dim, k_dim=joint_dim, v_dim=vertx_dim, kv_num = num_vertx, num_heads=joint_num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                                drop=drop, attn_drop=attn_drop, drop_path=drop_path, has_mlp=True)
        self.vertx_CA_FFN = CrossAttentionBlock(q_dim=vertx_dim, k_dim=vertx_dim, v_dim=joint_dim, kv_num = num_joint, num_heads=vertx_num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                                drop=drop, attn_drop=attn_drop, drop_path=drop_path, has_mlp=True)

        self.proj_joint_feat2coor = nn.Linear(joint_dim, 3)
        self.proj_vertx_feat2coor = nn.Linear(vertx_dim, 3)

    def forward(self, joint, vertx, img_feat):

        joint_feat, vertx_feat = self.joint_proj(joint), self.vertx_proj(vertx) # [B,17,3] -> [B,17,64], [B,431,3] -> [B,431,64]
        
        # pos_embed
        joint_feat, vertx_feat = joint_feat + self.joint_pos_embed, vertx_feat + self.vertx_pos_embed

        # CA + FFN
        joint_feat, vertx_feat = self.joint_CA_FFN(joint_feat + self.j_Q_embed, self.proj_v2j_dim(vertx_feat) + self.v2j_K_embed, vertx_feat, img_feat), \
                                 self.vertx_CA_FFN(vertx_feat + self.v_Q_embed, self.proj_j2v_dim(joint_feat) + self.j2v_K_embed, joint_feat, img_feat)

        # SA + FFN
        joint_feat, vertx_feat = self.joint_SA_FFN(joint_feat, img_feat), self.vertx_SA_FFN(vertx_feat, img_feat) # [B,17,64], [B,431,64]

        joint, vertx = self.proj_joint_feat2coor(joint_feat) + joint[:,:,:3], self.proj_vertx_feat2coor(vertx_feat) + vertx[:,:,:3] # [B,17,3], [B,431,3]

        return joint, vertx

class Pose2Mesh(nn.Module):
    def __init__(self, num_joint, embed_dim=256, SMPL_MEAN_vertices=osp.join(BASE_DATA_DIR, 'smpl_mean_vertices.npy')):
        super(Pose2Mesh, self).__init__()

        self.mesh = Mesh()
        
        # downsample mesh vertices from 6890 to 431
        init_vertices = torch.from_numpy(np.load(SMPL_MEAN_vertices)).cuda()
        downsample_verts_1723 = self.mesh.downsample(init_vertices)                    # [1723, 3]
        downsample_verts_431 = self.mesh.downsample(downsample_verts_1723, n1=1, n2=2) # [431, 3]
        self.register_buffer('init_vertices', downsample_verts_431)
        self.num_verts = downsample_verts_431.shape[0]

        # calculate the nearest joint of each vertex
        J_regressor = torch.from_numpy(np.load('data/Human36M/J_regressor_h36m_correct.npy').astype(np.float32)).cuda()
        self.joints_template = torch.matmul(J_regressor, init_vertices)
        self.vj_relation, self.jv_sets = build_verts_joints_relation(self.joints_template.cpu().numpy(), downsample_verts_431.cpu().numpy())

        self.coevoblock1 = CoevoBlock(num_joint, num_vertx=self.num_verts, joint_dim=cfg.MODEL.joint_dim, vertx_dim=cfg.MODEL.vertx_dim)
        self.coevoblock2 = CoevoBlock(num_joint, num_vertx=self.num_verts, joint_dim=cfg.MODEL.joint_dim, vertx_dim=cfg.MODEL.vertx_dim)
        self.coevoblock3 = CoevoBlock(num_joint, num_vertx=self.num_verts, joint_dim=cfg.MODEL.joint_dim, vertx_dim=cfg.MODEL.vertx_dim)
        self.upsample_conv = nn.Conv1d(self.num_verts, 6890, kernel_size=3, padding=1)
        
        self.gru_cur = nn.GRU(
            input_size=2048,
            hidden_size=1024,
            bidirectional=True,
            num_layers=2
        )
        self.linear_cur1 = nn.Linear(1024 * 2, 6890)
        self.linear_cur2 = nn.Linear(1024 * 2, 6890)
        self.linear_cur3 = nn.Linear(1024 * 2, 6890)
        
    def forward(self, joints, img_feats):
        # image feature aggregation
        y, _ = self.gru_cur(img_feats.permute(1,0,2))
        img_feat = y[cfg.DATASET.seqlen // 2]
        
        # reinitialize each vertex to its nearest joint
        vertxs = joints[:, self.vj_relation, :3]
        
        # co-evolution blocks
        joints1, vertxs = self.coevoblock1(joints, vertxs, img_feat)
        joints2, vertxs = self.coevoblock2(joints, vertxs, img_feat)
        joints3, vertxs = self.coevoblock3(joints, vertxs, img_feat)
        vertxs = self.upsample_conv(vertxs)
        
        # add residual vertices projected from image feature
        y_cur1 = self.linear_cur1(F.relu(y[cfg.DATASET.seqlen // 2])).unsqueeze(-1) # [B, 2048] -> [B, 6890]
        y_cur2 = self.linear_cur2(F.relu(y[cfg.DATASET.seqlen // 2])).unsqueeze(-1) # [B, 2048] -> [B, 6890]
        y_cur3 = self.linear_cur3(F.relu(y[cfg.DATASET.seqlen // 2])).unsqueeze(-1) # [B, 2048] -> [B, 6890]
        vertxs_w_res = vertxs + torch.cat([y_cur1, y_cur2, y_cur3], dim=-1)
        
        return joints3, vertxs_w_res  # B x 6890 x 3


def get_model(num_joint, embed_dim):
    model = Pose2Mesh(num_joint, embed_dim)

    return model