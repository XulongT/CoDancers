"""
GPT model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier
"""

import math
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F
from models.init_generator import PositionEncoderGPT
from models.block import MusicEncoder, Block, FusionBlock
# from .gpt3p import condGPT3Part
# logger = logging.getLogger(__name__)
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, input, target):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(input, target)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

class CrossCondGPT2(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, config):
        super().__init__()
        self.gpt_base = CrossCondGPTBase(config.base)
        self.gpt_head = CrossCondGPTHead(config.base)
        self.block_size = config.block_size
        self.theta = config.theta
        self.k = config.k
        self.xyz_emb = PositionEncoderGPT()
        self.music_encode = MusicEncoder(config.base)
        self.fuse_block = FusionBlock(config.base)
        self.cls_loss = FocalLoss(alpha=0.6)
        self.apply(self._init_weights)


    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def get_block_size(self):
        return self.block_size

    def sample(self, xs, cond, vqvae, xyz, person_k, class_f=None):
        shift = None
        block_size = self.get_block_size() - 1
        if shift is not None:
            block_shift = min(shift, block_size)
        else:
            block_shift = block_size
        x_up, x_down, x_root = xs

        if class_f is not None:
            theta = self.theta
            up_cf, down_cf, root_cf = class_f['up'].cuda(), class_f['down'].cuda(), class_f['root'].cuda()
            up_cf, down_cf, root_cf = torch.log1p(up_cf*theta), torch.log1p(down_cf*theta), torch.log1p(root_cf*theta)

        for k in range(cond.size(1)):
            x_cond_up = x_up if x_up.size(1) <= block_size else x_up[:, -(block_shift+(k-block_size-1)%(block_size-block_shift+1)):]
            x_cond_down = x_down if x_down.size(1) <= block_size else x_down[:, -(block_shift+(k-block_size-1)%(block_size-block_shift+1)):]
            x_cond_root = x_root if x_root.size(1) <= block_size else x_root[:, -(block_shift+(k-block_size-1)%(block_size-block_shift+1)):]
            xyz = xyz if xyz.size(1) <= block_size else xyz[:, -(block_shift+(k-block_size-1)%(block_size-block_shift+1)):]

            cond_input = cond[:, :k+1] if k < block_size else cond[:, k-(block_shift+(k-block_size-1)%(block_size-block_shift+1))+1:k+1]

            logits, _ = self.forward((x_cond_up, x_cond_down, x_cond_root), cond_input, xyz, person_k)
            logit_up, logit_down, logit_root = logits
            logit_up = logit_up[:, -1, :]
            logit_down = logit_down[:, -1, :]
            logit_root = logit_root[:, -1, :]

            probs_up = F.softmax(logit_up, dim=-1)
            probs_down = F.softmax(logit_down, dim=-1)
            probs_root = F.softmax(logit_root, dim=-1)

            if class_f is not None:
                probs_up = torch.log1p(probs_up * theta) / up_cf
                probs_down = torch.log1p(probs_down * theta) / down_cf
                probs_root = torch.log1p(probs_root * theta) / root_cf
                # probs_up = F.softmax(probs_up, dim=-1)
                # probs_down = F.softmax(probs_down, dim=-1)
                # probs_root = F.softmax(probs_root, dim=-1)


            # if class_f is not None: 
            #     up_cf, down_cf, root_cf = class_f['up'], class_f['down'], class_f['root']
            #     up_cf += torch.sum(probs_up[~person_k.view(-1), :], dim=0).cpu().detach()
            #     down_cf += torch.sum(probs_down[~person_k.view(-1), :], dim=0).cpu().detach()
            #     root_cf += torch.sum(probs_root[~person_k.view(-1), :], dim=0).cpu().detach()

            # prob_up, ix_up = torch.topk(probs_up, k=self.k, dim=-1)
            # prob_down, ix_down = torch.topk(probs_down, k=self.k, dim=-1)
            # prob_root, ix_root = torch.topk(probs_root, k=self.k, dim=-1)

            # up_samples = torch.distributions.Categorical(prob_up).sample()
            # up_sample = ix_up[torch.arange(ix_up.size(0)), up_samples].reshape(-1, 1)
            # down_samples = torch.distributions.Categorical(prob_down).sample()
            # down_sample = ix_down[torch.arange(ix_down.size(0)), down_samples].reshape(-1, 1)
            # root_samples = torch.distributions.Categorical(prob_root).sample()
            # root_sample = ix_root[torch.arange(ix_root.size(0)), root_samples].reshape(-1, 1)

            # x_up = torch.cat((x_up, up_sample), dim=1)
            # x_down = torch.cat((x_down, down_sample), dim=1)
            # x_root = torch.cat((x_root, root_sample), dim=1)
            # pose_xyz = torch.sum(vqvae.module.decode(([up_sample], [down_sample], [root_sample]))[:, :, :3], dim=1, keepdim=True)


            _, ix_up = torch.topk(probs_up, k=1, dim=-1)
            _, ix_down = torch.topk(probs_down, k=1, dim=-1)
            _, ix_root = torch.topk(probs_root, k=1, dim=-1)

            ix_up[person_k], ix_down[person_k], ix_root[person_k] = 16, 643, 172

            x_up = torch.cat((x_up, ix_up), dim=1)
            x_down = torch.cat((x_down, ix_down), dim=1)
            x_root = torch.cat((x_root, ix_root), dim=1)
            pose_xyz = torch.sum(vqvae.module.decode(([ix_up], [ix_down], [ix_root]))[:, :, :3], dim=1, keepdim=True)

            current_xyz = xyz[:, -1:, :] + pose_xyz
            xyz = torch.cat((xyz, current_xyz), dim=1)

        return ([x_up[:, 1:]], [x_down[:, 1:]], [x_root[:, 1:]])


    def forward(self, idxs, cond, xyz, person_k, targets=None):
        idx_up, idx_down, idx_root = idxs

        targets_up, targets_down, targets_root = None, None, None
        loss = 0

        music_feat = self.music_encode(cond).repeat_interleave(7, dim=0)
        pose_feat = self.gpt_base(idx_up, idx_down, idx_root)
        xyz_feat = self.xyz_emb(xyz)
        logits_up, logits_down, logits_root = self.gpt_head(pose_feat, xyz_feat, music_feat)
        # logits_up, logits_down, logits_root = self.fuse_block(pose_feat, music_feat, person_k)

        if targets is not None:
            person_k = person_k.view(-1)
            logits_up, logits_down, logits_root = logits_up[~person_k, :, :], logits_down[~person_k, :, :], logits_root[~person_k, :, :]
            b, t, _ = logits_up.shape

            targets_up, targets_down, targets_root = targets
            targets_up, targets_down, targets_root = targets_up[~person_k, :], targets_down[~person_k, :], targets_root[~person_k, :]

            loss_up = self.cls_loss(logits_up.reshape(b*t, -1), targets_up.reshape(-1))
            loss_down = self.cls_loss(logits_down.reshape(b*t, -1), targets_down.reshape(-1))
            loss_root = self.cls_loss(logits_root.reshape(b*t, -1), targets_root.reshape(-1))
            loss = loss_up + loss_down + loss_root

        return (logits_up, logits_down, logits_root), loss



class CrossCondGPTBase(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, config):
        super().__init__()

        self.tok_emb_up = nn.Linear(512, config.n_embd)
        self.tok_emb_down = nn.Linear(512, config.n_embd)
        self.tok_emb_root = nn.Linear(512, config.n_embd)

        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size*3, config.n_embd))
        self.cond_emb = nn.Linear(config.n_music, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])


        self.block_size = config.block_size

        input_dim = 768
        hidden_dim = 512
        num_heads = 4
        dropout = 0.1
        num_layers = 2
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout)
        self.group_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.linear = nn.Linear(config.n_embd, config.n_embd)

    def get_codebook(self, vqvae):
        vqvae_codebooks =  vqvae.module.vqvae_up.bottleneck.level_blocks[0].k, \
                            vqvae.module.vqvae_down.bottleneck.level_blocks[0].k, \
                            vqvae.module.vqvae_root.bottleneck.level_blocks[0].k
        self.vqvae_codebooks = vqvae_codebooks

    def get_block_size(self):
        return self.block_size

    def forward(self, idx_up, idx_down, idx_root):
        
        b, t = idx_up.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."
        b, t = idx_down.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."

        idx_up = self.vqvae_codebooks[0][idx_up]
        idx_down = self.vqvae_codebooks[1][idx_down]
        idx_root = self.vqvae_codebooks[2][idx_root]

        token_embeddings_up = self.tok_emb_up(idx_up)
        token_embeddings_down = self.tok_emb_down(idx_down)
        token_embeddings_root = self.tok_emb_down(idx_root)
        token_embeddings = torch.cat([token_embeddings_up, token_embeddings_down, token_embeddings_root], dim=1)
        position_embeddings = torch.cat([self.pos_emb[:, :t, :], self.pos_emb[:, self.block_size:self.block_size+t, :], \
                                            self.pos_emb[:, self.block_size*2:self.block_size*2+t, :], ], dim=1)

        x = self.drop(token_embeddings + position_embeddings)
        x = self.blocks(x)

        y = x.reshape(-1, 7, 3*t, 768)

        y_fuse = y[:, :, 0:t, :] + y[:, :, t:2*t, :] + y[:, :, 2*t:3*t, :]
        y_fuse_group = self.group_transformer(y_fuse.permute(0, 2, 1, 3).reshape(-1, 7, 768))
        y_fuse_group = y_fuse_group.reshape(b//7, t, 7, 768).permute(0, 2, 1, 3).reshape(-1, t, 768)

        x = self.linear(y_fuse_group)

        return x

class CrossCondGPTHead(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, config):
        super().__init__()

        self.linear = nn.Sequential(
            nn.Linear(768, 768),
            nn.Linear(768, 3*768),
        )
        self.drop = nn.Dropout(config.embd_pdrop)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.block_size = config.block_size

    def get_block_size(self):
        return self.block_size

    def forward(self, pose, xyz, music):
        
        _, t, c = pose.shape

        # x = torch.cat([pose, xyz, music], dim=1)
        # x = self.blocks(x)
        # x = x[:, :t, :] + x[:, t:2*t, :] + x[:, 2*t:3*t, :]

        x = pose + xyz + music
        x = self.linear(x)

        return x[:, :, 0:c], x[:, :, c:2*c], x[:, :, 2*c:3*c]
