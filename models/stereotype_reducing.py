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
# from .gpt3p import condGPT3Part
# logger = logging.getLogger(__name__)
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
class CrossCondGPT2(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, config):
        super().__init__()
        self.gpt_base = CrossCondGPTBase(config.base)
        self.gpt_head = CrossCondGPTHead(config.head)
        self.block_size = config.block_size
        print(config)
        self.theta = config.theta
    def get_block_size(self):
        return self.block_size

    def sample(self, xs, cond, vqvae, xyz, class_f=None):
        shift = None
        block_size = self.get_block_size() - 1
        if shift is not None:
            block_shift = min(shift, block_size)
        else:
            block_shift = block_size
        x_up, x_down, x_root = xs
        if class_f is not None:
            theta = self.theta
            up_cf, down_cf, root_cf = class_f
            up_cf, down_cf, root_cf = torch.exp(up_cf*theta), torch.exp(down_cf*theta), torch.exp(root_cf*theta)

        for k in range(cond.size(1)):
            x_cond_up = x_up if x_up.size(1) <= block_size else x_up[:, -(block_shift+(k-block_size-1)%(block_size-block_shift+1)):]
            x_cond_down = x_down if x_down.size(1) <= block_size else x_down[:, -(block_shift+(k-block_size-1)%(block_size-block_shift+1)):]  # crop context if needed
            x_cond_root = x_down if x_root.size(1) <= block_size else x_root[:, -(block_shift+(k-block_size-1)%(block_size-block_shift+1)):]  # crop context if needed

            cond_input = cond[:, :k+1] if k < block_size else cond[:, k-(block_shift+(k-block_size-1)%(block_size-block_shift+1))+1:k+1]

            logits, _ = self.forward((x_cond_up, x_cond_down, x_cond_root), cond_input, xyz)
            logit_up, logit_down, logit_root = logits
            logit_up = logit_up[:, -1, :]
            logit_down = logit_down[:, -1, :]
            logit_root = logit_root[:, -1, :]

            probs_up = F.softmax(logit_up, dim=-1)
            probs_down = F.softmax(logit_down, dim=-1)
            probs_root = F.softmax(logit_root, dim=-1)

            if class_f is not None:
                probs_up = torch.exp(probs_up * theta) / up_cf
                probs_down = torch.exp(probs_down * theta) / down_cf
                probs_root = torch.exp(probs_root * theta) / root_cf

            _, ix_up = torch.topk(probs_up, k=1, dim=-1)
            _, ix_down = torch.topk(probs_down, k=1, dim=-1)
            _, ix_root = torch.topk(probs_root, k=1, dim=-1)

            # print('probs_up', probs_up.shape)
            # print('up', up_cf.shape)
            # for i in range(7):
            #     up_cf[ix_up[i][0]] += 1
            #     down_cf[ix_down[i][0]] += 1
            #     root_cf[ix_root[i][0]] += 1
            # if class_f is not None:
            #     up_cf += torch.sum(probs_up, dim=0).cpu().detach()
            #     down_cf += torch.sum(probs_down, dim=0).cpu().detach()
            #     root_cf += torch.sum(probs_root, dim=0).cpu().detach()
            # import sys
            # sys.exit()

            x_up = torch.cat((x_up, ix_up), dim=1)
            x_down = torch.cat((x_down, ix_down), dim=1)
            x_root = torch.cat((x_root, ix_root), dim=1)

            pose_xyz = torch.sum(vqvae.module.decode(([ix_up], [ix_down], [ix_root]))[:, :, :3], dim=1, keepdim=True)
            xyz += pose_xyz

        return ([x_up[:, 1:]], [x_down[:, 1:]], [x_root[:, 1:]])

    # # Pure Top K Sample
    # def sample(self, xs, cond, vqvae, xyz):
    #     shift = None
    #     block_size = self.get_block_size() - 1
    #     if shift is not None:
    #         block_shift = min(shift, block_size)
    #     else:
    #         block_shift = block_size
    #     x_up, x_down, x_root = xs
    #     for k in range(cond.size(1)):
    #         x_cond_up = x_up if x_up.size(1) <= block_size else x_up[:, -(block_shift+(k-block_size-1)%(block_size-block_shift+1)):]
    #         x_cond_down = x_down if x_down.size(1) <= block_size else x_down[:, -(block_shift+(k-block_size-1)%(block_size-block_shift+1)):]  # crop context if needed
    #         x_cond_root = x_down if x_root.size(1) <= block_size else x_root[:, -(block_shift+(k-block_size-1)%(block_size-block_shift+1)):]  # crop context if needed

    #         cond_input = cond[:, :k+1] if k < block_size else cond[:, k-(block_shift+(k-block_size-1)%(block_size-block_shift+1))+1:k+1]

    #         logits, _ = self.forward((x_cond_up, x_cond_down, x_cond_root), cond_input, xyz)
    #         # pluck the logits at the final step and scale by temperature
    #         logit_up, logit_down, logit_root = logits
    #         logit_up = logit_up[:, -1, :]
    #         logit_down = logit_down[:, -1, :]
    #         logit_root = logit_root[:, -1, :]

    #         probs_up = F.softmax(logit_up, dim=-1)
    #         probs_down = F.softmax(logit_down, dim=-1)
    #         probs_root = F.softmax(logit_root, dim=-1)

    #         prob_up, ix_up = torch.topk(probs_up, k=10, dim=-1)
    #         prob_down, ix_down = torch.topk(probs_down, k=10, dim=-1)
    #         prob_root, ix_root = torch.topk(probs_root, k=10, dim=-1)

    #         up_samples = torch.distributions.Categorical(prob_up).sample()
    #         up_sample = ix_up[torch.arange(ix_up.size(0)), up_samples].reshape(-1, 1)
    #         down_samples = torch.distributions.Categorical(prob_down).sample()
    #         down_sample = ix_down[torch.arange(ix_down.size(0)), down_samples].reshape(-1, 1)
    #         root_samples = torch.distributions.Categorical(prob_root).sample()
    #         root_sample = ix_root[torch.arange(ix_root.size(0)), root_samples].reshape(-1, 1)

    #         # append to the sequence and continue
    #         x_up = torch.cat((x_up, up_sample), dim=1)
    #         x_down = torch.cat((x_down, down_sample), dim=1)
    #         x_root = torch.cat((x_root, root_sample), dim=1)

    #         pose_xyz = torch.sum(vqvae.module.decode(([ix_up], [ix_down], [ix_root]))[:, :, :3], dim=1, keepdim=True)
    #         xyz += pose_xyz

    #     return ([x_up[:, 1:]], [x_down[:, 1:]], [x_root[:, 1:]])


    def forward(self, idxs, cond, xyz, targets=None):
        idx_up, idx_down, idx_root = idxs

        targets_up, targets_down, targets_root = None, None, None
        if targets is not None:
            targets_up, targets_down, targets_root = targets
        
        feat = self.gpt_base(idx_up, idx_down, idx_root, cond, xyz)
        logits_up, logits_down, logits_root, loss_up, loss_down, loss_root = self.gpt_head(feat, targets)
        
        if loss_up is not None and loss_down is not None:
            loss = loss_up + loss_down + loss_root
        else:
            loss = None

        return (logits_up, logits_down, logits_root), loss

class CausalCrossConditionalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        # self.mask = se
        self.n_head = config.n_head

    def forward(self, x, layer_past=None):
        B, T, C = x.size()  # T = 3*t (music up down)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        t = T // 4
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:,:,:t,:t].repeat(1, 1, 4, 4) == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection

        y = self.resid_drop(self.proj(y))
        return y

class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalCrossConditionalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x):
        # print(f"x 0 , {x.shape}")
        x = x + self.attn(self.ln1(x))
        # print(f"x 1 , {x.shape}")
        x = x + self.mlp(self.ln2(x))
        # print(f"x 2 , {x.shape}")
        return x

class CrossCondGPTBase(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, config):
        super().__init__()


        self.tok_emb_up = nn.Embedding(config.vocab_size_up, config.n_embd  )
        self.tok_emb_down = nn.Embedding(config.vocab_size_down, config.n_embd)
        self.tok_emb_root = nn.Embedding(config.vocab_size_down, config.n_embd)

        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size*4, config.n_embd))
        self.cond_emb = nn.Linear(config.n_music, config.n_embd)
        self.xyz_emb = PositionEncoderGPT()
        self.drop = nn.Dropout(config.embd_pdrop)
        # transformer
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        # decoder head
        # self.ln_f = nn.LayerNorm(config.n_embd)

        self.block_size = config.block_size

        input_dim = 768
        hidden_dim = 512
        num_heads = 4
        dropout = 0.1
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout)
        self.group_transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)

        self.apply(self._init_weights)


        # logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # module.weight.data.uniform_(math.sqrt(6.0/sum(module.weight.size())))
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    def forward(self, idx_up, idx_down, idx_root, cond, xyz):
        b, t = idx_up.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."
        b, t = idx_down.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."


        token_embeddings_up = self.tok_emb_up(idx_up) # each index maps to a (learnable) vector
        token_embeddings_down = self.tok_emb_down(idx_down) # each index maps to a (learnable) vector
        token_embeddings_root = self.tok_emb_down(idx_root) # each index maps to a (learnable) vector

        expanded_Cond_emb = self.cond_emb(cond) + self.xyz_emb(xyz)
        token_embeddings = torch.cat([expanded_Cond_emb, token_embeddings_up, token_embeddings_down, token_embeddings_root ], dim=1)

        position_embeddings = torch.cat([self.pos_emb[:, :t, :], self.pos_emb[:, self.block_size:self.block_size+t, :], \
                                        self.pos_emb[:, self.block_size*2:self.block_size*2+t, :], self.pos_emb[:, self.block_size*3:self.block_size*3+t, :]], dim=1) # each position maps to a (learnable) vector
        # print("position_embeddings shape:", position_embeddings.shape)

        x = self.drop(token_embeddings + position_embeddings)

        x = self.blocks(x)

        y = x.reshape(-1, 7, 4*t, 768)

        y_fuse = y[:, :, 0:t, :] + y[:, :, t:2*t, :] + y[:, :, 2*t:3*t, :] + y[:, :, 3*t:4*t, :]
        y_fuse_group = self.group_transformer(y_fuse.permute(0, 2, 1, 3).reshape(-1, 7, 768))
        y_fuse_group = y_fuse_group.reshape(b//7, t, 7, 768).permute(0, 2, 1, 3).reshape(-1, t, 768)

        x[:, t:2*t, :] += y_fuse_group
        x[:, 2*t:3*t, :] += y_fuse_group
        x[:, 3*t:4*t, :] += y_fuse_group

        return x

class CrossCondGPTHead(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, config):
        super().__init__()

        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)

        self.block_size = config.block_size
        self.head_up = nn.Linear(config.n_embd, config.vocab_size_up, bias=False)
        self.head_down = nn.Linear(config.n_embd, config.vocab_size_down, bias=False)
        self.head_root = nn.Linear(config.n_embd, config.vocab_size_down, bias=False)

        self.apply(self._init_weights)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    def forward(self, x, targets=None):

        x = self.blocks(x)
        x = self.ln_f(x)
        N, T, C = x.size()
        t = T//4
        logits_up = self.head_up(x[:, t:t*2, :])
        logits_down = self.head_down(x[:, t*2:t*3, :]) # down half 
        logits_root = self.head_root(x[:, t*3:t*4, :]) # down half 

        # if we are given some desired targets also calculate the loss
        loss_up, loss_down, loss_root = None, None, None

        if targets is not None:
            targets_up, targets_down, targets_root = targets

            loss_up = F.cross_entropy(logits_up.view(-1, logits_up.size(-1)), targets_up.view(-1))
            loss_down = F.cross_entropy(logits_down.view(-1, logits_down.size(-1)), targets_down.view(-1))
            loss_root = F.cross_entropy(logits_root.view(-1, logits_root.size(-1)), targets_root.view(-1))
            

        return logits_up, logits_down, logits_root, loss_up, loss_down, loss_root

        

