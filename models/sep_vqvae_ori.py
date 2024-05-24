import numpy as np
import torch
import torch.nn as nn

# from .encdec import Encoder, Decoder, assert_shape
# from .bottleneck import NoBottleneck, Bottleneck
# from .utils.logger import average_metrics
# from .utils.audio_utils import  audio_postprocess

from .vqvae_ori import VQVAE

smpl_root = [0]
smpl_down = [1, 2, 4, 5, 7, 8, 10, 11]
smpl_up = [3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]


class SepVQVAE(nn.Module):
    def __init__(self, hps):
        super().__init__()
        self.hps = hps
        # self.cut_dim = hps.up_half_dim
        # self.use_rotmat = hps.use_rotmat if (hasattr(hps, 'use_rotmat') and hps.use_rotmat) else False
        self.chanel_num = hps.joint_channel
        self.vqvae_up = VQVAE(hps.up_half, len(smpl_up) * self.chanel_num)
        self.vqvae_down = VQVAE(hps.down_half, len(smpl_down) * self.chanel_num)
        self.vqvae_root = VQVAE(hps.down_half, len(smpl_root) * self.chanel_num)
        # self.use_rotmat = hps.use_rotmat if (hasattr(hps, 'use_rotmat') and hps.use_rotmat) else False
        # self.chanel_num = 9 if self.use_rotmat else 3

    def decode(self, zs, start_level=0, end_level=None, bs_chunks=1):
        
        K = 7
        zup = zs[0]
        zdown = zs[1]
        zroot = zs[2]

        # print(zup[0].shape)
        # print(zdown[0].shape)
        # BK, T = zup[0].size()
        # B = BK // K

        xup = self.vqvae_up.decode(zup)
        xdown = self.vqvae_down.decode(zdown)
        xroot = self.vqvae_root.decode(zroot)

        b, t, cup = xup.size()
        _, _, cdown = xdown.size()
        _, _, croot = xroot.size()

        x = torch.zeros(b, t, (cup + cdown + croot) // self.chanel_num, self.chanel_num).cuda()

        x[:, :, smpl_up] = xup.view(b, t, cup // self.chanel_num, self.chanel_num)
        x[:, :, smpl_down] = xdown.view(b, t, cdown // self.chanel_num, self.chanel_num)
        x[:, :, smpl_root] = xroot.view(b, t, croot // self.chanel_num, self.chanel_num)

        x = x.view(b, t, -1)

        return x

    def encode(self, x, start_level=0, end_level=None, bs_chunks=1):
        b, t, c = x.size()
        zup = self.vqvae_up.encode(x.view(b, t, c // self.chanel_num, self.chanel_num)[:, :, smpl_up].view(b, t, -1),
                                    start_level, end_level, bs_chunks)
        zdown = self.vqvae_down.encode(x.view(b, t, c // self.chanel_num, self.chanel_num)[:, :, smpl_down].view(b, t, -1), 
                                    start_level, end_level, bs_chunks)
        zroot = self.vqvae_root.encode(x.view(b, t, c // self.chanel_num, self.chanel_num)[:, :, smpl_root].view(b, t, -1), 
                                    start_level, end_level, bs_chunks)                                 
        return (zup, zdown, zroot)

    def sample(self, n_samples):
        xup = self.vqvae_up.sample(n_samples)
        xdown = self.vqvae_up.sample(n_samples)
        b, t, cup = xup.size()
        _, _, cdown = xdown.size()
        x = torch.zeros(b, t, (cup + cdown) // self.chanel_num, self.chanel_num).cuda()
        x[:, :, smpl_up] = xup.view(b, t, cup // self.chanel_num, self.chanel_num)
        x[:, :, smpl_down] = xdown.view(b, t, cdown // self.chanel_num, self.chanel_num)
        return x


    def forward(self, x):
        b, t, c = x.size()
        x = x.view(b, t, c // self.chanel_num, self.chanel_num)
        xup = x[:, :, smpl_up, :].view(b, t, -1)
        xdown = x[:, :, smpl_down, :].view(b, t, -1)
        xroot = x[:, :, smpl_root, :].view(b, t, -1)
        # xup[:] = 0

        x_out_up, loss_up, metrics_up = self.vqvae_up(xup)
        x_out_down, loss_down, metrics_down = self.vqvae_down(xdown)
        x_out_root, loss_root, metrics_root = self.vqvae_root(xroot)

        _, _, cup = x_out_up.size()
        _, _, cdown = x_out_down.size()
        _, _, croot = x_out_root.size()

        xout = torch.zeros(b, t, (cup + cdown + croot) // self.chanel_num, self.chanel_num).cuda().float()
        xout[:, :, smpl_up] = x_out_up.view(b, t, cup // self.chanel_num, self.chanel_num)
        xout[:, :, smpl_down] = x_out_down.view(b, t, cdown // self.chanel_num, self.chanel_num)
        xout[:, :, smpl_root] = x_out_root.view(b, t, croot // self.chanel_num, self.chanel_num)

        # xout[:, :, smpl_up] = xup.view(b, t, cup//self.chanel_num, self.chanel_num).float()
        # xout[:, :, smpl_down] = xdown.view(b, t, cdown//self.chanel_num, self.chanel_num).float()

        return xout.view(b, t, -1), (loss_up + loss_down + loss_root) * 0.5, [metrics_up, metrics_down]

