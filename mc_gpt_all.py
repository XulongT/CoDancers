# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this open-source project.


""" This script handling the training process. """
import os
import time
import random
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from dataset.md_seq import MoDaSeq, paired_collate_fn
# from models.gpt2 import condGPT2
from utils.log import Logger
from utils.functional_ori import str2bool, load_data, load_data_aist, check_data_distribution, visualizeAndWrite, \
    load_test_data_aist, load_test_data
from torch.optim import *
import warnings
from tqdm import tqdm
import itertools
import pdb
import numpy as np
import models
import datetime
from models.init_generator import PositionEncoder

warnings.filterwarnings('ignore')

import torch.nn.functional as F
import matplotlib.pyplot as plt


class MCTall():
    def __init__(self, args):
        self.config = args
        torch.backends.cudnn.benchmark = True
        self._build()

    def eval(self):
        with torch.no_grad():
            vqvae = self.model.eval()
            gpt = self.model2.eval()
            init_generator = PositionEncoder().cuda().eval()

            config = self.config
            checkpoint = torch.load(config.vqvae_weight)
            vqvae.load_state_dict(checkpoint['model'], strict=False)
            epoch_tested = config.testing.ckpt_epoch

            ckpt_path = os.path.join(self.ckptdir, f"epoch_{epoch_tested}.pt")
            self.device = torch.device('cuda' if config.cuda else 'cpu')
            print("Evaluation...")
            checkpoint = torch.load(ckpt_path)
            gpt.load_state_dict(checkpoint['model'])
            init_generator.load_state_dict(checkpoint['init_generator'])
            gpt.module.gpt_base.get_codebook(vqvae)

            results = []
            with torch.no_grad():
                # cf = None
                cf = torch.load('./querybank/train.pt')
                for i_eval, batch_eval in enumerate(tqdm(self.test_loader, desc='Generating Dance Poses')):
                    music_seq, pose_seq_k = batch_eval
                    music_seq = music_seq.to(self.device)
                    pose_seq_k = pose_seq_k.to(self.device)

                    pose_seq = pose_seq_k.reshape(-1, pose_seq_k.size(2), pose_seq_k.size(3))
                    root_xyz = pose_seq[:, :1, :3].clone().float()
                    if config.global_vel:
                        pose_seq[:, :-1, :3] = pose_seq[:, 1:, :3] - pose_seq[:, :-1, :3]
                        pose_seq[:, -1, :3] = pose_seq[:, -2, :3]
                    else:
                        pose_seq[:, :, :3] = 0
                    quants = vqvae.module.encode(pose_seq)
                    up, down, root = quants
                    
                    up_index = (torch.ones(up[0].shape[0], 1) * 7).long().cuda()
                    down_index = (torch.ones(down[0].shape[0], 1) * 4).long().cuda()
                    root_index = (torch.ones(root[0].shape[0], 1) * 342).long().cuda()
                    up_tensor = torch.cat([up_index, up[0].clone()], dim=1)
                    down_tensor = torch.cat([down_index, down[0].clone()], dim=1)
                    root_tensor = torch.cat([root_index, root[0].clone()], dim=1)
                    quants = ([up_tensor], [down_tensor], [root_tensor])
                    x = tuple(quants[i][0][:, :1].clone() for i in range(len(quants)))

                    person_k = self.test_person_dict[self.test_dance_names[i_eval]]

                    x[0][person_k:7] = torch.ones_like(x[0][person_k:7]) * 16
                    x[1][person_k:7] = torch.ones_like(x[1][person_k:7]) * 643
                    x[2][person_k:7] = torch.ones_like(x[2][person_k:7]) * 172

                        
                    music_init = init_generator(music_seq.float())
                    conds_music_root = torch.cat([music_init, music_seq], dim=1).float()
                    conds_music_root = conds_music_root[:, :-1, :]

                    person_list = torch.ones((7, 1), dtype=torch.bool)
                    person_list[:person_k, 0] = False

                    zs = gpt.module.sample(x, conds_music_root, vqvae, root_xyz, person_list, cf)

                    # pose_sample = vqvae.module.decode(quants)
                    pose_sample = vqvae.module.decode(zs)

                    length = int(self.test_dance_names[i_eval].split('.')[0].split('_')[-1])
                    pose_sample = pose_sample[:, :length, :]

                    if config.global_vel:
                        global_vel = pose_sample[:, :, :3].clone()
                        pose_sample[:, 0, :3] = root_xyz.clone().squeeze(1)
                        for iii in range(1, pose_sample.size(1)):
                            pose_sample[:, iii, :3] = pose_sample[:, iii-1, :3] + global_vel[:, iii-1, :]

                    results.append(pose_sample)

                epoch_i = epoch_tested
                visualizeAndWrite(results, config, self.evaldir, self.test_dance_names, epoch_i, quants, self.test_person_dict)

    def visgt(self, ):
        config = self.config
        print("Visualizing ground truth")

        results = []
        random_id = 0  # np.random.randint(0, 1e4)

        for i_eval, batch_eval in enumerate(tqdm(self.test_loader, desc='Generating Dance Poses')):
            # Prepare data
            # pose_seq_eval = map(lambda x: x.to(self.device), batch_eval)
            _, pose_seq_eval = batch_eval
            # src_pos_eval = pose_seq_eval[:, :] #
            # global_shift = src_pos_eval[:, :, :3].clone()
            # src_pos_eval[:, :, :3] = 0

            # pose_seq_out, loss, _ = model(src_pos_eval)  # first 20 secs
            # quants = model.module.encode(pose_seq_eval)[0].cpu().data.numpy()[0]
            # all_quants = np.append(all_quants, quants) if quants is not None else quants
            # pose_seq_out[:, :, :3] = global_shift
            results.append(pose_seq_eval)
            # moduel.module.encode

            # quants = model.module.encode(src_pos_eval)[0].cpu().data.numpy()[0]

            # exit()
        # weights = np.histogram(all_quants, bins=1, range=[0, config.structure.l_bins], normed=False, weights=None, density=None)
        visualizeAndWrite(results, config, self.gtdir, self.dance_names, 0)

    def analyze_code(self, ):
        config = self.config
        print("Analyzing codebook")

        epoch_tested = config.testing.ckpt_epoch
        ckpt_path = os.path.join(self.ckptdir, f"epoch_{epoch_tested}.pt")
        checkpoint = torch.load(ckpt_path)
        self.model.load_state_dict(checkpoint['model'])
        model = self.model.eval()

        training_data = self.training_data
        all_quants = None

        torch.cuda.manual_seed(config.seed)
        self.device = torch.device('cuda' if config.cuda else 'cpu')
        random_id = 0  # np.random.randint(0, 1e4)

        for i_eval, batch_eval in enumerate(tqdm(self.training_data, desc='Generating Dance Poses')):
            # Prepare data
            # pose_seq_eval = map(lambda x: x.to(self.device), batch_eval)
            pose_seq_eval = batch_eval.to(self.device)

            quants = model.module.encode(pose_seq_eval)[0].cpu().data.numpy()
            all_quants = np.append(all_quants, quants.reshape(-1)) if all_quants is not None else quants.reshape(-1)

        print(all_quants)
        # exit()
        # visualizeAndWrite(results, config,self.gtdir, self.dance_names, 0)
        plt.hist(all_quants, bins=config.structure.l_bins, range=[0, config.structure.l_bins])
        log = datetime.datetime.now().strftime('%Y-%m-%d')
        plt.savefig(self.histdir1 + '/hist_epoch_' + str(epoch_tested) + '_%s.jpg' % log)  # 图片的存储
        plt.close()

    def sample(self, ):
        config = self.config
        print("Analyzing codebook")

        epoch_tested = config.testing.ckpt_epoch
        ckpt_path = os.path.join(self.ckptdir, f"epoch_{epoch_tested}.pt")
        checkpoint = torch.load(ckpt_path)
        self.model.load_state_dict(checkpoint['model'])
        model = self.model.eval()

        quants = {}

        results = []

        if hasattr(config, 'analysis_array') and config.analysis_array is not None:
            # print(config.analysis_array)
            names = [str(ii) for ii in config.analysis_array]
            print(names)
            for ii in config.analysis_array:
                print(ii)
                zs = [(ii * torch.ones((1, self.config.sample_code_length), device='cuda')).long()]
                print(zs[0].size())
                pose_sample = model.module.decode(zs)
                if config.global_vel:
                    global_vel = pose_sample[:, :, :3]
                    pose_sample[:, 0, :3] = 0
                    for iii in range(1, pose_sample.size(1)):
                        pose_sample[:, iii, :3] = pose_sample[:, iii - 1, :3] + global_vel[:, iii - 1, :]

                quants[str(ii)] = zs[0].cpu().data.numpy()[0]

                results.append(pose_sample)
        else:
            names = ['rand_seq_' + str(ii) for ii in range(10)]
            for ii in range(10):
                zs = [torch.randint(0, self.config.structure.l_bins, size=(1, self.config.sample_code_length),
                                    device='cuda')]
                pose_sample = model.module.decode(zs)
                quants['rand_seq_' + str(ii)] = zs[0].cpu().data.numpy()[0]
                if config.global_vel:
                    global_vel = pose_sample[:, :, :3]
                    pose_sample[:, 0, :3] = 0
                    for iii in range(1, pose_sample.size(1)):
                        pose_sample[:, iii, :3] = pose_sample[:, iii - 1, :3] + global_vel[:, iii - 1, :]
                results.append(pose_sample)
        visualizeAndWrite(results, config, self.sampledir, names, epoch_tested, quants)

    def _build(self):
        config = self.config
        self.start_epoch = 0
        self._dir_setting()
        self._build_model()
        if not (hasattr(config, 'need_not_train_data') and config.need_not_train_data):
            self._build_train_loader()
        if not (hasattr(config, 'need_not_eval_data') and config.need_not_eval_data):
            self._build_eval_loader()
        if not (hasattr(config, 'need_not_test_data') and config.need_not_test_data):
            self._build_test_loader()
        self._build_optimizer()

    def _build_model(self):
        """ Define Model """
        config = self.config
        if hasattr(config.structure, 'name') and hasattr(config.structure_generate, 'name'):
            print(f'using {config.structure.name} and {config.structure_generate.name} ')
            model_class = getattr(models, config.structure.name)
            model = model_class(config.structure)

            model_class2 = getattr(models, config.structure_generate.name)
            model2 = model_class2(config.structure_generate)
        else:
            raise NotImplementedError("Wrong Model Selection")

        model = nn.DataParallel(model)
        model2 = nn.DataParallel(model2)
        self.model2 = model2.cuda()
        self.model = model.cuda()

    def _build_train_loader(self):

        data = self.config.data
        if data.name == "aist":
            print("train with AIST++ dataset!")
            external_wav_rate = self.config.ds_rate // self.config.external_wav_rate if hasattr(self.config,
                                                                                                'external_wav_rate') else 1
            external_wav_rate = self.config.music_relative_rate if hasattr(self.config,
                                                                           'music_relative_rate') else external_wav_rate
            train_music_data, train_dance_data, _, dance_names, person_list = load_data_aist(
                data.train_dir, interval=data.seq_len, move=self.config.move if hasattr(self.config, 'move') else 64,
                rotmat=self.config.rotmat, \
                external_wav=self.config.external_wav if hasattr(self.config, 'external_wav') else None, \
                external_wav1=self.config.external_wav1 if hasattr(self.config, 'external_wav1') else None, \
                external_wav_rate=external_wav_rate, \
                music_normalize=self.config.music_normalize if hasattr(self.config, 'music_normalize') else False, \
                wav_padding=self.config.wav_padding * (
                        self.config.ds_rate // self.config.music_relative_rate) if hasattr(self.config,
                                                                                           'wav_padding') else 0)
        else:
            train_music_data, train_dance_data = load_data(
                args_train.train_dir,
                interval=data.seq_len,
                data_type=data.data_type)
        self.training_data = prepare_dataloader(train_music_data, train_dance_data, self.config.batch_size, person_list)
        self.train_dance_names = dance_names
        self.train_sub_person_list = person_list

    def _build_eval_loader(self):
        config = self.config
        data = self.config.data
        if data.name == "aist":
            print("eval with AIST++ dataset!")
            music_data, dance_data, dance_names, person_list = load_test_data_aist(
                data.eval_dir, \
                move=config.move, \
                rotmat=config.rotmat, \
                external_wav=config.external_wav if hasattr(self.config, 'external_wav') else None, \
                external_wav1=config.external_wav1 if hasattr(self.config, 'external_wav1') else None, \
                external_wav_rate=self.config.external_wav_rate if hasattr(self.config, 'external_wav_rate') else 1, \
                music_normalize=self.config.music_normalize if hasattr(self.config, 'music_normalize') else False, \
                wav_padding=self.config.wav_padding * (
                        self.config.ds_rate // self.config.music_relative_rate) if hasattr(self.config,
                                                                                           'wav_padding') else 0)

        else:
            music_data, dance_data, dance_names = load_test_data(
                data.eval_dir, interval=None)
        person_dict = dict(zip(dance_names, person_list))
        # pdb.set_trace()

        self.eval_loader = torch.utils.data.DataLoader(
            MoDaSeq(music_data, dance_data),
            batch_size=1,
            shuffle=False
            # collate_fn=paired_collate_fn,
        )
        self.val_dance_names = dance_names
        self.val_person_dict = person_dict
        # pdb.set_trace()
        # self.training_data = self.test_loader

    def _build_test_loader(self):
        config = self.config
        data = self.config.data
        if data.name == "aist":
            print("test with AIST++ dataset!")
            music_data, dance_data, dance_names, person_list = load_test_data_aist( \
                data.test_dir, \
                move=config.move, \
                rotmat=config.rotmat, \
                external_wav=config.external_wav if hasattr(self.config, 'external_wav') else None, \
                external_wav1=config.external_wav1 if hasattr(self.config, 'external_wav1') else None, \
                external_wav_rate=self.config.external_wav_rate if hasattr(self.config, 'external_wav_rate') else 1, \
                music_normalize=self.config.music_normalize if hasattr(self.config, 'music_normalize') else False, \
                wav_padding=self.config.wav_padding * (
                            self.config.ds_rate // self.config.music_relative_rate) if hasattr(self.config,
                                                                                               'wav_padding') else 0)

        else:
            music_data, dance_data, dance_names = load_test_data(
                data.test_dir, interval=None)

        person_dict = dict(zip(dance_names, person_list))
        # pdb.set_trace()

        self.test_loader = torch.utils.data.DataLoader(
            MoDaSeq(music_data, dance_data),
            batch_size=1,
            shuffle=False
            # collate_fn=paired_collate_fn,
        )
        self.test_dance_names = dance_names
        self.test_person_dict = person_dict
        # pdb.set_trace()
        # self.training_data = self.test_loader

    def _build_optimizer(self):
        # model = nn.DataParallel(model).to(device)
        config = self.config.optimizer
        try:
            optim = getattr(torch.optim, config.type)
        except Exception:
            raise NotImplementedError('not implemented optim method ' + config.type)

        self.optimizer = optim(itertools.chain(self.model2.module.parameters(),
                                               ),
                               **config.kwargs)
        self.schedular = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, **config.schedular_kwargs)

    def _dir_setting(self):
        data = self.config.data
        self.expname = self.config.expname
        self.experiment_dir = os.path.join("./", "experiments")
        self.expdir = os.path.join(self.experiment_dir, self.expname)

        if not os.path.exists(self.expdir):
            os.mkdir(self.expdir)

        self.visdir = os.path.join(self.expdir, "vis")  # -- imgs, videos, jsons
        if not os.path.exists(self.visdir):
            os.mkdir(self.visdir)

        self.jsondir = os.path.join(self.visdir, "jsons")  # -- imgs, videos, jsons
        if not os.path.exists(self.jsondir):
            os.mkdir(self.jsondir)

        self.histdir = os.path.join(self.visdir, "hist")  # -- imgs, videos, jsons
        if not os.path.exists(self.histdir):
            os.mkdir(self.histdir)

        self.imgsdir = os.path.join(self.visdir, "imgs")  # -- imgs, videos, jsons
        if not os.path.exists(self.imgsdir):
            os.mkdir(self.imgsdir)

        self.videodir = os.path.join(self.visdir, "videos")  # -- imgs, videos, jsons
        if not os.path.exists(self.videodir):
            os.mkdir(self.videodir)

        self.ckptdir = os.path.join(self.expdir, "ckpt")
        if not os.path.exists(self.ckptdir):
            os.mkdir(self.ckptdir)

        self.evaldir = os.path.join(self.expdir, "eval")
        if not os.path.exists(self.evaldir):
            os.mkdir(self.evaldir)

        self.gtdir = os.path.join(self.expdir, "gt")
        if not os.path.exists(self.gtdir):
            os.mkdir(self.gtdir)

        self.jsondir1 = os.path.join(self.evaldir, "jsons")  # -- imgs, videos, jsons
        if not os.path.exists(self.jsondir1):
            os.mkdir(self.jsondir1)

        self.histdir1 = os.path.join(self.evaldir, "hist")  # -- imgs, videos, jsons
        if not os.path.exists(self.histdir1):
            os.mkdir(self.histdir1)

        self.imgsdir1 = os.path.join(self.evaldir, "imgs")  # -- imgs, videos, jsons
        if not os.path.exists(self.imgsdir1):
            os.mkdir(self.imgsdir1)

        self.videodir1 = os.path.join(self.evaldir, "videos")  # -- imgs, videos, jsons
        if not os.path.exists(self.videodir1):
            os.mkdir(self.videodir1)

        self.sampledir = os.path.join(self.evaldir, "samples")  # -- imgs, videos, jsons
        if not os.path.exists(self.sampledir):
            os.mkdir(self.sampledir)

        # self.ckptdir = os.path.join(self.expdir, "ckpt")
        # if not os.path.exists(self.ckptdir):
        #     os.mkdir(self.ckptdir)


def prepare_dataloader(music_data, dance_data, batch_size, person_list):
    data_loader = torch.utils.data.DataLoader(
        MoDaSeq(music_data, dance_data, person_list),
        num_workers=8,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True
        # collate_fn=paired_collate_fn,
    )

    return data_loader

# def train_m2d(cfg):
#     """ Main function """
#     parser = argparse.ArgumentParser()

#     parser.add_argument('--train_dir', type=str, default='data/train_1min',
#                         help='the directory of dance data')
#     parser.add_argument('--test_dir', type=str, default='data/test_1min',
#                         help='the directory of music feature data')
#     parser.add_argument('--data_type', type=str, default='2D',
#                         help='the type of training data')
#     parser.add_argument('--output_dir', metavar='PATH',
#                         default='checkpoints/layers2_win100_schedule100_condition10_detach')

#     parser.add_argument('--epoch', type=int, default=300000)
#     parser.add_argument('--batch_size', type=int, default=16)
#     parser.add_argument('--save_per_epochs', type=int, metavar='N', default=50)
#     parser.add_argument('--log_per_updates', type=int, metavar='N', default=1,
#                         help='log model loss per x updates (mini-batches).')
#     parser.add_argument('--seed', type=int, default=1234,
#                         help='random seed for data shuffling, dropout, etc.')
#     parser.add_argument('--tensorboard', action='store_false')

#     parser.add_argument('--d_frame_vec', type=int, default=438)
#     parser.add_argument('--frame_emb_size', type=int, default=800)
#     parser.add_argument('--d_pose_vec', type=int, default=24*3)
#     parser.add_argument('--pose_emb_size', type=int, default=800)

#     parser.add_argument('--d_inner', type=int, default=1024)
#     parser.add_argument('--d_k', type=int, default=80)
#     parser.add_argument('--d_v', type=int, default=80)
#     parser.add_argument('--n_head', type=int, default=10)
#     parser.add_argument('--n_layers', type=int, default=2)
#     parser.add_argument('--lr', type=float, default=1e-4)
#     parser.add_argument('--dropout', type=float, default=0.1)

#     parser.add_argument('--seq_len', type=int, default=240)
#     parser.add_argument('--max_seq_len', type=int, default=4500)
#     parser.add_argument('--condition_step', type=int, default=10)
#     parser.add_argument('--sliding_windown_size', type=int, default=100)
#     parser.add_argument('--lambda_v', type=float, default=0.01)

#     parser.add_argument('--cuda', type=str2bool, nargs='?', metavar='BOOL', const=True,
#                         default=torch.cuda.is_available(),
#                         help='whether to use GPU acceleration.')
#     parser.add_argument('--aist', action='store_true', help='train on AIST++')
#     parser.add_argument('--rotmat', action='store_true', help='train rotation matrix')

#     args = parser.parse_args()
#     args.d_model = args.frame_emb_size


#     args_data = args.data
#     args_structure = args.structure






