import numpy as np
import pickle
from features.kinetic import extract_kinetic_features
from features.manual_new import extract_manual_features
from scipy import linalg
import json
# kinetic, manual
import os
from scipy.ndimage import gaussian_filter as G
from scipy.signal import argrelextrema

import matplotlib.pyplot as plt


def get_mb(key, music_root, length=None):
    path = os.path.join(music_root, key)
    with open(path) as f:
        # print(path)
        sample_dict = json.loads(f.read())
        if length is not None:
            beats = np.array(sample_dict['music_array'])[:, 53][:][:length]
        else:
            beats = np.array(sample_dict['music_array'])[:, 53]

        beats = beats.astype(bool)
        beat_axis = np.arange(len(beats))
        beat_axis = beat_axis[beats]

        return beat_axis

# import scipy.signal as scisignal
def calc_db(keypoints, name=''):
    keypoints = np.array(keypoints).reshape(-1, 24, 3)
    kinetic_vel = np.mean(np.sqrt(np.sum((keypoints[1:] - keypoints[:-1]) ** 2, axis=2)), axis=1)
    kinetic_vel = G(kinetic_vel, 5)
    motion_beats = argrelextrema(kinetic_vel, np.less)
    return motion_beats, len(kinetic_vel)


def BA(music_beats, motion_beats):
    ba = 0
    for bb in music_beats:
        ba += np.exp(-np.min((motion_beats[0] - bb) ** 2) / 2 / 9)
    return (ba / len(music_beats))

from tqdm import tqdm
def calc_ba_score(root, mode=None):
    # gt_list = []
    ba_scores = []
    if mode == 'eval':
        music_root = './data/aistpp_eval_wav'
    else:
        music_root = './data/aistpp_test_full_wav'
    people_num_file = './data/People_Num'
    pkl_files = sorted([pkl for pkl in os.listdir(root) if pkl.endswith(".npy")])
    for i in tqdm(range(0, len(pkl_files), 7)):
        num_file_path = os.path.join(people_num_file, pkl_files[i].replace('json_dancer0.pkl.npy', 'txt'))
        f = open(num_file_path, "r")
        people_num = int(f.read())
        for j in range(people_num):
            npy_path = os.path.join(root, pkl_files[i+j])
            joint3d = np.load(npy_path, allow_pickle=True).item()['pred_position'][:, :]
            dance_beats, length = calc_db(joint3d, pkl_files[i+j])
            music_beats = get_mb(pkl_files[i+j].split('.')[0]+'.json', music_root, length)
            ba_scores.append(BA(music_beats, dance_beats))

    return np.mean(ba_scores)


if __name__ == '__main__':

    pred_root = './experiments/cc_motion_gpt/vis/pkl/ep000250'

    # print('Calculating and saving features')
    print(calc_ba_score(pred_root))
