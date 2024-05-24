import numpy as np
import pickle
import sys
sys.path.append('./utils')
from beat_align_score import calc_ba_score
from features.kinetic import extract_kinetic_features
from scipy import linalg
import os
import json
from concurrent.futures import ProcessPoolExecutor


def normalize(feat, feat2):
    mean = feat.mean(axis=0)
    std = feat.std(axis=0)

    return (feat - mean) / (std + 1e-5), (feat2 - mean) / (std + 1e-5)


def quantized_metrics(predicted_pkl_root, gt_pkl_root):
    pred_features_k = []
    gt_features_k = []
    pred_features_k = [np.load(os.path.join(predicted_pkl_root, 'kinetic_features', pkl)) for pkl in
                       sorted(os.listdir(os.path.join(predicted_pkl_root, 'kinetic_features')))]
    gt_features_k = [np.load(os.path.join(gt_pkl_root, 'kinetic_features', pkl)) for pkl in
                      sorted(os.listdir(os.path.join(gt_pkl_root, 'kinetic_features')))]

    pred_features_k = np.stack(pred_features_k)  # Nx72 p40
    gt_features_k = np.stack(gt_features_k)  # N' x 72 N' >> N

    gt_features_k, pred_features_k = normalize(gt_features_k, pred_features_k)

    print('Calculating metrics')

    fid_k = calc_fid(pred_features_k, gt_features_k)
    div_k_gt = calculate_avg_distance(gt_features_k)
    div_k = calculate_avg_distance(pred_features_k)
    metrics = {'fid_k': fid_k, 'div_k': div_k, 'div_k_gt': div_k_gt}

    return metrics

def quantized_group_metrics(predicted_pkl_root, gt_pkl_root):
    pred_features_k = []
    gt_features_k = []
    pred_features_k = [np.load(os.path.join(predicted_pkl_root, 'group_kinetic_features', pkl)) for pkl in
                       sorted(os.listdir(os.path.join(predicted_pkl_root, 'group_kinetic_features')))]
    gt_features_k = [np.load(os.path.join(gt_pkl_root, 'group_kinetic_features', pkl)) for pkl in
                      sorted(os.listdir(os.path.join(gt_pkl_root, 'group_kinetic_features')))]
    
    pred_features_k = np.stack(pred_features_k)  # Nx72 p40
    gt_features_k = np.stack(gt_features_k)  # N' x 72 N' >> N

    # gt_features_k, pred_features_k= np.log1p(gt_features_k), np.log1p(pred_features_k)
    gt_features_k, pred_features_k = normalize(gt_features_k, pred_features_k)
    
    print('Calculating metrics')

    fid_k = calc_fid(pred_features_k, gt_features_k)
    metrics = {'GMR': fid_k}

    return metrics


def calc_fid(kps_gen, kps_gt):
    print(kps_gen.shape)
    print(kps_gt.shape)

    # kps_gt, kps_gen = normalize1(kps_gt, kps_gen)
    eps = 1e-3

    mu_gen = np.mean(kps_gen, axis=0)
    sigma_gen = np.cov(kps_gen, rowvar=False)

    mu_gt = np.mean(kps_gt, axis=0)
    sigma_gt = np.cov(kps_gt, rowvar=False)

    sigma_gen += np.eye(sigma_gen.shape[0]) * eps
    sigma_gt += np.eye(sigma_gt.shape[0]) * eps

    mu1, mu2, sigma1, sigma2 = mu_gen, mu_gt, sigma_gen, sigma_gt

    diff = mu1 - mu2
    
    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            # raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)

def calc_diversity(feats):
    feat_array = np.array(feats)
    n, c = feat_array.shape
    diff = np.array([feat_array] * n) - feat_array.reshape(n, 1, c)
    return np.sqrt(np.sum(diff ** 2, axis=2)).sum() / n / (n - 1)


def calculate_avg_distance(feature_list):
    
    feature_list = np.stack(feature_list)
    # feature_list = normalize(feature_list)

    n = feature_list.shape[0]

    dist = 0
    for i in range(n):
        for j in range(i + 1, n):
            dist += np.linalg.norm(feature_list[i] - feature_list[j])
    dist /= (n * n - n) / 2
    return dist

def process_file(pkl):
    joint3d = np.load(pkl, allow_pickle=True).item()['pred_position'][:600, :]
    roott = joint3d[:1, :3]
    joint3d = joint3d - np.tile(roott, (1, 24))
    filename = pkl.split('/')[-1]
    np.save(pkl.replace(filename, 'kinetic_features/{}'.format(filename)), extract_kinetic_features(joint3d.reshape(-1, 24, 3)))


def mean_and_std(root):
    pred_features_k = [np.load(os.path.join(root, 'kinetic_features', pkl)) for pkl in
                       sorted(os.listdir(os.path.join(root, 'kinetic_features')))]
    pred_features_k = np.stack(pred_features_k)
    mean = pred_features_k.mean(axis=0)
    std = pred_features_k.std(axis=0)
    return mean, std

def calc_and_save_feats(root):

    files = []
    people_num_file = './data/People_Num'
    pkl_files = sorted([pkl for pkl in os.listdir(root) if pkl.endswith(".npy")])
    for i in range(0, len(pkl_files), 7):
        num_file_path = os.path.join(people_num_file, pkl_files[i].replace('json_dancer0.pkl.npy', 'txt'))
        f = open(num_file_path, "r")
        people_num = int(f.read())
        for j in range(people_num):
            files.append(os.path.join(root, pkl_files[i+j]))
    os.makedirs(os.path.join(root, 'kinetic_features'), exist_ok=True)
    with ProcessPoolExecutor(max_workers=16) as executor:
        executor.map(process_file, files)

    mean, std = mean_and_std('./data/aist_features_zero_start_test')     
    os.makedirs(os.path.join(root, 'group_kinetic_features'), exist_ok=True)
    people_num_file = './data/People_Num'
    pkl_files = sorted([pkl for pkl in os.listdir(root) if pkl.endswith(".npy")])
    GMC = []
    for i in range(0, len(pkl_files), 7):
        num_file_path = os.path.join(people_num_file, pkl_files[i].replace('json_dancer0.pkl.npy', 'txt'))
        f = open(num_file_path, "r")
        people_num = int(f.read())
        group_kinetic_feature = np.zeros(72)
        group_kinetic = []
        for j in range(people_num):
            kinetic_feature = np.load(os.path.join(root, 'kinetic_features', pkl_files[i+j]))
            group_kinetic_feature += kinetic_feature
            group_kinetic.append(kinetic_feature)

        group_kinetic = np.concatenate([group_kinetic], axis=0)
        group_kinetic = (group_kinetic-mean)/std+1e-4
        # v = np.mean((np.corrcoef(group_kinetic, rowvar=True)+1)/2)
        corr_matrix = (np.corrcoef(group_kinetic, rowvar=True) + 1) / 2
        np.fill_diagonal(corr_matrix, np.nan)
        v = np.nanmean(corr_matrix)
        GMC.append(v)
        group_kinetic_feature /= (people_num)
        np.save(os.path.join(root, 'group_kinetic_features/{}'.format(pkl_files[i])), group_kinetic_feature)
    GMC_value = np.mean(GMC)
    print('GMC:', GMC_value*100)

def gt_process(source, destination):
    for file in sorted(os.listdir(source)):
        json_path = os.path.join(source, file)
        with open(json_path) as f:
            sample_dict = json.loads(f.read())
            np_dance = np.array(sample_dict['dance_array'])
            for i in range(np_dance.shape[0]):
                if os.path.exists(os.path.join(destination, file + '_dancer{}.pkl.npy'.format(i))):
                    return
                np.save(os.path.join(destination, file + '_dancer{}.pkl.npy'.format(i)), {'pred_position': np_dance[i]})


def fid(pred_path, mode=None):
    files = []
    people_num_file = './data/People_Num'
    pkl_files = sorted([pkl for pkl in os.listdir(pred_path) if pkl.endswith(".npy")])
    for i in range(0, len(pkl_files), 7):
        num_file_path = os.path.join(people_num_file, pkl_files[i].replace('json_dancer0.pkl.npy', 'txt'))
        f = open(num_file_path, "r")
        people_num = int(f.read())
        for j in range(people_num):
            files.append(os.path.join(pred_path, pkl_files[i+j]))
    os.makedirs(os.path.join(pred_path, 'kinetic_features'), exist_ok=True)
    with ProcessPoolExecutor(max_workers=16) as executor:
        executor.map(process_file, files)

    if mode == 'eval':
        gt_root = './data/aist_features_zero_start_eval'
    else:
        gt_root = './data/aist_features_zero_start_test'

    pred_features_k = []
    gt_features_k = []
    pred_features_k = [np.load(os.path.join(pred_path, 'kinetic_features', pkl)) for pkl in
                       sorted(os.listdir(os.path.join(pred_path, 'kinetic_features')))]
    gt_features_k = [np.load(os.path.join(gt_root, 'kinetic_features', pkl)) for pkl in
                      sorted(os.listdir(os.path.join(gt_root, 'kinetic_features')))]

    pred_features_k = np.stack(pred_features_k)  # Nx72 p40
    gt_features_k = np.stack(gt_features_k)  # N' x 72 N' >> N
    # gt_features_k, pred_features_k= np.log1p(gt_features_k), np.log1p(pred_features_k)
    gt_features_k, pred_features_k = normalize(gt_features_k, pred_features_k)

    fid_k = calc_fid(pred_features_k, gt_features_k)
    div_k = calculate_avg_distance(pred_features_k)
    bs = calc_ba_score(pred_path, mode)
    metrics = {'fid': fid_k, 'div': div_k, 'bs': bs}

    return metrics

def init():
    gt_source, gt_destination = './data/aistpp_eval_wav', './data/aist_features_zero_start_eval'
    os.makedirs(gt_destination, exist_ok=True)
    gt_process(gt_source, gt_destination)
    gt_root = './data/aist_features_zero_start_eval'
    print('Calculating and saving Test features')
    calc_and_save_feats(gt_root)

    gt_source, gt_destination = './data/aistpp_test_full_wav', './data/aist_features_zero_start_test'
    os.makedirs(gt_destination, exist_ok=True)
    gt_process(gt_source, gt_destination)
    gt_root = './data/aist_features_zero_start_test'
    print('Calculating and saving Eval features')
    calc_and_save_feats(gt_root)

def init_debug():
    gt_source, gt_destination = './debug_data/aistpp_eval_wav', './debug_data/aist_features_zero_start_eval'
    os.makedirs(gt_destination, exist_ok=True)
    gt_process(gt_source, gt_destination)
    gt_root = './debug_data/aist_features_zero_start_eval'
    print('Calculating and saving Test features')
    calc_and_save_feats(gt_root)

    gt_source, gt_destination = './debug_data/aistpp_test_full_wav', './debug_data/aist_features_zero_start_test'
    os.makedirs(gt_destination, exist_ok=True)
    gt_process(gt_source, gt_destination)
    gt_root = './debug_data/aist_features_zero_start_test'
    print('Calculating and saving Eval features')
    calc_and_save_feats(gt_root)


if __name__ == '__main__':

    gt_root = './data/aist_features_zero_start_test'
    pred_root = './experiments/cc_motion_gpt/eval/pkl/ep000250'
    calc_and_save_feats(pred_root)

    print('Calculating metrics')
    print(quantized_metrics(pred_root, gt_root))
    print(quantized_group_metrics(pred_root, gt_root))

    # Beat Similarity
    print('Beat Similarity')
    print(calc_ba_score(pred_root))

