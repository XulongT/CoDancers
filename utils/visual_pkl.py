import os
import numpy as np
import pickle as pkl
# import vedo
import matplotlib.pyplot as plt
import numpy as np
import subprocess
from tqdm import tqdm
import shutil


def adjust_pose(joint):
    np_dance_trans = np.zeros([2, 25]).copy()
    joint = np.transpose(joint)

    # head
    np_dance_trans[:, 0] = joint[:, 15]

    # neck
    np_dance_trans[:, 1] = joint[:, 12]

    # left up
    np_dance_trans[:, 2] = joint[:, 16]
    np_dance_trans[:, 3] = joint[:, 18]
    np_dance_trans[:, 4] = joint[:, 20]

    # right up
    np_dance_trans[:, 5] = joint[:, 17]
    np_dance_trans[:, 6] = joint[:, 19]
    np_dance_trans[:, 7] = joint[:, 21]

    np_dance_trans[:, 8] = joint[:, 0]

    np_dance_trans[:, 9] = joint[:, 1]
    np_dance_trans[:, 10] = joint[:, 4]
    np_dance_trans[:, 11] = joint[:, 7]

    np_dance_trans[:, 12] = joint[:, 2]
    np_dance_trans[:, 13] = joint[:, 5]
    np_dance_trans[:, 14] = joint[:, 8]

    np_dance_trans[:, 15] = joint[:, 15]
    np_dance_trans[:, 16] = joint[:, 15]
    np_dance_trans[:, 17] = joint[:, 15]
    np_dance_trans[:, 18] = joint[:, 15]

    np_dance_trans[:, 19] = joint[:, 11]
    np_dance_trans[:, 20] = joint[:, 11]
    np_dance_trans[:, 21] = joint[:, 8]

    np_dance_trans[:, 22] = joint[:, 10]
    np_dance_trans[:, 23] = joint[:, 10]
    np_dance_trans[:, 24] = joint[:, 7]

    np_dance_trans = np.transpose(np_dance_trans)

    return np_dance_trans


pose_edge_list = [
    [0, 1], [1, 8],  # body
    [1, 2], [2, 3], [3, 4],  # right arm
    [1, 5], [5, 6], [6, 7],  # left arm
    [8, 9], [9, 10], [10, 11], [11, 24], [11, 22], [22, 23],  # right leg
    [8, 12], [12, 13], [13, 14], [14, 21], [14, 19], [19, 20]  # left leg
]
pose_color_list = [
    [153, 0, 51], [153, 0, 0],
    [153, 51, 0], [153, 102, 0], [153, 153, 0],
    [102, 153, 0], [51, 153, 0], [0, 153, 0],
    [0, 153, 51], [0, 153, 102], [0, 153, 153], [0, 153, 153], [0, 153, 153], [0, 153, 153],
    [0, 102, 153], [0, 51, 153], [0, 0, 153], [0, 0, 153], [0, 0, 153], [0, 0, 153]
]


def plot_line(joint, ax):
    for i, e in enumerate(pose_edge_list):
        ax.plot([joint[e[0]][0], joint[e[1]][0]], [joint[e[0]][1], joint[e[1]][1]],
                color=(pose_color_list[i][0] / 255, pose_color_list[i][1] / 255, pose_color_list[i][2] / 255))


def vis(root_path, mus_path, output_path):
    pkl_files = [file for file in sorted(os.listdir(root_path)) if file.endswith(".npy")]

    for i in tqdm(range(0, len(pkl_files), 7)):
        motion_seq = []
        for j in range(7):
            pkl = pkl_files[i + j]
            joint3d = np.load(os.path.join(root_path, pkl), allow_pickle=True).item()['pred_position'].reshape(-1, 24,
                                                                                                               3)
            flag = False
            for k in range(0, 320, 80):
                if np.sum(joint3d[k + 80] - joint3d[k]) > 0.3:
                    flag = True
                    break
            if flag == False:
                continue
            joint3d = np.expand_dims(joint3d, axis=0)
            motion_seq.append(joint3d)

        all_joints3d = np.concatenate(motion_seq, axis=0)
        print(all_joints3d.shape)
        img_path = './img/{}'.format(pkl_files[i])
        os.makedirs(img_path, exist_ok=True)

        fig, ax = plt.subplots()
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')

        for k in range(all_joints3d.shape[1]):
            joints3d = all_joints3d[:, k]  # first frame
            for joint in joints3d:
                joint = adjust_pose(joint[:, :2])
                ax.scatter(joint[:, 0], joint[:, 1], color='white')
                plot_line(joint, ax)

            plt.savefig(os.path.join(img_path, '{}.png'.format(k)))
            plt.cla()

        # music_path = mus_path + '/{}.wav'.format(pkl_files[i].replace('.json_dancer0.pkl.npy', ''))
        # video_path = output_path + '/{}.mp4'.format(pkl_files[i].replace('.json_dancer0.pkl.npy', ''))
        # cmd = f"ffmpeg -r 30 -i {img_path}/%d.png -vb 20M -vcodec mpeg4 -y {video_path} -loglevel quiet"
        # os.system(cmd)
        # video_path_new = video_path.replace('.mp4', '_audio.mp4')
        # cmd_audio = f"ffmpeg -i {video_path} -i {music_path} -map 0:v -map 1:a -c:v copy -shortest -y {video_path_new} -loglevel quiet"
        # os.system(cmd_audio)
        if os.path.exists(img_path):
            shutil.rmtree(img_path)


if __name__ == '__main__':
    mus_path = '../aist_plusplus_final/all_musics'
    # pkl_path = '../experiments/cc_motion_gpt/eval/pkl/ep000010'
    pkl_path = '../experiments/sep_vqvae/eval/pkl/ep000020'
    video_path = '../test_video'
    os.makedirs(video_path, exist_ok=True)
    vis(pkl_path, mus_path, video_path)

