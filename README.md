# CoDancers

Code for ICMR 2024 paper "CoDancers: Music-Driven Coherent Group Dance Generation"

[[Paper]](https://dl.acm.org/doi/10.1145/3652583.3657998) | [[Video Demo]](https://youtu.be/ZETMiNsC93g?si=ig5LIbDzcSybN0EX)

<a href="https://www.youtube.com/watch?v=ZETMiNsC93g" target="_blank">
    <img src="https://github.com/XulongT/CoDancers/blob/main/demo/play_demo.png" alt="Watch the video" width="400"/>
</a>


# Code

## Set up code environment

To set up the necessary environment for running the project, follow these steps:

1. **Create a new conda environment**:  

   ```
   conda create -n CoD_env python=3.8
   conda activate CoD_env
   ```

2. **Install PyTorch and dependencies**

   ```
   conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge
   conda install --file requirements.txt
   ```

## Download

Directly download our preprocessed feature from [here](https://drive.google.com/file/d/1KGCeQMH3dy62zwKDXafb8ToenbY-RBuC/view?usp=drive_link) into ./data folder.

To test with our pretrained models, please download the weights from [here](https://drive.google.com/file/d/1AtxnGhVQ7Wa-i5obwqIY8mQGvoqHg4jO/view?usp=drive_link) (Google Drive) and place them into ./experiments folder.

## Directory Structure

After downloading the corresponding data and weights, please move the relevant files to their respective directories.

The file directory structure is as follows:

```
|-- configs
|-- data
    |-- aistpp_music_librosa_3.75fps
    |-- aistpp_music_mert_3.75fps
    |-- aistpp_test_full_wav
    |-- aist_features_zero_start_test
        |-- group_kinetic_features
        |-- kinetic_features
    |-- People_Num
|-- dataset
|-- experiments
    |-- cc_motion_gpt
        |-- ckpt
    |-- sep_vqvae
        |-- ckpt
|-- models
    |-- utils
|-- querybank
|-- utils
    |-- features
```

## Training

    Coming soon...

## Evaluation

### 1. Generate Dancing Results

To test the VQ-VAE, use the following command:

    python -u main.py --config configs/sep_vqvae.yaml --eval

To test GPT, use the following command:

    python -u main_gpt_all.py --config configs/cc_motion_gpt.yaml --eval

### 2. Dance quality evaluations

After generating the dance in the above step, run the following codes.

To evaluate the VQ-VAE, use the following command:

    python ./utils/vqvae_metrics.py

To evaluate the GPT, use the following command:

```
python ./utils/gpt_metrics.py
```

For calculating the Trajectory Intersection Frequency (TIF) metric and performing Inverse Kinematics, due to the lengthy computation time, this repository does not provide demonstrations. You can refer to [vedo](https://github.com/marcomusy/vedo) and [Pose to SMPL](https://github.com/Dou-Yiming/Pose_to_SMPL) for further information. 

If you have any questions, don't hesitate to submit an issue or contact me.

# Acknowledgments

Our code is based on [Bailando](https://github.com/lisiyao21/Bailando/tree/main) , and some of the data is provided by [AIOZ-GDANCE](https://github.com/aioz-ai/AIOZ-GDANCE). We sincerely appreciate for their contributions.

# Citation

    @inproceedings{yang2024codancers,
      title={CoDancers: Music-Driven Coherent Group Dance Generation with Choreographic Unit},
      author={Yang, Kaixing and Tang, Xulong and Diao, Ran and Liu, Hongyan and He, Jun and Fan, Zhaoxin},
      booktitle={Proceedings of the 2024 International Conference on Multimedia Retrieval},
      pages={675--683},
      year={2024}
    }
