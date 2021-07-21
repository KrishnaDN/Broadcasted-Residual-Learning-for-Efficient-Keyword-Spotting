import os
import numpy as np
import torchaudio.compliance.kaldi as kaldi
import argparse
import codecs
import copy
import logging
import random
import numpy as np
import torch
import torchaudio
import yaml
from PIL import Image
from PIL.Image import BICUBIC
from torch.utils.data import Dataset, DataLoader
import kaldi_io
import math
import sys
import numpy as np
from typing import Optional, List, Dict
from dataset.helpers import create_dict, map_key2label,collate_fun
import warnings
warnings.filterwarnings('ignore')

class Augmentation(object):
    def __init__(self,):
        pass
    
    def _spec_augmentation(self,x,
                        warp_for_time=False,
                        num_t_mask=2,
                        num_f_mask=2,
                        max_t=20,
                        max_f=5,
                        max_w=20):
        y = np.copy(x)
        max_frames = y.shape[0]
        max_freq = y.shape[1]

        # time warp
        if warp_for_time and max_frames > max_w * 2:
            center = random.randrange(max_w, max_frames - max_w)
            warped = random.randrange(center - max_w, center + max_w) + 1

            left = Image.fromarray(x[:center]).resize((max_freq, warped), BICUBIC)
            right = Image.fromarray(x[center:]).resize((max_freq,
                                                    max_frames - warped),
                                                    BICUBIC)
            y = np.concatenate((left, right), 0)
        # time mask
        for i in range(num_t_mask):
            start = random.randint(0, max_frames - 1)
            length = random.randint(1, max_t)
            end = min(max_frames, start + length)
            y[start:end, :] = 0
        # freq mask
        for i in range(num_f_mask):
            start = random.randint(0, max_freq - 1)
            length = random.randint(1, max_f)
            end = min(max_freq, start + length)
            y[:, start:end] = 0
        return y

    @staticmethod
    def _spec_substitute(x, max_t=5, num_t_sub=3):
        y = np.copy(x)
        max_frames = y.shape[0]
        for i in range(num_t_sub):
            start = random.randint(0, max_frames - 1)
            length = random.randint(1, max_t)
            end = min(max_frames, start + length)
            # only substitute the earlier time chosen randomly for current time
            pos = random.randint(0, start)
            y[start:end, :] = y[start - pos:end - pos, :]
        return y

    def _load_wav_with_speed(self,wav_file, speed):
        if speed == 1.0:
            return torchaudio.load_wav(wav_file)
        else:
            si, _ = torchaudio.info(wav_file)
            E = torchaudio.sox_effects.SoxEffectsChain()
            E.append_effect_to_chain('speed', speed)
            E.append_effect_to_chain("rate", si.rate)
            E.set_input_file(wav_file)
            wav, sr = E.sox_build_flow_effects()
            wav = wav * (1 << 15)
            return wav, sr


    @property
    def _compute_cmvn_stats(self,):
        print('Computing CMVN stats')
        features = []
        for item in self.wav_paths:
            wav_file = os.path.join(self.root_folder,item.split(' ')[1])
            waveform, sample_rate = torchaudio.load_wav(wav_file)
            mat = kaldi.fbank(
                    waveform,
                    num_mel_bins=self.feature_extraction_conf['mel_bins'],
                    frame_length=self.feature_extraction_conf['frame_length'],
                    frame_shift=self.feature_extraction_conf['frame_shift'],
                    energy_floor=0.0,
                    sample_frequency=sample_rate
                )
            features.append(mat.detach().numpy())
        feats_stack = np.concatenate((features), axis=0)
        mean_vec = np.mean(feats_stack, axis=0)
        std_vec = np.std(feats_stack, axis=0)
        datum = dict()
        datum['mean_vec'] = mean_vec
        datum['std_vec'] = std_vec
        return datum
        
    
    def _load_cmvn(self, cmvn_file):
        print('*****Loading CMVN stats*****')
        stats = np.load(cmvn_file, allow_pickle=True).item()
        self.mean, self.std = np.expand_dims(stats['mean_vec'],axis=0), np.expand_dims(stats['std_vec'], axis=0)

    
    def _load_feature(self, x):
        key = x[0]
        wav_file = os.path.join(self.root_folder,x[1])
        if self.speed_perturb:
            weights = [1, 1, 1]
            speed = random.choices(self.speed_perturb_params, weights, k=1)[0]
            waveform, sample_rate = self._load_wav_with_speed(wav_file, speed)
        else:
            waveform, sample_rate = torchaudio.load_wav(wav_file)
            
        mat = kaldi.fbank(
                waveform,
                num_mel_bins=self.feature_extraction_conf['mel_bins'],
                frame_length=self.feature_extraction_conf['frame_length'],
                frame_shift=self.feature_extraction_conf['frame_shift'],
                energy_floor=0.0,
                sample_frequency=sample_rate
            )
        assert mat.shape[1] == self.mean.shape[1]
        mat = mat.detach().numpy()
        
        mat_norm = (mat-self.mean)/(self.std+10e-10)
        label = self.label_dict[map_key2label(key)]
        return mat_norm, label
        

class KaldiDataset(Augmentation):
    def __init__(self,
                 data_folder: str,
                 data_file: str,
                 labels: List,
                 speed_perturb: bool,
                 speed_perturb_params: Optional[List],
                 spec_augment: Optional[bool],
                 spec_augment_conf: Optional[Dict],
                 spec_substitute: Optional[bool],
                 feature_extraction_conf: Optional[Dict],
                 ):
        super().__init__()
        self.root_folder = data_folder
        self.data_file = data_file
        self.labels = labels
        self.speed_perturb = speed_perturb
        self.speed_perturb_params = speed_perturb_params
        self.spec_augment = spec_augment
        self.spec_augment_conf = spec_augment_conf
        self.spec_substitute = spec_substitute
        self.feature_extraction_conf = feature_extraction_conf
        self.wav_paths = [line.rstrip('\n') for line in open(self.data_file)]
        self.max_frames = self.feature_extraction_conf['max_frames']
        self.label_dict = create_dict(self.labels)
        
        
    def __len__(self):
        return len(self.wav_paths)

    def __getitem__(self, idx):
        wav_path = self.wav_paths[idx].split(' ')
        x, label = self._load_feature(wav_path)
        
        if x.shape[0] >=self.max_frames:
            feat = x[:self.max_frames,:]
        else:
            feat = np.concatenate((x, np.zeros((self.max_frames - x.shape[0], x.shape[1]))), axis=0)
        features = list()
        labels = list()
        features.append(feat)
        labels.append(label)
        if self.spec_augment:
            features.append(self._spec_augmentation(feat))
            labels.append(label)

        if self.spec_substitute:
            features.append(self._spec_substitute(feat))
            labels.append(label)
        return torch.Tensor(features).unsqueeze(1), torch.LongTensor(labels)


if __name__=='__main__':
    import yaml
    config = '/home/krishna/Krishna/keyword-detection/egs/speech_commands/conf/transformer_v2_35.yaml'
    with open(config, 'r') as f:
        params =yaml.safe_load(f)
    data_folder = params['data']['data_folder']
    data_file = params['data']['train']
    labels = params['data']['labels']
    kaldi_dataset = KaldiDataset(data_folder, data_file, labels, **params['dataset_conf']['kaldi_online_conf'])
    #mat, label = kaldi_dataset.__getitem__(1)
    feats = kaldi_dataset._compute_cmvn_stats