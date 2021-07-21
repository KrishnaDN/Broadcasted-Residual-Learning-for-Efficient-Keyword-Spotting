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
from dataset.helpers import create_dict, map_key2label,collate_fun
from torch.utils.data import DataLoader

class Augmentation(object):
    def __init__(self,cmvn_file, 
                 labels):
        self.cmvn_file = cmvn_file
        self._load_kaldi_cmvn
        self.label_dict = create_dict(labels)
        
    @property
    def _load_kaldi_cmvn(self,):
        means = []
        variance = []
        with open(self.cmvn_file, 'r') as fid:
            # kaldi binary file start with '\0B'
            if fid.read(2) == '\0B':
                logging.error('kaldi cmvn binary file is not supported, please '
                            'recompute it by: compute-cmvn-stats --binary=false '
                            ' scp:feats.scp global_cmvn')
                sys.exit(1)
            fid.seek(0)
            arr = fid.read().split()
            assert (arr[0] == '[')
            assert (arr[-2] == '0')
            assert (arr[-1] == ']')
            feat_dim = int((len(arr) - 2 - 2) / 2)
            for i in range(1, feat_dim + 1):
                means.append(float(arr[i]))
            count = float(arr[feat_dim + 1])
            for i in range(feat_dim + 2, 2 * feat_dim + 2):
                variance.append(float(arr[i]))

        for i in range(len(means)):
            means[i] /= count
            variance[i] = variance[i] / count - means[i] * means[i]
            if variance[i] < 1.0e-20:
                variance[i] = 1.0e-20
            variance[i] = 1.0 / math.sqrt(variance[i])
        self.mean = means
        self.istd = variance


    @staticmethod
    def _spec_augmentation(x,
                        warp_for_time=False,
                        num_t_mask=2,
                        num_f_mask=2,
                        max_t=50,
                        max_f=10,
                        max_w=80):
        """ Deep copy x and do spec augmentation then return it
        Args:
            x: input feature, T * F 2D
            num_t_mask: number of time mask to apply
            num_f_mask: number of freq mask to apply
            max_t: max width of time mask
            max_f: max width of freq mask
            max_w: max width of time warp

        Returns:
            augmented feature
        """

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
    def _spec_substitute(x, max_t=20, num_t_sub=3):
        """ Deep copy x and do spec substitute then return it

        Args:
            x: input feature, T * F 2D
            max_t: max width of time substitute
            num_t_sub: number of time substitute to apply

        Returns:
            augmented feature
        """
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

    def _load_feature(self,x):

        """ Load acoustic feature from files.
        The features have been prepared in previous step, usualy by Kaldi.
        Args:
            batch: a list of tuple (wav id , feature ark path).
        Returns:
            (keys, feats, labels)
        """
        key = x[0]
        mat = kaldi_io.read_mat(x[1])
        mat = mat - self.mean
        mat = mat * self.istd
        label = self.label_dict[map_key2label(key)]
        return mat, label

class AudioDataset(Augmentation):
    def __init__(self, 
                 data_file, 
                 cmvn_file, 
                 labels, 
                 feat_dim, 
                 spec_augment=True, 
                 spec_substitute=True, 
                 max_frames=98):
        """ Dataset

        Args:
            data_file: Input data file (feat.scp)
            cmvn_file: cmvn file computed after feature extraction step (global_cmvn)
            labels: list of all the labels. for example ['cat', 'dog',.......]
            feat_dim: integer (40)
            spec_augment: Do you want to apply spec_augment? (True of False)
            spec_substitute: Do you want to apply spectral substitution? (True of False)
            max_frames: integer ( for example 98)
        """
        
        super(AudioDataset, self).__init__(cmvn_file, 
                                           labels) 
        self.data_file = data_file
        self.spec_augment = spec_augment
        self.spec_substitute = spec_substitute
        self.ark_paths = [line.rstrip('\n') for line in open(self.data_file)]
        self.max_frames = max_frames

    def __len__(self):
        return len(self.ark_paths)

    def __getitem__(self, idx):
        ark_path = self.ark_paths[idx].split(' ')
        x, label = self._load_feature(ark_path)
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
    from dataset.dataset import AudioDataset
    config_file = './egs/speech_commands/conf/transformer_v2_35.yaml'
    with open(config_file) as f:
        params = yaml.safe_load(f)
    cmvn_file = params['data']['cmvn_file']
    data_file = params['data']['train']
    labels = params['data']['labels']
    train_dataset = AudioDataset(data_file, cmvn_file, labels, **params['dataset_conf']['kaldi_offline_conf'] )
    print(train_dataset.__getitem__(1))
    labels = params['data']['labels']
    data_loader = DataLoader(train_dataset,shuffle=True, batch_size=10, num_workers=4, collate_fn=collate_fun)
    