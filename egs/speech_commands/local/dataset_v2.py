#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 20:24:15 2021

@author: krishna
"""

import os
import numpy as np
import glob
import argparse

class SpeechCommandDataset:
    IGNORE_CLASS='_background_noise_'
    NUM_CLASSES=35
    def __init__(self,):
        pass
    
    @staticmethod
    def create_kaldi(files, mode='train'):
        os.makedirs('data/'+mode, exist_ok=True)
        with open('data/'+mode+'/wav.scp','w') as f_wav,open('data/'+mode+'/utt2spk','w') as f_u2s, open('data/'+mode+'/spk2utt','w') as f_s2u, open('data/'+mode+'/text','w') as f_txt:
            for filepath in files:
                f_wav.write(filepath.split('/')[-2]+'_'+filepath.split('/')[-1]+' '+filepath+'\n')
                f_u2s.write(filepath.split('/')[-2]+'_'+filepath.split('/')[-1]+' '+filepath.split('/')[-2]+'_'+filepath.split('/')[-1]+'\n')
                f_s2u.write(filepath.split('/')[-2]+'_'+filepath.split('/')[-1]+' '+filepath.split('/')[-2]+'_'+filepath.split('/')[-1]+'\n')
                f_txt.write(filepath.split('/')[-2]+'_'+filepath.split('/')[-1]+' '+filepath.split('/')[-2]+'\n')
    
    @classmethod
    def process_data(cls,data_folder, valid_file, test_file):
        files_list = list()
        train_files = []
        test_files = []
        valid_files = []
        
        class_names =[filepath.split('/')[-2] for filepath in sorted(glob.glob(data_folder+'/*/'))]
        class_names.remove(cls.IGNORE_CLASS)
        
        for class_name in class_names:
            if os.path.exists(data_folder+'/'+class_name):
                files_list+=glob.glob(data_folder+'/'+class_name+'/*.wav')
            
        with open(valid_file) as fid:
            for line in fid:
                filename = line.rstrip('\n')
                if data_folder+'/'+filename in files_list:
                    valid_files.append(data_folder+'/'+filename)
                    
        with open(test_file) as fid:
            for line in fid:
                filename = line.rstrip('\n')
                if data_folder+'/'+filename in files_list:
                    test_files.append(data_folder+'/'+filename)

        train_files = list(set(files_list) - set(valid_files) -  set(test_files))
        return train_files, test_files, valid_files

        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='./data2/')
    cmd_args = parser.parse_args()
    data_folder = cmd_args.dataset_path
    valid_file = data_folder+'/validation_list.txt'
    test_file = data_folder+'/testing_list.txt'
    dataset = SpeechCommandDataset()
    train_files, test_files, valid_files = dataset.process_data(data_folder, valid_file, test_file)
    dataset.create_kaldi(sorted(train_files), mode='train')
    dataset.create_kaldi(sorted(valid_files), mode='valid')
    dataset.create_kaldi(sorted(test_files), mode='test')
