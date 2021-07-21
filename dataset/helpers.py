import os
import numpy as np
import torch

def create_dict(labels_list):
    label_dict = dict()
    for i in range(len(labels_list)):
        label_dict[labels_list[i]] = i
    return label_dict

def map_key2label(key):
    if len(key.split('-'))==2:
        return key.split('-')[1].split('_')[0]
    else:
        return key.split('_')[0]
        

def collate_fun(batch):
    feats = torch.cat([item[0] for item in batch])
    labels = torch.cat([item[1] for item in batch])
    return feats, labels
    
