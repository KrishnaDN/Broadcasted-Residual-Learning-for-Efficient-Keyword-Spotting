from __future__ import print_function

import argparse
import copy
import logging
import os
import warnings
import torch
import torch.distributed as dist
import torch.optim as optim
import numpy as np
import yaml
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from dataset import AudioDataset, KaldiDataset
from dataset.helpers import collate_fun
from model import Models
from utils import load_checkpoint, save_checkpoint, average_models
from bin import BuildOptimizer, BuildScheduler
from bin import Executor

warnings.filterwarnings("ignore")

class Trainer:
    def __init__(self, params, args):
        self.params = params
        self.args = args
    
    @property
    def initialize_model(self,):
        model = Models[self.params['model']['model_type']](self.params['model'])
        return model
    
    def evaluate(self,):
        if self.params['dataset_conf']['kaldi_offline']:
            test_dataset_conf = copy.deepcopy(self.params['dataset_conf']['kaldi_offline_conf'])
            test_dataset_conf['spec_augment'] = False
            test_dataset_conf['spec_substitute'] = False
            cmvn_file = self.params['data']['cmvn_file']
            data_file = self.params['data']['test']
            labels = self.params['data']['labels']
            test_dataset = AudioDataset(data_file, cmvn_file, labels, **test_dataset_conf)
        elif self.params['dataset_conf']['kaldi_online']:
            data_folder = self.params['data']['data_folder']
            labels = self.params['data']['labels']
            test_dataset_conf = copy.deepcopy(self.params['dataset_conf']['kaldi_online_conf'])
            test_dataset_conf['speed_perturb'] = False
            test_dataset_conf['spec_augment'] = False
            test_dataset_conf['spec_substitute'] = False
            data_file = self.params['data']['valid']
            cmvn_file = os.path.join(self.params['train']['exp_dir'], self.params['train']['model_dir'],'global_cmvn.npy')
            test_dataset = KaldiDataset(data_folder, data_file, labels, **test_dataset_conf)
            test_dataset._load_cmvn(cmvn_file)
        else:
            print('*****Unknown Dataloader***')
            print('***** Please check your config file ***')
        
        test_sampler = None
        test_data_loader = DataLoader(test_dataset,
                                    collate_fn=collate_fun,
                                    sampler=test_sampler,
                                    shuffle=False,
                                    batch_size=1,
                                    num_workers=self.args.num_workers)
        model = self.initialize_model
        src_path = os.path.join(self.params['train']['exp_dir'], self.params['train']['model_dir'])
        self.params['eval']['average_model']['dst_model'] = os.path.join(src_path, self.params['eval']['average_model']['dst_model'])
        try:
            average_models(src_path, **self.params['eval']['average_model']) 
            load_checkpoint(model=model, path=self.params['eval']['average_model']['dst_model'])
        except:
            raise 
        use_cuda = self.args.gpu >= 0 and torch.cuda.is_available()
        device = torch.device('cuda' if use_cuda else 'cpu')
        model = model.to(device)
        executor = Executor()
        test_acc = executor.evaluation(model, test_data_loader, device)
        print(f'Evaluation data accuracy {test_acc*100}')

        
    def train(self,):
        if self.params['dataset_conf']['kaldi_offline']:
            cmvn_file = self.params['data']['cmvn_file']
            data_file = self.params['data']['train']
            labels = self.params['data']['labels']
            train_dataset = AudioDataset(data_file, cmvn_file, labels, **self.params['dataset_conf']['kaldi_offline_conf'])
            cv_dataset_conf = copy.deepcopy(self.params['dataset_conf']['kaldi_offline_conf'])
            cv_dataset_conf['spec_augment'] = False
            cv_dataset_conf['spec_substitute'] = False
            cmvn_file = self.params['data']['cmvn_file']
            data_file = self.params['data']['valid']
            labels = self.params['data']['labels']
            cv_dataset = AudioDataset(data_file, cmvn_file, labels, **cv_dataset_conf)
        elif self.params['dataset_conf']['kaldi_online']:
            data_folder = self.params['data']['data_folder']
            data_file = self.params['data']['train']
            labels = self.params['data']['labels']
            train_dataset = KaldiDataset(data_folder, data_file, labels, **self.params['dataset_conf']['kaldi_online_conf'])
            cmvn_stats = train_dataset._compute_cmvn_stats
            os.makedirs(os.path.join(self.params['train']['exp_dir'], self.params['train']['model_dir']), exist_ok=True)
            cmvn_file = os.path.join(self.params['train']['exp_dir'], self.params['train']['model_dir'],'global_cmvn.npy')
            print(f'Saving CMVN stats at {cmvn_file}')
            np.save(cmvn_file, cmvn_stats)
            train_dataset._load_cmvn(cmvn_file)
            cv_dataset_conf = copy.deepcopy(self.params['dataset_conf']['kaldi_online_conf'])
            cv_dataset_conf['speed_perturb'] = False
            cv_dataset_conf['spec_augment'] = False
            cv_dataset_conf['spec_substitute'] = False
            data_file = self.params['data']['valid']
            cmvn_file = os.path.join(self.params['train']['exp_dir'], self.params['train']['model_dir'],'global_cmvn.npy')
            cv_dataset = KaldiDataset(data_folder, data_file, labels, **cv_dataset_conf)
            cv_dataset._load_cmvn(cmvn_file)
            
        else:
            print('*****Unknown Dataloader***')
            print('***** Please check your config file ***')
            
        distributed = self.args.world_size > 1
        if distributed:
            logging.info('training on multiple gpus, this gpu {}'.format(self.args.gpu))
            dist.init_process_group(self.args.dist_backend,
                                    init_method=self.args.init_method,
                                    world_size=self.args.world_size,
                                    rank=self.args.rank)
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset, shuffle=True)
            cv_sampler = torch.utils.data.distributed.DistributedSampler(
                cv_dataset, shuffle=False)
        else:
            train_sampler = None
            cv_sampler = None

        train_data_loader = DataLoader(train_dataset,
                                    collate_fn=collate_fun,
                                    sampler=train_sampler,
                                    shuffle=(train_sampler is None),
                                    batch_size=self.params['train']['batch_size'],
                                    num_workers=self.args.num_workers)
        cv_data_loader = DataLoader(cv_dataset,
                                    collate_fn=collate_fun,
                                    sampler=cv_sampler,
                                    shuffle=False,
                                    batch_size=self.params['train']['batch_size'],
                                    num_workers=self.args.num_workers)
        model = self.initialize_model
        print(model)
        executor = Executor()
        if self.args.checkpoint is not None:
            infos = load_checkpoint(model, self.args.checkpoint)
        else:
            infos = {}
        start_epoch = infos.get('epoch', -1) + 1
        cv_loss = infos.get('cv_loss', 0.0)
        step = infos.get('step', -1)

        num_epochs = self.params['train']['epochs']
        model_dir = os.path.join(self.params['train']['exp_dir'],self.params['train']['model_dir'])
        writer = None
        if self.args.rank == 0:
            os.makedirs(model_dir, exist_ok=True)
            exp_id = os.path.basename(model_dir)
            writer = SummaryWriter(os.path.join(model_dir, exp_id))

        if distributed:
            assert (torch.cuda.is_available())
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(
                model, find_unused_parameters=True)
            device = torch.device("cuda")
        else:
            use_cuda = self.args.gpu >= 0 and torch.cuda.is_available()
            device = torch.device('cuda' if use_cuda else 'cpu')
            model = model.to(device)

        optimizer = BuildOptimizer[self.params['train']['optimizer_type']](
            filter(lambda p: p.requires_grad, model.parameters()), **self.params['train']['optimizer']
        )
        scheduler = BuildScheduler[self.params['train']['scheduler_type']](optimizer, **self.params['train']['scheduler'])
        final_epoch = None
        self.params['rank'] = self.args.rank
        self.params['is_distributed'] = distributed
        if start_epoch == 0 and self.args.rank == 0:
            save_model_path = os.path.join(model_dir, 'init.pt')
            save_checkpoint(model, save_model_path)
        
        executor.step = step
        scheduler.step()
        for epoch in range(start_epoch, num_epochs):
            if distributed:
                train_sampler.set_epoch(epoch)
            lr = optimizer.param_groups[0]['lr']
            logging.info('Epoch {} TRAIN info lr {}'.format(epoch, lr))
            executor.train(model, optimizer, scheduler, train_data_loader, device,
                        writer, self.params)
            total_loss, num_seen_utts = executor.cv(model, cv_data_loader, device,
                                                    self.params)
            if self.args.world_size > 1:
                # all_reduce expected a sequence parameter, so we use [num_seen_utts].
                num_seen_utts = torch.Tensor([num_seen_utts]).to(device)
                # the default operator in all_reduce function is sum.
                dist.all_reduce(num_seen_utts)
                total_loss = torch.Tensor([total_loss]).to(device)
                dist.all_reduce(total_loss)
                cv_loss = total_loss[0] / num_seen_utts[0]
                cv_loss = cv_loss.item()
            else:
                cv_loss = total_loss / num_seen_utts

            logging.info('Epoch {} CV info cv_loss {}'.format(epoch, cv_loss))
            if self.args.rank == 0:
                save_model_path = os.path.join(model_dir, '{}.pt'.format(epoch))
                save_checkpoint(
                    model, save_model_path, {
                        'epoch': epoch,
                        'lr': lr,
                        'cv_loss': cv_loss,
                        'step': executor.step
                    })
                writer.add_scalars('epoch', {'cv_loss': cv_loss, 'lr': lr}, epoch)
            final_epoch = epoch

        if final_epoch is not None and self.args.rank == 0:
            final_model_path = os.path.join(model_dir, 'final.pt')
            os.symlink('{}.pt'.format(final_epoch), final_model_path)
        
    
    

    
