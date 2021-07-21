import logging
from contextlib import nullcontext
import torch
from torch.nn.utils import clip_grad_norm_
from sklearn.metrics import accuracy_score
import tqdm

class Executor:
    def __init__(self):
        self.step = 0

    def train(self, model, optimizer, scheduler, data_loader, device, writer,
              args):
        ''' Train one epoch
        '''
        model.train()
        clip = args.get('grad_clip', 50.0)
        log_interval = args.get('log_interval', 10)
        rank = args.get('rank', 0)
        accum_grad = args.get('accum_grad', 1)
        is_distributed = args.get('is_distributed', True)
        logging.info('using accumulate grad, new batch size is {} times'
                     'larger than before'.format(accum_grad))
        num_seen_utts = 0
        num_total_batch = len(data_loader)
        for batch_idx, (feats, target) in enumerate(data_loader):
            feats = feats.to(device)
            target = target.to(device)
            num_utts = target.size(0)
            if num_utts == 0:
                continue
            context = None
            if is_distributed and batch_idx % accum_grad != 0 :
                context = model.no_sync
            else:
                context = nullcontext
            with context():
                loss, predictions = model(feats, 
                                          target)
                loss = loss / accum_grad
                loss.backward()

            num_seen_utts += num_utts
            if batch_idx % accum_grad == 0:
                if rank == 0 and writer is not None:
                    writer.add_scalar('train_loss', loss, self.step)
                grad_norm = clip_grad_norm_(model.parameters(), clip)
                if torch.isfinite(grad_norm):
                    optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                self.step += 1

            if batch_idx % log_interval == 0:
                lr = optimizer.param_groups[0]['lr']
                logging.debug('TRAIN Batch {}/{} loss {:.6f} lr {:.8f} rank {}'.format(
                                  batch_idx, num_total_batch,
                                  loss.item(), lr, rank))

    def cv(self, model, data_loader, device, args):
        ''' Cross validation on
        '''
        model.eval()
        log_interval = args.get('log_interval', 10)
        num_seen_utts = 0
        total_loss = 0.0
        num_total_batch = len(data_loader)
        with torch.no_grad():
            for batch_idx, (feats, target) in enumerate(data_loader):
                feats = feats.to(device)
                target = target.to(device)
                num_utts = target.size(0)
                if num_utts == 0:
                    continue
                loss, predictions = model(feats, 
                                          target)
                if torch.isfinite(loss):
                    num_seen_utts += num_utts
                    total_loss += loss.item() * num_utts
                if batch_idx % log_interval == 0:
                    logging.debug('CV Batch {}/{} loss {:.6f} history loss {:.6f}'.format(
                                      batch_idx, num_total_batch, loss.item(),
                                      total_loss / num_seen_utts))

        return total_loss, num_seen_utts
    
    
    def evaluation(self, model, data_loader, device):
        ''' Evaluation
        '''
        model.eval()
        num_seen_utts = 0
        total_loss = 0.0
        num_total_batch = len(data_loader)
        gt_labels , pred_labels = list(), list()
        with torch.no_grad():
            for batch_idx, (feats, target) in enumerate(tqdm.tqdm(data_loader)):
                assert feats.shape[0]==1
                feats = feats.to(device)
                num_utts = target.size(0)
                if num_utts == 0:
                    continue
                predictions= model.inference(feats)
                gt_labels.append(int(target[0].cpu().numpy()))
                pred_labels.append(int(predictions[0].cpu().numpy()))
        test_acc = accuracy_score(gt_labels, pred_labels)
        return test_acc