import numpy as np
import torch
import torch.nn.functional as F
import dgl
from tqdm import tqdm
import itertools
import more_itertools as mit
from functools import partial
from collections import defaultdict

from base import BaseTrainer
from model.model import *
from model.loss import *
from data_loader.data_loaders import *

MAX_NUM_OF_CANDIDATES = 100000

class Trainer(BaseTrainer):
    def __init__(self, model, loss, metrics, pre_metric, optimizer, config, data_loader, lr_scheduler=None):
        super(Trainer, self).__init__(model, loss, metrics, optimizer, config)
        self.config = config
        self.data_loader = data_loader
        self.pre_metric = pre_metric
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_mode = self.config['lr_scheduler']['args']['mode']
        self.do_validation = True
        self.test_batch_size = config['trainer']['test_batch_size']
        self.log_step = len(data_loader) // 5

    def eval_metrics(self, output, target):
        metrics = np.zeros(len(self.metrics))
        all_ranks = self.pre_metric
        for i, metric in enumerate(self.metrics):
            metrics[i] += metric(all_ranks)
        return metrics

def rearrange(scores, candidate_position_idx, true_position_idx):
    tmp = np.array([[x == y for x in candidate_position_idx] for y in true_position_idx]).any(0)
    correct = np.where(tmp)[0]
    incorrect = np.where(~tmp)[0]
    labels = torch.cat((torch.ones(len(correct)), torch.zeros(len(incorrect)))).int()
    scores = torch.cat((scores[correct], scores[incorrect]))
    return scores, labels

class MatchTrainer(Trainer):
    def __init__(self, input_mode, model, loss, metrics, pre_metric, optimizer, config, data_loader, lr_scheduler=None):
        super(MatchTrainer, self).__init__(model, loss, metrics, pre_metric, optimizer, config, data_loader, lr_scheduler)
        dataset = self.data_loader.dataset

        self.input_mode = input_mode
        self.candidate_positions = data_loader.dataset.all_edges
        if len(self.candidate_positions) > MAX_NUM_OF_CANDIDATES:
            val_pos = set(itertools.chain.from_iterable(dataset.valid_node2pos.values()))
            val_neg = list(set(self.candidate_positions).difference(val_pos))
            val_sample_size = max(MAX_NUM_OF_CANDIDATES - len(val_pos), 0)
            self.valid_candidate_positions = random.sample(val_neg, val_sample_size) + list(val_pos)
        else:
            self.valid_candidate_positions = self.candidate_positions

        if 'g' in input_mode:
            self.all_nodes = sorted(list(dataset.core_graph.nodes))
            self.edge2subgraph = {e: dataset.get_egonet_node_feature(-1, e[0], e[1]) for e in tqdm(self.candidate_positions)}

    def _train_epoch(self, epoch):
        self.model.train()

        total_loss = 0
        for batch_idx, batch in enumerate(self.data_loader):
            query, label, u, v, bgu, bgv, lens = batch
            self.optimizer.zero_grad()
            pred = self.model(query, u, v, bgu, bgv, lens)
            label = label[:, 0].to(self.device)
            loss = self.loss(pred, label)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

            if batch_idx % self.log_step == 0 or batch_idx == len(self.data_loader)-1:
                self.logger.debug(
                    'Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.4f}'.format(
                        epoch,
                        batch_idx * self.data_loader.batch_size,
                        self.data_loader.n_samples,
                        100.0 * batch_idx / len(self.data_loader),
                        loss.item()))

        log = {'loss': total_loss / len(self.data_loader)}

        if self.do_validation:
            val_log = {'val_metrics': self._test('validation')}
            log = {**log, **val_log}

        if self.lr_scheduler is not None:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                if self.lr_scheduler_mode == "min":
                    self.lr_scheduler.step(log['val_metrics'][0])
                else:
                    self.lr_scheduler.step(log['lr_scheduler'][-1])
            else:
                self.lr_scheduler.step()

        return log

    def _test(self, mode, gpu=True):
        assert mode in ['test', 'validation']
        torch.cuda.empty_cache()
        model = self.model if gpu else self.model.cpu()
        model.to(self.device)
        self.drop_last = False
        self.shuffle = False
        batch_size = self.test_batch_size

        model.eval()
        with torch.no_grad():
            dataset = self.data_loader.dataset
            node_features = dataset.node_features
            if mode == 'test':
                vocab = dataset.test_node_list
                node2pos = dataset.test_node2pos
                candidate_positions = self.candidate_positions
                self.logger.info(f'number of candidate positions: {len(candidate_positions)}')

            else:
                vocab = dataset.val_node_list
                node2pos = dataset.val_node2pos
                candidate_positions = self.valid_candidate_positions

            batched_model = []
            batched_positions = []
            for edges in tqdm(mit.sliced(candidate_positions, batch_size), desc="graph encoding"):
                edges = list(edges)
                us, vs, bgu, bgv, lens = None, None, None, None, None

                if 't' in self.input_mode:
                    us, vs = zip(*edges)
                    us = torch.tensor(us)
                    vs = torch.tensor(vs)
                if 'g' in self.input_mode:
                    bgs = [self.edge2subgraph[e] for e in edges]
                    bgu, bgv = zip(*bgs)

                ur, vr = self.model.forward_encoders(us, vs, bgu, bgv, lens)
                batched_model.append((ur.detach().cpu(), vr.detach().cpu()))
                batched_positions.append(len(edges))

            all_ranks = []
            for i, query in tqdm(enumerate(vocab), desc="testing"):
                batched_scores = []
                nf = node_features[query, :].to(self.device)
                for (ur, vr), n_pos in zip(batched_model, batched_positions):
                    nf_expand = nf.expand(n_pos, -1)
                    ur = ur.to(self.device)
                    vr = vr.to(self.device)
                    scores = model.match(ur, vr, nf_expand)
                    batched_scores.append(scores)
                batched_scores = torch.cat(batched_scores)
                batched_scores, labels = rearrange(batched_scores, candidate_positions, node2pos[query])
                all_ranks.extend(self.pre_metric(batched_scores, labels))
            total_metrics = [metric(all_ranks) for metric in self.metrics]
        return total_metrics

class TMNTrainer(MatchTrainer):
    def __init__(self, input_mode, model, loss, metrics, pre_metric, optimizer, config, data_loader, lr_scheduler=None):
        super(TMNTrainer, self).__init__(input_mode, model, loss, metrics, pre_metric, optimizer, config, data_loader, lr_scheduler)
        self.l1 = config['trainer']['l1']
        self.l2 = config['trainer']['l2']
        self.l3 = config['trainer']['l3']
        self.device, device_ids = self._prepare_device(config['n_gpu'])

        self.model = model.to(self.device)
        self.model.set_device(self.device)

        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

    def _train_epoch(self, epoch):
        self.model.train()
        total_loss = 0

        for batch_idx, batch in enumerate(self.data_loader):
            query, label, u, v, bgu, bgv, lens = batch
            self.optimizer.zero_grad()
            scores, scores_parent, scores_child, scores_pair = self.model(query, u, v, bgu, bgv, lens)
            label = label.to(self.device)
            loss_pair = self.loss(scores_pair, label[:, 0])
            loss_parent = self.loss(scores_parent, label[:, 1])
            loss_child = self.loss(scores_child, label[:, 2])
            loss = self.loss(scores, label[:, 0]) + self.l1 * loss_parent + self.l2 * loss_child + self.l3 * loss_pair
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            if batch_idx % self.log_step == 0 or batch_idx == len(self.data_loader) - 1:
                self.logger.debug(
                    'Epoch: {} [{}/{} ({:.0f}%)] Total loss: {:.3f} Pair matching loss: {:.3f} Parent matching loss: {:.3f} Child matching loss: {:.3f}'
                        .format(epoch, batch_idx * self.data_loader.batch_size, self.data_loader.n_samples,
                                100.0 * batch_idx / len(self.data_loader),
                                loss.item(), loss_pair.item(), loss_parent.item(), loss_child.item()))

        log = {'loss': total_loss / len(self.data_loader)}

        if self.do_validation:
            val_log = {'val_metrics': self._test('validation')}
            log = {**log, **val_log}

        if self.lr_scheduler is not None:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                if self.lr_scheduler_mode == "min":
                    self.lr_scheduler.step(log['val_metrics'][0])
                else:
                    self.lr_scheduler.step(log['val_metrics'][-1])
            else:
                self.lr_scheduler.step()

        return log