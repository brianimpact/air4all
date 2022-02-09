#!/usr/bin/env python
# coding:utf-8

import os
import sys
import time
import warnings

import numpy as np
import torch
from transformers import AutoTokenizer

import utils.logger as logger
from model import hiagm
from utils.configuration import Configuration
from utils.dataloader import HTCDataset, collate_fn
from utils.file_and_ckpt import *
from utils.loss import LabelContradictionPenalty, RecursiveRegularizationLoss
from utils.metric import f1_scores


def train(config):
    # READ FILES ACCORDING TO CONFIG
    label_ids = make_label_indices(config)
    hierarchy = read_hierarchy(config, label_ids)
    label_sequences = make_label_sequences(hierarchy, label_ids)
    # DATASET AND DATALOADER GENERATION
    tokenizer = AutoTokenizer.from_pretrained(config.model.embedding.type)
    collate_function = collate_fn(config, tokenizer, label_ids)
    train_dataset = HTCDataset(config, 'train', label_ids)
    val_dataset = HTCDataset(config, 'val', label_ids)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, config.training.batch_size, shuffle=True, num_workers=config.device.num_workers, collate_fn=collate_function, drop_last=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, config.training.batch_size, shuffle=False, num_workers=config.device.num_workers, collate_fn=collate_function, drop_last=False)
    logger.info('DATASET LOADED')
    # MODEL, CRITERION, OPTIMIZER AND SCHEDULER DEFINITION
    model = hiagm.HiAGM(config, label_ids)
    criterion = RecursiveRegularizationLoss(config, hierarchy)
    penalty = LabelContradictionPenalty(config, hierarchy)
    optimizer = getattr(torch.optim, config.training.optimizer.type)(params=model.parameters(), lr=config.training.optimizer.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=config.training.schedule.decay)
    logger.info('MODEL GENERATED')
    if config.mode == 'train':
        # LOAD CHECKPOINT FROM DIRECTORY
        best_epoch = {'micro-f1': -1, 'macro-f1': -1, 'average-f1': -1}
        best_performances = {'micro-f1': 0., 'macro-f1': 0., 'average-f1': 0.}
        epoch = 0
        checkpoint = os.path.join(config.path.checkpoints, config.path.initial_checkpoint)
        if os.path.isfile(checkpoint):
            logger.info('CHECKPOINT LOADING FROM %s' % checkpoint)
            epoch, best_performances = load_checkpoint(checkpoint, model, optimizer)
            logger.info('CHECKPOINT LOADED FROM EPOCH %d' % (epoch - 1))
            logger.info('\tMICRO-F1: %.5f | MACRO-F1: %.5f | AVERAGE-F1: %.5f'
                            % (best_performances['micro-f1'], best_performances['macro-f1'], best_performances['average-f1']))
            best_epoch = {'micro-f1': epoch - 1, 'macro-f1': epoch - 1, 'average-f1': epoch - 1}
        # USE PARALLEL GPUS
        if torch.cuda.is_available():
            model = torch.nn.parallel.DataParallel(model.cuda())
            logger.info('CUDA AVAILABLE: DATA PARALLEL MODEL DEFINED')
            optim_state_dict = optimizer.state_dict()
            for state in optim_state_dict['state'].values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()
        else:
            logger.info('CUDA UNAVAILABLE: RUNNING ON CPU')
        # START TRAINING
        suboptimal_epochs = 0
        for epoch in range(epoch, config.training.num_epochs):
            logger.info('EPOCH %d START' % epoch)
            start_time = time.time()
            model.train()
            total_loss = 0.
            epoch_labels = []
            epoch_logits = []
            for step, batch in enumerate(train_dataloader):
                # BATCH: (batch_input_ids, batch_token_type_ids, batch_attention_mask, batch_labels, batch_length)
                forward_start = time.time()
                # FORWARD
                logits = model(batch)
                loss = criterion(logits, batch[3], model.module.information_aggregation.classifiers)
                if penalty.penalty_weight > 0:
                    loss += penalty(logits)
                total_loss += loss.item()
                epoch_labels.append(batch[3])
                epoch_logits.append(logits.detach().cpu())
                backward_start = time.time()
                # BACKWARD
                for p in model.parameters():
                    p.grad = None
                loss.backward()
                optimizer.step()
                logger.info('EPOCH %d STEP %d / %d LOSS %f (FORWARD %.3fs | BACKWARD %.3fs)' % (epoch, step + 1, len(train_dataset) // config.training.batch_size, loss.item(), backward_start - forward_start, time.time() - backward_start))
            # F1 SCORE CALCULATION
            epoch_labels = torch.cat(epoch_labels, dim=0).to(torch.bool).cpu().numpy()
            epoch_logits = torch.cat(epoch_logits, dim=0).cpu().numpy()
            epoch_predictions = 1. / (1. + np.exp(- epoch_logits)) >= 0.5
            train_performances = f1_scores(epoch_labels, epoch_logits, epoch_predictions, label_ids, label_sequences, [0, 0, 0])
            logger.info('EPOCH %d TRAINING FINISHED. LOSS: %f | MICRO-F1: %f | MACRO-F1: %f | AVERAGE-F1: %f'
                        % (epoch, total_loss / (len(train_dataset) // config.training.batch_size), train_performances['micro-f1'], train_performances['macro-f1'], train_performances['average-f1']))
            # VALIDATION
            total_loss = 0.
            epoch_labels = []
            epoch_logits = []
            with torch.no_grad():
                model.eval()
                for step, batch in enumerate(val_dataloader):
                    # BATCH: (batch_input_ids, batch_token_type_ids, batch_attention_mask, batch_labels, batch_length)
                    logits = model(batch)
                    loss = criterion(logits, batch[3], model.module.information_aggregation.classifiers)
                    if penalty.penalty_weight > 0:
                        loss += penalty(logits)
                    total_loss += loss.item()
                    epoch_labels.append(batch[3])
                    epoch_logits.append(logits.detach().cpu())
                # F1 SCORE CALCULATION
                epoch_labels = torch.cat(epoch_labels, dim=0).to(torch.bool).cpu().numpy()
                epoch_logits = torch.cat(epoch_logits, dim=0).cpu().numpy()
                epoch_predictions = 1. / (1. + np.exp(- epoch_logits)) >= 0.5
                val_performances = f1_scores(epoch_labels, epoch_logits, epoch_predictions, label_ids, label_sequences, [0, 0, 0])
                logger.info('EPOCH %d VALIDATION FINISHED. LOSS: %f | MICRO-F1: %f | MACRO-F1: %f | AVERAGE-F1: %f'
                            % (epoch, total_loss / (len(val_dataset) // config.training.batch_size), val_performances['micro-f1'], val_performances['macro-f1'], val_performances['average-f1']))
            # PERFORMANCE COMPARISON WITH PREVIOUS CHECKPOINTS
            if val_performances['micro-f1'] < best_performances['micro-f1'] or \
            val_performances['macro-f1'] < best_performances['macro-f1'] or \
            val_performances['average-f1'] < best_performances['average-f1']:
                suboptimal_epochs += 1
                # LEARNING RATE DECAY
                if suboptimal_epochs % config.training.schedule.patience == 0:
                    scheduler.step()
                    logger.info('%d CONSECUTIVE EPOCHS WITHOUT PERFORMANCE IMPROVEMENT: LEARNING RATE DECAYED TO %f' % (suboptimal_epochs, optimizer.param_groups[0]['lr']))
                # EARLY STOPPING
                if suboptimal_epochs == config.training.schedule.early_stopping:
                    logger.info('%d CONSECUTIVE EPOCHS WITHOUT PERFORMANCE IMPROVEMENT: EARLY STOPPING' % suboptimal_epochs)
                    break
            else:
                # BEST CHECKPOINT UPDATE
                suboptimal_epochs = 0
                if val_performances['micro-f1'] > best_performances['micro-f1']:
                    logger.info('MICRO-F1 IMPROVED FROM %.5f (EPOCH %d) TO %.5f (EPOCH %d)' 
                                % (best_performances['micro-f1'], best_epoch['micro-f1'], val_performances['micro-f1'], epoch))
                    best_performances['micro-f1'] = val_performances['micro-f1']
                    best_epoch['micro-f1'] = epoch
                    save_checkpoint(os.path.join(config.path.checkpoints, 'best_micro_f1.ckpt'), epoch, val_performances, model, optimizer)
                if val_performances['macro-f1'] > best_performances['macro-f1']:
                    logger.info('MACRO-F1 IMPROVED FROM %.5f (EPOCH %d) TO %.5f (EPOCH %d)'
                                % (best_performances['macro-f1'], best_epoch['macro-f1'], val_performances['macro-f1'], epoch))
                    best_performances['macro-f1'] = val_performances['macro-f1']
                    best_epoch['macro-f1'] = epoch
                    save_checkpoint(os.path.join(config.path.checkpoints, 'best_macro_f1.ckpt'), epoch, val_performances, model, optimizer)
                if val_performances['average-f1'] > best_performances['average-f1']:
                    logger.info('AVERAGE-F1 IMPROVED FROM %.5f (EPOCH %d) TO %.5f (EPOCH %d)' 
                                % (best_performances['average-f1'], best_epoch['average-f1'], val_performances['average-f1'], epoch))
                    best_performances['average-f1'] = val_performances['average-f1']
                    best_epoch['average-f1'] = epoch
                    save_checkpoint(os.path.join(config.path.checkpoints, 'best_average_f1.ckpt'), epoch, val_performances, model, optimizer)
            if epoch % config.training.schedule.auto_save == config.training.schedule.auto_save - 1:
                save_checkpoint(os.path.join(config.path.checkpoints, 'epoch_%d.ckpt' % epoch), epoch, val_performances, model, optimizer)
            
            logger.info('EPOCH %d COMPLETED (%d SECONDS)' % (epoch, time.time() - start_time))
    # TESTING
    # USE PARALLEL GPUS
    if config.mode == 'test':
        if torch.cuda.is_available():
            model = torch.nn.parallel.DataParallel(model.cuda())
            logger.info('CUDA AVAILABLE: DATA PARALLEL MODEL DEFINED')
        else:
            logger.info('CUDA UNAVAILABLE: RUNNING ON CPU')
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()
    postproc = [{'none': 0, 'remove': 1, 'connect': 2}[config.postprocessing.isolated_leaf],
                {'none': 0, 'remove': 1}[config.postprocessing.unfinished_path],
                {'none': 0, 'argmax_leaf': 1, 'argmax_path': 2}[config.postprocessing.if_empty]]
    # TEST DATASET AND DATALOADER GENERATED
    test_dataset = HTCDataset(config, 'test', label_ids)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, config.training.batch_size, shuffle=False, num_workers=config.device.num_workers, collate_fn=collate_function, drop_last=False)

    best_micro_f1_model = os.path.join(config.path.checkpoints, 'best_micro_f1.ckpt')
    best_macro_f1_model = os.path.join(config.path.checkpoints, 'best_macro_f1.ckpt')
    best_average_f1_model = os.path.join(config.path.checkpoints, 'best_average_f1.ckpt')
    all_performances = []

    best_micro_epoch = load_checkpoint(best_micro_f1_model, model, optimizer, 'test')
    logger.info('BEST MICRO F1 CHECKPOINT FROM EPOCH %d LOADED FOR TEST' % best_micro_epoch)
    epoch_labels = []
    epoch_logits = []
    with torch.no_grad():
        model.eval()
        for step, batch in enumerate(test_dataloader):
            logits = model(batch)
            epoch_labels.append(batch[3])
            epoch_logits.append(logits.detach().cpu())
        epoch_labels = torch.cat(epoch_labels, dim=0).to(torch.bool).cpu().numpy()
        epoch_logits = torch.cat(epoch_logits, dim=0).cpu().numpy()
        epoch_predictions = 1. / (1. + np.exp(- epoch_logits)) >= 0.5
        test_performances = f1_scores(epoch_labels, epoch_logits, epoch_predictions, label_ids, label_sequences, postproc)
        logger.info('TEST FOR BEST MICRO F1 CHECKPOINT (EPOCH %d) FINISHED. MICRO-F1: %f | MACRO-F1: %f | AVERAGE-F1: %f'
                    % (best_micro_epoch, test_performances['micro-f1'], test_performances['macro-f1'], test_performances['average-f1']))
        all_performances.append(test_performances)

    best_macro_epoch = load_checkpoint(best_macro_f1_model, model, optimizer, 'test')
    logger.info('BEST MACRO F1 CHECKPOINT FROM EPOCH %d LOADED FOR TEST' % best_macro_epoch)
    if best_macro_epoch != best_micro_epoch:
        epoch_labels = []
        epoch_logits = []
        with torch.no_grad():
            model.eval()
            for step, batch in enumerate(test_dataloader):
                logits = model(batch)
                epoch_labels.append(batch[3])
                epoch_logits.append(logits.detach().cpu())
            epoch_labels = torch.cat(epoch_labels, dim=0).to(torch.bool).cpu().numpy()
            epoch_logits = torch.cat(epoch_logits, dim=0).cpu().numpy()
            epoch_predictions = 1. / (1. + np.exp(- epoch_logits)) >= 0.5
            test_performances = f1_scores(epoch_labels, epoch_logits, epoch_predictions, label_ids, label_sequences, postproc)
            logger.info('TEST FOR BEST MACRO F1 CHECKPOINT (EPOCH %d) FINISHED. MICRO-F1: %f | MACRO-F1: %f | AVERAGE-F1: %f'
                        % (best_micro_epoch, test_performances['micro-f1'], test_performances['macro-f1'], test_performances['average-f1']))
            all_performances.append(test_performances)
    else:
        logger.info('TEST FOR BEST MACRO F1 CHECKPOINT (EPOCH %d) FINISHED. MICRO-F1: %f | MACRO-F1: %f | AVERAGE-F1: %f'
                     % (best_macro_epoch, all_performances[0]['micro-f1'], all_performances[0]['macro-f1'], all_performances[0]['average-f1']))

    best_average_epoch = load_checkpoint(best_average_f1_model, model, optimizer, 'test')
    logger.info('BEST AVERAGE F1 CHECKPOINT FROM EPOCH %d LOADED FOR TEST' % best_average_epoch)
    if best_average_epoch != best_micro_epoch and best_average_epoch != best_macro_epoch:    
        epoch_labels = []
        epoch_logits = []
        with torch.no_grad():
            model.eval()
            for step, batch in enumerate(test_dataloader):
                logits = model(batch)
                epoch_labels.append(batch[3])
                epoch_logits.append(logits.detach().cpu())
            epoch_labels = torch.cat(epoch_labels, dim=0).to(torch.bool).cpu().numpy()
            epoch_logits = torch.cat(epoch_logits, dim=0).cpu().numpy()
            epoch_predictions = 1. / (1. + np.exp(- epoch_logits)) >= 0.5
            test_performances = f1_scores(epoch_labels, epoch_logits, epoch_predictions, label_ids, label_sequences, postproc)
            logger.info('TEST FOR BEST AVERAGE F1 CHECKPOINT (EPOCH %d) FINISHED. MICRO-F1: %f | MACRO-F1: %f | AVERAGE-F1: %f'
                        % (best_average_epoch, test_performances['micro-f1'], test_performances['macro-f1'], test_performances['average-f1']))
    elif best_average_epoch == best_micro_epoch: 
        logger.info('TEST FOR BEST AVERAGE F1 CHECKPOINT (EPOCH %d) FINISHED. MICRO-F1: %f | MACRO-F1: %f | AVERAGE-F1: %f'
                     % (best_average_epoch, all_performances[0]['micro-f1'], all_performances[0]['macro-f1'], all_performances[0]['average-f1']))
    else:
        logger.info('TEST FOR BEST AVERAGE F1 CHECKPOINT (EPOCH %d) FINISHED. MICRO-F1: %f | MACRO-F1: %f | AVERAGE-F1: %f'
                     % (best_average_epoch, all_performances[1]['micro-f1'], all_performances[1]['macro-f1'], all_performances[1]['average-f1']))

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    config_path = sys.argv[1]
    assert os.path.isfile(config_path), "CONFIGURATION FILE DOES NOT EXIST"
    with open(config_path, 'r') as fin:
        config = json.load(fin)
    config = Configuration(config)
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(x) for x in config.device.cuda])
    torch.manual_seed(1227)
    torch.cuda.manual_seed(1227)
    logger.add_filehandler(config.path.log)
    logger.logging_verbosity(1)

    if not os.path.isdir(config.path.checkpoints):
        os.mkdir(config.path.checkpoints)

    train(config)
