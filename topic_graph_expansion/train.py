import numpy as np
import torch
import collections
import data_loader.data_loaders as model_data
import model.model as model_arch
import model.loss as model_loss
import model.metric as model_metric
import trainer.trainer as trainer_arch
import argparse
import time
from functools import partial
from parse_config import ConfigParser

def main(config):
    logger = config.get_logger('train')
    train_data_loader = config.initialize('train_data_loader', model_data, config['input_mode'], config['path'])
    logger.info(train_data_loader)

    node_features = train_data_loader.dataset.node_features
    vocab_size, embed_dim = node_features.size()
    model = config.initialize('architecture', model_arch, config['input_mode'])
    model.set_embedding(vocab_size=vocab_size, embed_dim=embed_dim, pretrained_embedding=node_features)
    logger.info(model)

    loss = getattr(model_loss, config['loss'])
    if config['loss'].startswith("bce_loss"):
        pre_metric = partial(model_metric.get_ranks, flag=1)
    else:
        pre_metric = partial(model_metric.get_ranks, flag=0)
    metrics = [getattr(model_metric, metric) for metric in config['metrics']]
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.initialize('optimizer', torch.optim, trainable_params)
    lr_scheduler = config.initialize('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    start_time = time.time()
    Trainer = config.initialize_trainer('architecture', trainer_arch)
    trainer = Trainer(config['input_mode'], model, loss, metrics, pre_metric, optimizer,
                      config=config,
                      data_loader=train_data_loader,
                      lr_scheduler=lr_scheduler)
    evaluations = trainer.train()
    end_time = time.time()
    logger.info(f"Finish training in {end_time-start_time} seconds")

    return evaluations


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=None, type=str)
    parser.add_argument('--resume', default=None, type=str)
    parser.add_argument('--device', default=None, type=str)
    parser.add_argument('--n_trials', default=1, type=int)

    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')

    options = [
        CustomArgs(['--train_data'], type=str, target=('train_data_loader', 'args', 'path')),
        CustomArgs(['--batch_size'], type=int, target=('train_data_loader', 'args', 'batch_size')),
        CustomArgs(['--neg_size'], type=int, target=('train_data_loader', 'args', 'neg_size')),
        CustomArgs(['--num_neighbors'], type=int, target=('train_data_loader', 'args', 'num_neighbors')),
        CustomArgs(['--drop_last'], type=bool, target=('train_data_loader', 'args', 'drop_last')),
        CustomArgs(['--sampling'], type=int, target=('train_data_loader', 'args', 'sampling')),
        CustomArgs(['--cache_refresh_time'], type=int, target=('train_data_loader', 'args', 'cache_refresh_time')),
        CustomArgs(['--num_workers'], type=int, target=('train_data_loader', 'args', 'num_workers')),
        CustomArgs(['--graph_propagation_method'], type=str, target=('architecture', 'args', 'graph_propagation_method')),
        CustomArgs(['--readout_method'], type=str, target=('architecture', 'args', 'graph_readout_method')),
        CustomArgs(['--matching_method'], type=str, target=('architecture', 'args', 'matching_method')),
        CustomArgs(['--k'], type=int, target=('architecture', 'args', 'k')),
        CustomArgs(['--in_dim'], type=int, target=('architecture', 'args', 'in_dim')),
        CustomArgs(['--hidden_dim'], type=int, target=('architecture', 'args', 'hidden_dim')),
        CustomArgs(['--out_dim'], type=int, target=('architecture', 'args', 'out_dim')),
        CustomArgs(['--pos_dim'], type=int, target=('architecture', 'args', 'pos_dim')),
        CustomArgs(['--num_heads'], type=int, target=('architecture', 'args', 'heads', 0)),
        CustomArgs(['--feat_drop'], type=float, target=('architecture', 'args', 'feat_drop')),
        CustomArgs(['--attn_drop'], type=float, target=('architecture', 'args', 'attn_drop')),
        CustomArgs(['--hidden_drop'], type=float, target=('architecture', 'args', 'hidden_drop')),
        CustomArgs(['--out_drop'], type=float, target=('architecture', 'args', 'out_drop')),
        CustomArgs(['--input_mode'], type=str, target=('input_mode',)),
        CustomArgs(['--loss'], type=str, target=('loss',)),
        CustomArgs(['--epochs'], type=int, target=('trainer', 'epochs')),
        CustomArgs(['--early_stop'], type=int, target=('trainer', 'early_stop')),
        CustomArgs(['--test_batch_size'], type=int, target=('trainer', 'test_batch_size')),
        CustomArgs(['--verbose_level'], type=int, target=('trainer', 'verbosity')),
        CustomArgs(['--l1'], type=float, target=('trainer', 'l1')),
        CustomArgs(['--l2'], type=float, target=('trainer', 'l2')),
        CustomArgs(['--l3'], type=float, target=('trainer', 'l3')),
        CustomArgs(['--learning_rate'], type=float, target=('optimizer', 'args', 'lr')),
        CustomArgs(['--weight_decay'], type=float, target=('optimizer', 'args', 'weight_decay'))
    ]
    config = ConfigParser(parser, options)

    args = parser.parse_args()
    n_trials = args.n_trials

    if n_trials > 0:
        config.get_logger('train').info(f'number of trials: {n_trials}')
        metrics = config['metrics']
        save_file = config.log_dir / 'evaluations.txt'
        fin = open(save_file, 'w')
        fin.write('\t'.join(metrics))

        evaluations = []
        for i in range(n_trials):
            config.set_save_dir(i+1)
            res = main(config)
            evaluations.append(res)
            fin.write('\t'.join([f'{i:.3f}' for i in res]))

        evaluations = np.array(evaluations)
        means = evaluations.mean(axis=0)
        stds = evaluations.std(axis=0)
        final_output = '  '.join([f'& {i:.3f} +- {j:.3f}' for i, j in zip(means, stds)])
        fin.write(final_output)
        config.get_logger('train').info(final_output)
    else:
        main(config)
