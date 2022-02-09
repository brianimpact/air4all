import torch
from numpy import inf
from abc import abstractmethod
import copy

class BaseTrainer:
    def __init__(self, model, loss, metrics, optimizer, config):
        self.config = config
        self.logger = config.get_logger('trainer', config['trainer']['verbosity'])

        self.device, device_ids = self._prepare_device(config['n_gpu'])

        self.model = model.to(self.device)
        self.model.set_device(self.device)
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        self.loss = loss
        self.metrics = metrics
        self.optimizer = optimizer

        config_trainer = config['trainer']
        self.epochs = config_trainer['epochs']
        self.monitor = config_trainer.get('monitor', 'off')
        self.save_period = config_trainer['save_period']
        if self.monitor == 'off':
            self.monitor_mode = 'off'
            self.monitor_best = 0
        else:
            self.monitor_mode, self.monitor_metric = self.monitor.split()
            assert self.monitor_mode in ['min', 'max']

            self.monitor_best = inf if self.monitor_mode == 'min' else -inf
            self.early_stop = config_trainer.get('early_stop', inf)

        self.start_epoch = 1
        self.checkpoint_dir = config.save_dir

        if config.resume is not None:
            self._resume_checkpoint(config.resume)

    def _prepare_device(self, n_gpu_use):
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self.logger.warning("Warning: No GPU available on this machine,"
                                "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            self.logger.warning("Warning: The number of GPU\'s configured to use is {}, but only {} are available "
                                "on this machine.".format(n_gpu_use, n_gpu))
            n_gpu_use = n_gpu

        device = torch.device('cuda' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    @abstractmethod
    def _train_epoch(self, epoch):
        raise NotImplementedError

    @abstractmethod
    def _test(self, mode):
        raise NotImplementedError

    def _save_checkpoint(self, epoch, save_best=False):
        arch = type(self.model).__name__
        state = {
            'architecture': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.monitor_best,
            'config': self.config
        }

        if save_best:
            filename = str(self.checkpoint_dir / 'model_best.pth')
            #if self.save_flag:
            torch.save(state, filename)
            self.logger.info("Saving current best: model_best.pth ...")
        else:
            filename = str(self.checkpoint_dir / 'checkpoint-epoch{}.pth'.format(epoch))
            #if self.save_flag:
            torch.save(state, filename)
            self.logger.info("Save checkpoint: {} ...".format(filename))

        return filename

    def _resume_checkpoint(self, resume_path):
        resume_path = str(resume_path)
        self.logger.info("Load checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)

        self.start_epoch = checkpoint['epoch'] + 1
        self.monitor_best = checkpoint['monitor_best']

        if checkpoint['config']['architecture'] != self.config['architecture']:
            self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")
        self.model.load_state_dict(checkpoint['state_dict'])
        if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
            self.logger.warning("Warning: Optimizer type given in config file is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))

    def train(self):
        no_improvement = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            res = self._train_epoch(epoch)
            log = {'epoch':epoch}

            for k, v in res.items():
                if k == 'metrics':
                    log.update({mtr.__name__: v[i] for i, mtr in enumerate(self.metrics)})
                elif k == 'val_metrics':
                    log.update({'val_' + mtr.__name__: v[i] for i, mtr in enumerate(self.metrics)})
                elif k == 'edge_val_metrics':
                    log.update({'egde_val_' + mtr.__name__: v[i] for i, mtr in enumerate(self.metrics)})
                else:
                    log[k] = v

            is_best = False
            if self.monitor_mode != 'off':
                try:
                    improved = (self.monitor_mode == 'min' and log[self.monitor_metric] <= self.monitor_best) or \
                               (self.monitor_mode == 'max' and log[self.monitor_metric] >= self.monitor_best)
                except KeyError:
                    self.logger.warning("Warning: Metric '{}' is not found. "
                                        "Model performance monitoring is disabled.".format(self.monitor_metric))
                    self.monitor_mode = 'off'
                    improved = False
                    no_improvement = 0

                if improved:
                    self.monitor_best = log[self.monitor_metric]
                    no_improvement = 0
                    is_best = True
                else:
                    no_improvement += 1
                if no_improvement > self.early_stop:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                     "Training stops.".format(self.early_stop))
                    break

            if epoch % self.save_period == 0 or is_best:
                saved = self._save_checkpoint(epoch, save_best=is_best)
                if is_best:
                    best_state = copy.deepcopy(self.model.state_dict())
                    best_saved = saved

        self.logger.info("Testing with best model...")
        self.model.load_state_dict(best_state)
        evaluations = self.test()
        self.logger.info("The best model saved in: {} ...".format(best_saved))

        return evaluations

    def test(self):
        test_values = self._test('test')
        for i, metric in enumerate(self.metrics):
            self.logger.info('    {:15s}: {:.3f}'.format('test_' + metric.__name__, test_values[i]))
        return test_values
