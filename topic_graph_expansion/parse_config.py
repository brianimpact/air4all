import os
import json
from collections import OrderedDict
from pathlib import Path
from functools import reduce
from operator import getitem
from datetime import datetime
import logging
from logger import set_logging

def read_json(file_name):
    with file_name.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, file_name):
    with file_name.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

class ConfigParser:
    def __init__(self, args, options='', timestamp=True):
        for option in options:
            args.add_argument(*option.flags, type=option.type, default=None)
        args = args.parse_args()

        if args.device:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.device

        if args.resume:
            self.resume = Path(args.resume)
            if args.config is not None:
                self.config_file_name = Path(args.config)
            else:
                self.config_file_name = self.resume.parent.parent / 'config.json'
        else:
            assert args.config is not None
            self.resume = None
            self.config_file_name = Path(args.config)

        config = read_json(self.config_file_name)
        self.__config = _update_config(config, options, args)

        save_dir = Path(self.config['trainer']['save_dir'])
        experiment_name = self.config['name']
        timestamp = datetime.now().strftime(r'%m%d_%H%M%S') if timestamp else ''
        self.__save_dir = save_dir / experiment_name / timestamp / 'models'
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.origin_save_dir = self.save_dir
        self.__log_dir = save_dir / experiment_name / timestamp / 'log'
        self.log_dir.mkdir(parents=True, exist_ok=True)
        set_logging(self.log_dir)
        self.log_levels = {
            0: logging.WARNING,
            1: logging.INFO,
            2: logging.DEBUG
        }

        write_json(self.config, self.save_dir / 'config.json')

    def initialize(self, name, module, *args):
        module_config = self[name]
        return getattr(module, module_config['type'])(*args, **module_config['args'])

    def initialize_trainer(self, name, module):
        module_config = self[name]
        return getattr(module, module_config.get('trainer', 'Trainer'))

    def set_save_dir(self, trial_id):
        self.__save_dir = self.origin_save_dir / f'trial{trial_id}'
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def get_logger(self, name, verbosity=2):
        msg_verbosity = 'verbosity option {} is invalid. Valid options are {}.'.format(verbosity, self.log_levels.keys())
        assert verbosity in self.log_levels, msg_verbosity
        logger = logging.getLogger(name)
        logger.setLevel(self.log_levels[verbosity])
        return logger

    def __getitem__(self, name):
        return self.config[name]

    @property
    def config(self):
        return self.__config

    @property
    def save_dir(self):
        return self.__save_dir

    @property
    def log_dir(self):
        return self.__log_dir

def _update_config(config, options, args):
    for option in options:
        value = getattr(args, _get_option_name(option.flags))
        if value is not None:
            _set_by_path(config, option.target, value)
    return config

def _get_option_name(flags):
    for flag in flags:
        if flag.startswith('--'):
            return flag.replace('--', '')
    return flags[0].replace('--', '')

def _set_by_path(tree, keys, value):
    _get_by_path(tree, keys[:-1])[keys[-1]] = value

def _get_by_path(tree, keys):
    return reduce(getitem, keys, tree)