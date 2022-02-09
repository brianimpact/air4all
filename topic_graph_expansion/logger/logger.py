import logging
import logging.config
import json
from collections import OrderedDict
from pathlib import Path

def read_json(fname):
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def set_logging(save_path, logger_config='configs/logger_config.json', default_level=logging.INFO):
    logger_config = Path(logger_config)
    if logger_config.is_file():
        config = read_json(logger_config)
        # modify logging paths based on run config
        for _, handler in config['handlers'].items():
            if 'filename' in handler:
                handler['filename'] = str(save_path / handler['filename'])
        logging.config.dictConfig(config)
    else:
        print("Warning: logging configuration file is not found in {}.".format(logger_config))
        logging.basicConfig(level=default_level)