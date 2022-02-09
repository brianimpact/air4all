import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.dataloader import default_collate

class BaseDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, shuffle, drop_last, valid_split, num_workers, collate_fn=default_collate):

        self.shuffle = shuffle
        self.drop_last = drop_last
        self.valid_split = valid_split
        self.batch_idx = 0
        self.n_samples = len(dataset)

        self.sampler, self.valid_sampler = self.sampler_split(self.valid_split)

        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'drop_last': self.drop_last,
            'collate_fn': collate_fn,
            'num_workers': num_workers
        }
        super(BaseDataLoader, self).__init__(sampler=self.sampler, **self.init_kwargs)


    def sampler_split(self, split_size):
        idx = np.arange(self.n_samples)
        np.random.seed(0)
        np.random.shuffle(idx)

        if split_size == 0.0:
            return None, None

        if isinstance(split_size, int):
            assert split_size > 0 and split_size < self.n_samples, "validation set size is larger thant the entire dataset size."
            valid_length = split_size
        else:
            valid_length = int(self.n_samples * split_size)

        train_ids = np.delete(idx, np.arange(0, valid_length))
        valid_ids = idx[0:valid_length]
        self.n_samples = len(train_ids)

        train_sampler = SubsetRandomSampler(train_ids)
        valid_sampler = SubsetRandomSampler(valid_ids)

        self.shuffle = False

        return train_sampler, valid_sampler

    def valid_split(self):
        if self.valid_sampler is None:
            return None
        else:
            return DataLoader(sampler=self.valid_sampler, **self.init_kwargs)


