import torch
from torch.utils.data import DataLoader
from data_loader.dataset import *

class TaxoCompDataLoader(DataLoader):
    def __init__(self, input_mode, path, sampling=1, batch_size=32, neg_size=3, max_pos_size=10, num_neighbors=4,
                 shuffle=True, drop_last=True, num_workers=8, normalize_embed=False, cache_refresh_time=16, test=0):

        self.input_mode = input_mode
        self.sampling = sampling
        self.batch_size = batch_size
        self.max_pos_size = max_pos_size
        self.neg_size = neg_size
        self.num_neighbors = num_neighbors
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.normalize_embed = normalize_embed
        self.num_workers = num_workers

        flag = 'test' if test else 'train'

        raw_graph_dataset = AIDataset(taxonomy_name="", dataset_path=path, load_file=False)
        if 'g' in input_mode:
            msk_graph_dataset = GraphDataset(raw_graph_dataset, mode=flag, sampling=sampling, max_pos_size=max_pos_size, neg_size=neg_size, num_neighbors=num_neighbors, normalize_embed=normalize_embed, cache_refresh_time=cache_refresh_time)
        else:
            msk_graph_dataset = RawDataset(raw_graph_dataset, mode=flag, sampling=sampling, max_pos_size=max_pos_size, neg_size=neg_size, num_neighbors=num_neighbors, normalize_embed=normalize_embed, cache_refresh_time=cache_refresh_time)

        self.dataset = msk_graph_dataset

        super(TaxoCompDataLoader, self).__init__(dataset=self.dataset, batch_size=self.batch_size, shuffle=self.shuffle, drop_last=self.drop_last, collate_fn=self.collate_fn, num_workers=self.num_workers, pin_memory=True)
        self.n_samples = len(self.dataset)


    def collate_fn(self, samples):
        if 'g' in self.input_mode:
            us, vs, u_graphs, v_graphs, queries, labels = map(list, zip(*chain(*samples)))
            return torch.tensor(queries), torch.tensor(labels), torch.tensor(us), torch.tensor(vs), u_graphs, v_graphs, None

        else:
            us, vs, queries, labels = map(list, zip(*chain(*samples)))
            return torch.tensor(queries), torch.tensor(labels), torch.tensor(us), torch.tensor(vs), None, None, None


