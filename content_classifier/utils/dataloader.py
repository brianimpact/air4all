#!/usr/bin/env python
# coding:utf-8

import json
import torch

from torch.utils.data import Dataset


class HTCDataset(Dataset):
    # DATASET FOR HIERARCHICAL TEXT CLASSIFICATION
    def __init__(self, config, mode, label_ids):
        super(HTCDataset, self).__init__()
        if mode == 'train':
            path = config.path.data.train
        elif mode == 'val':
            path = config.path.data.val
        elif mode == 'test':
            path = config.path.data.test
        else:
            raise NotImplementedError
        self.label_ids = label_ids
        self.text = []
        self.labels = []
        with open(path, 'r', encoding='utf8') as f:
            lines = f.readlines()
            for line in lines:
                data = json.loads(line)
                if len(data['text']) > 100:
                    self.text.append(data['text'])
                    self.labels.append([self.label_ids[label] for label in data['label']])
    
    def __len__(self):
        return len(self.text)
    
    def __getitem__(self, i):
        return {'text': self.text[i], 'label': self.labels[i]}


class collate_fn:
    def __init__(self, config, tokenizer, label_ids):
        self.config = config
        self.tokenizer = tokenizer
        self.label_ids = label_ids
    
    def __call__(self, batch):
        # COLLATE FUNCTION USES BERT TOKENIZER TO EMBED EACH CONTENT
        batch_text = [sample['text'] for sample in batch]
        batch_inputs = self.tokenizer(batch_text, padding=True, truncation=True, max_length=self.config.training.max_text_length, return_tensors='pt')
        batch_length = batch_inputs['attention_mask'].sum(1).to(torch.int64)
        batch_input_ids = batch_inputs['input_ids']
        batch_token_type_ids = batch_inputs['token_type_ids']
        batch_attention_mask = batch_inputs['attention_mask']
        batch_labels = [[1. if i in sample['label'] else 0. for i in range(len(self.label_ids))] for sample in batch]
        return batch_input_ids, batch_token_type_ids, batch_attention_mask, torch.tensor(batch_labels, dtype=torch.float32), batch_length
