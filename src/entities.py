from collections import OrderedDict

import torch
from torch.utils.data import Dataset as TorchDataset


class Sample:
    def __init__(self, id: int, source: str, target: str = None):
        self._id = id
        self._source = source
        self._target = target

    @property
    def source(self):
        return self._source

    @property
    def target(self):
        return self._target

    def __str__(self):
        return self._source + '->' + self._target


class Dataset(TorchDataset):
    TRAIN_MODE = 'train'
    EVAL_MODE = 'eval'

    def __init__(self, name: str, tokenizer):
        self._mode = Dataset.TRAIN_MODE
        self._name = name
        self._tokenizer = tokenizer
        self._id = 0
        self._samples = OrderedDict()

    def add_sample(self, source: str, target: str = None):
        sample = Sample(self._id, source, target)
        self._samples[self._id] = sample
        self._id += 1

        return sample

    def set_mode(self, mode: str):
        self._mode = mode

    @property
    def samples(self):
        return list(self._samples.values())

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        if self._mode == 'train':
            return create_train_sample(self.samples[item], self._tokenizer)
        else:
            return create_eval_sample(self.samples[item], self._tokenizer)


def create_train_sample(sample, tokenizer):
    encoding = tokenizer(sample.source).input_ids
    attention_mask = torch.ones(len(encoding), dtype=torch.bool)
    with tokenizer.as_target_tokenizer():
        label = tokenizer(sample.target).input_ids

    encoding = torch.tensor(encoding, dtype=torch.long)
    label = torch.tensor(label, dtype=torch.long)

    return dict(encoding=encoding, attention_mask=attention_mask, label=label)


def create_eval_sample(sample, tokenizer):
    encoding = tokenizer(sample.source).input_ids
    encoding = torch.tensor(encoding, dtype=torch.long)
    attention_mask = torch.ones(len(encoding), dtype=torch.bool)

    return dict(encoding=encoding, attention_mask=attention_mask)
