import random
import time

import torch.nn as nn
from transformers import Trainer
from transformers.trainer import get_parameter_names
from torch.utils.data import DataLoader, DistributedSampler


class LVLMTrainer(Trainer):
    def create_optimizer(self):
        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, [nn.LayerNorm])
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                    "name": "decay_parameters"
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                    "name": "no_decay_parameters"
                },
            ]

            optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        return self.optimizer
    

class LVLMMULTITrainer(Trainer):
    def __init__(
        self,
        datasets=None,
        collate_fn=None,
        special_args=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.datasets = datasets
        self.collate_fn = collate_fn
        self.special_args = special_args
        self._dataloaders = None

    def create_optimizer(self):
        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, [nn.LayerNorm])
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                    "name": "decay_parameters"
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                    "name": "no_decay_parameters"
                },
            ]

            optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        return self.optimizer

    def get_train_dataloader(self):
        batch_size = self.special_args.per_device_train_batch_size
        datasets = self.datasets
        batch_sizes = {name: [batch_size for i in data] for name, data in datasets.items()}
        split_names = sorted(self.datasets.keys())

        datasets = [self.datasets[split] for split in split_names]
        batch_sizes = [batch_sizes[split] for split in split_names]
        is_trains = [split in "train" for split in split_names]

        collate_fns = [[self.collate_fn for i in split] for split in split_names]

        dataloaders = self.create_loaders(
            datasets=datasets,
            num_workers=self.special_args.dataloader_num_workers,
            batch_sizes=batch_sizes,
            is_trains=is_trains,
            collate_fns=collate_fns,
        )   

        self._dataloaders = {k: v for k, v in zip(split_names, dataloaders)}

        return self._dataloaders["train"]

    def create_loaders(
        self,
        datasets,
        num_workers,
        batch_sizes,
        is_trains,
        collate_fns,
        dataset_ratios=None,
    ):
        """
        Create dataloaders for training and validation.
        """

        def _create_loader(dataset, num_workers, bsz, is_train, collate_fn):
            if self.special_args.use_distributed:
                sampler = DistributedSampler(
                    dataset,
                    shuffle=is_train,
                    num_replicas=self.special_args.world_size,
                    rank=self.special_args.process_index,
                )
            else:
                sampler = None

            loader = DataLoader(
                dataset,
                batch_size=bsz,
                num_workers=num_workers,
                pin_memory=True,
                sampler=sampler,
                shuffle=sampler is None and is_train,
                collate_fn=collate_fn,
                drop_last=True if is_train else False,
            )

            if is_train:
                loader = IterLoader(loader, use_distributed=self.special_args.use_distributed)
                loader.task=dataset.task  

            return loader

        loaders = []

        for dataset, bsz, is_train, collate_fn in zip(datasets, batch_sizes, is_trains, collate_fns):
            if isinstance(dataset, list) or isinstance(dataset, tuple):
                if hasattr(dataset[0], 'sample_ratio') and dataset_ratios is None:
                    dataset_ratios = [d.sample_ratio for d in dataset]
                loader = MultiIterLoader(
                    loaders=[
                        _create_loader(d, num_workers, bsz[i], is_train, collate_fn[i])
                        for i, d in enumerate(dataset)
                    ],
                    ratios=dataset_ratios,
                )
            else:
                loader = _create_loader(dataset, num_workers, bsz, is_train, collate_fn)

            loaders.append(loader)

        return loaders


class MultiIterLoader:
    """
    A simple wrapper for iterating over multiple iterators.

    Args:
        loaders (List[Loader]): List of Iterator loaders.
        ratios (List[float]): List of ratios to sample from each loader. If None, all loaders are sampled uniformly.
    """

    def __init__(self, loaders, ratios=None):
        # assert all loaders has __next__ method
        for loader in loaders:
            assert hasattr(loader, "__next__"), "Loader {} has no __next__ method.".format(loader)

        if ratios is None:
            ratios = [1.0] * len(loaders)
        else:
            assert len(ratios) == len(loaders)
            ratios = [float(ratio) / sum(ratios) for ratio in ratios]

        self.loaders = loaders
        self.ratios = ratios

    def __iter__(self):
        return self

    def __next__(self):
        # random sample from each loader by ratio
        loader_idx = random.choices(range(len(self.loaders)), self.ratios, k=1)[0]
        # print(loader_idx)
        # print(self.loaders[loader_idx].task) 
        return next(self.loaders[loader_idx])


class IterLoader:
    """
    A wrapper to convert DataLoader as an infinite iterator.

    Modified from:
        https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/iter_based_runner.py
    """

    def __init__(self, dataloader: DataLoader, use_distributed: bool = False):
        self._dataloader = dataloader
        self.iter_loader = iter(self._dataloader)
        self._use_distributed = use_distributed
        self._epoch = 0

    @property
    def epoch(self) -> int:
        return self._epoch

    def __next__(self):
        try:
            data = next(self.iter_loader)
        except StopIteration:
            self._epoch += 1
            if hasattr(self._dataloader.sampler, "set_epoch") and self._use_distributed:
                self._dataloader.sampler.set_epoch(self._epoch)
            time.sleep(2)  # Prevent possible deadlock during epoch transition
            self.iter_loader = iter(self._dataloader)
            data = next(self.iter_loader)

        return data

    def __iter__(self):
        return self

    def __len__(self):
        return len(self._dataloader)