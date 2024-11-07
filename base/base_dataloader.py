import numpy as np 
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import RandomSampler

class BaseDataLoader(DataLoader):
    """
    Base class for all data loaders
    """
    def __init__(
        self, 
        dataset: torch.utils.data.Dataset, 
        batch_size: int = 8, 
        shuffle: bool = True, 
        validation_split: float = 0.0, 
        num_workers: int = 1, 
        collate_fn: callable = default_collate
    ):
        """
        Initializes the base trainer with the given dataset and parameters.
        Args:
            dataset (torch.utils.data.Dataset): The dataset to be used for training.
            batch_size (int, optional): Number of samples per batch. Defaults to 8.
            shuffle (bool, optional): Whether to shuffle the dataset. Defaults to True.
            validation_split (float, optional): Fraction of the dataset to be used for validation. Defaults to 0.0.
            num_workers (int, optional): Number of subprocesses to use for data loading. Defaults to 1.
            collate_fn (callable, optional): Function to merge a list of samples to form a mini-batch. Defaults to default_collate.
        Attributes:
            batch_idx (int): Index of the current batch.
            n_samples (int): Total number of samples in the dataset.
            sampler (torch.utils.data.Sampler): Sampler for the training data.
            valid_sampler (torch.utils.data.Sampler): Sampler for the validation data.
            init_kwargs (dict): Dictionary of initialization keyword arguments.
        """
        self.validation_split = validation_split
        self.shuffle = shuffle

        self.batch_idx = 0
        self.n_samples = len(dataset)

        self.sampler, self.valid_sampler = self._split_sampler(self.validation_split)

        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers
        }
        super().__init__(sampler=self.sampler, **self.init_kwargs)

    def _split_sampler(
        self, 
        split
    ) -> (torch.utils.data.Sampler, torch.utils.data.Sampler):
        """
        Split the dataset into train and validation sets
        """
        if split == 0.0:
            return RandomSampler(self.dataset), None

        idx_full = np.arange(self.n_samples)

        np.random.seed(0)
        np.random.shuffle(idx_full)

        len_valid = int(self.n_samples * split)

        valid_idx = idx_full[0:len_valid]
        train_idx = np.delete(idx_full, np.arange(0, len_valid))

        train_sampler = RandomSampler(train_idx)
        valid_sampler = RandomSampler(valid_idx)

        return train_sampler, valid_sampler

    def split_validation(self) -> DataLoader:
        """
        Split the dataset into training and validation sets
        """
        if self.valid_sampler is None:
            return None

        valid_loader = DataLoader(sampler=self.valid_sampler, **self.init_kwargs)
        return valid_loader