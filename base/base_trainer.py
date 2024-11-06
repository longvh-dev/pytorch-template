import torch
from abc import abstractmethod
from numpy import inf
from logger import TensorboardWriter

class BaseTrainer:
    """
    Base class for all trainers
    """
    def __init__(
        self, 
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        metric_ftns: list,
        optimizer: torch.optim.Optimizer,
        config: dict, 
    ):
    """
    Initializes the base trainer with the given model, criterion, metric functions, optimizer and configuration.
    Args:
        model (torch.nn.Module): The model to be trained.
        criterion (torch.nn.Module): The loss function.
        metric_ftns (list): List of metric functions.
        optimizer (torch.optim.Optimizer): The optimizer to be used.
        config (dict): Configuration dictionary.
    """
    self.config = config
    self.logger = config.get_logger('trainer', config['trainer']['verbosity'])