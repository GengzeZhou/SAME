"""
The Trainer class is used to train the Agent, and different training methods can be implemented
by creating different trainers.
"""
from .base_trainer import BaseTrainer

def load_trainer(name, *args, **kwargs):
    cls = BaseTrainer.registry[name]
    return cls(*args, **kwargs)
