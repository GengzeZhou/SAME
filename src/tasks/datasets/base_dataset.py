import torch

class MetaDataset(type):
    registry = {}

    def __init__(cls, name, bases, attrs):
        super().__init__(name, bases, attrs)
        if 'name' in attrs:
            MetaDataset.registry[attrs['name']] = cls

class BaseDataset(torch.utils.data.Dataset, metaclass=MetaDataset):
    def load_data(self, anno_file, debug=False):
        '''
        Load data from the annotation file.
        '''
        raise NotImplementedError
