from .base_dataset import MetaDataset

# import the dataset class here
from .r2r import R2RDataset
from .rxr import RXRDataset
from .cvdn import CVDNDataset
from .soon import SOONDataset
from .reverie import REVERIEDataset
from .objnav_mp3d import ObjNavMP3DDataset
from .r2r_scalevln import R2RScaleVLNDataset
from .r2r_prevalent import R2RAugDataset
from .reverie_scalevln import REVERIEScaleVLNDataset

__all__ = list(MetaDataset.registry.keys())

def load_dataset(name, *args, **kwargs):
    cls = MetaDataset.registry[name]
    return cls(*args, **kwargs)