import os
import h5py
import lmdb
from typing import Dict, Union
from omegaconf.dictconfig import DictConfig
import numpy as np
import msgpack
# import msgpack_numpy
# msgpack_numpy.patch()
from .datasets.mp3d_envs import angle_feature, convert_elevation, convert_heading


class ImageFeaturesDB(object):
    def __init__(self, img_ft_file: str, image_feat_size: str):
        self.image_feat_size = image_feat_size
        self.img_ft_file = img_ft_file
        self._feature_store = {}

    def get_image_feature(self, scan:str, viewpoint: str=None, load_in_memory: bool=False) -> np.ndarray:
        key = '%s_%s' % (scan, viewpoint) if viewpoint is not None else scan
        if key in self._feature_store:
            ft = self._feature_store[key]
        else:
            if self.img_ft_file.endswith('.hdf5'):
                with h5py.File(self.img_ft_file, 'r') as f:
                    ft = f[key]
                    if len(ft.shape)==1:
                        ft = ft[:self.image_feat_size].astype(np.float32)
                    else:
                        ft = ft[:, :self.image_feat_size].astype(np.float32)
                    if load_in_memory:
                        self._feature_store[key] = ft
            elif self.img_ft_file.endswith('.lmdb'):
                with lmdb.open(self.img_ft_file, readonly=True, lock=False).begin() as txn:
                    unpacked_data = msgpack.unpackb(txn.get(key.encode('ascii')))
                    ft = np.frombuffer(unpacked_data[b'data'], dtype=unpacked_data[b'type'])
                    ft = ft.reshape(unpacked_data[b'shape'])
        return ft


def create_feature_db(config: DictConfig) -> Dict[str, ImageFeaturesDB]:
    ft_databases = {}
    ft_database_path = config.feature.feature_database
    for source in ft_database_path:
        path = ft_database_path[source] \
                if ft_database_path[source].startswith("/") \
                else os.path.join(config.experiment.data_dir, ft_database_path[source])
        ft_databases[source] = ImageFeaturesDB(
            path, 
            config.feature.image_feat_size
        )
    return ft_databases


class REVERIEObjectFeatureDB(object):
    def __init__(self, obj_ft_file, obj_feat_size, im_width=640, im_height=480):
        self.obj_feat_size = obj_feat_size
        self.obj_ft_file = obj_ft_file
        self._feature_store = {}
        self.im_width = im_width
        self.im_height = im_height

    def load_feature(self, scan, viewpoint, max_objects=None):
        key = '%s_%s' % (scan, viewpoint)
        if key in self._feature_store:
            obj_fts, obj_attrs = self._feature_store[key]
        else:
            with lmdb.open(self.obj_ft_file, readonly=True, lock=False).begin() as txn:
                obj_data = txn.get(key.encode('ascii'))
            if obj_data is not None:
                obj_data = msgpack.unpackb(obj_data)
                obj_fts = obj_data['fts'][:, :self.obj_feat_size].astype(np.float32)
                obj_attrs = {k: v for k, v in obj_data.items() if k != 'fts'}
            else:
                obj_fts = np.zeros((0, self.obj_feat_size), dtype=np.float32)
                obj_attrs = {}
            self._feature_store[key] = (obj_fts, obj_attrs)

        if max_objects is not None:
            obj_fts = obj_fts[:max_objects]
            obj_attrs = {k: v[:max_objects] for k, v in obj_attrs.items()}
        return obj_fts, obj_attrs

    def get_object_feature(
        self, scan, viewpoint, base_heading, base_elevation, angle_feat_size,
        max_objects=None
    ):
        obj_fts, obj_attrs = self.load_feature(scan, viewpoint, max_objects=max_objects)
        obj_ang_fts = np.zeros((len(obj_fts), angle_feat_size), dtype=np.float32)
        obj_box_fts = np.zeros((len(obj_fts), 3), dtype=np.float32)
        obj_ids = []
        if len(obj_fts) > 0:
            for k, obj_ang in enumerate(obj_attrs['centers']):
                obj_ang_fts[k] = angle_feature(
                    obj_ang[0] - base_heading, obj_ang[1] - base_elevation, angle_feat_size
                )
                w, h = obj_attrs['bboxes'][k][2:]
                obj_box_fts[k, :2] = [h/self.im_height, w/self.im_width]
                obj_box_fts[k, 2] = obj_box_fts[k, 0] * obj_box_fts[k, 1]
            obj_ids = obj_attrs['obj_ids']
        return obj_fts, obj_ang_fts, obj_box_fts, obj_ids


class SOONObjectFeatureDB(object):
    # TODO: This class requires adapting to current modification.
    def __init__(self, obj_ft_file, obj_feat_size):
        self.obj_feat_size = obj_feat_size
        self.obj_ft_file = obj_ft_file
        self._feature_store = {}

    def load_feature(self, scan, viewpoint, max_objects=None):
        key = '%s_%s' % (scan, viewpoint)
        if key in self._feature_store:
            obj_fts, obj_attrs = self._feature_store[key]
        else:
            with lmdb.open(self.obj_ft_file, readonly=True, lock=False).begin() as txn:
                obj_data = txn.get(key.encode('ascii'))
            if obj_data is not None:
                obj_data = msgpack.unpackb(obj_data)
                obj_fts = obj_data['fts'][:, :self.obj_feat_size].astype(np.float32)
                obj_attrs = {
                    'directions': obj_data['2d_centers'],
                    'obj_ids': obj_data['obj_ids'],
                    'bboxes': np.array(obj_data['xyxy_bboxes']),
                }
            else:
                obj_attrs = {}
                obj_fts = np.zeros((0, self.obj_feat_size), dtype=np.float32)
            self._feature_store[key] = (obj_fts, obj_attrs)

        if max_objects is not None:
            obj_fts = obj_fts[:max_objects]
            obj_attrs = {k: v[:max_objects] for k, v in obj_attrs.items()}
        return obj_fts, obj_attrs

    def get_object_feature(
        self, scan, viewpoint, base_heading, base_elevation, angle_feat_size,
        max_objects=None
    ):
        obj_fts, obj_attrs = self.load_feature(scan, viewpoint, max_objects=max_objects)
        obj_ang_fts = np.zeros((len(obj_fts), angle_feat_size), dtype=np.float32)
        obj_loc_fts = np.zeros((len(obj_fts), 3), dtype=np.float32)
        obj_directions, obj_ids = [], []
        if len(obj_fts) > 0:
            for k, obj_ang in enumerate(obj_attrs['directions']):
                obj_ang_fts[k] = angle_feature(
                    obj_ang[0] - base_heading, obj_ang[1] - base_elevation, angle_feat_size
                )
                x1, y1, x2, y2 = obj_attrs['bboxes'][k]
                h = y2 - y1
                w = x2 - x1
                obj_loc_fts[k, :2] = [h/224, w/224]
                obj_loc_fts[k, 2] = obj_loc_fts[k, 0] * obj_loc_fts[k, 1]
            obj_directions = [[convert_heading(x[0]), convert_elevation(x[1])] for x in obj_attrs['directions']]
            obj_ids = obj_attrs['obj_ids']
        return obj_fts, obj_ang_fts, obj_loc_fts, obj_directions, obj_ids

def create_object_feature_db(config: DictConfig) -> Dict[str, Union[REVERIEObjectFeatureDB, SOONObjectFeatureDB]]:
    obj_databases = {}
    obj_database_path = config.feature.object_database
    for source in obj_database_path:
        path = obj_database_path[source] \
                if obj_database_path[source].startswith("/") \
                else os.path.join(config.experiment.data_dir, obj_database_path[source])
        if source == 'reverie':
            obj_databases[source] = REVERIEObjectFeatureDB(
                path, 
                config.feature.obj_feat_size
            )
        elif source == 'soon':
            obj_databases[source] = SOONObjectFeatureDB(
                path,
                config.feature.obj_feat_size
            )
    return obj_databases