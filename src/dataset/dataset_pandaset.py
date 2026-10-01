from dataclasses import dataclass
from typing import Literal

from .dataset_temporal18 import DatasetTemporal18, DatasetTemporal18Cfg


@dataclass
class DatasetPandaSetCfg(DatasetTemporal18Cfg):
    name: Literal["pandaset"]


class DatasetPandaSet(DatasetTemporal18):
    pass
