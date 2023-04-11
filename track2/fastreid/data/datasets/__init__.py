# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from ...utils.registry import Registry

DATASET_REGISTRY = Registry("DATASET")
DATASET_REGISTRY.__doc__ = """
Registry for datasets
It must returns an instance of :class:`Backbone`.
"""

# Person re-id datasets
# from .cuhk03 import CUHK03
# from .dukemtmcreid import DukeMTMC
# from .market1501 import Market1501
# from .msmt17 import MSMT17
# from .AirportALERT import AirportALERT
# from .iLIDS import iLIDS
# from .pku import PKU
# from .prai import PRAI
# from .prid import PRID
# from .grid import GRID
# from .saivt import SAIVT
# from .sensereid import SenseReID
# from .sysu_mm import SYSU_mm
# from .thermalworld import Thermalworld
# from .pes3d import PeS3D
# from .caviara import CAVIARa
# from .viper import VIPeR
# from .lpw import LPW
# from .shinpuhkan import Shinpuhkan
# from .wildtracker import WildTrackCrop
# from .cuhk_sysu import cuhkSYSU

# Vehicle re-id datasets
# from .veri import VeRi
# from .vehicleid import VehicleID, SmallVehicleID, MediumVehicleID, LargeVehicleID
# from .veriwild import VeRiWild, SmallVeRiWild, MediumVeRiWild, LargeVeRiWild

# from .retri import SOP, InShop, CUB, Cars196

# from .ms1mv2 import MS1MV2, LFW, CPLFW, CALFW, CFP_FF, CFP_FP, AgeDB_30, VGG2_FP
# from .ms1mv3 import MS1MV3
# from .veriall import VeriAll
# from .personall import PersonAll
# from .decathlon import  Aircraft, Cifar100, Daimlerpedcls, Dtd, Gtsrb, Imagenet12, Omniglot, Svhn, Ucf101, VggFlowers
# from .intern_train import DF20, IWildCam2020, TsingHuaDogs, FoodX251, CompCars
# from .imagenet import ImageNet1k, ImageNet1kBD
# from .intern_test import  PatchCamelyon, GTSRB, Fer2013, Retinopathy, Resisc45, \
#     EuroSAT, SVHN, FGVCaircraft, Caltech101, DTD, SUN397, \
#            Oxford102Flower, OxfordPet, Food101, CIFAR10, CIFAR100

# from .luperson import LUPerson
from .cocodet import COCODataSet

__all__ = [k for k in globals().keys() if "builtin" not in k and not k.startswith("_")]
