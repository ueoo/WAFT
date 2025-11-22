import math
import os
import os.path as osp
import random

from glob import glob

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data

from tqdm import tqdm

from dataloader.template import FlowDataset
from utils import frame_utils


class KITTI(FlowDataset):
    def __init__(self, aug_params=None, split="training", root="datasets/KITTI/2015"):
        super(KITTI, self).__init__(aug_params)
        if split == "testing":
            self.is_test = True

        root = osp.join(root, split)
        images1 = sorted(glob(osp.join(root, "image_2/*_10.png")))
        images2 = sorted(glob(osp.join(root, "image_2/*_11.png")))

        for img1, img2 in zip(images1, images2):
            frame_id = img1.split("/")[-1]
            self.extra_info += [[frame_id]]
            self.image_list += [[img1, img2]]

        if split == "training":
            self.flow_list = sorted(glob(osp.join(root, "flow_occ/*_10.png")))

    def read_flow(self, index):
        flow, valid = frame_utils.readFlowKITTI(self.flow_list[index])
        return flow, valid
