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


class MpiSintel(FlowDataset):
    def __init__(self, aug_params=None, split="training", root="datasets/Sintel", dstype="clean"):
        super(MpiSintel, self).__init__(aug_params)
        flow_root = osp.join(root, split, "flow")
        image_root = osp.join(root, split, dstype)

        if split == "test":
            self.is_test = True

        for scene in os.listdir(image_root):
            image_list = sorted(glob(osp.join(image_root, scene, "*.png")))
            for i in range(len(image_list) - 1):
                self.image_list += [[image_list[i], image_list[i + 1]]]
                self.extra_info += [(scene, i)]  # scene and frame_id

            if split != "test":
                self.flow_list += sorted(glob(osp.join(flow_root, scene, "*.flo")))

    def read_flow(self, index):
        flow = frame_utils.read_gen(self.flow_list[index])
        valid = (np.abs(flow[..., 0]) < 1000) & (np.abs(flow[..., 1]) < 1000)
        return flow, valid
