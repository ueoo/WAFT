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


class HD1K(FlowDataset):
    def __init__(self, aug_params=None, root="datasets/HD1K"):
        super(HD1K, self).__init__(aug_params, sparse=True)

        seq_ix = 0
        while 1:
            flows = sorted(glob(os.path.join(root, "hd1k_flow_gt", "flow_occ/%06d_*.png" % seq_ix)))
            images = sorted(glob(os.path.join(root, "hd1k_input", "image_2/%06d_*.png" % seq_ix)))

            if len(flows) == 0:
                break

            for i in range(len(flows) - 1):
                self.flow_list += [flows[i]]
                self.image_list += [[images[i], images[i + 1]]]

            seq_ix += 1

    def read_flow(self, index):
        flow, valid = frame_utils.readFlowKITTI(self.flow_list[index])
        return flow, valid
