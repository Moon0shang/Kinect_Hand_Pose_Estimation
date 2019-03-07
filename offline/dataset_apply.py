import torch.utils.data as data
import os
import os.path
import torch
import numpy as np


class HandPointDataset(data.Dataset):
    def __init__(self, opt, train=False):

        self.OUTPUT = opt.OUTPUT
        self.SAMPLE_NUM = opt.SAMPLE_NUM
        self.INPUT_FEATURE_NUM = opt.INPUT_FEATURE_NUM
        self.JOINT_NUM = opt.JOINT_NUM

        self.points_cloud = np.empty(
            [1, self.SAMPLE_NUM, self.INPUT_FEATURE_NUM], dtype=np.float32)
        self.points_cloud[0, :, :] = np.load('./normal_pc.npy')

    def __getitem__(self, index):
        return self.points_cloud[index, :, :]

    def __len__(self):
        return self.points_cloud.shape[0]
