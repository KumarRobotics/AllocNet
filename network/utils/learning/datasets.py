import os
import h5py
import numpy as np
import open3d as o3d
import torch
from torch.utils.data import Dataset


class ConvMultiMapMinSnapDataset(Dataset):
    def __init__(self, path, seq_len=5, max_hpoly_length=None, max_vpoly_length=None):
        self.root_dir = path
        hdf5_path = os.path.join(path, "dataset.h5")
        # Check if file exist
        if not os.path.exists(hdf5_path):
            raise FileNotFoundError("File not found: {}".format(hdf5_path))

        self.hdf5_path = hdf5_path

        self.seq_len = seq_len

    def __len__(self):
        with h5py.File(self.hdf5_path, 'r') as hdf5_file:
            return len(hdf5_file.keys())

    def __getitem__(self, idx):
        with h5py.File(self.hdf5_path, 'r') as hdf5_file:
            group = hdf5_file[f"idx_{idx}"]
            # state = torch.tensor(group["state"][:])
            stacked_state = torch.tensor(group["stacked_state"][:])
            stacked_hpolys = torch.tensor(group["stacked_hpolys"][:])
            traj_times = torch.tensor(group["traj_times"][:])
            # print(traj_times)
            # print(state)

            # Padding
            if traj_times.shape[0] < self.seq_len:
                traj_times = torch.cat((traj_times, torch.zeros(self.seq_len - traj_times.shape[0])), dim=0)

            if stacked_hpolys.shape[2] < self.seq_len:
                stacked_hpolys = torch.cat((stacked_hpolys, torch.zeros(50, 4, self.seq_len - stacked_hpolys.shape[2])), dim=2)

        return stacked_hpolys, stacked_state, traj_times
