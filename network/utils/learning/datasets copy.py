import os
import h5py
import numpy as np
import open3d as o3d
import torch
from torch.utils.data import Dataset


class SingleMapDataset(Dataset):
    def __init__(self, path):
        pcd_path = os.path.join(path, "map.pcd")
        hdf5_path = os.path.join(path, "dataset.h5")
        # Check if files exist
        if not os.path.exists(pcd_path):
            raise FileNotFoundError("File not found: {}".format(pcd_path))
        if not os.path.exists(hdf5_path):
            raise FileNotFoundError("File not found: {}".format(hdf5_path))

        self.hdf5_path = hdf5_path

        max_hpoly_length = 0
        max_vpoly_length = 0
        with h5py.File(self.hdf5_path, 'r') as hdf5_file:
            for key in hdf5_file.keys():
                group = hdf5_file[key]

                max_hpoly_length = max(max_hpoly_length, len(group['hpoly_end_probs'][:]))
                max_vpoly_length = max(max_vpoly_length, len(group['vpoly_end_probs'][:]))

        self.max_hpoly_length = max_hpoly_length
        self.max_vpoly_length = max_vpoly_length

        pcd = o3d.io.read_point_cloud(pcd_path)
        self.pcd_data = np.asarray(pcd.points)

    def __len__(self):
        with h5py.File(self.hdf5_path, 'r') as hdf5_file:
            return len(hdf5_file.keys())

    def __getitem__(self, idx):
        with h5py.File(self.hdf5_path, 'r') as hdf5_file:
            group = hdf5_file[f"idx_{idx}"]
            state = torch.tensor(group["state"][:])
            hpoly_elems = torch.tensor(group["hpoly_elems"][:])
            hpoly_end_probs = torch.tensor(group["hpoly_end_probs"][:])
            hpoly_seq_end_probs = torch.tensor(group["hpoly_seq_end_probs"][:])
            vpoly_elems = torch.tensor(group["vpoly_elems"][:])
            vpoly_end_probs = torch.tensor(group["vpoly_end_probs"][:])
            vpoly_seq_end_probs = torch.tensor(group["vpoly_seq_end_probs"][:])

        padded_hpoly_elems, padded_hpoly_end_probs, padded_hpoly_seq_end_probs, padded_vpoly_elems, padded_vpoly_end_probs, padded_vpoly_seq_end_probs = self.padding(hpoly_elems, hpoly_end_probs, hpoly_seq_end_probs, vpoly_elems, vpoly_end_probs, vpoly_seq_end_probs)

        return self.pcd_data, state, padded_hpoly_elems, padded_hpoly_end_probs, padded_hpoly_seq_end_probs, padded_vpoly_elems, padded_vpoly_end_probs, padded_vpoly_seq_end_probs

    def get_max_hpoly_length(self):
        return self.max_hpoly_length

    def get_max_vpoly_length(self):
        return self.max_vpoly_length

    def padding(self, hpoly_elems, hpoly_end_probs, hpoly_seq_end_probs, vpoly_elems, vpoly_end_probs, vpoly_seq_end_probs):
        if hpoly_elems.size(0) < self.max_hpoly_length:
            hpoly_elems_padding = torch.zeros((self.max_hpoly_length - hpoly_elems.size(0), 4), dtype=hpoly_elems.dtype)
            padded_hpoly_elems = torch.cat((hpoly_elems, hpoly_elems_padding), dim=0)
        else:
            padded_hpoly_elems = hpoly_elems

        if hpoly_end_probs.size(0) < self.max_hpoly_length:
            hpoly_end_probs_padding = torch.zeros(self.max_hpoly_length - hpoly_end_probs.size(0), dtype=hpoly_end_probs.dtype)
            padded_hpoly_end_probs = torch.cat((hpoly_end_probs, hpoly_end_probs_padding))
        else:
            padded_hpoly_end_probs = hpoly_end_probs

        if hpoly_seq_end_probs.size(0) < self.max_hpoly_length:
            hpoly_seq_end_probs_padding = torch.zeros(self.max_hpoly_length - hpoly_seq_end_probs.size(0), dtype=hpoly_seq_end_probs.dtype)
            padded_hpoly_seq_end_probs = torch.cat((hpoly_seq_end_probs, hpoly_seq_end_probs_padding))
        else:
            padded_hpoly_seq_end_probs = hpoly_seq_end_probs

        if vpoly_elems.size(0) < self.max_vpoly_length:
            vpoly_elems_padding = torch.zeros((self.max_vpoly_length - vpoly_elems.size(0), 3), dtype=vpoly_elems.dtype)
            padded_vpoly_elems = torch.cat((vpoly_elems, vpoly_elems_padding), dim=0)
        else:
            padded_vpoly_elems = vpoly_elems

        if vpoly_end_probs.size(0) < self.max_vpoly_length:
            vpoly_end_probs_padding = torch.zeros(self.max_vpoly_length - vpoly_end_probs.size(0), dtype=vpoly_end_probs.dtype)
            padded_vpoly_end_probs = torch.cat((vpoly_end_probs, vpoly_end_probs_padding))
        else:
            padded_vpoly_end_probs = vpoly_end_probs

        if vpoly_seq_end_probs.size(0) < self.max_vpoly_length:
            vpoly_seq_end_probs_padding = torch.zeros(self.max_vpoly_length - vpoly_seq_end_probs.size(0), dtype=vpoly_seq_end_probs.dtype)
            padded_vpoly_seq_end_probs = torch.cat((vpoly_seq_end_probs, vpoly_seq_end_probs_padding))
        else:
            padded_vpoly_seq_end_probs = vpoly_seq_end_probs

        return padded_hpoly_elems, padded_hpoly_end_probs, padded_hpoly_seq_end_probs, padded_vpoly_elems, padded_vpoly_end_probs, padded_vpoly_seq_end_probs


class MultiMapDataset(Dataset):
    def __init__(self, path):
        self.root_dir = path
        pcd_file_dir = os.path.join(path, "maps")
        hdf5_path = os.path.join(path, "dataset.h5")
        # Check if files exist
        if not os.path.exists(pcd_file_dir):
            raise FileNotFoundError("File not found: {}".format(pcd_file_dir))
        if not os.path.exists(hdf5_path):
            raise FileNotFoundError("File not found: {}".format(hdf5_path))

        self.hdf5_path = hdf5_path

        max_hpoly_length = 0
        max_vpoly_length = 0
        with h5py.File(self.hdf5_path, 'r') as hdf5_file:
            for key in hdf5_file.keys():
                group = hdf5_file[key]

                max_hpoly_length = max(max_hpoly_length, len(group['hpoly_end_probs'][:]))
                max_vpoly_length = max(max_vpoly_length, len(group['vpoly_end_probs'][:]))

        self.max_hpoly_length = max_hpoly_length
        self.max_vpoly_length = max_vpoly_length

        self.max_points = 0
        for root, dirs, files in os.walk(pcd_file_dir):
            for file in files:
                if file.endswith(".pcd"):
                    pcd_path = os.path.join(root, file)
                    curr_cloud = o3d.io.read_point_cloud(pcd_path)
                    curr_points = np.asarray(curr_cloud.points)
                    self.max_points = max(self.max_points, curr_points.shape[0])

    def __len__(self):
        with h5py.File(self.hdf5_path, 'r') as hdf5_file:
            return len(hdf5_file.keys())

    def __getitem__(self, idx):
        with h5py.File(self.hdf5_path, 'r') as hdf5_file:
            group = hdf5_file[f"idx_{idx}"]
            pcd_rela_path = group["pcd_rela_path"][()].decode("utf-8")
            state = torch.tensor(group["state"][:])
            hpoly_elems = torch.tensor(group["hpoly_elems"][:])
            hpoly_end_probs = torch.tensor(group["hpoly_end_probs"][:])
            hpoly_seq_end_probs = torch.tensor(group["hpoly_seq_end_probs"][:])
            vpoly_elems = torch.tensor(group["vpoly_elems"][:])
            vpoly_end_probs = torch.tensor(group["vpoly_end_probs"][:])
            vpoly_seq_end_probs = torch.tensor(group["vpoly_seq_end_probs"][:])

        pcd_path = os.path.join(self.root_dir, pcd_rela_path)
        cloud = o3d.io.read_point_cloud(pcd_path)
        pcd_data = np.asarray(cloud.points)

        padded_pcd_data, padded_hpoly_elems, padded_hpoly_end_probs, padded_hpoly_seq_end_probs, padded_vpoly_elems, padded_vpoly_end_probs, padded_vpoly_seq_end_probs = self.padding(pcd_data, hpoly_elems, hpoly_end_probs, hpoly_seq_end_probs, vpoly_elems, vpoly_end_probs, vpoly_seq_end_probs)
        # print("Max points: {}".format(self.max_points))
        # print("dimension of padded_pc_data: ", padded_pcd_data.shape)

        return padded_pcd_data, state, padded_hpoly_elems, padded_hpoly_end_probs, padded_hpoly_seq_end_probs, padded_vpoly_elems, padded_vpoly_end_probs, padded_vpoly_seq_end_probs

    def get_max_hpoly_length(self):
        return self.max_hpoly_length

    def get_max_vpoly_length(self):
        return self.max_vpoly_length

    def padding(self, pcd_data, hpoly_elems, hpoly_end_probs, hpoly_seq_end_probs, vpoly_elems, vpoly_end_probs, vpoly_seq_end_probs):
        if pcd_data.shape[0] < self.max_points:
            pcd_data_padding = np.zeros((self.max_points - pcd_data.shape[0], 3), dtype=pcd_data.dtype)
            padded_pcd_data = np.concatenate((pcd_data, pcd_data_padding), axis=0)
        else:
            padded_pcd_data = pcd_data

        if hpoly_elems.size(0) < self.max_hpoly_length:
            hpoly_elems_padding = torch.zeros((self.max_hpoly_length - hpoly_elems.size(0), 4), dtype=hpoly_elems.dtype)
            padded_hpoly_elems = torch.cat((hpoly_elems, hpoly_elems_padding), dim=0)
        else:
            padded_hpoly_elems = hpoly_elems

        if hpoly_end_probs.size(0) < self.max_hpoly_length:
            hpoly_end_probs_padding = torch.zeros(self.max_hpoly_length - hpoly_end_probs.size(0), dtype=hpoly_end_probs.dtype)
            padded_hpoly_end_probs = torch.cat((hpoly_end_probs, hpoly_end_probs_padding))
        else:
            padded_hpoly_end_probs = hpoly_end_probs

        if hpoly_seq_end_probs.size(0) < self.max_hpoly_length:
            hpoly_seq_end_probs_padding = torch.zeros(self.max_hpoly_length - hpoly_seq_end_probs.size(0), dtype=hpoly_seq_end_probs.dtype)
            padded_hpoly_seq_end_probs = torch.cat((hpoly_seq_end_probs, hpoly_seq_end_probs_padding))
        else:
            padded_hpoly_seq_end_probs = hpoly_seq_end_probs

        if vpoly_elems.size(0) < self.max_vpoly_length:
            vpoly_elems_padding = torch.zeros((self.max_vpoly_length - vpoly_elems.size(0), 3), dtype=vpoly_elems.dtype)
            padded_vpoly_elems = torch.cat((vpoly_elems, vpoly_elems_padding), dim=0)
        else:
            padded_vpoly_elems = vpoly_elems

        if vpoly_end_probs.size(0) < self.max_vpoly_length:
            vpoly_end_probs_padding = torch.zeros(self.max_vpoly_length - vpoly_end_probs.size(0), dtype=vpoly_end_probs.dtype)
            padded_vpoly_end_probs = torch.cat((vpoly_end_probs, vpoly_end_probs_padding))
        else:
            padded_vpoly_end_probs = vpoly_end_probs

        if vpoly_seq_end_probs.size(0) < self.max_vpoly_length:
            vpoly_seq_end_probs_padding = torch.zeros(self.max_vpoly_length - vpoly_seq_end_probs.size(0), dtype=vpoly_seq_end_probs.dtype)
            padded_vpoly_seq_end_probs = torch.cat((vpoly_seq_end_probs, vpoly_seq_end_probs_padding))
        else:
            padded_vpoly_seq_end_probs = vpoly_seq_end_probs

        return padded_pcd_data, padded_hpoly_elems, padded_hpoly_end_probs, padded_hpoly_seq_end_probs, padded_vpoly_elems, padded_vpoly_end_probs, padded_vpoly_seq_end_probs


class MultiMapMinSnapDataset(Dataset):
    def __init__(self, path, max_hpoly_length=None, max_vpoly_length=None):
        self.root_dir = path
        hdf5_path = os.path.join(path, "dataset.h5")
        # Check if file exist
        if not os.path.exists(hdf5_path):
            raise FileNotFoundError("File not found: {}".format(hdf5_path))

        self.hdf5_path = hdf5_path

        if max_hpoly_length is None or max_vpoly_length is None:
            max_hpoly_length = 0

            max_vpoly_length = 0

            with h5py.File(self.hdf5_path, 'r') as hdf5_file:
                for key in hdf5_file.keys():
                    group = hdf5_file[key]

                    max_hpoly_length = max(max_hpoly_length, len(group['hpoly_end_probs'][:]))
                    max_vpoly_length = max(max_vpoly_length, len(group['vpoly_end_probs'][:]))

            self.max_hpoly_length = max_hpoly_length
            self.max_vpoly_length = max_vpoly_length
        else:
            self.max_hpoly_length = max_hpoly_length
            self.max_vpoly_length = max_vpoly_length


    def __len__(self):
        with h5py.File(self.hdf5_path, 'r') as hdf5_file:
            return len(hdf5_file.keys())

    def __getitem__(self, idx):
        with h5py.File(self.hdf5_path, 'r') as hdf5_file:
            group = hdf5_file[f"idx_{idx}"]
            state = torch.tensor(group["state"][:])
            hpoly_elems = torch.tensor(group["hpoly_elems"][:])
            hpoly_end_probs = torch.tensor(group["hpoly_end_probs"][:])
            vpoly_elems = torch.tensor(group["vpoly_elems"][:])
            vpoly_end_probs = torch.tensor(group["vpoly_end_probs"][:])

        padded_hpoly_elems, padded_hpoly_end_probs, padded_vpoly_elems, padded_vpoly_end_probs = self.padding(hpoly_elems, hpoly_end_probs, vpoly_elems, vpoly_end_probs)

        # return state, padded_hpoly_elems, padded_hpoly_end_probs
        return state, padded_hpoly_elems, padded_hpoly_end_probs, padded_vpoly_elems, padded_vpoly_end_probs

    def padding(self, hpoly_elems, hpoly_end_probs, vpoly_elems, vpoly_end_probs):
        if hpoly_elems.size(0) < self.max_hpoly_length:
            hpoly_elems_padding = torch.zeros((self.max_hpoly_length - hpoly_elems.size(0), 4), dtype=hpoly_elems.dtype)
            padded_hpoly_elems = torch.cat((hpoly_elems, hpoly_elems_padding), dim=0)
        else:
            padded_hpoly_elems = hpoly_elems

        if hpoly_end_probs.size(0) < self.max_hpoly_length:
            hpoly_end_probs_padding = torch.zeros(self.max_hpoly_length - hpoly_end_probs.size(0), dtype=hpoly_end_probs.dtype)
            padded_hpoly_end_probs = torch.cat((hpoly_end_probs, hpoly_end_probs_padding))
        else:
            padded_hpoly_end_probs = hpoly_end_probs

        if vpoly_elems.size(0) < self.max_vpoly_length:
            vpoly_elems_padding = torch.zeros((self.max_vpoly_length - vpoly_elems.size(0), 3), dtype=vpoly_elems.dtype)
            padded_vpoly_elems = torch.cat((vpoly_elems, vpoly_elems_padding), dim=0)
        else:
            padded_vpoly_elems = vpoly_elems
        
        if vpoly_end_probs.size(0) < self.max_vpoly_length:
            vpoly_end_probs_padding = torch.zeros(self.max_vpoly_length - vpoly_end_probs.size(0), dtype=vpoly_end_probs.dtype)
            padded_vpoly_end_probs = torch.cat((vpoly_end_probs, vpoly_end_probs_padding))
        else:
            padded_vpoly_end_probs = vpoly_end_probs

        return padded_hpoly_elems, padded_hpoly_end_probs, padded_vpoly_elems, padded_vpoly_end_probs

    def filter_indices(self):
        valid_indices = []
        with h5py.File(self.hdf5_path, 'r') as hdf5_file:
            for idx, key in enumerate(hdf5_file.keys()):
                group = hdf5_file[key]
                if len(group['hpoly_end_probs'][:]) <= self.max_hpoly_length and \
                        len(group['vpoly_end_probs'][:]) <= self.max_vpoly_length:
                    valid_indices.append(idx)
        return valid_indices



class ConvMultiMapMinSnapDataset(Dataset):
    def __init__(self, path, max_hpoly_length=None, max_vpoly_length=None):
        self.root_dir = path
        hdf5_path = os.path.join(path, "dataset.h5")
        # Check if file exist
        if not os.path.exists(hdf5_path):
            raise FileNotFoundError("File not found: {}".format(hdf5_path))

        self.hdf5_path = hdf5_path

        if max_hpoly_length is None or max_vpoly_length is None:
            max_hpoly_length = 0

            max_vpoly_length = 0

            with h5py.File(self.hdf5_path, 'r') as hdf5_file:
                for key in hdf5_file.keys():
                    group = hdf5_file[key]

                    max_hpoly_length = max(max_hpoly_length, len(group['hpoly_end_probs'][:]))
                    max_vpoly_length = max(max_vpoly_length, len(group['vpoly_end_probs'][:]))

            self.max_hpoly_length = max_hpoly_length
            self.max_vpoly_length = max_vpoly_length
        else:
            self.max_hpoly_length = max_hpoly_length
            self.max_vpoly_length = max_vpoly_length

    def __len__(self):
        with h5py.File(self.hdf5_path, 'r') as hdf5_file:
            return len(hdf5_file.keys())

    def __getitem__(self, idx):
        with h5py.File(self.hdf5_path, 'r') as hdf5_file:
            group = hdf5_file[f"idx_{idx}"]
            state = torch.tensor(group["state"][:])
            hpoly_elems = torch.tensor(group["hpoly_elems"][:])
            hpoly_end_probs = torch.tensor(group["hpoly_end_probs"][:])
            vpoly_elems = torch.tensor(group["vpoly_elems"][:])
            vpoly_end_probs = torch.tensor(group["vpoly_end_probs"][:])
            stacked_state = torch.tensor(group["stacked_state"][:])
            stacked_hpolys = torch.tensor(group["stacked_hpolys"][:])

        padded_hpoly_elems, padded_hpoly_end_probs, padded_vpoly_elems, padded_vpoly_end_probs = self.padding(
            hpoly_elems, hpoly_end_probs, vpoly_elems, vpoly_end_probs)

        # return state, padded_hpoly_elems, padded_hpoly_end_probs
        return state, padded_hpoly_elems, padded_hpoly_end_probs, padded_vpoly_elems, padded_vpoly_end_probs, stacked_hpolys, stacked_state

    def padding(self, hpoly_elems, hpoly_end_probs, vpoly_elems, vpoly_end_probs):
        if hpoly_elems.size(0) < self.max_hpoly_length:
            hpoly_elems_padding = torch.zeros((self.max_hpoly_length - hpoly_elems.size(0), 4), dtype=hpoly_elems.dtype)
            padded_hpoly_elems = torch.cat((hpoly_elems, hpoly_elems_padding), dim=0)
        else:
            padded_hpoly_elems = hpoly_elems

        if hpoly_end_probs.size(0) < self.max_hpoly_length:
            hpoly_end_probs_padding = torch.zeros(self.max_hpoly_length - hpoly_end_probs.size(0),
                                                  dtype=hpoly_end_probs.dtype)
            padded_hpoly_end_probs = torch.cat((hpoly_end_probs, hpoly_end_probs_padding))
        else:
            padded_hpoly_end_probs = hpoly_end_probs

        if vpoly_elems.size(0) < self.max_vpoly_length:
            vpoly_elems_padding = torch.zeros((self.max_vpoly_length - vpoly_elems.size(0), 3), dtype=vpoly_elems.dtype)
            padded_vpoly_elems = torch.cat((vpoly_elems, vpoly_elems_padding), dim=0)
        else:
            padded_vpoly_elems = vpoly_elems

        if vpoly_end_probs.size(0) < self.max_vpoly_length:
            vpoly_end_probs_padding = torch.zeros(self.max_vpoly_length - vpoly_end_probs.size(0),
                                                  dtype=vpoly_end_probs.dtype)
            padded_vpoly_end_probs = torch.cat((vpoly_end_probs, vpoly_end_probs_padding))
        else:
            padded_vpoly_end_probs = vpoly_end_probs

        return padded_hpoly_elems, padded_hpoly_end_probs, padded_vpoly_elems, padded_vpoly_end_probs

    def filter_indices(self):
        valid_indices = []
        with h5py.File(self.hdf5_path, 'r') as hdf5_file:
            for idx, key in enumerate(hdf5_file.keys()):
                group = hdf5_file[key]
                if len(group['hpoly_end_probs'][:]) <= self.max_hpoly_length and \
                        len(group['vpoly_end_probs'][:]) <= self.max_vpoly_length:
                    valid_indices.append(idx)
        return valid_indices
