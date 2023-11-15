import os
import glob
import h5py
import pathlib
import open3d as o3d
import numpy as np
from utils.corridor_generator import CorridorGenerator
from utils.pcd_segmentation import get_bounding_box

dataset_folder_dir = "datasets/multi_map_dataset"


samples_per_map = 50
start_goal_dist_min = 2.0
safe_distance = 0.3
max_num_corridors = 5
planner_timeout_threshold = 10.0

import yaml
with open("utils/params.yaml", "r") as stream:
    try:
        planning_params = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)
        
max_vel = planning_params['physical_limits']['max_vel']
max_acc = planning_params['physical_limits']['max_acc']


def find_pcd_files(directory):
    directory = os.path.normpath(directory) + os.sep

    pcd_files_sample_times_dict = {}

    for root, dirs, files in os.walk(directory):
        for file in glob.glob(root + os.sep + '*.pcd'):
            rel_path = os.path.relpath(file, directory)
            pcd_files_sample_times_dict[rel_path] = 0

    return pcd_files_sample_times_dict


def retrieve_sampling_times_for_maps(hdf5_filepath, pcd_dict):
    with h5py.File(hdf5_filepath, 'r') as hdf5_file:
        for group_name in hdf5_file:
            group = hdf5_file[group_name]
            pcd_rela_path = group['pcd_rela_path'][()].decode('utf-8')
            # print("pcd_rela_path: ", pcd_rela_path)
            # print("type(pcd_rela_path): ", type(pcd_rela_path))
            if pcd_rela_path in pcd_dict:
                pcd_dict[pcd_rela_path] += 1
            else:
                print("pcd_rela_path: ", pcd_rela_path, " not found in pcd_dict!")
        total_sample_num = len(hdf5_file.keys())

    return pcd_dict, total_sample_num



pcd_files_sample_times_dict = find_pcd_files(dataset_folder_dir)

output_hdf5_dataset_dir = os.path.join(dataset_folder_dir, "dataset.h5")

if pathlib.Path(output_hdf5_dataset_dir).exists():
    pcd_files_sample_times_dict, total_sample_num = retrieve_sampling_times_for_maps(output_hdf5_dataset_dir, pcd_files_sample_times_dict)
    # print("pcd_files_sample_times_dict: ", pcd_files_sample_times_dict)
    # print("len(pcd_files_sample_times_dict): ", len(pcd_files_sample_times_dict))
else:
    total_sample_num = 0

sampled_map_count = 0
for pcd_file_rela_path, sample_times in pcd_files_sample_times_dict.items():
    print("====== map: ", sampled_map_count, "/", len(pcd_files_sample_times_dict), " ======")
    curr_pcd_file_dir = os.path.join(dataset_folder_dir, pcd_file_rela_path)
    print("map directory: ", curr_pcd_file_dir)

    cloud = o3d.io.read_point_cloud(curr_pcd_file_dir)
    if cloud.is_empty():
        print("cloud is empty! Skipping...")
        exit()

    points = np.asarray(cloud.points)

    range_x_min, range_y_min, range_z_min, range_x_max, range_y_max, range_z_max = get_bounding_box(points)
    print("map boundary: ", range_x_min, range_y_min, range_z_min, ' - ', range_x_max, range_y_max, range_z_max)

    map_range = ([range_x_min, range_y_min, range_z_min], [range_x_max, range_y_max, range_z_max])

    curr_sample_idx = sample_times

    while curr_sample_idx < samples_per_map:
        print("------ sampling: ", curr_sample_idx, "/", samples_per_map, " corridor on map ", sampled_map_count, "/", len(pcd_files_sample_times_dict), " ------")

        rand_start_x = np.random.uniform(range_x_min, range_x_max)
        rand_start_y = np.random.uniform(range_y_min, range_y_max)
        rand_start_z = np.random.uniform(range_z_min, range_z_max)

        rand_goal_x = np.random.uniform(range_x_min, range_x_max)
        rand_goal_y = np.random.uniform(range_y_min, range_y_max)
        rand_goal_z = np.random.uniform(range_z_min, range_z_max)

        start = np.array([rand_start_x, rand_start_y, rand_start_z])
        goal = np.array([rand_goal_x, rand_goal_y, rand_goal_z])

        if np.linalg.norm(start - goal) < start_goal_dist_min:
            print("start and goal are too close! Skipping...")
            continue

        sfc_generator = CorridorGenerator(start, goal, map_range, safe_distance, points, max_num_corridors, planner_timeout_threshold)

        print("start: ", start)
        print("goal: ", goal)

        res = sfc_generator.get_corridor()
        pts, new_goal = sfc_generator.get_inner_pts()


        if res is None or pts is None:
            print("corridor generation failed! Skipping...")
            continue
        else:
            hpoly_elems, hpoly_end_probs, hpoly_seq_end_probs, vpoly_elems, vpoly_end_probs, vpoly_seq_end_probs = res
            print("hpoly_elems shape: ", hpoly_elems.shape)
            print("hpoly_end_probs shape: ", hpoly_end_probs.shape)
            print("hpoly_seq_end_probs shape: ", hpoly_seq_end_probs.shape)
            print("vpoly_elems shape: ", vpoly_elems.shape)
            print("vpoly_end_probs shape: ", vpoly_end_probs.shape)
            print("vpoly_seq_end_probs shape: ", vpoly_seq_end_probs.shape)
        


        # get random vel and acc
        rand_vel_x = np.random.uniform(-max_vel, max_vel)
        rand_vel_y = np.random.uniform(-max_vel, max_vel)
        rand_vel_z = np.random.uniform(-max_vel, max_vel)
        rand_acc_x = np.random.uniform(-max_acc, max_acc)
        rand_acc_y = np.random.uniform(-max_acc, max_acc)
        rand_acc_z = np.random.uniform(-max_acc, max_acc)
        start_state = np.transpose(np.array([start,[rand_vel_x, rand_vel_y, rand_vel_z], [rand_acc_x, rand_acc_y, rand_acc_z]]))
        end_state = np.zeros(start_state.shape)
        end_state[:, 0] = new_goal


        state =  np.concatenate((start_state, end_state), axis=None)
        #print("state is ", state)

        if not pathlib.Path(output_hdf5_dataset_dir).exists():
            with h5py.File(output_hdf5_dataset_dir, 'w') as f:
                group = f.create_group(f"idx_{total_sample_num}")
                group.create_dataset("pcd_rela_path", data=pcd_file_rela_path, dtype=h5py.string_dtype(encoding='utf-8'))
                group.create_dataset("state", data=state)
                group.create_dataset("hpoly_elems", data=hpoly_elems)
                group.create_dataset("hpoly_end_probs", data=hpoly_end_probs)
                group.create_dataset("hpoly_seq_end_probs", data=hpoly_seq_end_probs)
                group.create_dataset("vpoly_elems", data=vpoly_elems)
                group.create_dataset("vpoly_end_probs", data=vpoly_end_probs)
                group.create_dataset("vpoly_seq_end_probs", data=vpoly_seq_end_probs)
                group.create_dataset("traj_inner_points", data=pts)
        else:
            with h5py.File(output_hdf5_dataset_dir, 'a') as f:
                group = f.create_group(f"idx_{total_sample_num}")
                group.create_dataset("pcd_rela_path", data=pcd_file_rela_path, dtype=h5py.string_dtype(encoding='utf-8'))
                group.create_dataset("state", data=state)
                group.create_dataset("hpoly_elems", data=hpoly_elems)
                group.create_dataset("hpoly_end_probs", data=hpoly_end_probs)
                group.create_dataset("hpoly_seq_end_probs", data=hpoly_seq_end_probs)
                group.create_dataset("vpoly_elems", data=vpoly_elems)
                group.create_dataset("vpoly_end_probs", data=vpoly_end_probs)
                group.create_dataset("vpoly_seq_end_probs", data=vpoly_seq_end_probs)
                group.create_dataset("traj_inner_points", data=pts)  
        print('data daved to: ', output_hdf5_dataset_dir)

        curr_sample_idx += 1
        sample_times += 1
        total_sample_num += 1

    sampled_map_count += 1