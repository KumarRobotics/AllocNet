import sys
import h5py
import pathlib
import open3d as o3d
import numpy as np
from utils.corridor_generator import CorridorGenerator
from utils.pcd_segmentation import get_bounding_box
from utils.bilevel_traj_opt import BiLevelTrajOpt


pcd_dir = "datasets/single_map_dataset/map.pcd"
output_hdf5_dataset_dir = "datasets/trajs/dataset_traj.h5"

max_samples = 10000

start_goal_dist_min = 2.0

cloud = o3d.io.read_point_cloud(pcd_dir)
if cloud.is_empty(): exit()
points = np.asarray(cloud.points)

range_x_min, range_y_min, range_z_min, range_x_max, range_y_max, range_z_max = get_bounding_box(points)
print("map boundary: ", range_x_min, range_y_min, range_z_min, ' - ', range_x_max, range_y_max, range_z_max)

curr_sample_idx = 0

# If the dataset exist, get the largest saved index and continue from there
if pathlib.Path(output_hdf5_dataset_dir).exists():
    with h5py.File(output_hdf5_dataset_dir, 'r') as f:
        curr_sample_idx = len(f.keys())

# start = np.array([0.0, 0.0, 0.0])
# goal = np.array([0.0, 0.0, 0.0])
map_range = ([range_x_min, range_y_min, range_z_min], [range_x_max, range_y_max, range_z_max])



# planning parameters
import yaml


np.set_printoptions(threshold=np.inf)
with open("utils/params.yaml", "r") as stream:
    try:
        params = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)
        
max_vel = params['physical_limits']['max_vel']
max_acc = params['physical_limits']['max_acc']




while curr_sample_idx < max_samples:
    print("------ curr_sample_idx: ", curr_sample_idx, " ------")

    rand_start_x = np.random.uniform(range_x_min, range_x_max)
    rand_start_y = np.random.uniform(range_y_min, range_y_max)
    rand_start_z = np.random.uniform(range_z_min, range_z_max)

    rand_end_x = np.random.uniform(range_x_min, range_x_max)
    rand_end_y = np.random.uniform(range_y_min, range_y_max)
    rand_end_z = np.random.uniform(range_z_min, range_z_max)

    start = np.array([rand_start_x, rand_start_y, rand_start_z])
    goal = np.array([rand_end_x, rand_end_y, rand_end_z])

    if np.linalg.norm(start - goal) < start_goal_dist_min:
        print("start and goal are too close! Try again!")
        continue

    sfc_generator = CorridorGenerator(start, goal, map_range, 0.3, points, 5, 10.0)

    print("start: ", start)
    print("goal: ", goal)

    res = sfc_generator.get_corridor()
    

    if res is None:
        print("Failed to get corridor! Try again!")

        continue
    else:
        hpoly_elems, hpoly_end_probs, hpoly_seq_end_probs, vpoly_elems, vpoly_end_probs, vpoly_seq_end_probs = res
        print("hpoly_elems shape: ", hpoly_elems.shape)
        print("hpoly_end_probs shape: ", hpoly_end_probs.shape)
        print("hpoly_seq_end_probs shape: ", hpoly_seq_end_probs.shape)
        # print("vpoly_elems shape: ", vpoly_elems.shape)
        # print("vpoly_end_probs shape: ", vpoly_end_probs.shape)
        # print("vpoly_seq_end_probs shape: ", vpoly_seq_end_probs.shape)
    
    
    pts = sfc_generator.get_inner_pts()

    # get random vel and acc
    rand_vel_x = np.random.uniform(-max_vel, max_vel)
    rand_vel_y = np.random.uniform(-max_vel, max_vel)
    rand_vel_z = np.random.uniform(-max_vel, max_vel)

    rand_acc_x = np.random.uniform(-max_acc, max_acc)
    rand_acc_y = np.random.uniform(-max_acc, max_acc)
    rand_acc_z = np.random.uniform(-max_acc, max_acc)

    start_state = np.array([start,[rand_vel_x, rand_vel_y, rand_vel_z], [rand_acc_x, rand_acc_y, rand_acc_z]])


    qp_traj = BiLevelTrajOpt(start_state, goal, pts, hpoly_elems, params).qp_traj
    
    status = qp_traj.solve()
    #qp_traj.vis_traj(res = 0.1, i = curr_sample_idx)

    #parameters are Csqrt,G,h,A,b
    
    # coeff, t_allo = qp_traj.get_ct()


    state = np.concatenate((start, goal), axis=None)

    # If the dataset file does not exist, create it
    if not pathlib.Path(output_hdf5_dataset_dir).exists():
        with h5py.File(output_hdf5_dataset_dir, 'w') as f:
            group = f.create_group(f"idx_{curr_sample_idx}")
            group.create_dataset("state", data=state)
            # group.create_dataset("hpoly_elems", data=hpoly_elems)
            # group.create_dataset("hpoly_end_probs", data=hpoly_end_probs)
            # group.create_dataset("hpoly_seq_end_probs", data=hpoly_seq_end_probs)
            # group.create_dataset("vpoly_elems", data=vpoly_elems)
            # group.create_dataset("vpoly_end_probs", data=vpoly_end_probs)
            # group.create_dataset("vpoly_seq_end_probs", data=vpoly_seq_end_probs)
            group.create_dataset("traj_Csqrt", data=qp_traj.Csqrt)
            #group.create_dataset("traj_G", data=qp_traj.G)
            #group.create_dataset("traj_h", data=qp_traj.h)
            group.create_dataset("traj_A", data=qp_traj.A)
            group.create_dataset("traj_b", data=qp_traj.b)
            group.create_dataset("traj_coeff_matrices", data=coeff)

    else:
        with h5py.File(output_hdf5_dataset_dir, 'a') as f:
            group = f.create_group(f"idx_{curr_sample_idx}")
            group.create_dataset("state", data=state)
            group.create_dataset("hpoly_elems", data=hpoly_elems)
            # group.create_dataset("hpoly_end_probs", data=hpoly_end_probs)
            # group.create_dataset("hpoly_seq_end_probs", data=hpoly_seq_end_probs)
            # group.create_dataset("vpoly_elems", data=vpoly_elems)
            # group.create_dataset("vpoly_end_probs", data=vpoly_end_probs)
            # group.create_dataset("vpoly_seq_end_probs", data=vpoly_seq_end_probs)
            group.create_dataset("traj_Csqrt", data=qp_traj.Csqrt)
            #group.create_dataset("traj_G", data=qp_traj.G)
            #group.create_dataset("traj_h", data=qp_traj.h)
            group.create_dataset("traj_A", data=qp_traj.A)
            group.create_dataset("traj_b", data=qp_traj.b)
            group.create_dataset("traj_coeff_matrices", data=coeff)


    print("data saved to hdf5 file!")

    curr_sample_idx += 1

