from utils.learning.minsnap_network import MinimalSnapNetwork
from utils.pcd_segmentation import get_bounding_box
import torch
from utils.corridor_generator import CorridorGenerator
import numpy as np
import open3d as o3d
import yaml
import time


models_dir = "models/minsnap_185m_3l2c_20_ac2"
pcd_dir = "datasets/single_map_dataset/map.pcd"
model_idx = 3


start_goal_dist_min = 2.0

model_dir = models_dir + "/checkpoint" + str(model_idx) + ".pt"
config_dir = models_dir + "/config.yaml"
config = yaml.load(open(config_dir), Loader=yaml.FullLoader)

max_vel = config['physical_limits']['max_vel']
max_acc = config['physical_limits']['max_acc']
seg = config['planning']['seg']


max_hpoly_length = config['max_hpoly_length']

cloud = o3d.io.read_point_cloud(pcd_dir)
if cloud.is_empty():
    exit()
points = np.asarray(cloud.points)
range_x_min, range_y_min, range_z_min, range_x_max, range_y_max, range_z_max = get_bounding_box(points)
map_range = ([range_x_min, range_y_min, range_z_min], [range_x_max, range_y_max, range_z_max])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def sample_input():
    rand_start_x = np.random.uniform(range_x_min, range_x_max)
    rand_start_y = np.random.uniform(range_y_min, range_y_max)
    rand_start_z = np.random.uniform(range_z_min, range_z_max)

    rand_end_x = np.random.uniform(range_x_min, range_x_max)
    rand_end_y = np.random.uniform(range_y_min, range_y_max)
    rand_end_z = np.random.uniform(range_z_min, range_z_max)

    start = np.array([rand_start_x, rand_start_y, rand_start_z])
    goal = np.array([rand_end_x, rand_end_y, rand_end_z])

    # get random vel and acc
    rand_vel_x = np.random.uniform(-max_vel, max_vel)
    rand_vel_y = np.random.uniform(-max_vel, max_vel)
    rand_vel_z = np.random.uniform(-max_vel, max_vel)
    rand_acc_x = np.random.uniform(-max_acc, max_acc)
    rand_acc_y = np.random.uniform(-max_acc, max_acc)
    rand_acc_z = np.random.uniform(-max_acc, max_acc)
    start_state = np.transpose(
        np.array([start, [rand_vel_x, rand_vel_y, rand_vel_z], [rand_acc_x, rand_acc_y, rand_acc_z]]))
    end_state = np.zeros(start_state.shape)
    end_state[:, 0] = goal

    state = torch.from_numpy(np.concatenate((start_state, end_state), axis=None))

    if np.linalg.norm(start - goal) < start_goal_dist_min:
        print("start and goal are too close! Try again!")
        return state, None

    sfc_generator = CorridorGenerator(start, goal, map_range, 0.3, points, 5, 5.0)

    res = sfc_generator.get_corridor()
    pts = sfc_generator.get_inner_pts()
    # sfc_generator.vis_corridor()
    # print(type(hpolys[0]))

    hpoly_elems, hpoly_end_probs, _, _, _, _ = res

    if res is None or pts is None:
        return None

    # Add one dimension to the front of state
    state = state.unsqueeze(0)
    hpoly_elems = torch.from_numpy(hpoly_elems).unsqueeze(0)
    hpoly_end_probs = torch.from_numpy(hpoly_end_probs).unsqueeze(0)

    # Pad hpoly_elems and hpoly_end_probs to max_hpoly_length
    if hpoly_elems.size(1) < max_hpoly_length:
        pad = torch.zeros((1, max_hpoly_length - hpoly_elems.size(1), 4), dtype=hpoly_elems.dtype)
        hpoly_elems = torch.cat((hpoly_elems, pad), dim=1)
        pad = torch.zeros((1, max_hpoly_length - hpoly_end_probs.size(1)), dtype=hpoly_end_probs.dtype)
        hpoly_end_probs = torch.cat((hpoly_end_probs, pad), dim=1)

    return state.to(device), hpoly_elems.to(device), hpoly_end_probs.to(device)


def main():
    np.set_printoptions(threshold=np.inf)

    state, hpoly_elems, hpoly_end_probs = sample_input()

    model = MinimalSnapNetwork(seg=seg, max_hpoly_length=max_hpoly_length)
    model.to(device)
    checkpoint = torch.load(model_dir)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    inference_start_time = time.time()
    solution = model(state, hpoly_elems, hpoly_end_probs)
    inference_end_time = time.time()
    print("Total computation time: ", inference_end_time - inference_start_time)

    # model.qp_traj.vis_sfc_with_trajs(i=1)


if __name__=='__main__':
    main()