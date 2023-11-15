from utils.learning.minsnap_network_conv_lstm import ConvLSTMMinimalSnapNetwork
from utils.pcd_segmentation import get_bounding_box
import torch
from utils.corridor_generator import CorridorGenerator
import numpy as np
import open3d as o3d
import yaml
import time


models_dir = "models/minsnap_dlstm_10"
pcd_dir = "datasets/test_map.pcd"
model_idx = 2114


start_goal_dist_min = 2.0

safe_distance = 0.3

max_num_corridors = 5

planner_timeout_threshold = 10.0

batch_size = 100

model_dir = models_dir + "/checkpoint" + str(model_idx) + ".pt"
config_dir = models_dir + "/config.yaml"
config = yaml.load(open(config_dir), Loader=yaml.FullLoader)

seg = config['planning']['seg']

max_vel = config['physical_limits']['max_vel']
max_acc = config['physical_limits']['max_acc']
seg = config['planning']['seg']


max_hpoly_length = 50

cloud = o3d.io.read_point_cloud(pcd_dir)
if cloud.is_empty():
    exit()
points = np.asarray(cloud.points)
range_x_min, range_y_min, range_z_min, range_x_max, range_y_max, range_z_max = get_bounding_box(points)
map_range = ([range_x_min, range_y_min, range_z_min], [range_x_max, range_y_max, range_z_max])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def reconstruct_hpoly(hpoly_elems, hpoly_end_probs):
    hpolys = None

    curr_elems = None
    for i in range(len(hpoly_end_probs)):
        if hpoly_end_probs[i] == 0:
            if curr_elems is None:
                curr_elems = hpoly_elems[i]
            else:
                curr_elems = np.vstack((curr_elems, hpoly_elems[i]))
        elif hpoly_end_probs[i] == 1:
            curr_elems = np.vstack((curr_elems, hpoly_elems[i]))

            if curr_elems.shape[0] < max_hpoly_length:
                curr_elems = np.pad(curr_elems, ((0, max_hpoly_length - curr_elems.shape[0]), (0, 0)), 'constant', constant_values=0)
            elif curr_elems.shape[0] == max_hpoly_length:
                pass
            else:
                print("Error: curr_elems.shape[0] > max_hpoly_length!")

            if hpolys is None:
                hpolys = curr_elems
            else:
                hpolys = np.dstack((hpolys, curr_elems))

            curr_elems = None

    #print("hpoly shape before padding: ", hpolys.shape)

    if hpolys.shape[2] < max_num_corridors:
        hpolys = np.pad(hpolys, ((0, 0), (0, 0), (0, max_num_corridors - hpolys.shape[2])), 'constant', constant_values=0)
    elif hpolys.shape[2] == max_num_corridors:
        pass
    else:
        print("Error: hpolys.shape[0] > max_num_corridors!")
        return None

    #print("hpoly shape after padding: ", hpolys.shape)

    #hpolys = np.transpose(hpolys, (2, 1, 0))

    return hpolys


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
        np.array([rand_start_x, rand_start_y, rand_start_z, rand_vel_x, rand_vel_y, rand_vel_z, rand_acc_x, rand_acc_y, rand_acc_z]))
    end_state = np.zeros(start_state.shape)
    end_state[0:3] = goal

    stacked_state = np.vstack((start_state, end_state))


    stacked_state = np.transpose(stacked_state, (1, 0))

    if np.linalg.norm(start - goal) < start_goal_dist_min:
        print("start and goal are too close! Try again!")
        return None, None

    sfc_generator = CorridorGenerator(start, goal, map_range, safe_distance, points, max_num_corridors, planner_timeout_threshold)

    res = sfc_generator.get_corridor()
    pts = sfc_generator.get_inner_pts()

    if res is None or pts is None:
        return None, None

    hpoly_elems, hpoly_end_probs, _, _, _, _ = res

    stacked_hpolys = reconstruct_hpoly(hpoly_elems, hpoly_end_probs)




    # Add one dimension to the front of state
    stacked_state = torch.from_numpy(stacked_state).unsqueeze(0)
    stacked_hpolys = torch.from_numpy(stacked_hpolys).unsqueeze(0)

    return stacked_state.to(device), stacked_hpolys.to(device)


def main():
    num_correct_stop_tokens = 0

    for i in range(batch_size):
        np.set_printoptions(threshold=np.inf)

        stacked_state, stacked_hpolys = sample_input()

        while stacked_state is None or stacked_hpolys is None:
            stacked_state, stacked_hpolys = sample_input()

        model = ConvLSTMMinimalSnapNetwork(seg=seg, max_poly_length=max_hpoly_length, hidden_size=256)
        model.to(device)
        checkpoint = torch.load(model_dir)
        model.load_state_dict(checkpoint["model_state_dict"])

        comp_start_time = time.time()
        solution, obj1_val, objt_val, objc_val, stop_token_loss = model(stacked_state, stacked_hpolys)
        comp_end_time = time.time()
        print("Total computation time: ", comp_end_time - comp_start_time)

        print("solution: ", solution)
        print("obj1_val: ", obj1_val)
        #print("objt_val: ", objt_val)
        print("objc_val: ", objc_val)
        # model.qp_traj.vis_sfc_with_trajs(i=1)
        print("stop_token_loss: ", stop_token_loss)

        if stop_token_loss.item() < 1.0:
            num_correct_stop_tokens += 1

    print("====== Percentage of correct stop tokens: ", num_correct_stop_tokens / batch_size, " ======")


if __name__=='__main__':
    main()