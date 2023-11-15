import os
import torch
import yaml
import random
import numpy as np
import resource
import sys
import gc
import objgraph

from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from utils.learning.minsnap_network_conv_mlp_as import ConvMLPMinimalSnapNetwork4AblationStudy
from utils.learning.datasets import ConvMultiMapMinSnapDataset

import tracemalloc

import os
import glob
import logging



class MinSnapNetworkTrainingManager():
    def __init__(self, config_dir):
        print("===== Init MinSnapNetworkTrainingManager =====")
        print("Parse config file")

        #torch.set_default_dtype(torch.float32)
        # Check for previous config
        config = yaml.load(open(config_dir), Loader=yaml.FullLoader)
        self.checkpoint_dir = config["checkpoint_dir"]
        session_saved_config_dir = os.path.join(self.checkpoint_dir, "config.yaml")
        if os.path.exists(session_saved_config_dir):
            print("Found previous config file, load it")
            config = yaml.load(open(session_saved_config_dir), Loader=yaml.FullLoader)
            max_hpoly_length = config["max_hpoly_length"]
            max_vpoly_length = config["max_vpoly_length"]
        else:
            max_hpoly_length = None
            max_vpoly_length = None

        self.poly_mode = config["poly_mode"]

        self.dataset_dir = config["dataset_dir"]
        self.log_dir = config["log_dir"]
        max_vel = config['physical_limits']['max_vel']
        max_acc = config['physical_limits']['max_acc']
        seg = config['planning']['seg']

        random_seed = config['random_seed']

        self.use_scheduler = config["use_scheduler"]

        self.max_epochs = config["training"]["max_epochs"]
        self.save_freq = config["training"]["save_freq"]
        training_data_ratio = config["training"]["training_data_ratio"]
        dataloader_batch_size = config["training"]["dataloader_batch_size"]
        dataloader_shuffle = config["training"]["dataloader_shuffle"]
        self.validation_batch_size = config["training"]["validation_batch_size"]

        hidden_size = config["training"]["hidden_size"]
        learning_rate = config["training"]["learning_rate"]

        if isinstance(learning_rate, str):
            learning_rate = float(learning_rate)

        s_T_0 = config["scheduler"]["T_0"]
        s_T_mult = config["scheduler"]["T_mult"]
        s_eta_min = config["scheduler"]["eta_min"]
        s_last_epoch = config["scheduler"]["last_epoch"]

        # Load dataset
        print("Load dataset")
        # if os.path.exists(self.dataset_dir):
        #     if max_hpoly_length is None:
        #         dataset = ConvMultiMapMinSnapDataset(self.dataset_dir)
        #         max_hpoly_length = 50
        #         #max_vpoly_length = dataset.max_vpoly_length
        #         prev_recorded_len = False
        #     else:
        #         dataset = ConvMultiMapMinSnapDataset(self.dataset_dir, max_hpoly_length, max_vpoly_length)
        #         prev_recorded_len = True
        # else:
        #     raise Exception("Dataset not found")
        dataset = ConvMultiMapMinSnapDataset(self.dataset_dir)
        # Prepare dataloaders
        print("Prepare dataloaders")
        #processed_dataset = None

        # if prev_recorded_len:
        #     valid_indices = dataset.filter_indices()
        #     processed_dataset = torch.utils.data.Subset(dataset, valid_indices)
        # else:
        
        
        processed_dataset = dataset


 
        if training_data_ratio < 1.0:
            training_dataset_size = int(training_data_ratio * len(processed_dataset))
            validation_dataset_size = len(processed_dataset) - training_dataset_size
            training_dataset, validation_dataset = random_split(dataset=processed_dataset,
                                                                lengths=[training_dataset_size, validation_dataset_size],
                                                                generator=torch.Generator().manual_seed(random_seed))
        else:
            training_dataset = processed_dataset
            validation_dataset = processed_dataset
        
        
        self.training_dataloader = DataLoader(dataset=training_dataset,
                                              batch_size=dataloader_batch_size,
                                              shuffle=dataloader_shuffle)

        self.validation_dataloader = DataLoader(dataset=validation_dataset,
                                                batch_size=self.validation_batch_size,
                                                shuffle=dataloader_shuffle)

        self.validation_dataloader_iterator = iter(self.validation_dataloader)

        self.curr_epoch_idx = 0

        # Initialize model
        print("Initialize model")
        loss_dim_reduction = config["training"]["loss_dim_reduction"]
        w1 = config["obj_weights"]["w1"]

        wt = config["obj_weights"]["wt"]

        wc = config["obj_weights"]["wc"]

        wp = config["obj_weights"]["wp"]

        self.model = ConvMLPMinimalSnapNetwork4AblationStudy(seg=seg,
                                                            max_poly_length=max_hpoly_length, lr=learning_rate, T_0=s_T_0,
                                                            T_mult=s_T_mult, eta_min=s_eta_min, last_epoch=s_last_epoch,
                                                            hidden_size=256, use_scheduler=self.use_scheduler,
                                                            loss_dim_reduction=loss_dim_reduction, w1=0.0, wt=1.0, wc=0.0, wp=0.0)


        # Set random seed
        print("Set random seed: ", random_seed)
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)

        self.save_index = 0
        self.curr_step_idx = 0

        print("Check for previous training sessions")
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
            config["max_hpoly_length"] = max_hpoly_length
            config["max_vpoly_length"] = max_vpoly_length
            with open(session_saved_config_dir, 'w') as f:
                yaml.dump(config, f)
        else:
            checkpoint_file_list = os.listdir(self.checkpoint_dir)
            checkpoint_file_list = [file for file in checkpoint_file_list if not file.endswith(".yaml")]
            if len(checkpoint_file_list) > 0:
                checkpoint_file_list.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
                last_checkpoint_file_path = os.path.join(self.checkpoint_dir, checkpoint_file_list[-1])
                print("Loading checkpoint: {}".format(last_checkpoint_file_path))
                checkpoint = torch.load(last_checkpoint_file_path)
                self.model.load_state_dict(checkpoint["model_state_dict"])
                self.model.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                if self.use_scheduler:
                    self.model.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                self.curr_epoch_idx = checkpoint["epoch_idx"]
                self.save_index = checkpoint["save_index"] + 1
                self.curr_step_idx = checkpoint["step_idx"] + 1


        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.writer = SummaryWriter(self.log_dir)

        # objt outlier debug
        # self.objt_debug_log_thresh = 35.0
        # self.objt_log_dir = os.path.join(self.checkpoint_dir, "objt_debug_log")
        # if not os.path.exists(self.objt_log_dir):
        #     os.makedirs(self.objt_log_dir)


        print("===== Init MinSnapNetworkTrainingManager Done =====")

    def save_checkpoint(self):
        checkpoint_file_path = os.path.join(self.checkpoint_dir, "checkpoint" + str(self.save_index) + ".pt")
        if self.use_scheduler:
            torch.save({
                "step_idx": self.curr_step_idx,
                "epoch_idx": self.curr_epoch_idx,
                "save_index": self.save_index,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.model.optimizer.state_dict(),
                "scheduler_state_dict": self.model.scheduler.state_dict(),
            }, checkpoint_file_path)
        else:
            torch.save({
                "step_idx": self.curr_step_idx,
                "epoch_idx": self.curr_epoch_idx,
                "save_index": self.save_index,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.model.optimizer.state_dict(),
            }, checkpoint_file_path)
        print("------ checkpoint saved at: {} ------".format(checkpoint_file_path))
        self.save_index += 1

    def train_one_epoch(self):
        total_obj_val = 0.0
        total_obj1_val = 0.0
        total_objt_val = 0.0
        total_objc_val = 0.0

        # print("------ MEMORY USAGE: train_one_epoch begins: {} MB ------".format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024))
        for i, data in enumerate(self.training_dataloader):
            #tracemalloc.start()

            # print("------ MEMORY USAGE: train_one_epoch dataloader enumeration begins: {} MB ------".format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024))
            stacked_hpolys, stacked_state, traj_times = data

            # x = x.to(self.device)

            # hpoly_elems = hpoly_elems.to(self.device)
            # hpoly_end_probs = hpoly_end_probs.to(self.device)

            stacked_hpolys = stacked_hpolys.to(self.device)
            stacked_state = stacked_state.to(self.device)

            traj_times = traj_times.to(self.device)



            curr_obj_val, curr_obj1_val, curr_objt_val, curr_objc_val, curr_success_rate, objt_debug_log, curr_padding_loss = \
                self.model.train_model(stacked_state, stacked_hpolys, traj_times)

            self.writer.add_scalar("obj_val training / iteration", curr_obj_val, self.curr_step_idx)
            self.writer.add_scalar("obj1_val training / iteration", curr_obj1_val, self.curr_step_idx)
            self.writer.add_scalar("objt_val training / iteration", curr_objt_val, self.curr_step_idx)
            self.writer.add_scalar("objc_val training / iteration", curr_objc_val, self.curr_step_idx)
            self.writer.add_scalar("success_rate training / iteration", curr_success_rate, self.curr_step_idx)
            self.writer.add_scalar("padding_loss training / iteration", curr_padding_loss, self.curr_step_idx)
            # print("------ MEMORY USAGE: train_one_epoch after train_model: {} MB ------".format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024))

            total_obj_val += curr_obj_val
            total_obj1_val += curr_obj1_val
            total_objt_val += curr_objt_val
            total_objc_val += curr_objc_val

            if self.curr_step_idx % self.save_freq == 0:
                self.save_checkpoint()
                # print("------ MEMORY USAGE: train_one_epoch after save_checkpoint: {} MB ------".format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024))

            del curr_obj_val
            del curr_obj1_val
            del curr_objt_val

            gc.collect()
            torch.cuda.empty_cache()
            
            self.curr_step_idx = self.curr_step_idx + 1

        avg_obj_val = total_obj_val / len(self.training_dataloader)
        avg_obj1_val = total_obj1_val / len(self.training_dataloader)
        avg_objt_val = total_objt_val / len(self.training_dataloader)
        avg_objc_val = total_objc_val / len(self.training_dataloader)

        self.curr_epoch_idx += 1

        del total_obj_val
        del total_obj1_val
        del total_objt_val

        gc.collect()
        torch.cuda.empty_cache()

        return avg_obj_val, avg_obj1_val, avg_objt_val, avg_objc_val

    def train(self):
        print("===== Start Training =====")
        while self.curr_epoch_idx < self.max_epochs:
            print("Epoch: {}".format(self.curr_epoch_idx))
            avg_obj_val, avg_obj1_val, avg_objt_val, avg_objc_ = self.train_one_epoch()

            obj_vali_vals = []
            obj1_vali_vals = []
            objt_vali_vals = []
            objc_vali_vals = []

            for _ in range(self.validation_batch_size):
                try:
                    vali_data = next(self.validation_dataloader_iterator)
                except StopIteration:
                    self.validation_dataloader_iterator = iter(self.validation_dataloader)
                    vali_data = next(self.validation_dataloader_iterator)

                vali_states, vali_hpolys, vali_times = vali_data

                vali_states = vali_states.to(self.device)
                vali_hpolys = vali_hpolys.to(self.device)
                vali_times  = vali_times.to(self.device)

                curr_obj_vali_val, curr_obj1_vali_val, curr_objt_vali_val, curr_objc_vali_val \
                      = self.model.eval_model(vali_states, vali_hpolys, vali_times)

                obj_vali_vals.append(curr_obj_vali_val)
                obj1_vali_vals.append(curr_obj1_vali_val)
                objt_vali_vals.append(curr_objt_vali_val)
                objc_vali_vals.append(curr_objc_vali_val)

            obj_vali_val = sum(obj_vali_vals) / len(obj_vali_vals)
            obj1_vali_val = sum(obj1_vali_vals) / len(obj1_vali_vals)
            objt_vali_val = sum(objt_vali_vals) / len(objt_vali_vals)
            objc_vali_val = sum(objc_vali_vals) / len(objc_vali_vals)

            print("obj_vali_val: {}".format(obj_vali_val))
            print("obj1_vali_val: {}".format(obj1_vali_val))
            print("objt_vali_val: {}".format(objt_vali_val))

            self.writer.add_scalar("obj_val validation / epoch", obj_vali_val, self.curr_epoch_idx)
            self.writer.add_scalar("obj1_val validation / epoch", obj1_vali_val, self.curr_epoch_idx)
            self.writer.add_scalar("objt_val validation / epoch", objt_vali_val, self.curr_epoch_idx)
            self.writer.add_scalar("objc_val validation / epoch", objc_vali_val, self.curr_epoch_idx)

            self.writer.add_scalar("obj_val training / epoch", avg_obj_val, self.curr_epoch_idx)
            self.writer.add_scalar("obj1_val training / epoch", avg_obj1_val, self.curr_epoch_idx)
            self.writer.add_scalar("objt_val training / epoch", avg_objt_val, self.curr_epoch_idx)
            self.writer.add_scalar("objc_val training / epoch", avg_objc_, self.curr_epoch_idx)

            # self.save_checkpoint()

        self.writer.close()
        print("===== Finish Training =====")


if __name__ == '__main__':
    config_dir = "configs/minsnap_conv_params.yaml"
    training_manager = MinSnapNetworkTrainingManager(config_dir)
    training_manager.train()

