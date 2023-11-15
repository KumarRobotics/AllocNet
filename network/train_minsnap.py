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
from utils.learning.minsnap_network import MinimalSnapNetwork
from utils.learning.datasets import MultiMapMinSnapDataset

import tracemalloc


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
        if os.path.exists(self.dataset_dir):
            if max_hpoly_length is None or max_vpoly_length is None:
                dataset = MultiMapMinSnapDataset(self.dataset_dir)
                max_hpoly_length = dataset.max_hpoly_length
                max_vpoly_length = dataset.max_vpoly_length
                prev_recorded_len = False
            else:
                dataset = MultiMapMinSnapDataset(self.dataset_dir, max_hpoly_length, max_vpoly_length)
                prev_recorded_len = True
        else:
            raise Exception("Dataset not found")

        # Prepare dataloaders
        print("Prepare dataloaders")
        processed_dataset = None

        if prev_recorded_len:
            valid_indices = dataset.filter_indices()
            processed_dataset = torch.utils.data.Subset(dataset, valid_indices)
        else:
            processed_dataset = dataset


 
        if training_data_ratio < 1.0:
            training_dataset_size = int(training_data_ratio * len(dataset))
            validation_dataset_size = len(dataset) - training_dataset_size
            training_dataset, validation_dataset = random_split(dataset=dataset,
                                                                lengths=[training_dataset_size, validation_dataset_size],
                                                                generator=torch.Generator().manual_seed(random_seed))
        else:
            training_dataset = dataset
            validation_dataset = dataset
        
        
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
        w2 = config["obj_weights"]["w2"]
        # w3 = config["obj_weights"]["w3"]
        wt = config["obj_weights"]["wt"]
        #w4 = config["obj_weights"]["w4"]

        if self.poly_mode == "hpoly":
            self.model = MinimalSnapNetwork(seg=seg,
                                            max_poly_length=max_hpoly_length, lr=learning_rate, T_0=s_T_0,
                                            T_mult=s_T_mult, eta_min=s_eta_min, last_epoch=s_last_epoch,
                                            hidden_size=hidden_size, use_scheduler=self.use_scheduler,
                                            loss_dim_reduction=loss_dim_reduction, w1=w1, w2=w2, wt=wt, poly_mode=self.poly_mode)
        elif self.poly_mode == "vpoly":
            self.model = MinimalSnapNetwork(seg=seg,
                                            max_poly_length=max_vpoly_length, lr=learning_rate, T_0=s_T_0,
                                            T_mult=s_T_mult, eta_min=s_eta_min, last_epoch=s_last_epoch,
                                            hidden_size=hidden_size, use_scheduler=self.use_scheduler,
                                            loss_dim_reduction=loss_dim_reduction, w1=w1, w2=w2, wt=wt, poly_mode=self.poly_mode)

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
        total_obj2_val = 0.0
        #total_obj3_val = 0.0
        total_objt_val = 0.0

        # print("------ MEMORY USAGE: train_one_epoch begins: {} MB ------".format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024))
        for i, data in enumerate(self.training_dataloader):
            #tracemalloc.start()

            # print("------ MEMORY USAGE: train_one_epoch dataloader enumeration begins: {} MB ------".format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024))
            x, hpoly_elems, hpoly_end_probs, vpoly_elems, vpoly_end_probs = data
            x = x.to(self.device)

            hpoly_elems = hpoly_elems.to(self.device)
            hpoly_end_probs = hpoly_end_probs.to(self.device)
            vpoly_elems = vpoly_elems.to(self.device)
            vpoly_end_probs = vpoly_end_probs.to(self.device)


            curr_obj_val, curr_obj1_val, curr_obj2_val, curr_objt_val = self.model.train_model(x, hpoly_elems, hpoly_end_probs, vpoly_elems, vpoly_end_probs)

            # print("------ MEMORY USAGE: train_one_epoch prepare data: {} MB ------".format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024))

            

            self.writer.add_scalar("obj_val training / iteration", curr_obj_val, self.curr_step_idx)
            self.writer.add_scalar("obj1_val training / iteration", curr_obj1_val, self.curr_step_idx)
            self.writer.add_scalar("obj2_val training / iteration", curr_obj2_val, self.curr_step_idx)
            #self.writer.add_scalar("obj3_val training / iteration", curr_obj3_val, self.curr_step_idx)
            self.writer.add_scalar("objt_val training / iteration", curr_objt_val, self.curr_step_idx)
            #self.writer.add_scalar("obj4_val training / iteration", curr_obj4_val, self.curr_step_idx)

            # print("------ MEMORY USAGE: train_one_epoch after train_model: {} MB ------".format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024))

            total_obj_val += curr_obj_val
            total_obj1_val += curr_obj1_val
            total_obj2_val += curr_obj2_val
            #total_obj3_val += curr_obj3_val
            total_objt_val += curr_objt_val

            if self.curr_step_idx % self.save_freq == 0:
                self.save_checkpoint()
                # print("------ MEMORY USAGE: train_one_epoch after save_checkpoint: {} MB ------".format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024))

            del x
            del hpoly_elems
            del hpoly_end_probs
            del curr_obj_val
            del curr_obj1_val
            del curr_obj2_val
            #del curr_obj3_val
            del curr_objt_val

            gc.collect()
            torch.cuda.empty_cache()
            
            self.curr_step_idx = self.curr_step_idx + 1

            # Memory Leak Debugging
            #objgraph.show_growth()
            #objgraph.show_backrefs(objgraph.get_leaking_objects(),  max_depth=5,  filename='get_leaking_objects.png')
            #snapshot = tracemalloc.take_snapshot()
            #top_stats = snapshot.statistics('lineno')
            #print("[ Top 10 ]")
            #for stat in top_stats[:10]:
            #    print(stat)

        avg_obj_val = total_obj_val / len(self.training_dataloader)
        avg_obj1_val = total_obj1_val / len(self.training_dataloader)
        avg_obj2_val = total_obj2_val / len(self.training_dataloader)
        #avg_obj3_val = total_obj3_val / len(self.training_dataloader)
        avg_objt_val = total_objt_val / len(self.training_dataloader)

        self.curr_epoch_idx += 1

        del total_obj_val
        del total_obj1_val
        del total_obj2_val
        #del total_obj3_val
        del total_objt_val

        gc.collect()
        torch.cuda.empty_cache()

        return avg_obj_val, avg_obj1_val, avg_obj2_val, avg_objt_val

    def train(self):
        print("===== Start Training =====")
        while self.curr_epoch_idx < self.max_epochs:
            print("Epoch: {}".format(self.curr_epoch_idx))
            avg_obj_val, avg_obj1_val, avg_obj2_val, avg_objt_val = self.train_one_epoch()

            obj_vali_vals = []
            obj1_vali_vals = []
            obj2_vali_vals = []
            #obj3_vali_vals = []
            objt_vali_vals = []

            for _ in range(self.validation_batch_size):
                try:
                    vali_data = next(self.validation_dataloader_iterator)
                except StopIteration:
                    self.validation_dataloader_iterator = iter(self.validation_dataloader)
                    vali_data = next(self.validation_dataloader_iterator)

                vali_x, vali_hpoly_elems, vali_hpoly_end_probs = vali_data

                vali_x = vali_x.to(self.device)
                vali_hpoly_elems = vali_hpoly_elems.to(self.device)
                vali_hpoly_end_probs = vali_hpoly_end_probs.to(self.device)

                curr_obj_vali_val, curr_obj1_vali_val, curr_obj2_vali_val, curr_objt_vali_val  = self.model.eval_model(vali_x, vali_hpoly_elems, vali_hpoly_end_probs)

                obj_vali_vals.append(curr_obj_vali_val)
                obj1_vali_vals.append(curr_obj1_vali_val)
                obj2_vali_vals.append(curr_obj2_vali_val)
                #obj3_vali_vals.append(curr_obj3_vali_val)
                objt_vali_vals.append(curr_objt_vali_val)

            obj_vali_val = sum(obj_vali_vals) / len(obj_vali_vals)
            obj1_vali_val = sum(obj1_vali_vals) / len(obj1_vali_vals)
            obj2_vali_val = sum(obj2_vali_vals) / len(obj2_vali_vals)
            #obj3_vali_val = sum(obj3_vali_vals) / len(obj3_vali_vals)
            objt_vali_val = sum(objt_vali_vals) / len(objt_vali_vals)

            print("obj_vali_val: {}".format(obj_vali_val))
            print("obj1_vali_val: {}".format(obj1_vali_val))
            print("obj2_vali_val: {}".format(obj2_vali_val))
            #print("obj3_vali_val: {}".format(obj3_vali_val))
            print("objt_vali_val: {}".format(objt_vali_val))

            self.writer.add_scalar("obj_val validation / epoch", obj_vali_val, self.curr_epoch_idx)
            self.writer.add_scalar("obj1_val validation / epoch", obj1_vali_val, self.curr_epoch_idx)
            self.writer.add_scalar("obj2_val validation / epoch", obj2_vali_val, self.curr_epoch_idx)
            #self.writer.add_scalar("obj3_val validation / epoch", obj3_vali_val, self.curr_epoch_idx)
            self.writer.add_scalar("objt_val validation / epoch", objt_vali_val, self.curr_epoch_idx)

            self.writer.add_scalar("obj_val training / epoch", avg_obj_val, self.curr_epoch_idx)
            self.writer.add_scalar("obj1_val training / epoch", avg_obj1_val, self.curr_epoch_idx)
            self.writer.add_scalar("obj2_val training / epoch", avg_obj2_val, self.curr_epoch_idx)
            #self.writer.add_scalar("obj3_val training / epoch", avg_obj3_val, self.curr_epoch_idx)
            self.writer.add_scalar("objt_val training / epoch", avg_objt_val, self.curr_epoch_idx)

            # self.save_checkpoint()

        self.writer.close()
        print("===== Finish Training =====")


if __name__ == '__main__':
    config_dir = "configs/minsnap_params.yaml"
    training_manager = MinSnapNetworkTrainingManager(config_dir)
    training_manager.train()

