import torch
import torch.nn as nn
import cvxpy as cp
from utils.learning.layers import OsqpLayer
from utils.min_traj_opt import MinTrajOpt
from functools import reduce
import operator
import numpy as np
import torch
#from cvxpylayers.torch import CvxpyLayer
import torch.optim as optim
import resource
import gc
from memory_profiler import profile
import objgraph
import tracemalloc
import copy
import sys
import time
from torch.autograd import Function, Variable




class ConvMLPMinimalSnapNetwork4AblationStudy(nn.Module):
    def __init__(self, seg, max_poly_length, lr=1e-3,
                 T_0=500, T_mult=1, eta_min=1e-5, last_epoch=-1, hidden_size=128, use_scheduler=True,
                 loss_dim_reduction='mean', w1=1, wt= 5.0, wc=1.0, wp=1.0):
        super().__init__()
        #init

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.seg = seg

        self.output_module = nn.Sequential(
            nn.Linear(38, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(p=0.2),

            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(p=0.2),

            nn.Linear(hidden_size, self.seg),
            nn.Softplus(beta=2)
        )

        for layer in self.output_module:
            if type(layer) == nn.Linear:
                nn.init.kaiming_normal_(layer.weight)

        self.output_module.to(self.device)

        self.max_poly_length = max_poly_length


 
        self.hpoly_input_module = nn.Sequential(
            # yuwei: revise the size here
            nn.Conv2d(50, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(16, 32)
        )
        self.state_input_module = nn.Sequential(
            nn.Conv1d(9, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(8, 6)
        )

        self.hpoly_input_module.to(self.device)
        self.state_input_module.to(self.device)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        if use_scheduler:
            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer,
                                                                            T_0=T_0,
                                                                            T_mult=T_mult,
                                                                            eta_min=eta_min,
                                                                            last_epoch=last_epoch)

        self.loss_dim_reduction = loss_dim_reduction

        self.w1 = w1

        self.wt = wt

        self.wc = wc

        self.wp = wp

        self.to(self.device)

        self.phase = 2

    def forward(self, stacked_state, stacked_hpolys):
        with torch.no_grad():
            stacked_state = stacked_state.float()
            stacked_hpolys = stacked_hpolys.float()

            state_embeddings = self.state_input_module(stacked_state)
            hpoly_embeddings = self.hpoly_input_module(stacked_hpolys)

            state_embeddings = state_embeddings.squeeze(-1)
            hpoly_embeddings = hpoly_embeddings.squeeze(-1)

            combined = torch.cat((state_embeddings, hpoly_embeddings), dim=1).type(torch.float32)

            tfs = self.output_module(combined)

            return tfs


