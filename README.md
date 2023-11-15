# AllocNet: Learning Time Allocation for Trajectory Optimization

## About

AllocNet is a lightweight learning-based trajectory optimization framework. 

__Authors__: [Yuwei Wu](https://github.com/yuwei-wu), [Xiatao Sun](https://github.com/M4D-SC1ENTIST), Igor Spasojevic, and Vijay Kumar from the [Kumar Lab](https://www.kumarrobotics.org/).

__Video Links__:  [Youtube](https://www.youtube.com/watch?v=tYFui5t9zJk)


__Related Paper__: Wu, Y., Sun, X., Spasojevic, I. and Kumar, V., 2023. Learning Optimal Trajectories for Quadrotors. arXiv preprint arXiv:2309.15191.
[arxiv Preprint](https://arxiv.org/pdf/2309.15191.pdf)

If this repo helps your research, please cite our paper at:

```bibtex
@misc{wu2023learning,
      title={Learning Optimal Trajectories for Quadrotors}, 
      author={Yuwei Wu and Xiatao Sun and Igor Spasojevic and Vijay Kumar},
      year={2023},
      eprint={2309.15191},
      archivePrefix={arXiv},
      primaryClass={cs.RO}
}
```

## Acknowledgements


- Dataset: The raw point cloud dataset from [M3ED](https://m3ed.io/)
- Front-end Path Planning: We use [OMPL](https://ompl.kavrakilab.org/) planning library
- Planning Modules and Visualization: We use the module in [GCOPTER](https://github.com/ZJU-FAST-Lab/GCOPTER)


## Run our pre-trained Model in Simulation

The repo has been tested on 20.04 with ros-desktop-full installation.

### 1. Prerequisites

#### 1.1 ROS and OMPL

Follow the guidance to install [ROS](https://wiki.ros.org/ROS/Installation) and install OMPL:
```
sudo apt install libompl-dev
```

#### 1.2 libtorch

Download the libtorch and put it into the planner folder: [GPU version](https://download.pytorch.org/libtorch/test/cu117/libtorch-cxx11-abi-shared-with-deps-latest.zip), or [CPU version](https://download.pytorch.org/libtorch/test/cpu/libtorch-cxx11-abi-shared-with-deps-latest.zip)

#### 1.3 QP solver 

We use osqp to solve quadratic programming, install by:

```
git clone -b release-0.6.3 https://github.com/osqp/osqp.git
cd osqp
git submodule init
git submodule update
mkdir build & cd build
cmake ..
sudo make install

cd ../..
git clone https://github.com/robotology/osqp-eigen.git
cd osqp-eigen
mkdir build & cd build
cmake ..
sudo make install
```

### 2. Build on ROS 

```
git clone git@github.com:yuwei-wu/AllocNet.git && cd AllocNet/src
wstool init && wstool merge utils.rosinstall && wstool update
catkin build
```
The default mode is set to the GPU version. To switch to the CPU, navigate to line 29 in the 'learning_planning.cpp' file and replace 'device(torch::kGPU)' with 'device(torch::kCPU)'. After making this change, recompile the code for the updates to take effect.

### 3. Run

```
source devel/setup.bash
roslaunch planner learning_planning.launch
```

click 2D Nav Goal to trigger planning:

<p align="center">
  <img src="docs/sim_vis.gif"/>
</p>


## Train new models

### 0. Folder Structure

```plaintext
network/
│
├── config/                - Configuration files for training and testing.
│
│
├── utils/                 - Utility functions and classes.
│   └── learning/          - Contains network classes and layers
│
└── sample_<...>.py        - Scripts for sampling data
└── train_minsnap_<...>.py - Scripts for training
└── test_minsnap_<...>.py  - Scripts for testing
└── ts_conversion_<...>.py - Scripts for converting to TorchScript
```

### 1. Pre-requisites
- Ubuntu 20.04 / Windows 11
  - If using WSL2 with simulation running in windows, please add `export WSL_HOST_IP=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}')` to your `.bashrc` file to allow communication between Windows and the subsystem.
- Python 3.8
- CUDA 11.7

### 2. Setup

#### 2.1 Install Dependencies

- Install Ubuntu packages
  - `sudo apt-get install python3-dev python3-venv`
- Create a virtual environment
  - `python3 -m venv venv`
- Activate the virtual environment
  - `source venv/bin/activate`
- Install the requirements
  - `pip install wheel`
  - `pip install numpy==1.24.2`
  - `pip install -r requirements.txt`
  

#### 2.2 Setup Iris
Follow the instructions to install, and you may need to change the *CMakeLists.txt* in *iris-distro/CMakeLists.txt*
```
iris: https://github.com/rdeits/iris-distro
```
For AMD CPU, if you encounter a core dump, please refer to instructions in this link:
```
https://github.com/rdeits/iris-distro/issues/81
```
```
pip install -U kaleido
```

### 3. Run
- For training, please run ```python train_minsnap_<model_configuration>.py```
- For testing, please run ```python test_minsnap_<model_configuration>.py```
- For converting the learned model to TorchScript, please run ```python ts_conversion_<model_configuration>.py```


## Maintaince

For any technical issues, please contact Yuwei Wu (yuweiwu@seas.upenn.edu, yuweiwu20001@outlook.com).
