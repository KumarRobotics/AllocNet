# AllocNet: Learning Time Allocation for Trajectory Optimization

A lightweight learning-based trajectory optimization framework. 

## About



## Acknowledges


- dataset: the raw point cloud dataset from [M3ED](https://m3ed.io/)
- front-end and visualization: we use the module in [GCOPTER](https://github.com/ZJU-FAST-Lab/GCOPTER)



## Run the Simulation

### 1. Prerequisites

#### 1.1 libtorch

download the libtorch and put it into the planner folder

[GPU VERSION](https://download.pytorch.org/libtorch/test/cu117/libtorch-cxx11-abi-shared-with-deps-latest.zip), [CPU VERSION](https://download.pytorch.org/libtorch/test/cpu/libtorch-cxx11-abi-shared-with-deps-latest.zip)

#### 1.2 qp solver 

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

### 3. Run

```
source devel/setup.bash
roslaunch planner learning_planning.launch
```













