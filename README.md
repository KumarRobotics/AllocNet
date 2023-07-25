# AllocNet: Learning Time Allocation for Trajectory Optimization

A lightweight learning-based trajectory optimization framework. The code will be released after the acceptance of this paper.


## 1. Prerequisites

### 1.1 libtorch

download the libtorch and put it into planner folder

[GPU VERSION](https://download.pytorch.org/libtorch/test/cu117/libtorch-cxx11-abi-shared-with-deps-latest.zip)

[CPU VERSION](https://download.pytorch.org/libtorch/test/cpu/libtorch-cxx11-abi-shared-with-deps-latest.zip)

### 1.2 qp solver 

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
### 1.3 raw dataset

[M3ED](https://m3ed.io/)
