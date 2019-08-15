#!/bin/bash  
#sudo apt-get update  
#sudo apt-get install python3-setuptools python3-dev -y  
#sudo easy_install3 pip  
#pip3 install numpy  
  
#sudo apt-get install build-essential -y  
#sudo apt-get install cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev -y  
  
cd ~  
#wget https://github.com/Itseez/opencv/archive/3.0.0.zip  
#注意这里不是 cd 3.0.0  , 而是 cd opencv-3.0.0  
#unzip 3.0.0.zip && 
cd opencv-3.0.0
#mkdir build 
cd build 
cmake -D CMAKE_BUILD_TYPE=Release \
-D CMAKE_INSTALL_PREFIX=/usr/local  \
PYTHON3_EXECUTABLE=/usr/bin/python3 \
PYTHON_INCLUDE_DIR=/usr/include/python3.5 \
PYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.5m.so \
PYTHON3_NUMPY_INCLUDE_DIRS=/usr/local/lib/python3.5/dist-packages/numpy/core/include ..
  
# make -j [N] :表示在那个一时间内进行编译的任务数 ，如果-j 后面不跟任何数字，则不限制处理器并行编译的任务数,现在发现直接用make -j4 安装会失败  
#所以还是用 make -j1 成功了  
make -j1  
sudo make install  
