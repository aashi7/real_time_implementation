### Dependencies 

1. Ubuntu 16.04 
2. ROS Kinetic 
3. CUDA-9.0
4. ZED SDK for Ubuntu 16 v2.8 
5. zed-ros-wrapper (https://github.com/stereolabs/zed-ros-wrapper) 

### Pedestrian Detection 
Faster RCNN (https://github.com/endernewton/tf-faster-rcnn)

Issues: 
1) ImportError: No module named Cython.Distutils
Ran the following command 
`sudo apt-get install cython`

### Server details 
On MAC 
`vim ~/.zshrc`    
On Ubuntu      
`vim ~/.bashrc`    
`alias cube='ssh -Y aashi@128.2.177.239'`    
`alias bheem='ssh -Y aashi@128.2.194.131'`   

### Virtual Environment 
1) `bash ./Anaconda3-2019.03-Linux-x86_64.sh`
Ananconda installation location - "/home/klab/anaconda3" (Deleted the previous one) 
2) `conda create -n venv python=3.5 anaconda`
3) `conda activate venv`
4) `pip install easydict`
5) `pip install opencv-python`
6) `cd catkin_ws/src/tf-faster-rcnn`
7) `CUDA_VISIBLE_DEVICES=0 ./tools/demo.py`


