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

### Launching ZED Node in ROS 
1) `cd catkin_ws/`
2) `roslaunch zed_display_rviz display_zed.launch`
3) To run Faster RCNN, I have to do - `unset PYTHONPATH`

### Write a script to implement command line 
1) Useful blog - Using ROS with Conda (http://yuqli.com/?p=2817)
Note - `source /opt/ros/kinetic/lib/python2.7/dist-packages/` sets the `PYHONPATH` variable to `/opt/ros/kinetic/...` location. The solution is to unset the `PYTHONPATH` in `~/.bashrc`, right after sourcing ROS, but before sourcing Anaconda env `conda activate venv`. Use `unset PYTHONPATH`      
NOT NEEDED!

### Ways to solve
1) One way is that I eliminate ROS as dependency - won't be able to use LiDAR then
WILL TRY SHORTER PATHS FIRST   
2)
`source activate venv`   
`pip install -U rospkg`  
SEEMS TO WORK!!

### TO DO: Pass the images coming from ROS Node into the Faster RCNN 
1) Create a python file where I subscribe to a ROS Node and pass it in the Faster RCNN using demo.py. Have this file at the same place as demo.py. 

### May 14, 2019 
`source activate venv`  
`source /home/klab/catkin_ws/devel/setup.bash`  
`roslaunch zed_display_rviz display_zed.launch`      
`rosrun cam_to_tf run_on_cam.py`

### Issues
1)    
cv_bridge is available for python2 and cannot be used in py3. You can try running in python2.
 => Have a python2 virtual environment    
`conda create -n venv python=2.7 anaconda`     
`pip install tensorflow-gpu`    
`conda install cudatoolkit`    
`conda install cudnn`    
`pip install --upgrade tensorflow-gpu==1.8.0`  
Downgrading to 1.8.0 solved the problem!

2) 
`fatal: Pathspec '.' is in submodule 'src/tf-faster-rcnn'`
`git rm --cached tf-faster-rcnn`  

3) 
Cannot push data folder; max limit reached on github  
`git add .`     
`git rm -r --cached src/tf-faster-rcnn/data/voc_2007_trainval+voc_2012_trainval`

### TO DO: Run my prediction network on input frames 
1)   
a) Locate my model and test code   
b) Bring it on predator, should be able to run it in the same python2.7 virtual environment python2env   
c) Then, pass input frames through my network   
d) How will I store a tuple of frames in callback?    

## a) Locate my model
