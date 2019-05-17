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
`git rm -r --cached src/tf-faster-rcnn/data`

### TO DO: Run my prediction network on input frames 
1)   
a) Locate my model and test code   
b) Bring it on predator, should be able to run it in the same python2.7 virtual environment python2env   
c) Then, pass input frames through my network   
d) How will I store a tuple of frames in callback?    

## a) Locate my model
`tf-faster-rcnn/data/6Image6s_027`  
`tf-faster-rcnn/data/vgg_on_voc800`  

## b) Run the model on h5 in `python2env`   
`python predictionNet.py`

## TO DO: How to store a tuple of frames in callback or later? 


### May 16, 2019 

Few points:

1) SingleImage6s_047 is 545M whereas 6Image6s_027 is 94M : 
One of the explanation for this could be that the vgg model is not saved in 6Image6s_027. Does this mean the VGG network is not fine-tuned on my dataset? I can confirm by comparing the weights with vgg_on_voc800.

2) Imported from my old repo aashi7/short_term_prediction 
Data on /mnt/hdd1/aashi/cmu_data/
#### Training and Test Data Info 
|N| Left/Right images  | Labels       | LIDAR/Stereo | Bounding boxes      |       
|-| ------------------ | -------      | ------------ | ------------------  |
|1|  left_imgs         | labels       |   Stereo     | dets_mask_rcnn      |
|2|  left_imgs_3       | labels_3     |   Stereo     | dets_mask_rcnn_test |
|3|  left_imgs_nov_2   | labels_left_2|   LIDAR      | det_left_2          |
|4|                    | labels_right_2|  LIDAR      | det_right_2         | 


3) What to do when github is ahead by X commits?  
`git status`   
`git pull --rebase`  - Didn't help   
`git checkout`
`git reset --hard`


### May 17, 2019 

## TO DO: Fix GPU allocation for single image 

1) Launch the ZED wrapper without Rviz (Rviz takes 698MiB)
`roslaunch zed_wrapper zed.launch`

2) To kill a process on GPU
`kill -9 pid`