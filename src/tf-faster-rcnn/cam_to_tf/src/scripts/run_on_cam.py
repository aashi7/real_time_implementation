#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import cv2

import rospy 
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

### Imports from demo.py ### 
sys.path.insert(0, '../../../tools')

import _init_paths
from model.config import cfg
from model.test import im_detect
from model.nms_wrapper import nms

from utils.timer import Timer
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os, cv2
import argparse

from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1


##### Imports for prediction network #####
import torch
import torchvision.models as models
import h5py 
from logger import Logger
from torchvision.transforms import transforms 
import torch.utils.data as data
import numpy as np 
import pdb
import torch.nn as nn 
import torch.optim as optim 
from torch.autograd import Variable
import shutil
import os 
import random
import torch.nn.functional as F
##########################################

## latency 
import timeit 

## For multiple frames 
from collections import deque 

## For publishing time 
from std_msgs.msg import String 
##########################################


CLASSES = ('__background__',
		   'aeroplane', 'bicycle', 'bird', 'boat',
		   'bottle', 'bus', 'car', 'cat', 'chair',
		   'cow', 'diningtable', 'dog', 'horse',
		   'motorbike', 'person', 'pottedplant',
		   'sheep', 'sofa', 'train', 'tvmonitor')

NETS = {'vgg16': ('vgg16_faster_rcnn_iter_70000.ckpt',),'res101': ('res101_faster_rcnn_iter_110000.ckpt',)}
DATASETS= {'pascal_voc': ('voc_2007_trainval',),'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval',)}

## I want to load the model only once 
class predictNearCollision:

	def __init__(self):

		model = models.vgg16(pretrained=False)
		num_final_in = model.classifier[-1].in_features

		model.classifier[-1] = nn.Linear(num_final_in, 1) ## Regressed output
		num_features = model.classifier[6].in_features
		features = list(model.classifier.children())[:-1] # Remove last layer
		features.extend([nn.Linear(num_features, 2048), nn.ReLU(), nn.Linear(2048, 1)]) # Add our layer with 4 outputs
		model.classifier = nn.Sequential(*features) # Replace the model classifier

		epoch_num = 47 ## Absolute mean error = 0.3022 
		MODEL_PATH = '../../../data/model_files/SingleImage6s_' + str(epoch_num).zfill(3)
		
		model.load_state_dict(torch.load(MODEL_PATH))
		
		self.model = model.cuda()
		self.model.eval()

		## Get the mean and variance
		hfp_test = h5py.File('../../../data/SingleImageTest.h5','r')
		self.mean = hfp_test["Mean"][()]
		self.var = hfp_test["Variance"][()]

	def getNearCollisionTime(self, cv_image):

		## Preprocessing  
		## Resize 
		r_img = cv2.resize(cv_image, (224, 224))
		normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
			std=[0.229, 0.224, 0.225])
		transform = transforms.Compose([transforms.ToTensor(), normalize])
		rgb = transform(r_img)

		## unsqueeze it to become torch.Size([1, 3, 224, 224])
		rgb = rgb.unsqueeze(0)
		rgb = rgb.float().cuda()

		outputs = self.model(rgb)

		pred = outputs[0].data.cpu().numpy()*self.var + self.mean 

		return pred[0]


class image_converter:

	def __init__(self):

		## Object of predictNearCollision class will be created here 
		self.predNet = predictNearCollision()
		# self.nstream = MultiStreamNearCollision().cuda()

		# self.nstream.load_state_dict(torch.load('../../../data/model_files/4Image6s_004'))

		self.bridge = CvBridge()
		self.image_sub = rospy.Subscriber("/zed/zed_node/rgb/image_rect_color", Image, 
			self.callback)
		cfg.TEST.HAS_PRN = True 
		args = self.parse_args()

		# model path 
		demonet = args.demo_net 
		dataset = args.dataset 
		tfmodel = os.path.join('../../../output', demonet, 
			DATASETS[dataset][0], 'default', NETS[demonet][0])

		if not os.path.isfile(tfmodel + '.meta'):
			raise IOError(('{:s} not found.\nDid you download the proper networks from '
				'our server and place them properly?').format(tfmodel + '.meta'))		

		# set config 
		tfconfig = tf.ConfigProto(allow_soft_placement = True)
		tfconfig.gpu_options.allow_growth = True 

		# init session 
		self.sess = tf.Session(config = tfconfig)

		# load network 
		if demonet == 'vgg16':
			self.net = vgg16()
		elif demonet == 'res101':
			self.net = resnetv1(num_layers=101)
		else:
			raise NotImplementedError

		self.net.create_architecture("TEST", 21, 
			tag='default', anchor_scales = [8, 16, 32])

		saver = tf.train.Saver()
		saver.restore(self.sess, tfmodel)

		print('Loaded network {:s}'.format(tfmodel))

		self.counter = 0 ## Intializing a counter, alternately I can initialize a queue 

		self.stack_imgs = deque(maxlen=4)   ## 4 frames 

		## To check the frequency 
		#self.image_pub = rospy.Publisher("image_topic_2", Image)
		self.time_pub = rospy.Publisher('near_collision_time', String, queue_size = 10)


	def parse_args(self):
		"""Parse input arguments."""
		parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
		parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16 res101]',
							choices=NETS.keys(), default='res101')
		parser.add_argument('--dataset', dest='dataset', help='Trained dataset [pascal_voc pascal_voc_0712]',
							choices=DATASETS.keys(), default='pascal_voc_0712')
		args = parser.parse_args()

		return args

	def vis_detections(self, im, class_name, dets, time, thresh=0.5):
		"""Draw detected bounding boxes."""
		inds = np.where(dets[:, -1] >= thresh)[0]
		if len(inds) == 0:
			return


		for i in inds:
			bbox = dets[i, :4]
			score = dets[i, -1]

			cv2.rectangle(im, (int(bbox[0]), int(bbox[1])), 
				(int(bbox[2]), int(bbox[3])),
				(255, 255, 0), 2)

			cv2.putText(im,  str(time) + " s", (int(bbox[0]), int(bbox[1]-2)), 
				cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3) ## thickness = 3, blue color 


		cv2.imshow("Image window", im)
		#cv2.imwrite('predictions/'+str(self.counter)+'.png', im)
		cv2.waitKey(1)


	def callback(self,data):

		#pdb.set_trace() ## Trace to see the GPU allocation 

		try:
			cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
		except CvBridgeError as e:
			print(e)

		## cv_image is my im

		scores, boxes = im_detect(self.sess, 
			self.net, cv_image) 


		# # Visualize detections for each class 
		CONF_THRESH = 0.8 
		NMS_THRESH = 0.3 

		## cv_image is my im to be passed into the network


		# spatial_features = self.nstream.preprocess(cv_image)

		# self.stack_imgs.append(spatial_features)
		#print(len(self.stack_imgs)) ## will keep on discarding the previous frames and keep the latest four 
		
		t = self.predNet.getNearCollisionTime(cv_image)
		# if (len(self.stack_imgs) == 4):
		# 	t = self.nstream(stack_imgs)
		# else:
		# 	t = 1000  
		
		#print(self.counter)

		######### Visualization ############

		## Only concerned with the pedestrian class 

		cls_ind = 15
		cls = CLASSES[cls_ind]
		# cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
		cls_scores = scores[:, cls_ind]

		# dets = np.hstack((cls_boxes, cls_scores[:, 
		# 	np.newaxis])).astype(np.float32)
		# keep = nms(dets, NMS_THRESH)
		# dets = dets[keep,:]
		# self.vis_detections(cv_image, cls, dets, t, thresh=CONF_THRESH)	
		self.counter = self.counter + 1

		## If I publish this on a node, I can see the frequency of this callback 
		# try: 
		# 	self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
		# except CvBridgeError as e:
		# 	print(e)

		self.time_pub.publish(str(t))


def main(args):

	ic = image_converter()
	rospy.init_node('run_on_cam', anonymous=True)
	try:
		rospy.spin()
	except KeyboardInterrupt:
		print("Shutting down")
	cv2.destroyAllWindows()

if __name__ == '__main__':
	main(sys.argv)

