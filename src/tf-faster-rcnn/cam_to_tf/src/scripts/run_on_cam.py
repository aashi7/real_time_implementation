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

CLASSES = ('__background__',
		   'aeroplane', 'bicycle', 'bird', 'boat',
		   'bottle', 'bus', 'car', 'cat', 'chair',
		   'cow', 'diningtable', 'dog', 'horse',
		   'motorbike', 'person', 'pottedplant',
		   'sheep', 'sofa', 'train', 'tvmonitor')

NETS = {'vgg16': ('vgg16_faster_rcnn_iter_70000.ckpt',),'res101': ('res101_faster_rcnn_iter_110000.ckpt',)}
DATASETS= {'pascal_voc': ('voc_2007_trainval',),'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval',)}

class image_converter:

	def __init__(self):

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


	def parse_args(self):
		"""Parse input arguments."""
		parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
		parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16 res101]',
							choices=NETS.keys(), default='res101')
		parser.add_argument('--dataset', dest='dataset', help='Trained dataset [pascal_voc pascal_voc_0712]',
							choices=DATASETS.keys(), default='pascal_voc_0712')
		args = parser.parse_args()

		return args

	def vis_detections(self, im, class_name, dets, thresh=0.5):
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

			cv2.putText(im, str(class_name)+str(score), (int(bbox[0]), int(bbox[1]-2)), 
				cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255))


		cv2.imshow("Image window", im)
		cv2.waitKey(1)


	def callback(self,data):

		try:
			cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
		except CvBridgeError as e:
			print(e)

		## cv_image is my im

		scores, boxes = im_detect(self.sess, 
			self.net, cv_image) 

		# Visualize detections for each class 
		CONF_THRESH = 0.8 
		NMS_THRESH = 0.3 

		# Only concerned with pedestrian class 
		cls_ind = 15
		cls = CLASSES[cls_ind]
		cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
		cls_scores = scores[:, cls_ind]
		dets = np.hstack((cls_boxes, cls_scores[:, 
			np.newaxis])).astype(np.float32)
		keep = nms(dets, NMS_THRESH)
		dets = dets[keep,:]
		self.vis_detections(cv_image, cls, dets, thresh=CONF_THRESH)		


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

