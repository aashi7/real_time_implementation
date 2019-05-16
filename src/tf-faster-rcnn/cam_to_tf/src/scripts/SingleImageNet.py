
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

class FrameDataset(data.Dataset):

	def __init__(self, f, transform=None):

		self.f = f 
		self.transform = transform 
		
	def __getitem__(self, index):

		rgb = np.array(self.f["rgb"][index])
		label = np.array((self.f["labels"][index] - self.f["Mean"])/self.f["Variance"])

		t_rgb = torch.zeros(3, 224, 224)

		if self.transform is not None:
			t_rgb[:,:,:] = self.transform(rgb[:,:,:])

		return rgb, t_rgb, label 

	def __len__(self):
		return len(self.f["rgb"])

## First: I want to try without loading VGG model 

def test():

	model = models.vgg16(pretrained=True)
	num_final_in = model.classifier[-1].in_features

	model.classifier[-1] = nn.Linear(num_final_in, 1) ## Regressed output
	num_features = model.classifier[6].in_features
	features = list(model.classifier.children())[:-1] # Remove last layer
	features.extend([nn.Linear(num_features, 2048), nn.ReLU(), nn.Linear(2048, 1)]) # Add our layer with 4 outputs
	model.classifier = nn.Sequential(*features) # Replace the model classifier

	epoch_num = 47 ## Absolute mean error = 0.3022 
	MODEL_PATH = '../../../data/model_files/SingleImage6s_' + str(epoch_num).zfill(3)
	
	model.load_state_dict(torch.load(MODEL_PATH))
	
	model = model.cuda()
	hfp_test = h5py.File('../../../data/SingleImageTest.h5','r')

	mean = hfp_test["Mean"][()]
	var = hfp_test["Variance"][()]

	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	test_loader = data.DataLoader(FrameDataset(f = hfp_test, transform = 
		transforms.Compose([transforms.ToTensor(), normalize])), batch_size = 1)

	model.eval()

	err = []

	for iter, (img, rgb, label) in enumerate(test_loader, 0):
		rgb = rgb.float().cuda()
		label = label.float().cuda()
		label = label.unsqueeze(-1)
		outputs = model(rgb)

		pred = outputs[0].data.cpu().numpy()*var + mean 
		gt = label[0].data.cpu().numpy()*var + mean

		err.append(abs(pred[0] - gt[0]))

	print(np.mean(err))
	print(np.std(err))

if __name__ == '__main__':
	test()
