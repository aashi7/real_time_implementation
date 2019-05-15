import torch 
from logger import Logger 
from torchvision.transforms import transforms
import numpy as np 
import torch.nn as nn 
import torch.optim as optim 
import torch.utils.data as data 
import h5py
import torchvision.models as models 
import pdb 


class FrameDataset(data.Dataset):

	def __init__(self, f, transform=None, test=False):

		self.f = f 
		self.transform = transform 
		self.test = test 

	def __getitem__(self, index):

		rgb = np.array(self.f["rgb"][index])
		label = np.array((self.f["labels"][index] - self.f["Mean"]))

		t_rgb = torch.zeros(rgb.shape[0], 3, 224, 224)

		if self.transform is not None:
			for i in range(rgb.shape[0]):
				t_rgb[i,:,:,:] = self.transform(rgb[i,:,:,:])

		return rgb, t_rgb, label 

	def __len__(self):

		return len(self.f["rgb"])


def load_vgg_voc_weights(MODEL_PATH):
    checkpoint_dict = torch.load(MODEL_PATH)
    vgg_model.load_state_dict(checkpoint_dict)

vgg_model = models.vgg16(pretrained=True)
num_final_in = vgg_model.classifier[-1].in_features
NUM_CLASSES = 20 ## in VOC
vgg_model.classifier[-1] = nn.Linear(num_final_in, NUM_CLASSES)
#model_path = '/home/aashi/the_conclusion/model_files/' + 'vgg_on_voc' + str(800)
model_path = '../../../data/vgg_on_voc800'
load_vgg_voc_weights(model_path)

class VGGNet(nn.Module):

	def __init__(self):

		super(VGGNet, self).__init__()

		# self.vgg_model = models.vgg16(pretrained=True)
		# num_final_in = self.vgg_model.classifier[-1].in_features 
		# NUM_CLASSES = 20 
		# self.vgg_model.classifier[-1] = nn.Linear(num_final_in, NUM_CLASSES) 

		# model_path = '../../../data/vgg_on_voc800'

		# self.vgg_model.load_state_dict(torch.load(model_path))

		self.rgb_net = self.get_vgg_features()

		kernel_size = 3 
		padding = int((kernel_size - 1)/2)

		self.conv_layer = nn.Conv2d(512, 16, kernel_size, 1, padding, bias=True)
		self.feature_size = 16*7*7*6 
		self.final_layer = nn.Sequential(

			nn.ReLU(),
			nn.Linear(self.feature_size, 2048),
			nn.ReLU(),
			nn.Dropout(),
			nn.Linear(2048,1)

			) 


	def forward(self, rgb):

		four_imgs = []
		for i in range(rgb.shape[1]):
			img_features = self.rgb_net(rgb[:,i,:,:,:])
			channels_reduced = self.conv_layer(img_features)
			img_features = channels_reduced.view((-1, 16*7*7))
			four_imgs.append(img_features)

		concat_output = torch.cat(four_imgs, dim=1)
		out = self.final_layer(concat_output)

		return out	


	def get_vgg_features(self):

		modules = list(vgg_model.children())[:-1]
		vgg16 = nn.Sequential(*modules)
		return vgg16.type(torch.Tensor)


def test():

	model = VGGNet().cuda()

	epoch_num = 27 
	MODEL_PATH = '6Image6s_' + str(epoch_num).zfill(3)

	checkpoint_dict = torch.load(MODEL_PATH)
	model.load_state_dict(checkpoint_dict)

	hfp_test = h5py.File('../../../data/6ImageTest.h5')

	mean = hfp_test["Mean"][()]
	var = hfp_test["Variance"][()]
	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
	test_loader = data.DataLoader(FrameDataset(f = hfp_test, 
		transform = transforms.Compose([transforms.ToTensor(), normalize]),
		test=True), batch_size = 1)

	model.eval()

	print("Loaded the prediction network")

	err = []

	for iter, (img, rgb, label) in enumerate(test_loader, 0):

		rgb = rgb.float().cuda()
		label = label.float().cuda()
		label = label.unsqueeze(-1)
		outputs = model(rgb)

		gt = label[0].data.cpu().numpy() + mean 
		pred = outputs[0].data.cpu().numpy() + mean 

		err.append(abs(pred[0] - gt[0]))

	print(np.mean(err))
	print(np.std(err))


if __name__ == '__main__':
	test()