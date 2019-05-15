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


class VGGNet(nn.Module):

	def __init__(self):

		super(VGGNet, self).__init__()

		self.vgg_model = models.vgg16(pretrained=True)
		# num_final_in = self.vgg_model.classifier[-1].in_features 
		# NUM_CLASSES = 20 
		# self.vgg_model.classifier[-1] = nn.Linear(num_final_in, NUM_CLASSES) 

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

		modules = list(self.vgg_model.children())[:-1]
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

	err = []

	for iter, (img, rgb, label) in enumerate(test_loader, 0):

		rgb = rgb.float().cuda()
		label = label.float().cuda()
		label = label.unsqueeze(-1)
		outputs = model(rgb)

		gt = label[0].data.cpu().numpy() + mean 
		pred = label[0].data.cpu().numpy() + mean 

		err.append(abs(pred[0] - gt[0]))

	print(np.mean(err))
	pritn(np.std(err))


if __name__ == '__main__':
	test()