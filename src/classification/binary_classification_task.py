
# coding: utf-8

# In[11]:


import torch
import torchvision.models as models
import h5py 
from logger import Logger
from torchvision.transforms import transforms 
import torch.utils.data as data
import numpy as np 
import pdb
import matplotlib.pyplot as plt
import torch.nn as nn 
import torch.optim as optim 
from torch.autograd import Variable
import shutil
import os 
import random
import torch.nn.functional as F

from sklearn.metrics import roc_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score


# In[12]:


## If there is a near-collision in next two seconds or not 

class FrameDataset(data.Dataset):
    
    def __init__(self, f, transform=None, test=False):
        
        self.f = f 
        self.transform = transform
        self.test = test 
        
    def __getitem__(self, index):
        
        rgb = np.array(self.f["rgb"][index])
        label = np.array(self.f["labels"][index], dtype=np.uint8)
        
        t_label = torch.zeros(2)
        
        if (label[0] or label[1]):
            t_label[0] = 1 ## Near-collision within next 2 seconds
        else:
            t_label[1] = 1 ## No Near-collision within next 2 seconds 
            
        t_rgb = torch.zeros(rgb.shape[0], 3, 224, 224)
        
        prob = random.uniform(0, 1)
        
        if self.transform is not None:
            
            for i in range(rgb.shape[0]):
                if (prob > 0.5 and not self.test):
                    flip_transform = transforms.Compose([transforms.ToPILImage(), transforms.RandomHorizontalFlip(1.0)])
                    rgb[i,:,:,:] = flip_transform(rgb[i,:,:,:])
                t_rgb[i,:,:,:] = self.transform(rgb[i,:,:,:])
                
        return t_rgb, t_label
    
    def __len__(self):
        return len(self.f["rgb"])


# In[20]:


hfp_train = h5py.File('/mnt/hdd1/aashi/cmu_data/threeSecsTrain.h5', 'r')
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
batch_size = 36

labels_train = hfp_train["labels"]

weights = []
for i in range(len(labels_train)):
    if (labels_train[i][0] or labels_train[i][1]):
        weights.append(0.6)
    else:
        weights.append(0.4)
        
weights = torch.DoubleTensor(weights)
sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

train_loader = data.DataLoader(FrameDataset(f = hfp_train, transform = transforms.Compose([transforms.ToTensor(), normalize]),test = False),
                              batch_size=batch_size, sampler = sampler)

hfp_test = h5py.File('/mnt/hdd1/aashi/cmu_data/threeSecsTest.h5', 'r')
test_loader = data.DataLoader(FrameDataset(f = hfp_test, transform = transforms.Compose([transforms.ToTensor(), normalize]), test = True), 
                               batch_size=1)


# In[21]:


def load_vgg_voc_weights(MODEL_PATH):
    checkpoint_dict = torch.load(MODEL_PATH)
    vgg_model.load_state_dict(checkpoint_dict)

vgg_model = models.vgg16(pretrained=True)
num_final_in = vgg_model.classifier[-1].in_features
NUM_CLASSES = 20 ## in VOC
vgg_model.classifier[-1] = nn.Linear(num_final_in, NUM_CLASSES)
model_path = '/home/aashi/the_conclusion/model_files/' + 'vgg_on_voc' + str(800)
load_vgg_voc_weights(model_path)

class VGGNet(nn.Module):
    
    def __init__(self):
        super(VGGNet, self).__init__()
        self.rgb_net = self.get_vgg_features()
        
        kernel_size = 3 
        padding = int((kernel_size - 1)/2)
        self.conv_layer = nn.Conv2d(512, 16, kernel_size, 1, padding, bias=True)
        self.conv_bn = nn.BatchNorm2d(16)
        self.feature_size = 16*7*7*4
        self.final_layer = nn.Sequential(
        nn.Dropout(),
        nn.Linear(self.feature_size, 256),
        nn.Linear(256, 2),  ## 4 classes instead of 2 
        nn.Softmax()  ## If loss function uses Softmax  
        )
        
    def forward(self, rgb): ## sequence of four images - last index is latest 
        four_imgs = []
        for i in range(rgb.shape[1]):
            img_features = self.rgb_net(rgb[:,i,:,:,:])
            channels_reduced = self.conv_bn(self.conv_layer(img_features))
            img_features = channels_reduced.view((-1, 16*7*7))
            four_imgs.append(img_features)
        concat_output = torch.cat(four_imgs, dim = 1)
        out = self.final_layer(concat_output)
        return out
        
    def get_vgg_features(self):

        modules = list(vgg_model.children())[:-1]
        vgg16 = nn.Sequential(*modules)
        
        return vgg16.type(torch.Tensor)


# In[22]:


def load_model_weights(epoch_num):
    model_file = '/mnt/hdd1/aashi/binary_classification_v2_' + str(epoch_num).zfill(3)
    checkpoint_dict = torch.load(model_file)
    model.load_state_dict(checkpoint_dict)


# In[23]:


model = VGGNet().cuda()
optimizer = optim.SGD(model.parameters(), 0.001)
criterion = nn.BCELoss()

if os.path.exists('binary_classification_curve'):
    shutil.rmtree('binary_classification_curve')
logger = Logger('binary_classification_curve', name='performance_curves')

def save_model_weights(epoch_num):
    model_file = '/mnt/hdd1/aashi/binary_classification_v2_' + str(epoch_num).zfill(3)
    torch.save(model.state_dict(), model_file)


# In[7]:


iterations = 0 
epochs = 10 

for e in range(epochs):
    for iter, (rgb, label) in enumerate(train_loader, 0):
        
        rgb = Variable(rgb.float().cuda())
        label = Variable(label.float().cuda())
        
        optimizer.zero_grad()
        # (1) Forward pass 
        outputs = model(rgb)
        # (2) Compute diff 
        loss = criterion(outputs, label)
        # (3) Compute gradients 
        loss.backward()
        # (4) update weights 
        optimizer.step()
        
        iterations += 1
        logger.scalar_summary('training_loss', loss.data.cpu().numpy(), iterations)
        
    if (e % 1 == 0):
        print(e)
        save_model_weights(e)
        model.eval()
        total_loss = 0.0
        for iter, (rgb, label) in enumerate(test_loader, 0):
            rgb = rgb.float().cuda()
            outputs = model(rgb)
            loss = criterion(outputs, label.float().cuda())
            total_loss += loss.data.cpu().numpy()
        logger.scalar_summary('test_loss', total_loss, e)
        model.train()


# In[16]:


############ After Training ##########

# hfp_test = h5py.File('/mnt/hdd1/aashi/cmu_data/threeSecsTest.h5', 'r')
# normalize = transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229, 0.224, 0.225])
# batch_size = 1
# transform = transforms.Compose([transforms.ToTensor(), normalize])
# test_loader = data.DataLoader(FrameDataset(f = hfp_test, transform = transforms.Compose([transforms.ToTensor(), normalize]), test = True), 
#                                batch_size=1)
# model = VGGNet().cuda()
# model.eval()

# e = 2 
# load_model_weights(2)


# # In[17]:


# confusion_matrix = np.zeros((2,2))

# for iter, (t_rgb, label) in enumerate(test_loader, 0):
#     t_rgb = t_rgb.float().cuda()
#     outputs = model(t_rgb)

#     outputs = outputs.detach().cpu().numpy()
    
#     true = np.argmax(label)
#     pred = np.argmax(outputs)
    
#     confusion_matrix[pred][true] += 1


# # In[18]:


# tp = confusion_matrix[0][0]
# fp = confusion_matrix[0][1]
# fn = confusion_matrix[1][0]
# precision = tp/(tp + fp)
# recall = tp/(tp + fn)
# f1Score = 2*precision*recall/(precision + recall)


# # In[24]:


# print(f1Score)

