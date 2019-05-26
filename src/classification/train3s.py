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

class FrameDataset(data.Dataset):
    
    def __init__(self, f, transform=None, test = False):
        self.f = f 
        self.transform = transform 
        self.test = test
        
    def __getitem__(self, index):
        rgb = np.array(self.f["rgb"][index])
        label = np.array(self.f["labels"][index], dtype=np.uint8)
        
        t_rgb = torch.zeros(rgb.shape[0], 3, 224, 224)
        
        prob = random.uniform(0, 1)
        prob2 = random.uniform(0, 1)

        if self.transform is not None:
            for i in range(rgb.shape[0]):
                if (prob > 0.5 and not self.test):
                    flip_transform = transforms.Compose([transforms.ToPILImage(), transforms.RandomHorizontalFlip(1.0)])
                    rgb[i,:,:,:] = flip_transform(rgb[i,:,:,:])
                if (prob2 > 0.5 and not self.test):
                	color_jitter_transform = transforms.Compose([transforms.ToPILImage() ,transforms.ColorJitter(brightness = 0.5, contrast = 0.5, saturation = 0.5, hue = 0.2)])
                	rgb[i,:,:,:] = color_jitter_transform(rgb[i,:,:,:])
                t_rgb[i,:,:,:] = self.transform(rgb[i,:,:,:])

                
        return t_rgb, label
    
    def __len__(self):
        return len(self.f["rgb"])

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
        ## input_channels, output_channels, kernel_size, stride, padding, bias
        self.feature_size = 16*7*7*4
        self.final_layer = nn.Sequential(
        nn.Linear(self.feature_size, 256),
        nn.Linear(256, 4)  ## 4 classes instead of 2 
        #nn.Sigmoid()
        #nn.Softmax()  ## If loss function uses Softmax  
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
#         return concat_output
        
    def get_vgg_features(self):

        ##vgg16 = load_vgg_voc_weights(vgg16, model_path)
        modules = list(vgg_model.children())[:-1]
        ## I can also freeze 
        ## high level layer, should I take a lower level?
        vgg16 = nn.Sequential(*modules)
        
        ## Uncommented this to let it fine-tune on my model 
        # for p in vgg16.parameters():
        #     p.requires_grad = False 
        
        return vgg16.type(torch.Tensor)

hfp_train = h5py.File('/mnt/hdd1/aashi/cmu_data/threeSecsTrain.h5', 'r')
hfp_test = h5py.File('/mnt/hdd1/aashi/cmu_data/threeSecsTest.h5', 'r')
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
test_loader = data.DataLoader(FrameDataset(f = hfp_test, transform = transforms.Compose([transforms.ToTensor(), normalize]), test = True), 
                               batch_size=1)
batch_size = 36
train_loader = data.DataLoader(FrameDataset(f = hfp_train, transform = transforms.Compose([transforms.ToTensor(), normalize]),test = False),
                              batch_size=batch_size, shuffle=True)
model = VGGNet().cuda()
optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), 0.001)

def save_model_weights(epoch_num):
    #model_file = '/home/aashi/the_conclusion/model_files/vgg_voc_flip_freeze_' + str(epoch_num).zfill(3)
    model_file = '/mnt/hdd1/aashi/3sW_' + str(epoch_num).zfill(3)
    torch.save(model.state_dict(), model_file)


def load_model_weights(epoch_num):
    #model_file = '/home/aashi/the_conclusion/model_files/vgg_voc_flip_freeze_' + str(epoch_num).zfill(3)
    model_file = '/mnt/hdd1/aashi/3sW_' + str(epoch_num).zfill(3)
    checkpoint_dict = torch.load(model_file)
    model.load_state_dict(checkpoint_dict)

if os.path.exists('3sW'):
    shutil.rmtree('3sW')
logger = Logger('3sW', name='performance_curves')

iterations = 0
epochs = 100 
weights = np.array([0.8365691332256922, 0.8362934667034222, 0.8410191785137636, 0.48611822155712203])
weights = weights/sum(weights)
print(weights)
weights = torch.from_numpy(weights)
weights = weights.float().cuda()
criterion = nn.MultiLabelSoftMarginLoss(weight=weights)

for e in range(epochs):
    for iter, (rgb, label) in enumerate(train_loader, 0):

        rgb = Variable(rgb.float().cuda())
        label = Variable(label.float().cuda())
        optimizer.zero_grad()
        label = label.squeeze(-1)
        outputs = model(rgb)
        loss = criterion(outputs, label) ## Only changes the loss function 
        loss.backward()
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
            label = label.float().cuda()
            label = label.squeeze(-1)
            outputs = model(rgb)
            # target = torch.argmax(label, dim=1)
            # loss = F.cross_entropy(outputs, target)
            loss = criterion(outputs, label)
            total_loss += loss.data.cpu().numpy()
        logger.scalar_summary('test_loss', total_loss, e)    
        model.train()