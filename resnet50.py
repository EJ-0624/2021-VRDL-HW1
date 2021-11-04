#!/usr/bin/env python
# coding: utf-8

# #  Bird image preprocessing

# In[1]:


import os
root = "C:/Users/User/OneDrive/桌面/NYCU/hw1"
os.chdir(root)



# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split, StratifiedKFold
import time
import copy
import zipfile
from PIL import Image


# In[3]:


import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.model_zoo import load_url as load_state_dict_from_url
from typing import Type, Any, Callable, Union, List, Optional
from torch.optim import lr_scheduler
import torchvision.models as models
from torchvision import *
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from efficientnet_pytorch import EfficientNet
import timm


# In[4]:


import warnings
warnings.filterwarnings('ignore')


# In[5]:


import GPUtil
GPUtil.showUtilization()


# In[7]:


import argparse
parser = argparse.ArgumentParser()
# settings
# parser.add_argument("--dir_name", type=str)
parser.add_argument('--local_rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument("--model1", type=str, default='efficientnet-b3')
# parser.add_argument("--model1", type=str, default='efficientnet_b3_pruned')
parser.add_argument("--model2", type=str, default='tf_efficientnet_b2_ns')
parser.add_argument("--folds", type=int, default=5)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--n_classes", type=int, default=200)
parser.add_argument("--label_smoothing", type=float, default=0.125)
parser.add_argument("--grad_clip", type=int, default=1)
parser.add_argument("--test", action='store_true', default=False)
# learning
# parser.add_argument("--alpha", type=float, default=0.01)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--epochs1", type=int, default=20)
parser.add_argument("--epochs2", type=int, default=20)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--max_lr1", type=float, default=1e-3)
parser.add_argument("--max_lr2", type=float, default=1e-4)
parser.add_argument("--weight_decay", type=float, default=1e-6)
args = parser.parse_args(args=[])


# In[8]:


# device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")


# In[9]:


train_dir = "training_images.zip"
test_dir  = "testing_images.zip"
train_image_file = "training_labels.txt"
test_image_file  = "testing_img_order.txt"
class_file = "classes.txt" 

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

train_transform = transforms.Compose([transforms.Resize([256,256]),
                                      transforms.RandomCrop([224,224], padding=4, padding_mode='reflect'),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomRotation(10),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225]),
                                      AddGaussianNoise(0.1, 0.08),
                                      transforms.RandomErasing(inplace=True)
                                      # transforms.ToPILImage()
                                      ])
valid_transform = transforms.Compose([transforms.Resize([224,224]),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])                                      
                                      ])




# # Read_data

# In[10]:


class BirdDataSet(Dataset):
    # def __init__(self, data_dir, image_list_file, test = False, transform = None):
    def __init__(self, data_dir, dataframe, test = False, transform = None):
        """
        Args:
            data_dir: path to image directory.
            dataframe : the dataframe containing images with corresponding labels.
            test: (bool) – If True, image_list_file is for test.
            transform: optional transform to be applied on a sample.
        """
        self.dir = data_dir
        self.df = dataframe
        self.test = test
        self.transform = transform                    
            
    def __getitem__(self, index):
        """
        Args:
            index: the index of item
        Returns:
            image and its labels
        """
        if not self.test:
            img, target = self.df.iloc[index, [0,2]].values
            with zipfile.ZipFile(self.dir, "r") as z:
                file_in_zip = z.namelist()
                if img in file_in_zip:
                    # print("Found image: ", img, " -- ")
                    filelikeobject = z.open(img, 'r')
                    image = Image.open(filelikeobject).convert('RGB')
#                     plt.imshow(image)
                else:
                    print("Not found " + img + "in " + self.dir)
                    
            if self.transform is not None:
                image = self.transform(image)
#                 image.show()
            return image, torch.tensor(target)
        
        else:
            img = self.df.iloc[index][0]
            with zipfile.ZipFile(self.dir, "r") as z:
                file_in_zip = z.namelist()
                if img in file_in_zip:
                    # print("Found image: ", img, " -- ")
                    filelikeobject = z.open(img, 'r')
                    image = Image.open(filelikeobject).convert('RGB')
                    # plt.imshow(image)
                else:
                    print("Not found " + img + "in " + self.dir)            
        
            if self.transform is not None:
                image = self.transform(image)
                # image.show()
            return image
    
    def __len__(self):
        return len(self.df)


#%%

class ResNet50(nn.Module):
    """Model modified.
    The architecture of our model is the same as standard ResNet50
    except the classifier layer which has an additional sigmoid function.
    """
    def __init__(self, out_size, grad = False):
        super(ResNet50, self).__init__()
        self.network = models.resnet50(pretrained=True)
        num_ftrs = self.network.fc.in_features
        # self.network.fc = nn.Linear(num_ftrs, out_size)
        self.network.fc = nn.Sequential(nn.BatchNorm1d(num_ftrs),
                                        nn.ReLU(inplace=True),
                                        nn.Dropout(0.3),
                                        nn.Linear(num_ftrs, 512),
                                        nn.BatchNorm1d(512),
                                        nn.ReLU(inplace=True),
                                        nn.Dropout(0.3),
                                        nn.Linear(512, out_size)
                                        )
    
    def forward(self, x):
        x = self.network(x)
        return x
    
    def freeze(self):
        # To freeze the residual layers
        for param in self.network.parameters():
            param.require_grad = False
        for param in self.network.fc.parameters():
            param.require_grad = True
    
    def unfreeze(self):
        # Unfreeze all layers
        for param in self.network.parameters():
            param.require_grad = True
    # def forward(self, x):
    #     x = self.DenseNet121(x)
    #     return x


# In[12]:


class LabelSmoothLoss(nn.Module):
    '''
    (1) When smoothing=0.0, the output is the same as nn.CrossEntropyLoss within precision 1e-5.
    (2) When smoothing>0.0, the sums of weights over different classes weight.sum(dim=-1) are always 1.
    '''
    
    def __init__(self, smoothing=0.0):
        super(LabelSmoothLoss, self).__init__()
        self.smoothing = smoothing
    
    def forward(self, input, target):
        log_prob = F.log_softmax(input, dim=-1)
        weight = input.new_ones(input.size()) *             self.smoothing / (input.size(-1) - 1.)
        weight.scatter_(-1, target.unsqueeze(-1), (1. - self.smoothing))
        loss = (-weight * log_prob).sum(dim=-1).mean()
        return loss    


# In[13]:


def draw_chart(chart_data, Fold):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    # -------------- draw loss image --------------
    ax[0].plot(chart_data['epoch'],chart_data['train_loss'],label='train_loss')
    ax[0].plot(chart_data['epoch'],chart_data['val_loss'],label='val_loss')
    ax[0].grid(True,axis="y",ls='--')
    ax[0].legend(loc= 'best')
    ax[0].set_title('Loss on Training and Validation Data', fontsize=10)
    ax[0].set_xlabel('epoch',fontsize=10)
    ax[0].set_ylabel('Loss',fontsize=10)

    # -------------- draw accuracy image --------------
    ax[1].plot(chart_data['epoch'],chart_data['val_acc'],label='val_acc')
    ax[1].plot(chart_data['epoch'],chart_data['train_acc'],label='train_acc')
    ax[1].grid(True,axis="y",ls='--')
    ax[1].legend(loc= 'best')
    ax[1].set_title('Accuracy  on Training and Validation Data',fontsize=10)
    ax[1].set_xlabel('epoch',fontsize=10)
    ax[1].set_ylabel('Accuracy',fontsize=10)
    
    plt.suptitle('Fold :{}'.format(Fold))
    plt.tight_layout()



#%%

def fit(model, dataloader, optimizer, criterion, scheduler, chart_data, acc_list, path_save, freeze = True):
    if freeze == True:
        epochs = args.epochs1
    else:
        epochs = args.epochs2
    total_epochs = args.epochs1 + args.epochs2

    for epoch in trange(epochs, desc="Epochs"):
        if freeze == True:
            chart_data['epoch'].append(epoch)
            print(f'Starting epoch {epoch+1}')
        else:
            chart_data['epoch'].append(epoch+args.epochs1)
            print(f'Starting epoch {epoch+args.epochs1+1}')            
        for phase in ['train', 'val']:
            if phase == "train":     # put the model in training mode
                model.train()
            else:     # put the model in validation mode
                model.eval()

            # keep track of training and validation loss
            running_loss = 0.0
            running_accuracy = 0.0
            
            for i, (data , target) in enumerate(dataloader[phase]):
                data , target = data.to(device), target.to(device)
                if phase == 'train':
                    optimizer.zero_grad()
                    
                output = model(data)
                _, preds = torch.max(output, dim=1)
                loss = criterion(output, target)
                
                if phase == 'train':
                    loss.backward()
                    # Gradient clipping
                    if args.grad_clip: 
                        nn.utils.clip_grad_value_(model.parameters(), args.grad_clip)
                    optimizer.step()                       
                    
                # statistics              
                running_loss += loss.item()
                running_accuracy += preds.eq(target).sum().item()
            epoch_loss = running_loss / len(dataloader[phase].dataset)
            epoch_acc = running_accuracy / len(dataloader[phase].dataset) 
            
            # visulize loss and accuracy
            if phase == 'train':
                chart_data['train_loss'].append((epoch_loss))
                chart_data['train_acc'].append(epoch_acc)
                curr_lr = optimizer.param_groups[0]['lr']
                print(f'LR:{curr_lr}')
                scheduler.step()
                
            if phase == 'val':
                chart_data['val_loss'].append((epoch_loss))
                chart_data['val_acc'].append(epoch_acc)
                acc_list.append(epoch_acc)
            if freeze == True:
                print('========================================================================') 
                print('Epoch [%d/%d]:%s Loss of the model:  %.4f %%' % (epoch+1, total_epochs, phase, 100 * epoch_loss))
                print('Epoch [%d/%d]:%s Accuracy of the model: %.4f %%' % (epoch+1, total_epochs,phase, 100 * epoch_acc)) 
                print('========================================================================')  
            else:
                print('========================================================================') 
                print('Epoch [%d/%d]:%s Loss of the model:  %.4f %%' % (epoch+args.epochs1+1, total_epochs, phase, 100 * epoch_loss))
                print('Epoch [%d/%d]:%s Accuracy of the model: %.4f %%' % (epoch+args.epochs1+1, total_epochs,phase, 100 * epoch_acc)) 
                print('========================================================================')                
                    
        # if freeze == True:
        #     torch.save({'model_state_dict':model.state_dict()}, 
        #                os.path.join(path_save,str(epoch+1)+'_'+str(epoch_acc)+'.pth'))              
        if freeze == False:
            torch.save({'model_state_dict':model.state_dict()}, 
                       os.path.join(path_save,str(epoch+args.epochs1+1)+'_'+str(epoch_acc)+'.pth'))                   
    return chart_data, acc_list
# # Training

# In[14]:


from tqdm import trange
def training():
    since = time.time()  
    df = pd.read_table(train_image_file, sep = "\s", header=None, 
                       names = ["Images", "Class_names"])
    df['labels'] = df.groupby(['Class_names']).ngroup()

    # kfolds
    kfolds = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    for fold in range(args.folds):
        # split
        train_idx, valid_idx = list(kfolds.split(np.arange(len(df)), df['labels']))[fold]
        train_set = BirdDataSet(train_dir, df.iloc[train_idx, :], transform = train_transform)
        valid_set = BirdDataSet(train_dir, df.iloc[valid_idx, :], transform = valid_transform)
        print('='*70) 
        print(f'FOLD {fold} for model')
        print('train_size: {}, valid_size: {}'.format(len(train_idx), len(valid_idx)))
        print('='*70)

        dataloader  = {"train" : DataLoader(dataset = train_set, batch_size = args.batch_size,
                                            shuffle = True, num_workers = args.num_workers),
                        "val"  : DataLoader(dataset = valid_set, batch_size = args.batch_size,
                                            shuffle = False, num_workers = args.num_workers)}
        since = time.time()
        path_save = f"resnet50/fold_{fold}"
        if not os.path.exists(path_save):
            os.makedirs(path_save)
        chart_data={"train_loss":[],"val_loss":[], "val_acc":[],"train_acc":[],"epoch":[]}
        
        model = ResNet50(args.n_classes).to(device)
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model, device_ids=[0,1,2,3])
        criterion = nn.CrossEntropyLoss()
#         criterion = LabelSmoothLoss(args.label_smoothing).to(device)
#         optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=4e-6)
    
#         scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, verbose=True)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-5)
        # scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.8685)
        # scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        # scheduler = lr_scheduler.OneCycleLR(optimizer, args.lr, epochs = args.epochs, 
        #                                         steps_per_epoch=len(dataloader['train']))
        
        
        acc_list = []        
        print('******************************model freeze******************************')
        model.freeze()
        if torch.cuda.device_count() > 1:
            model1 = nn.DataParallel(model, device_ids=[0,1,2,3])
        scheduler = lr_scheduler.OneCycleLR(optimizer, args.max_lr1, epochs = args.epochs1, 
                                            steps_per_epoch=len(dataloader['train']))
        chart_data, acc_list = fit(model, dataloader, optimizer, criterion, 
                                   scheduler, chart_data, acc_list, path_save)

        print('*****************************model unfreeze*****************************')
        model.unfreeze()
        if torch.cuda.device_count() > 1:
            model2 = nn.DataParallel(model, device_ids=[0,1,2,3])
        scheduler = lr_scheduler.OneCycleLR(optimizer, args.max_lr2, epochs = args.epochs2, 
                                            steps_per_epoch=len(dataloader['train']))
        chart_data, acc_list = fit(model, dataloader, optimizer, criterion, 
                                   scheduler, chart_data, acc_list, path_save, freeze = False)    
            
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc for epoch {}: {:.4f}%'.format(np.argmax(acc_list)+1, max(acc_list) * 100))
        draw_chart(chart_data, fold)
         
        # # Saving the model
        # torch.save({'model_state_dict':model.state_dict()}, 
        #             os.path.join(path_save,'Last_epoch'+str(epoch_acc)+'.pth'))
    # return np.argmax(acc_list), max(acc_list)


# In[15]:


def testing():   
    test = pd.read_table(test_image_file, sep = "\s", header=None, 
                         names = ["Images"])
    
    test_dataset = BirdDataSet(test_dir, test, test = True, transform = valid_transform)
    test_loader = DataLoader(test_dataset, batch_size = args.batch_size,
                             num_workers=args.num_workers, shuffle=False)

    # for fold in range(args.folds):    
    model = ResNet50(args.n_classes).to(device)
    path = "resnet50/fold_1/37_0.7183333333333334.pth"
    resume_file = torch.load(path, map_location = "cuda:0")
    model.load_state_dict(resume_file['model_state_dict'], False)
    model.eval()
    
    predicts = []
    with torch.no_grad():
        
        for j, image in enumerate(test_loader):
            image = image.to(device)
            output = model(image).cpu().detach()
            _, batch_prediction = torch.max(output, dim=1)
            predicts.extend(batch_prediction)

    # Corresponding labels to relative categories
    with open(class_file) as f:
         class_list = [x.strip() for x in f.readlines()]  

    # all the testing images    
    with open(test_image_file) as f:
         test_images = [x.strip() for x in f.readlines()]  

    submission = []
    for img, pred in zip(test_images, predicts):  # image order is important to your result
        predicted_class = class_list[pred]    # the predicted category
        submission.append([img, predicted_class])
    txt_name = 'answer-resbet50-fold-1-31_0.72.txt'
    np.savetxt(txt_name, submission, fmt='%s')
#         np.savetxt('answer.txt', submission, fmt='%s')


# In[ ]:


if __name__ == "__main__":
    if args.test:
        testing()
    else:
        training()
    torch.cuda.empty_cache()


# In[ ]:




