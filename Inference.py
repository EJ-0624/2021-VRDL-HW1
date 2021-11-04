# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 19:37:22 2021

@author: User
"""

import numpy as np
import pandas as pd
import zipfile
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import argparse
from efficientnet_pytorch import EfficientNet


parser = argparse.ArgumentParser()
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--n_classes", type=int, default=200)
parser.add_argument("--batch_size", type=int, default=64)
args = parser.parse_args(args=[])


train_dir = "training_images.zip"
test_dir  = "testing_images.zip"
train_image_file = "training_labels.txt"
test_image_file  = "testing_img_order.txt"
class_file = "classes.txt" 


valid_transform = transforms.Compose([transforms.Resize([224,224]),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])                                      
                                      ])

# device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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


class Efficientnet(nn.Module):
    """Model modified.
    The architecture of our model is the same as standard ResNet50
    except the classifier layer which has an additional sigmoid function.
    """
    def __init__(self, out_size, grad = False):
        super(Efficientnet, self).__init__()
        self.network = EfficientNet.from_pretrained(args.model, num_classes = args.n_classes)

    def forward(self, x):
        x = self.network(x)
        return x
    
    def freeze(self):
        # To freeze the residual layers
        for param in self.network.parameters():
            param.require_grad = False
        for param in self.network._fc.parameters():
            param.require_grad = True
    
    def unfreeze(self):
        # Unfreeze all layers
        for param in self.network.parameters():
            param.require_grad = True
            
        

def testing():   
    test = pd.read_table(test_image_file, sep = "\s", header=None, 
                         names = ["Images"])
    
    test_dataset = BirdDataSet(test_dir, test, test = True, transform = valid_transform)
    test_loader = DataLoader(test_dataset, batch_size = args.batch_size,
                             num_workers=args.num_workers, shuffle=False)

    model = Efficientnet(args.n_classes).to(device)
    path = "EfficientNet_b4_weights.pth"
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
    txt_name = 'answer.txt'
    np.savetxt(txt_name, submission, fmt='%s')


    
if __name__ == "__main__":
    testing()
    torch.cuda.empty_cache()    
