from PIL import Image
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.utils.data as data
from os import listdir
from os.path import join
import numpy as np
import torch
import os
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import scipy.misc
import skimage.color
import cv2

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

def default_loader(path, mode='RGB'):
    img = Image.open(path).convert('RGB')
    if mode=='RGB':
        return img
    elif mode=='LAB':
        arr = np.array(img)
        cv_img = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
        lab = Image.fromarray(cv_img)
        return lab

class Dataset(data.Dataset):
    def __init__(self,contentPath,stylePath,fineSize,mode):
        super(Dataset,self).__init__()
        self.contentPath = contentPath
        self.content_image_list = [x for x in sorted(listdir(contentPath)) if is_image_file(x)]
        self.stylePath = stylePath
        self.style_image_list = [x for x in sorted(listdir(stylePath)) if is_image_file(x)]

        self.mode = mode
        self.fineSize = fineSize
        #self.normalize = transforms.Normalize(mean=[103.939,116.779,123.68],std=[1, 1, 1])
        #normalize = transforms.Normalize(mean=[123.68,103.939,116.779],std=[1, 1, 1])
        self.prep = transforms.Compose([
                    transforms.Scale(fineSize),
                    transforms.ToTensor(),
                    #transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]), #turn to BGR
                    ])

    def __getitem__(self,index):
        contentImgPath = os.path.join(self.contentPath,self.content_image_list[index])
        styleImgPath = os.path.join(self.stylePath,self.style_image_list[index])
        contentImg = default_loader(contentImgPath, self.mode)
        styleImg = default_loader(styleImgPath, self.mode)

        # resize
        if(self.fineSize != 0):
            w,h = contentImg.size
            if(w > h):
                if(w != self.fineSize):
                    neww = self.fineSize
                    newh = int(h*neww/w)
                    contentImg = contentImg.resize((neww,newh))
                    styleImg = styleImg.resize((neww,newh))
            else:
                if(h != self.fineSize):
                    newh = self.fineSize
                    neww = int(w*newh/h)
                    contentImg = contentImg.resize((neww,newh))
                    styleImg = styleImg.resize((neww,newh))


        # Preprocess Images
        contentImg = transforms.ToTensor()(contentImg)
        styleImg = transforms.ToTensor()(styleImg)
        return contentImg.squeeze(0),styleImg.squeeze(0),self.content_image_list[index]

    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.content_image_list)
