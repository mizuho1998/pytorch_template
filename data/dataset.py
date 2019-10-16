import torch
import torch.utils.data as data

from PIL import Image
import numpy as np
import os
import json
import random
import csv
from tqdm import tqdm

from dataset import imageTransform 

class Dataset(data.Dataset):
    
    def __init__(self, data_path, anotation_path, label_path, class_num=200):
        self.data_path = data_path
        self.anotation_path = anotation_path
        self.label_path = label_path
        self.class_num = class_num
        self.data = self.makeDataset(self.data_path, self.anotation_path, self.label_path, self.batch_size)
        

    def __getitem__(self, index):
        data   = self.data[index]
        target = self.data[index]['label']
        sample = self.loadData(data)

        return sample, target

    def __len__(self):
        return len(self.data)

    def makeDataset(self, data_path, anotation_path, label_path):
        dataset = []

        # dataset = [{
        #     'path':,
        #     'label':
        #     ...},
        #     ...]

        return dataset


    def getWindow(self, path, s_frame):
        window = []

        for i in range(self.window_len):
            img_path = os.path.join(path, 'image_{:05}.jpg'.format(s_frame + i))
            clip = imageTransform.loadImage(img_path)
            clip = imageTransform.cropImage(clip)
            clip = imageTransform.resizeImage(clip)
            clip = imageTransform.toTensor(clip)
            window.append(clip)

        window = torch.stack(window, 0).permute(1, 0, 2, 3)

        return window


    def loadData(self, data):
        path = data['path']

        return self.getWindow(path, 0)





