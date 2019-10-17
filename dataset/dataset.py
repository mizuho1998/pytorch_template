import torch
import torch.utils.data as data

from PIL import Image
import numpy as np
import os
import json
import random
import csv
from tqdm import tqdm

from .imageTransform import ImageTransform 


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


    def loadLabel(self, path):
        labels = {}
        with open(path) as f:
            reader = csv.reader(f)

            label_name = []
            label_num  = []

            for row in reader:
                label_name.append(row[0])
                label_num.append(int(row[1]))
                labels = {k:v for k, v in zip(label_name, label_num)}

          return labels


    def makeDataset(self, data_path, anotation_path, label_path):
        dataset = []
        labels  = self.loadLabel(label_path)

        # dataset = [{
        #     'path':,
        #     'label':
        #     ...},
        #     ...]

        return dataset


    def getWindow(self, path, s_frame):
        window = []
        it = imageTransform()

        for i in range(self.window_len):
            img_path = os.path.join(path, 'image_{:05}.jpg'.format(s_frame + i))
            clip = it.load(img_path)
            clip = it.crop(clip)
            clip = it.resize(clip)
            clip = it.toTensor(clip)
            window.append(clip)

        window = torch.stack(window, 0).permute(1, 0, 2, 3)

        return window


    def loadData(self, data):
        path = data['path']

        return self.getWindow(path, 0)

