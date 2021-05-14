import nibabel as nib
import numpy as np
import pandas as pd
import os
import shutil
import matplotlib.pyplot as plt
import cv
import torch
import cv2
from cv2 import resize
from PIL import Image
import nrrd
import torchvision.transforms as transforms
from tqdm import tqdm

from config import *

class DataLoader():
    def __init__(self, path, shape = INPUT_SHAPE):
        self.path = path
    def rgb_loader():
        img = cv2.imread(self.path)
        img = cv2.resize(img, INPUT_SHAPE)
        return Image.fromarray(img)

    def transform(self):
        x = self.rgb_loader()
        operation = transforms.Compose([
                    transforms.Resize(INPUT_SHAPE),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406],
                                        [0.229, 0.224, 0.225])
        ])
        return operation(x).unsqueeze(0)

class Generator():
    def __init__(self, path, batch_size = BATCH_SIZE):
        self.path = path
    def gen(self):
        for filepath in os.listdir(self.path):
            try:
                scan = nib.load(self.path+filepath)
                yield np.array(scan.dataobj)
            except StopIteration:
                break;
    def construct_dataset(self, repeat = 5):
        data, labels = gen()
        gen = batch_data(data, labels, BATCH_SIZE)
        dataset = tf.data.Dataset.from_generator(
                lambda: gen,
                output_signature=(
                tf.TensorSpec(shape=(BATCH_SIZE, 3000,1), dtype=tf.float32),
                tf.TensorSpec(shape=(BATCH_SIZE, 5,1), dtype=tf.int32)))

        return dataset

    def split_dataset(self):
        dataset = self.construct_dataset()
        DATASET_SIZE = tf.data.experimental.cardinality(dataset).numpy()
        train_size = int(0.85 * DATASET_SIZE)
        val_size = int(0.15 * DATASET_SIZE)
        test_size = int(0.05 * DATASET_SIZE)

        train_dataset = dataset.take(train_size)
        test_dataset = dataset.skip(train_size)
        val_dataset = test_dataset.skip(val_size)
        test_dataset = test_dataset.take(test_size)
        return train_dataset, val_dataset, test_dataset

    def batch_data(self):
        x,y = self.gen()
        shuffle = np.random.permutation(len(x))
        start = 0
        x = x[shuffle]
        y = y[shuffle]
        while start + self.batch_size <= len(x):
            yield x[start:start+self.batch_size], y[start:start+self.batch_size]
            start += self.batch_size



