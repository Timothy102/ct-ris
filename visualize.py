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
import nrrd
import torchvision.transforms as transforms
from tqdm import tqdm

from config import *


def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=TRAIN_PATH,
                        help="File path to the CSV or NPY file that contains walking data.")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_VIS,
                        help="Directory where to save outputs.")
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    else:
        shutil.rmtree(args.output_dir)
        os.makedirs(args.output_dir)

    return args

class Visualizor():
    def __init__(self, root, filename, output_dir):
        self.root = root
        self.filename = filename
        self.output_dir = output_dir
    def initdir(self):
        path = self.output_dir + self.filename.replace(".nii.gz", "/")
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)
        return path

    def save_images_one(self, train = True):
        k_path = initdir(self.filename)
        filename_mask = self.filename.replace(".nii.gz", ".nrrd")
        if train:
            filepath_img = os.path.join(train_path, filename)
            seg = segmentation_path +"train/"
        else: 
            filepath_img = os.path.join(test_path, filename)
            seg = segmentation_path +  "test/"
        filepath_mask = os.path.join(seg, filename_mask)
        patient_number = filename_mask.split(".nrrd")[0]
        
        img = mask_original(filepath_img, filepath_mask)
        num_slices = img.shape[2]
        
        
        for i in tqdm(range(num_slices)):
            store_path = k_path + str(i)+"-"
            try:
                current_slice, bbox_pixels = bound_box_and_reshape(img, i)
            except:
                continue
            current_slice = np.rot90(np.squeeze(current_slice))
            percentage = percent_lungs_slice(filepath_mask, i)
            if percentage != 0:
                plt.imsave(store_path+"cur"+".png", current_slice, cmap = 'gray')
                current_slice = rgb_loader(store_path+"cur"+".png")
                seg = transform(current_slice)
                semi_mask = predict_net(seg)
                plt.imsave(store_path+"semi"+".png", semi_mask, cmap = 'gray')
                unet_mask = predict_unet(transform(rgb_loader(store_path+"semi"+".png")), seg)
                plt.imsave(store_path+"unet"+".png", unet_mask, cmap = 'gray')
                

                ggo_mask, consolidation_mask = split_class_imgs(unet_mask)
                plt.imsave(store_path+"ggo"+".png", ggo_mask, cmap = 'gray')
                plt.imsave(store_path+"conso"+".png", consolidation_mask, cmap = 'gray')

            
    def save_all_images(self):
        for k in os.tqdm(listdir(self.root)):
            save_images_one(k)


def main(args = sys.argv[1:]):
    args = parseArguments()
    vis = Visualizor(args.data_dir, args.filename, args.output_dir)


if name == "__main__":
    main()
    

