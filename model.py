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
from preprocessing import transform, rgb_loader
from inf_net import init_semi, init_unet, predict_semi, predict_unet

def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--f_path", type=str, default=TRAIN_PATH,
                        help="File path to the CSV or NPY file that contains walking data.")
    args = parser.parse_args()
    return args

class Master():
    def __init__(self, filename, seg_path = SEGMENTATION_PATH):
        self.filename = filename
    def percent_lungs_slice(self, filepath_mask, slice_number):
        M, _ = nrrd.read(filepath_mask)
        current_slice, _ = bound_box_and_reshape(M, slice_number)
        num_pixels = current_slice.shape[2] * current_slice.shape[1]
        white = np.sum(current_slice)
        return "{:.1f}".format(white / num_pixels)

    def bound_box_and_reshape(self, img, slice_idx):
        """
        Crop given slice of image to lung size (remove redundant empty space) and reshape to 512x512 pxls. Return edited img_slice.
        """
        img_slice = img[:,:,slice_idx]    
        rows = np.any(img_slice, axis=1)
        cols = np.any(img_slice, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        img_slice = resize(img_slice[rmin:rmax, cmin:cmax], (512,512))
        img_slice = np.transpose(img_slice[:, :, np.newaxis], axes = [2, 0, 1]).astype('float32')
        lung_pixels = abs((rmax-rmin) * (cmax - cmin))
        
        return img_slice, lung_pixels

    def mask_original(self, filepath_CT, filepath_mask):
        """
        Mask and normalize original CT. Select only preset number of central slices. 
        """
        I = np.array(nib.load(filepath_CT).dataobj)
        #I = normalize(I, -1350, 150)
        M, _ = nrrd.read(filepath_mask)
        
        nS = np.where(M==1, I, M)
        
        #z = nS.shape[2]//2
        #dz = nb_central_slices//2
        #nS = nS[:,:,z-dz:z+dz]
            
        return nS

    def calculate_mask_area_percentage(self, mask, NP_VALUE_OF_MASK):
        """
            Input je numpy array. Vrže ven odstotek, kolikšen
            del slike zajema anomalija.
        """
        mask = np.where(mask == NP_VALUE_OF_MASK, 1, 0)
        summa = np.sum(mask)
        percentage = float(summa) / (mask.shape[0]*mask.shape[1])
        return round(percentage,3)

    def process_one_patient(self, filename_img):
        '''
            Returns the end-to-end lung segmentation statistics for a single 
            3D CT scan given the input path image. Input should be in .npy format.  
        '''

        filename_mask = filename_img.replace(".nii.gz", ".nrrd")
        filepath_img = os.path.join(self.root, filename_img)
        filepath_mask = os.path.join(self.seg_path, filename_mask)
        patient_number = filename_mask.split(".nrrd")[0]
        img = mask_original(filepath_img, filepath_mask)
        num_slices = img.shape[2]
        
        label = get_label(patient_number)
        
        patient_lung_volume_score = 0.0
        patient_ggo_score = 0.0
        patient_consolidation_score = 0.0
            
        for i in tqdm(range(num_slices)):
            try:
                current_slice, bbox_pixels = bound_box_and_reshape(img, i)
            except:
                continue
        
            current_slice = np.rot90(np.squeeze(current_slice))
            percentage = percent_lungs_slice(filepath_mask, i)
            if percentage != 0:
                plt.imsave('cur.png', current_slice)
                t = transform(rgb_loader('cur.png'))
                r = predict_semi(t)
                plt.imsave('semi.png', r)
                e = transform(rgb_loader('semi.png'))
                pred = predict_unet(t,e)
                ggo_mask, consolidation_mask = split_class_imgs(pred)
                
                patient_lung_volume_score += float(percentage) * bbox_pixels
                
                ggo_percentage = calculate_mask_area_percentage(ggo_mask, NP_VALUE_OF_MASK)
                consolidation_percentage = calculate_mask_area_percentage(consolidation_mask, NP_VALUE_OF_MASK)
                
                patient_ggo_score += ggo_percentage * bbox_pixels
                patient_consolidation_score += consolidation_percentage * bbox_pixels
                
                plt.show()
        
        return {"filename_img" : filename_img,
                "label" : label,
                "lung_vol": patient_lung_volume_score, 
                "ggo_vol" : patient_ggo_score, 
                "cons_vol" : patient_consolidation_score}


    def process_all_patients(self):
        training_set = []
        for _,_,filenames in os.walk(self.path):
            for filename in tqdm(filenames):
                if filename.endswith(".nii.gz"):
                    patient_dict = process_one_patient(filename, path = self.path, seg_path = self.seg_path)
                    with open(training_csv, mode='a') as file_:
                        file_.write(str(patient_dict.values()))
                        file_.write("\n")
                    training_set.append(patient_dict)
        return training_set

    def printResults(self):
        print(self.process_one_patient(self.filename))
    def toPandas(self):
        return pd.DataFrame(self.process_one_patient(self.filename))
    
    def load_images_from_folder(self):
        images = []
        for filename in os.listdir(self.root):
            img = cv2.imread(os.path.join(self.root,filename))
            if img is not None:
                images.append(img)
        return images

    def display_by_slice(self):
        for k in self.load_images_from_folder(x):
            plt.imshow(k)

def main(args = sys.argv[1:]):
    master = Master(args.f_path)
    master.printResults()

if name == "__main__":
    main()
