##############################
# dataset utilities.
# adapted from https://github.com/daa233/generative-inpainting-pytorch/blob/master/data/dataset.py
##############################
import sys
import numpy as np
import torch.utils.data as data
from torch import narrow
from os import listdir
from inpainterutils.tools import default_loader, is_image_file, normalize
import os

import torchvision.transforms as transforms
from torchvision.transforms import functional as F

import random

class Dataset(data.Dataset):
    # master class for loading in data
    # modified for selecting random patches from full non-anomalous (normal) images
    def __init__(self, config, data_path, image_shape, with_subfolder=False, random_crop=True, return_name=False, subset_frac=None, return_label=False):
        super(Dataset, self).__init__()
        if with_subfolder:
            if subset_frac is not None:
                all_filenames = self._find_samples_in_subfolders(data_path)
                self.samples = random.sample(all_filenames, int(subset_frac*len(all_filenames)))
            else:
                self.samples = self._find_samples_in_subfolders(data_path)
        else:
            if subset_frac is not None:
                all_filenames = [x for x in listdir(data_path) if is_image_file(x)]
                self.samples = random.sample(all_filenames, int(subset_frac*len(all_filenames)))
            else: 
                self.samples = [x for x in listdir(data_path) if is_image_file(x)]
        self.config = config
        self.data_path = data_path
        self.image_shape = image_shape[:-1]
        self.random_crop = random_crop
        self.return_name = return_name
        self.return_label = return_label

    def __getitem__(self, index): # defines how this behaves as an iterable
        path = os.path.join(self.data_path, self.samples[index])
        try:
            initial_img = default_loader(path)
            img = initial_img

            useful_patch = False
            while not useful_patch: 
                # ensure that the used patch of the image has the actual object (eg, breast) within
                # so the patch isn't just background/empty space
                img = initial_img
                if self.random_crop: # crops img into desired shape to  be used for training
                    imgw, imgh = img.size
                    if imgh < self.image_shape[0] or imgw < self.image_shape[1]:
                        img = transforms.Resize(min(self.image_shape)) (img)
                    img = transforms.RandomCrop(self.image_shape)(img) 
                else:
                    img = transforms.Resize(self.image_shape)(img)
                    img = transforms.RandomCrop(self.image_shape)(img) 

                img = transforms.ToTensor()(img)  # turn the image to a tensor
                img = normalize(img)

                # check if the patch contains tissue
                useful_patch = img.max() >= -0.99

            # (fix for grayscale imgs) make sure channel number in img is correct, according to config
            if int(img.shape[0]) != self.config['train']['image_shape'][2]:
                img = narrow(img, 0, 0, 1)
            
            if self.return_name:
                return self.samples[index], img
            elif self.return_label:
                label = 'Normal'
                return img, label 
            else:
                return img
        except Exception as e:
            print(e)
            print('erroroneous file path: $$$$$$$$\n\n\n{}\n\n\n$$$$$$$$'.format(path))

            return

    def _find_samples_in_subfolders(self, dir):
        """
        Finds the class folders in a dataset.
        Args:
            dir (string): Root directory path.
        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.
        Ensures:
            No class is a subdirectory of another.
        """
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        samples = []
        for target in sorted(class_to_idx.keys()):
            d = os.path.join(dir, target)
            if not os.path.isdir(d):
                continue
            for root, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    if is_image_file(fname):
                        path = os.path.join(root, fname)
                        # item = (path, class_to_idx[target])
                        # samples.append(item)
                        samples.append(path)
        return samples

    def __len__(self):
        return len(self.samples)
