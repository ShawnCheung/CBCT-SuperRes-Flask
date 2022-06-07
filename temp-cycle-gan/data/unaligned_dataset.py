import os

import torch

from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import torchvision.transforms as transforms
import scipy.io as sio

class UnalignedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image
        # self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        # self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        if self.opt.input_nc == 1:
            A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
            if self.opt.serial_batches:   # make sure index is within then range
                index_B = index % self.B_size
            else:   # randomize the index for domain B to avoid fixed pairs.
                index_B = random.randint(0, self.B_size - 1)
            B_path = self.B_paths[index_B]
            A_img = Image.open(A_path)#.convert('RGB')
            B_img = Image.open(B_path)#.convert('RGB')
            # apply image transformation
            # A = self.transform_A(A_img)
            # B = self.transform_B(B_img)
            A = transforms.ToTensor()(A_img) / 12000  #6000
            B = transforms.ToTensor()(B_img) / 22000
            # A = transforms.Resize([75, 75])(A)  #A = transforms.Resize([300, 300])(A)
            # B = transforms.Resize([75, 75])(B)  #B = transforms.Resize([300, 300])(B)
            A = transforms.RandomCrop([52, 52])(A)
            B = transforms.RandomCrop([416, 416])(B)
            A = transforms.RandomHorizontalFlip(0.5)(A)
            B = transforms.RandomHorizontalFlip(0.5)(B)
            A = transforms.Normalize((0.5,), (0.5,))(A)
            B = transforms.Normalize((0.5,), (0.5,))(B)
        else:
            A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
            if self.opt.serial_batches:  # make sure index is within then range
                index_B = index % self.B_size
            else:  # randomize the index for domain B to avoid fixed pairs.
                index_B = random.randint(0, self.B_size - 1)
            B_path = self.B_paths[index_B]
            A_img = sio.loadmat(A_path)['data'].view('int16')
            B_img = sio.loadmat(B_path)['data'].view('int16')
            # apply image transformation
            # A = self.transform_A(A_img)
            # B = self.transform_B(B_img)
            # A = transforms.ToTensor()(A_img) / 6000  # 6000
            # B = transforms.ToTensor()(B_img) / 25000
            A = torch.from_numpy(A_img)/ self.opt.normalize_A
            B = torch.from_numpy(B_img)/ self.opt.normalize_B
            A = transforms.RandomCrop([self.opt.crop_size_A, self.opt.crop_size_A])(A)
            B = transforms.RandomCrop([self.opt.crop_size_B, self.opt.crop_size_B])(B)
            A = transforms.Resize([self.opt.resize_size, self.opt.resize_size], Image.NEAREST)(A)  # A = transforms.Resize([300, 300])(A)
            B = transforms.Resize([self.opt.resize_size, self.opt.resize_size])(B)  # B = transforms.Resize([300, 300])(B)

            A = transforms.RandomHorizontalFlip(0.5)(A)
            B = transforms.RandomHorizontalFlip(0.5)(B)
            A = transforms.RandomVerticalFlip(0.5)(A)
            B = transforms.RandomVerticalFlip(0.5)(B)
            A = transforms.Normalize((0.5,), (0.5,))(A)
            B = transforms.Normalize((0.5,), (0.5,))(B)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

        # return {'clinical': A, 'micro': B, 'clinical_paths': A_path, 'micro_paths': B_path}


    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
