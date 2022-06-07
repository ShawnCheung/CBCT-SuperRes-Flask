from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import torchvision.transforms as transforms
import scipy.io as sio
from os import listdir
import torch
import numpy as np
from base import readdcm
class SingleDataset(BaseDataset):
    """This dataset class can load a set of images specified by the path --dataroot /path/to/data.

    It can be used for generating CycleGAN results only for one side with the model option '-model test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        A_paths = sorted(listdir(opt.dataroot))
        A_paths = [opt.dataroot + "//" + A_paths[i] for i in range(len(A_paths))]
        A_paths = A_paths[0:-2]+A_paths[1:-1]+A_paths[2:]
        self.A_paths = [A_paths[i::len(A_paths)//3] for i in range(len(A_paths)//3)]
        input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        # self.transform = get_transform(opt, grayscale=(input_nc == 1))

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A and A_paths
            A(tensor) - - an image in one domain
            A_paths(str) - - the path of the image
        """

        A_path = self.A_paths[index]
        A_img = []
        for x in A_path:
            A_img.append(readdcm(x)+1024)
        A_img = np.array(A_img)
        # A_img = sio.loadmat(A_path)['data'].view('int16')
        A = torch.from_numpy(A_img) / self.opt.normalize_A    #6000
        A = transforms.Resize([int(self.opt.resize_size/self.opt.crop_size_A)*A.size()[1], int(self.opt.resize_size/self.opt.crop_size_A)*A.size()[2]], Image.NEAREST)(A)
        # A = transforms.CenterCrop([256, 256])(A)

        A = A[:, 0:A.size()[1] - (A.size()[1] % 4), :]
        A = A[:, :, 0:A.size()[2] - (A.size()[2] % 4)]
        A = transforms.Normalize((0.5,), (0.5,))(A)



        return {'A': A.float(), 'A_paths': A_path[1]}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.A_paths)
