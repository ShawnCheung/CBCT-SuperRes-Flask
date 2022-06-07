import torch
from options.test_options import TestOptions
from models import networks
import data.single_dataset as single_dataset
import torchvision.transforms as transforms
from base import writedcm
import os
import sys
from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import numpy as np
if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    # dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    # model = create_model(opt)      # create a model given opt.model and other options
    # model.setup(opt)               # regular setup: load and print networks; create schedulers
    # create a website
    model = networks.ResnetGenerator(opt.input_nc, opt.output_nc, opt.ngf, networks.get_norm_layer(norm_type=opt.norm),
                               not opt.no_dropout,  n_blocks=9)
    # model = networks.UnetGenerator(opt.input_nc, opt.output_nc, 8, opt.ngf,  networks.get_norm_layer(norm_type=opt.norm))
    dataset = single_dataset.SingleDataset(opt)
    model.load_state_dict(torch.load(ROOT / "checkpoints\\exp_of_NEW_mat16bit_upUseLinJin\\latest_net_G_A.pth"))
    # web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    # if opt.load_iter > 0:  # load_iter is 0 by default
    #     web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    # print('creating web directory', web_dir)
    # webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    dataloder = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        num_workers=int(opt.num_threads))
    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    for i, data in enumerate(dataloder):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        if torch.cuda.is_available():
            data['A'] = data['A'].cuda()
        out = model(data['A'])  # unpack data from data loader
        # model.test()           # run inference
        # visuals = model.get_current_visuals()  # get image results
        # img_path = model.get_image_paths()     # get image paths
        imagA = transforms.ToPILImage()(data['A'][0][(opt.input_nc-1)//2]/2+0.5)
        imagB = transforms.ToPILImage()(out[0][(opt.input_nc-1)//2]/2+0.5)
        # imagA.show()
        # imagB.show()
        imagA.save(opt.results_dir+"/a" + data['A_paths'][0].split('//')[-1].split('.')[0]+'.tif')
        imagB.save(opt.results_dir+"/b" + data['A_paths'][0].split('//')[-1].split('.')[0]+'.tif')
        savePath = opt.results_dir+"/" + data['A_paths'][0].split('//')[-1]
        resultData =  ((out[0][(opt.input_nc-1)//2]/2+0.5).cpu().detach().numpy() * opt.normalize_B).astype(np.int16)
        writedcm(data['A_paths'][0], savePath, resultData)
        pass
        # if i % 5 == 0:  # save images to an HTML file
        #     print('processing (%04d)-th image... %s' % (i, img_path))
        # save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
    # webpage.save()  # save the HTML
