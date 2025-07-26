"""
This file will help to demonstrate pipeline for testing microscopy data using the DeepCAD-RT algorithm.
The demo shows how to construct the params and call the relevant functions for testing DeepCAD-RT network.
The demo will automatically download tif file and corresponding model file for demo testing.
See inside for details.

* This demo is also available as a jupyter notebook (see demo_test_pipeline.ipynb) and Colab notebook (see
DeepCAD_RT_demo_colab.ipynb)

More information can be found in the companion paper.
"""
import os
from pyexpat import model
import torch
import torch.nn as nn
import argparse
import time
import datetime
import numpy as np
import math
import tifffile as tiff

from torch.autograd import Variable
from torch.utils.data import DataLoader
from skimage import io
from tqdm import tqdm
from SRDTrans.SRDTrans import SRDTrans
from SRDTrans.data_process import test_preprocess_lessMemoryNoTail_chooseOne, testset, singlebatch_test_save, \
    multibatch_test_save

from SRDTrans.sampling import *


def srd_denoise(reg_image,model):
    #############################################################################################################################################
    parser = argparse.ArgumentParser()
    parser.add_argument('--GPU', type=str, default='0',
                        help="the index of GPU used for computation (e.g., '0', '0,1', '0,1,2')")
    parser.add_argument('--patch_x', type=int, default=128, help="patch size in x and y")
    parser.add_argument('--patch_t', type=int, default=128, help="patch size in t")
    parser.add_argument('--overlap_factor', type=float, default=0.5,
                        help="the overlap factor between two adjacent patches")
    parser.add_argument('--pth_path', type=str, default='./SRDTrans/pth', help="the root path to save models")
    parser.add_argument('--test_datasize', type=int, default=10000000, help='how many slices to be tested')
    parser.add_argument('--scale_factor', type=int, default=1, help='the factor for image intensity scaling')
    opt = parser.parse_args()

    # use isotropic patch size by default
    opt.patch_y = opt.patch_x  # the height of 3D patches (patch size in y)
    opt.patch_t = opt.patch_t  # the length of 3D patches (patch size in t)
    # opt.gap_t (image gap) is the distance between two adjacent patches
    # use isotropic opt.gap by default
    opt.gap_t = int(opt.patch_t * (1 - opt.overlap_factor))
    opt.gap_x = int(opt.patch_x * (1 - opt.overlap_factor))
    opt.gap_y = int(opt.patch_y * (1 - opt.overlap_factor))
    opt.ngpu = str(opt.GPU).count(',') + 1
    opt.batch_size = opt.ngpu  # By default, the batch size is equal to the number of GPU for minimal memory consumption
    print('\033[1;31mParameters -----> \033[0m')
    print(opt)
    ########################################################################################################################
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.GPU)
    ##############################################################################################################################################################
    denoise_generator = SRDTrans(
        img_dim=opt.patch_x,
        img_time=opt.patch_t,
        in_channel=1,
        embedding_dim=128,
        num_heads=8,
        hidden_dim=128 * 4,
        window_size=7,
        num_transBlock=1,
        attn_dropout_rate=0.1,
        f_maps=[8, 16, 32, 64],
        input_dropout_rate=0
    )

    if torch.cuda.is_available():
        print('\033[1;31mUsing {} GPU(s) for testing -----> \033[0m'.format(torch.cuda.device_count()))
        denoise_generator = denoise_generator.cuda()
        denoise_generator = nn.DataParallel(denoise_generator, device_ids=range(opt.ngpu))
    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    ##############################################################################################################################################################

    # Start processing
    # load model
    model_name = os.path.join(opt.pth_path, model)

    if isinstance(denoise_generator, nn.DataParallel):
        denoise_generator.module.load_state_dict(torch.load(model_name))  # parallel
        denoise_generator.eval()
    else:
        denoise_generator.load_state_dict(torch.load(model_name))  # not parallel
        denoise_generator.eval()
    denoise_generator.cuda()

    # test all stacks
    name_list, reg_image, coordinate_list, img_mean, input_data_type = test_preprocess_lessMemoryNoTail_chooseOne(
        opt, reg_image)
    prev_time = time.time()
    time_start = time.time()
    denoise_img = np.zeros(reg_image.shape)

    #print("coordinate_list length:", len(coordinate_list))
    test_data = testset(name_list, coordinate_list, reg_image)
    testloader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False)
    with torch.no_grad():
        for iteration, (noise_patch, single_coordinate) in enumerate(testloader):
            noise_patch = noise_patch.cuda()

            real_A = noise_patch
            real_A = Variable(real_A)
            fake_B = denoise_generator(real_A)

            preditc_numpy = fake_B.cpu().detach().numpy().astype(np.float32)
            ################################################################################################################
            # Determine approximate time left
            batches_done = iteration
            batches_left = 1 * len(testloader) - batches_done
            time_left_seconds = int(batches_left * (time.time() - prev_time))
            time_left = datetime.timedelta(seconds=time_left_seconds)
            prev_time = time.time()
            ################################################################################################################
            if iteration % 1 == 0:
                time_end = time.time()
                time_cost = time_end - time_start  # datetime.timedelta(seconds= (time_end - time_start))
                print(
                    '\r[Patch %d/%d] [Time Cost: %.0d s] [ETA: %s s]     '
                    % (
                        iteration + 1,
                        len(testloader),
                        time_cost,
                        time_left_seconds
                    ), end=' ')

            if (iteration + 1) % len(testloader) == 0:
                print('\n', end=' ')
            ################################################################################################################
            output_image = np.squeeze(fake_B.cpu().detach().numpy())
            raw_image = np.squeeze(real_A.cpu().detach().numpy())
            if (output_image.ndim == 3):
                turn = 1
            else:
                turn = output_image.shape[0]
            # print(turn)
            if (turn > 1):
                for id in range(turn):
                    # print('shape of output_image -----> ',output_image.shape)
                    aaaa, bbbb, stack_start_w, stack_end_w, stack_start_h, stack_end_h, stack_start_s, stack_end_s = multibatch_test_save(
                        single_coordinate, id, output_image, raw_image)
                    aaaa = aaaa + img_mean
                    bbbb = bbbb + img_mean
                    denoise_img[stack_start_s:stack_end_s, stack_start_h:stack_end_h,
                    stack_start_w:stack_end_w] \
                        = aaaa * (np.sum(bbbb) / np.sum(aaaa)) ** 0.5
            else:
                aaaa, bbbb, stack_start_w, stack_end_w, stack_start_h, stack_end_h, stack_start_s, stack_end_s = singlebatch_test_save(
                    single_coordinate, output_image, raw_image)
                aaaa = aaaa + img_mean
                bbbb = bbbb + img_mean
                denoise_img[stack_start_s:stack_end_s, stack_start_h:stack_end_h, stack_start_w:stack_end_w] \
                    = aaaa * (np.sum(bbbb) / np.sum(aaaa)) ** 0.5

        del reg_image
        output_img = denoise_img.squeeze().astype(np.float32) * opt.scale_factor
        del denoise_img
        output_img = np.clip(output_img, 0, 65535).astype('int32')
        # Save inference image
        if input_data_type == 'uint16':
            output_img = np.clip(output_img, 0, 65535)
            output_img = output_img.astype('uint16')

        elif input_data_type == 'int16':
            output_img = np.clip(output_img, -32767, 32767)
            output_img = output_img.astype('int16')
        else:
            output_img = output_img.astype('int32')
        return output_img
