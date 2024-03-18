"""
This file will help to demonstrate pipeline for testing microscopy data using the DeepCAD-RT algorithm.
The demo shows how to construct the params and call the relevant functions for testing DeepCAD-RT network.
The demo will automatically download tif file and corresponding model file for demo testing.
See inside for details.

* This demo is also available as a jupyter notebook (see demo_test_pipeline.ipynb) and Colab notebook (see
DeepCAD_RT_demo_colab.ipynb)

More information can be found in the companion paper.
"""
from deepcad.test_collection import testing_class
from deepcad.movie_display import display
from deepcad.utils import get_first_filename, download_demo
import tkinter as tk
import os
from tkinter import filedialog


def main():
    ## -----------------------------------------pytho Folder Settings --------------------------------------------##
    # GUI choose  the path of datasets
    root = tk.Tk()
    root.withdraw()
    datasets_path = filedialog.askdirectory()
    if not datasets_path:
        print("\033[91m" +
              "No folder selected. Program is terminated. " + "\033[0m")
        return

    # set model dir
    pth_dir = r"E:\\DeepCAD_RT_pytorch\\pth"  # './pth'
    denoise_model = 'deepcad_model'  # A folder containing pth models to be tested

    # set output dir: datasets_path\DeepCAD-RT
    parent_dir = os.path.abspath(os.path.join(datasets_path, os.pardir))
    datasets_name = os.path.basename(datasets_path)
    output_dir = os.path.join(parent_dir, datasets_name + "_denoised")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    ## ----------------------------------------- Folder Settings --------------------------------------------##

    ## ----------------------------------------- Parameter Settings --------------------------------------------##
    # %% First setup some parameters for testing
    # the number of frames to be tested (test all frames if the number exceeds the total number of frames in a .tif file)
    test_datasize = 100000
    # the index of GPU you will use for computation (e.g. '0', '0,1', '0,1,2')
    GPU = '0'
    patch_xy = 150                        # the width and height of 3D patches
    patch_t = 150                         # the time dimension of 3D patches
    # the overlap factor between two adjacent patches.
    overlap_factor = 0.60
    # Since the receptive field of 3D-Unet is ~90, seamless stitching requires an overlap (patch_xyt*overlap_factor）of at least 90 pixels.
    # if you use Windows system, set this to 0.
    num_workers = 0

    # %% Setup some parameters for result visualization during testing period (optional)
    # choose whether to display inference performance after each epoch
    visualize_images_per_epoch = False
    # choose whether to save inference image after each epoch in pth path
    save_test_images_per_epoch = True

    ## ----------------------------------------- Parameter Settings --------------------------------------------##

    # %% Play the demo noise movie (optional)
    # playing the first noise movie using opencv.
    display_images = False

    if display_images:
        display_filename = get_first_filename(datasets_path)
        print('\033[1;31mDisplaying the first raw file -----> \033[0m')
        print(display_filename)
        display_length = 500  # the frames number of the noise movie
        # normalize the image and display
        display(display_filename, display_length=display_length,
                norm_min_percent=0.5, norm_max_percent=99.8)

    test_dict = {
        # dataset dependent parameters
        'patch_x': patch_xy,
        'patch_y': patch_xy,
        'patch_t': patch_t,
        'overlap_factor': overlap_factor,
        'scale_factor': 1,                   # the factor for image intensity scaling
        'test_datasize': test_datasize,
        'datasets_path': datasets_path,
        'pth_dir': pth_dir,                 # pth file root path
        'denoise_model': denoise_model,
        'output_dir': output_dir,         # result file root path
        # network related parameters
        'fmap': 16,                          # the number of feature maps
        'GPU': GPU,
        'num_workers': num_workers,
        'visualize_images_per_epoch': visualize_images_per_epoch,
        'save_test_images_per_epoch': save_test_images_per_epoch
    }
    # %%% Testing preparation
    # first we create a testing class object with the specified parameters
    tc = testing_class(test_dict)
    # start the testing process
    tc.run()


if __name__ == "__main__":
    main()
