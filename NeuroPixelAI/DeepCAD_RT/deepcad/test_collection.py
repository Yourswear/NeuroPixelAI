import os
import numpy as np
# import yaml
from .network import Network_3D_Unet
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import time
# import datetime
from .data_process import test_preprocess_chooseOne, testset, multibatch_test_save, singlebatch_test_save
# from skimage import io
# from DeepCAD_RT.deepcad.movie_display import test_img_display
from tifffile import imread,TiffWriter


class testing_class():
    """
    Class implementing testing process
    """

    def __init__(self, params_dict):
        """
        Constructor class for testing process

        Args:20230
           params_dict: dict
               The collection of testing params set by users
        Returns:
           self

        """
        self.overlap_factor = 0.5
        self.datasets_path = ''
        self.fmap = 16
        self.output_dir = './results'
        self.pth_dir = ''
        self.batch_size = 1
        self.patch_t = 150
        self.patch_x = 150
        self.patch_y = 150
        self.gap_y = 115
        self.gap_x = 115
        self.gap_t = 115
        self.GPU = '0'
        self.ngpu = 1
        self.num_workers = 0
        self.scale_factor = 1
        self.test_datasize = 400
        self.denoise_model = ''
        self.visualize_images_per_epoch = False
        self.save_test_images_per_epoch = False
        self.colab_display = False
        self.result_display = ''
        self.set_params(params_dict)


    def run(self,reg_image):
        """
        General function for testing DeepCAD network.
        """
        # # create some essential file for result storage
        # self.prepare_file()
        # get models for processing
        self.read_modellist()
        # save some essential testing parameters in para.yaml
        self.save_yaml_test()
        # initialize denoise network with testing parameters.
        self.initialize_network()
        # specifies the GPU for the testing program.
        self.distribute_GPU()
        # start testing and result visualization during testing period (optional)
        enhance_image = self.test(reg_image)
        return enhance_image

    # def prepare_file(self):
    #     """
    #     Make data folder to store testing results
    #     Important Fields:
    #         self.datasets_name: the sub folder of the dataset
    #         self.pth_path: the folder for pth file storage
    #
    #     """
    #     if self.datasets_path[-1]!='/':
    #        self.datasets_name=self.datasets_path.split("/")[-1]
    #     else:
    #        self.datasets_name=self.datasets_path.split("/")[-2]
    #     # self.datasets_name = self.datasets_path.split("/", 1)[1]
    #     # pth_name = self.datasets_name + '_' + datetime.datetime.now().strftime("%Y%m%d%H%M")
    #     # self.pth_path = self.pth_dir + '/' + pth_name
    #     # if not os.path.exists(self.pth_path):
    #     #     os.makedirs(self.pth_path)
    #
    #     if not os.path.exists(self.output_dir):
    #         os.mkdir(self.output_dir)
    #     current_time = datetime.datetime.now().strftime("%Y%m%d%H%M")
    #     self.output_path = self.output_dir + '//' + 'DataFolderIs_' + self.datasets_name + '_' + current_time + '_ModelFolderIs_' + self.denoise_model
    #     if not os.path.exists(self.output_path):
    #         os.mkdir(self.output_path)

    def set_params(self, params_dict):
        """
        Set the params set by user to the testing class object and calculate some default parameters for testing

        """
        for key, value in params_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)

        self.gap_x = int(self.patch_x * (1 - self.overlap_factor))  # patch gap in x
        self.gap_y = int(self.patch_y * (1 - self.overlap_factor))  # patch gap in y
        self.gap_t = int(self.patch_t * (1 - self.overlap_factor))  # patch gap in t
        self.ngpu = str(self.GPU).count(',') + 1  # check the number of GPU used for testing
        self.batch_size = self.ngpu  # By default, the batch size is equal to the number of GPU for minimal memory consumption
        print('\033[1;31mTesting parameters -----> \033[0m')
        print(self.__dict__)


    def read_modellist(self):

        model_path = self.pth_dir + '//' + self.denoise_model
        model_list = list(os.walk(model_path, topdown=False))[-1][-1]
        model_list.sort()

        # calculate the number of model file
        count_pth = 0
        for i in range(len(model_list)):
            aaa = model_list[i]
            if '.pth' in aaa:
                count_pth = count_pth + 1
        self.model_list = model_list
        self.model_list_length = count_pth

    def initialize_network(self):
        """
        Initialize U-Net 3D network, which is the main network architecture of DeepCAD

        Important Fields:
           self.fmap: the number of the feature map in U-Net 3D network.
           self.local_model: the denoise network

        """
        denoise_generator = Network_3D_Unet(in_channels=1,
                                            out_channels=1,
                                            f_maps=self.fmap,
                                            final_sigmoid=True)
        self.local_model = denoise_generator

    def save_yaml_test(self):
        """
        Save some essential params in para.yaml.

        """
        para = {'datasets_path': 0, 'test_datasize': 0, 'denoise_model': 0,
                'output_dir': 0, 'pth_dir': 0, 'GPU': 0, 'batch_size': 0,
                'patch_x': 0, 'patch_y': 0, 'patch_t': 0, 'gap_y': 0, 'gap_x': 0,
                'gap_t': 0, 'fmap': 0, 'scale_factor': 0, 'overlap_factor': 0}
        # para["datasets_path"] = self.datasets_path
        para["denoise_model"] = self.denoise_model
        para["test_datasize"] = self.test_datasize
        #para["output_dir"] = self.output_dir
        para["pth_dir"] = self.pth_dir
        para["GPU"] = self.GPU
        para["batch_size"] = self.batch_size
        para["patch_x"] = self.patch_x
        para["patch_y"] = self.patch_y
        para["patch_t"] = self.patch_t
        para["gap_x"] = self.gap_x
        para["gap_y"] = self.gap_y
        para["gap_t"] = self.gap_t
        para["fmap"] = self.fmap
        para["scale_factor"] = self.scale_factor
        para["overlap_factor"] = self.overlap_factor


    def distribute_GPU(self):
        """
        Allocate the GPU for the testing program. Print the using GPU information to the screen.
        For acceleration, multiple GPUs parallel testing is recommended.

        """
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.GPU)
        if torch.cuda.is_available():
            self.local_model = self.local_model.cuda()
            self.local_model = nn.DataParallel(self.local_model, device_ids=range(self.ngpu))
            print('\033[1;31mUsing {} GPU(s) for testing -----> \033[0m'.format(torch.cuda.device_count()))
        cuda = True if torch.cuda.is_available() else False
        # Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    def test(self,reg_image):
        """
        Pytorch testing workflow

        """
        pth_count=0
        for pth_index in range(len(self.model_list)):
            aaa = self.model_list[pth_index]
            if '.pth' in aaa:
                pth_count=pth_count+1
                pth_name = self.model_list[pth_index]

                # load model
                model_name = self.pth_dir + '//' + self.denoise_model + '//' + pth_name
                if isinstance(self.local_model, nn.DataParallel):
                    self.local_model.module.load_state_dict(torch.load(model_name))  # parallel
                    self.local_model.eval()
                else:
                    self.local_model.load_state_dict(torch.load(model_name))  # not parallel
                    self.local_model.eval()
                self.local_model.cuda()
                self.print_img_name = False
                # test all stacks
                #for N in range(len(self.img_list)):
                #name_list, noise_img, coordinate_list,test_im_name, img_mean, input_data_type = test_preprocess_chooseOne(self, reg_image)
                name_list, reg_image, img_mean, input_data_type, coordinate_list = test_preprocess_chooseOne(self, reg_image)
                # print(len(name_list))
                prev_time = time.time()
                time_start = time.time()
                denoise_img = np.zeros(reg_image.shape)
                input_img = np.zeros(reg_image.shape)

                test_data = testset(name_list, coordinate_list, reg_image)
                testloader = DataLoader(test_data, batch_size=self.batch_size, shuffle=False,
                                        num_workers=self.num_workers)
                for iteration, (noise_patch, single_coordinate) in enumerate(testloader):
                    noise_patch = noise_patch.cuda()
                    real_A = noise_patch

                    real_A = Variable(real_A)
                    fake_B = self.local_model(real_A)

                    # Determine approximate time left
                    batches_done = iteration
                    batches_left = 1 * len(testloader) - batches_done
                    time_left_seconds = int(batches_left * (time.time() - prev_time))
                    # time_left = datetime.timedelta(seconds=time_left_seconds)
                    prev_time = time.time()

                    if iteration % 1 == 0:
                        time_end = time.time()
                        time_cost = time_end - time_start  # datetime.timedelta(seconds= (time_end - time_start))
                        print(
                            '\r[Model %d/%d, %s]  [Patch %d/%d] [Time Cost: %.0d s] [ETA: %s s]     '
                            % (
                                pth_count,
                                self.model_list_length,
                                pth_name,
                                iteration + 1,
                                len(testloader),
                                time_cost,
                                time_left_seconds
                            ), end=' ')

                    if (iteration + 1) % len(testloader) == 0:
                        print('\n', end=' ')

                    # Enhanced sub-stacks are sequentially output from the network
                    output_image = np.squeeze(fake_B.cpu().detach().numpy())
                    raw_image = np.squeeze(real_A.cpu().detach().numpy())

                    if (output_image.ndim == 3):
                        postprocess_turn = 1
                    else:
                        postprocess_turn = output_image.shape[0]

                    # The final enhanced stack can be obtained by stitching all sub-stacks.
                    if (postprocess_turn > 1):
                        for id in range(postprocess_turn):
                            output_patch, raw_patch, stack_start_w, stack_end_w, stack_start_h, stack_end_h, stack_start_s, stack_end_s = multibatch_test_save(
                                single_coordinate, id, output_image, raw_image)
                            output_patch=output_patch+img_mean
                            raw_patch=raw_patch+img_mean
                            denoise_img[stack_start_s:stack_end_s, stack_start_h:stack_end_h,
                            stack_start_w:stack_end_w] \
                                = output_patch * (np.sum(raw_patch) / np.sum(output_patch)) ** 0.5
                            input_img[stack_start_s:stack_end_s, stack_start_h:stack_end_h,
                            stack_start_w:stack_end_w] \
                                = raw_patch
                    else:
                        output_patch, raw_patch, stack_start_w, stack_end_w, stack_start_h, stack_end_h, stack_start_s, stack_end_s = singlebatch_test_save(
                            single_coordinate, output_image, raw_image)
                        output_patch=output_patch+img_mean
                        raw_patch=raw_patch+img_mean
                        denoise_img[stack_start_s:stack_end_s, stack_start_h:stack_end_h, stack_start_w:stack_end_w] \
                            = output_patch * (np.sum(raw_patch) / np.sum(output_patch)) ** 0.5
                        input_img[stack_start_s:stack_end_s, stack_start_h:stack_end_h, stack_start_w:stack_end_w] \
                            = raw_patch

                # Stitching finish
                output_img = denoise_img.squeeze().astype(np.float32) * self.scale_factor
                del denoise_img


                # Normalize and display inference image
                # if (self.visualize_images_per_epoch):
                #     print('Displaying the denoised file ----->')
                #     display_length = 200
                #     test_img_display(output_img, display_length=display_length, norm_min_percent=1,
                #                      norm_max_percent=99)

                # Save inference image
                if (self.save_test_images_per_epoch):
                    if input_data_type == 'uint16':
                        output_img=np.clip(output_img, 0, 65535)
                        output_img = output_img.astype('uint16')

                    elif input_data_type == 'int16':
                        output_img=np.clip(output_img, -32767, 32767)
                        output_img = output_img.astype('int16')

                    else:
                        output_img = output_img.astype('int32')
        return output_img

        print('denoise finished.')
