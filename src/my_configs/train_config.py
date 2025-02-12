
import argparse


class trainConfig():
    '''
        This class is used to store the training configurations.
    '''
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self, parser):
        ## basic parameters
        parser.add_argument('--max_epoch', type=int, default=800, help='maximum epochs')
        parser.add_argument('--eval_freq', type=int, default=10, help='evaluation frequency')
        parser.add_argument('--gpu_id', type=str, default='1', help='gpu id')
        
        ## dataset path & output path & dataloader
        parser.add_argument('--train_path', type=str, default='./Data/Imaging_for_3DUNet_long/train', help='the path of training dataset')
        parser.add_argument('--val_path', type=str, default='./Data/Imaging_for_3DUNet_long/val', help='the path of validation dataset')
        parser.add_argument('--output_dir', type=str, default='./output', help='the path of output')
        parser.add_argument('--batch_size', type=int, default=None, help='batch size')
        parser.add_argument('--isDrop', type=bool, default=False, help='whether to drop the last batch')

        ## model parameters
        parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
        parser.add_argument('--classes', type=int, default=4, help='number of classes')
        parser.add_argument('--weights_loss', type=list, nargs='+', default=[1., 8., 3., 3.], help='weights for class loss')
        parser.add_argument('--mod', type=str, default='long_seg', help='model type, chosen from "long_seg" and "single_seg", others are invalid')
        parser.add_argument('--ignore_bg_label', type=bool, default=True, help='Whether to ignore the background label when computing loss')


        self.initialized = True
        return parser