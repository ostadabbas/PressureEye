import argparse
import os
from util import util
import torch
import models
import data


class BaseOptions():
    """This class defines options used during both training and test time.

    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
    """

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def initialize(self, parser):
        """Define the common options that are used in both training and test."""
        # basic parameters
        parser.add_argument('--dataroot', default=r'/home/liu.shu/datasets/SLP/danaLab', help='path to images (should have subfolders trainA, trainB, valA, valB, etc). for PM point to danaLab directly')
        parser.add_argument('--name', type=str, default='exp', help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        # model parameters
        parser.add_argument('--model', type=str, default='vis2PM', help='chooses which model to use. [cycle_gan | pix2pix | test | colorization | vis2PM ].')
        parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels: 3 for RGB and 1 for grayscale')
        parser.add_argument('--output_nc', type=int, default=1, help='# of output image channels: 3 for RGB and 1 for grayscale')
        parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in the last conv layer')
        parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in the first conv layer')
        parser.add_argument('--netD', type=str, default='basic', help='specify discriminator architecture [basic | n_layers | pixel]. The basic model is a 70x70 PatchGAN. n_layers allows you to specify the layers in the discriminator')
        parser.add_argument('--netG', type=str, default='resnet_9blocks', help='specify generator architecture [resnet_9blocks | resnet_6blocks | unet_256 | unet_128]')
        parser.add_argument('--n_layers_D', type=int, default=3, help='only used if netD==n_layers')
        parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization [instance | batch | none]')
        parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal | xavier | kaiming | orthogonal]')
        parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
        parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
        # dataset parameters
        parser.add_argument('--dataset_mode', type=str, default='pm', help='chooses how datasets are loaded. [unaligned | aligned | single | colorization | pm]')
        parser.add_argument('--direction', type=str, default='AtoB', help='AtoB or BtoA')
        parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        parser.add_argument('--num_threads', default=4, type=int, help='# threads for loading data')
        parser.add_argument('--batch_size', type=int, default=60, help='input batch size, default 1')
        parser.add_argument('--load_size', type=int, default=286, help='scale images to this size')
        parser.add_argument('--crop_size', type=int, default=256, help='then crop to this size, or the std input size')
        parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        parser.add_argument('--preprocess', type=str, default='resize_and_crop', help='scaling and cropping of images at load time [resize_and_crop | crop | scale_width | scale_width_and_crop | none], for pm [scale shift_rotate ], default will scale to 255, any combine will ran random transform')
        parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')
        parser.add_argument('--display_winsize', type=int, default=256, help='display window size for both visdom and HTML')
        # additional parameters
        parser.add_argument('--epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--load_iter', type=int, default='0', help='which iteration to load? if load_iter > 0, the code will load models by iter_[load_iter]; otherwise, the code will load models by [epoch]')
        parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
        # v1  pwrs
        parser.add_argument('--suffix', default='n_phy{n_phy}_stg{n_stg}_whtL-{type_whtL}-{lambda_L}{type_L}_lap{lambda_lap}_sum{lambda_sum}_ssim{lambda_ssim}_D{lambda_D}L{n_layers_D}', type=str, help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{load_size}_phy-{phyMode}')
        # data
        parser.add_argument('--mod_src', nargs='+', default=['RGB'], help='source modality list, can accept multiple modalities typical model [RGB|IR|depthRaw| PMarray]')
        parser.add_argument('--mod_tar', nargs='+', default=['PMarray'], help='target modality list')
        parser.add_argument('--n_phy', type=int, default=1, help='how many physics parameters to pass in, 1 for weight only, the physique order changed to w,gen,h... in dataset interface already, we can choose 0,1, or 10 generally')
        parser.add_argument('--depthPreproc', default='normalize',
                            help='depth preprocessin method, [Normalize | HHA| ...], normalize to bed height and depth range, deprecated ')
        parser.add_argument('--cov_li', nargs='+', default=['uncover'],
                            help='the cover condition list for data loading')
        parser.add_argument('--pmDsProc', default='clip01',
                            help='how to process pmDs data, could be cliped to 0~1 or -1 to 1 [clip01|clip11]')
        parser.add_argument('--if_align', default='w',
                            help='if align images or not')
        # pm loss
        parser.add_argument('--type_L', default='L2', help='loss type, [L1|L2], default MSE for regression, pwsr only l2 or pwsr')
        parser.add_argument('--type_whtL', default='n', help='the mode for L loss weight [auto| n| pwrs], two modes right now, can be extended')
        parser.add_argument('--whtScal', default=100, help='rescale normalized target value to weight L1 loss')
        parser.add_argument('--h_mode', type=int, default=0, help='histogram mode: 0:image base, 1:global, 2:gau_filted')
        parser.add_argument('--h_base', type=float, default=1.0, help='the histogram base, pseudo sample this much, the histogram base ')
        parser.add_argument('--lambda_sum', type=float, default=0.0, help='coefficent for sum loss')
        parser.add_argument('--lambda_L', type=float, default=100.0, help='weight for L loss, L1 or L2 shared')
        parser.add_argument('--lambda_ssim', type=float, default=0.0, help='weight for ssim loss')
        parser.add_argument('--lambda_D', type=float, default=0.0, help='weight for D loss, tune the GAN loss in G updating')
        parser.add_argument('--lambda_lap', type=float, default=0.001, help='weight of the laplacian smoothing')
        parser.add_argument('--kdeMode', type=int, default=0, help='kde mode add blurry to the ')
        parser.add_argument('--sig_kde', type=float, default=1., help='kde guassian kernel sigma')
        parser.add_argument('--if_phyMean', default='y', help='if use mean for weight equality, otherwise use sum')

        # net structure
        parser.add_argument('--n_stg', type=int, default=3, help='how many stages to build for visPM model ')
        parser.add_argument('--if_actiFn', default='wo', help='choose [w|wo] for the final layer activation function,  depends on the clip mode, it will give sigmoid or tanh for activation')
        parser.add_argument('--phyMode', default='enc', help='physique parameters injection mode for enc, effect for vis2PM only, [concat|fc_gated | sim_gated | sfg|dcfg], fc_gated: use fc pointwise gating, work with n_gateLayers together. sim_gated: simple times the weight scalar directly, sfg simple final simple gated, just scale everything in image to save value dcfg1:final layer nn followed by decov, final one suppose to work with sigmoid model, enc: encoder to cat')      # not very helpful , just  concat is good.  add additional layers.
        parser.add_argument('--n_gateLayers', type=int, default=1, help='how many gate layers for the physique input, used for the fc_gated of phyMode')
        parser.add_argument('--if_normPhy', type=str, default='wo', help='if normalize the physical vectors')
        parser.add_argument('--if_posInit', type=str, default='w', help='if use the [w|wo] positive init operation for sfg layer only')
        # testing specification
        parser.add_argument('--if_test', type=str, default='n', help='if this is only for test, otherwise run the training session')
        parser.add_argument('--n_testPM', type=int, default=12, help='how many subject is usded for test')
        parser.add_argument('--rg_PCS', type=float, default=0.1, help='the PCS test range around center')
        parser.add_argument('--pcs_test', type=float, default=0.05, help='PCS ratio for print out')
        parser.add_argument('--efs_rt', type=float, default=0.05, help='the thresh for pressure needed to be taken into consideration')
        parser.add_argument('--n_train', type=int, default=-1, help ='the max trained numbers in each epoch, -1 not restriction')
        parser.add_argument('--num_test_in_train', type=int, default=50, help='how many test samples needed during training')
        parser.add_argument('--num_test', type=int, default=-1, help='how many test images to run after training, -1 to run all')
        parser.add_argument('--num_imgSv', type=int, default=500, help='how many images to save during final test')
        parser.add_argument('--if_saveDiff', type=str, default='y', help='if save the diff result for PM test')
        parser.add_argument('--if_saveImg', type=str, default='y', help='if save images for testPM session')
        parser.add_argument('--if_saveWhtCmb', type=str, default='n', help='if save images for testPM session')
        # --- from other places
        parser.add_argument('--niter', type=int, default=25, help='# of iter (should be epoches) at starting learning rate')
        parser.add_argument('--niter_decay', type=int, default=5, help='# of iter  (epoches) to linearly decay learning rate to zero')
        parser.add_argument('--predNm', default='test_diffV2.npz', help='the prediction output result. V2 is used to differ from original with diffferent content')
        # parser.add_argument('--modSrc_li', nargs='+', help)
        self.initialized = True
        return parser

    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        if not self.initialized:  # check if it has been initialized, next call second time, parser will not be available, wrong logic?
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)
        # get the basic options
        opt, _ = parser.parse_known_args()
        # modify model-related parser options
        model_name = opt.model
        model_option_setter = models.get_option_setter(model_name)  # get modify option static function
        parser = model_option_setter(parser, self.isTrain)
        opt, _ = parser.parse_known_args()  # parse again with new defaults
        # modify dataset-related parser options
        dataset_name = opt.dataset_mode
        dataset_option_setter = data.get_option_setter(dataset_name)    # add more options
        parser = dataset_option_setter(parser, self.isTrain) # make checking complicated, put in one file is easier I think.
        # save and return the parser
        self.parser = parser
        opt, _ = parser.parse_known_args()
        # return parser.parse_args()
        return opt

    def print_options(self, opt):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        util.mkdirs(expr_dir)   # all option will mk dirs
        file_name = os.path.join(expr_dir, '{}_opt.txt'.format(opt.phase))
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device.
        calculate additional settings from basic input
        """
        opt = self.gather_options()  # initialize and parse all options out
        opt.isTrain = self.isTrain   # train or test, set gpu, return opt

        # updating  opts
        if 0 == opt.lambda_D:
            opt.if_D = 'nD'
        else:
            opt.if_D = 'D'

        # name forming
        covStr = ''
        if 'uncover' in opt.cov_li:
            covStr += 'u'
        if 'cover1' in opt.cov_li:
            covStr += '1'
        if 'cover2' in opt.cov_li:
            covStr += '2'
        covStr += 'c'  # cover suffix
        opt.name = opt.name + '_' + covStr  # add cover string to name
        # model comes first
        opt.name = opt.model + '_' + opt.name
        if 'pm' == opt.dataset_mode:  # if pm, update the ch_in/out
            opt.input_nc = len(opt.mod_src)
            if 'RGB' in opt.mod_src:
                opt.input_nc += 2
            print('input channel is', opt.input_nc)
            opt.output_nc = len(opt.mod_tar)
            if 'RGB' in opt.mod_tar:
                opt.output_nc += 2
            opt.name = opt.name + '_' +'-'.join(opt.mod_src + ['2'] + opt.mod_tar)
        # process opt.suffix
        if opt.suffix:
            suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else '' # vars return __dict__ attribute
            opt.name = opt.name + suffix
        self.print_options(opt)
        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])
        self.opt = opt
        return self.opt
