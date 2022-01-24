import torch
from .base_model import BaseModel
from . import networks
import os
import pytorch_ssim

# get from modellib.__dict__.items(): , case not that matter
class Vis2PMmodel(BaseModel):
    """ This class implements visual images to PM mapping pairs with given physique parameters. Physiques how to use physique parameters and discrimnator is controlled in opt.
    The model training requires '--dataset_mode PM' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).
    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf

    history:  20.7.28 , pwrs.  adapted phy_converter, -> code.   expander to size.
    so the decode part will be fixed.
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='unet_256_pm', dataset_mode='pm')    # default net_G unet_256_pm
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        if opt.if_D =='D':  # only D option, gives the full net
            self.loss_names = ['G_L', 'G_GAN',  'D']     # combine just to D
        elif opt.if_D == 'nD':
            self.loss_names = ['G_L']
        else:
            print('please select the opt.if_D from [D|nD], exit()')
            quit(-1)
        if opt.lambda_sum >0 :       # if with add this to loss name
            self.loss_names += ['sum_L']
            self.if_sum = True
        else:
            self.if_sum = False
        if opt.lambda_ssim > 0:
            self.loss_names += ['ssim']

        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        self.if_D = opt.if_D
        self.type_L = opt.type_L
        self.opt = opt      # keep all opt
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']

        actiType = None
        if opt.if_actiFn == 'w':
            if opt.pmDsProc == 'clip11':        # dataset processing to 0-1 or 1-1
                actiType = 'tanh'
            elif opt.pmDsProc == 'clip01':
                actiType = 'sigmoid'
        elif 'wo' == opt.if_actiFn:
            actiType = None
        else:
            print('please choose w or wo for activation func')
            quit(-1)    # wrong activation
        print('activate is', actiType)
        self.netG = networks.pm_G(opt.input_nc, opt.output_nc, opt.ngf, n_stg=opt.n_stg, norm=opt.norm, use_dropout=not opt.no_dropout, init_type=opt.init_type, init_gain=opt.init_gain, gpu_ids=opt.gpu_ids, n_phy=opt.n_phy, phyMode=opt.phyMode, n_gateLayers=opt.n_gateLayers, actiType=actiType, sz_std=opt.crop_size, if_posInit=opt.if_posInit)
        # the attribute name corresponds to 'net_' + models_name[i]
        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)      # cross entropy or distance depending on gan mode
            if 'auto' == opt.type_whtL:
                # self.criterionL1 = networks.whtL1Loss   # pass function handle
                print('use auto weighted mode')
                ## make the new L loss func based on L type
                self.criterionL = networks.autoWtL(whtScal=opt.whtScal, clipMod=opt.pmDsProc, type_L=opt.type_L)  # base weight default 1, tar comes later
            elif 'pwrs' == opt.type_whtL:
                print('pixelwise resampling weighting')
                # get net
                self.criterionL = networks.pwrsWtL(type_L=opt.type_L)  # base weight default 1, tar comes later
            elif 'n' == opt.type_whtL:
                if 'L2' == opt.type_L:
                    self.criterionL = torch.nn.L1Loss()    # give a suitable L1
                elif 'L1' == opt.type_L:
                    self.criterionL = torch.nn.MSELoss()
                else:
                    print('no such L implementation', opt.type_L)
                    quit(-1)
            else:
                print('no such implementation {} yet'.format(opt.type_whtL))
                quit(-1)

            self.crit_ssim = pytorch_ssim.SSIM()  # SSIM score, higher better, for loss make it negative
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.phyVec = input['phyVec'].to(self.device)
        self.wt = input['wt_pwrs'].to(self.device)
        # get input here

        self.image_paths = input['A_paths' if AtoB else 'B_paths']  # path also included

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_Bs = self.netG(self.real_A, self.phyVec)  # G(A) model(x) similar,
        # self.fake_B = self.fake_Bs[-1]  # keep last one
        self.fake_B= self.fake_Bs[-1]   # keep last one

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        loss_D_li = []
        for fake_B in self.fake_Bs:
            fake_AB = torch.cat((self.real_A, fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
            pred_fake = self.netD(fake_AB.detach())     # why detatch, it will push it through fake session?  update D don't hurt G net!
            loss_D_fake = self.criterionGAN(pred_fake, False)
            # Real
            real_AB = torch.cat((self.real_A, self.real_B), 1)
            pred_real = self.netD(real_AB)  # forward D
            loss_D_real = self.criterionGAN(pred_real, True)
            # combine loss and calculate gradients
            loss_D = (loss_D_fake + loss_D_real) * 0.5
            loss_D_li.append(loss_D)
        self.loss_D = sum(loss_D_li)
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # accumulator
        loss_L_li = []
        loss_sum_li =[]
        loss_ssim_li = []
        loss_GAN_li = []
        # loss_G_li = []
        #
        for fake_B in self.fake_Bs:
            fake_AB = torch.cat((self.real_A, fake_B), 1)
            if self.opt.if_D == 'D':
                pred_fake = self.netD(fake_AB)
                # self.loss_G_GAN = self.criterionGAN(pred_fake, True)
                loss_GAN_li.append(self.criterionGAN(pred_fake, True) * self.opt.lambda_D)      # add lambda D here
            # self.loss_G_L = self.criterionL(self.fake_B, self.real_B) * self.opt.lambda_L
            if 'pwrs' == self.opt.type_whtL:
                loss_L_li.append(self.criterionL(fake_B, self.real_B, self.wt) * self.opt.lambda_L) #
            else:
                loss_L_li.append(self.criterionL(fake_B, self.real_B) * self.opt.lambda_L)
            # self.loss_G = self.loss_G_L + torch.tensor(0)  # sum use L as starting point, use add operation
            # loss_sum_L = torch.tensor(0)    # initialization
            if 'L1' == self.type_L:
                # change it to the phyVec one relation.  try it later.  read in and test later.
                if self.opt.if_phyMean == 'y':
                    loss_sum_li.append(torch.abs(fake_B.mean([2,3]) - self.real_B.mean([2,3])).sum() * self.opt.lambda_sum) # mean to avoid the resolution issue, -> sum proportional to batch)
                else:
                    loss_sum_li.append(torch.abs(fake_B.sum([2,3]) - self.real_B.sum([2,3])).sum() * self.opt.lambda_sum) # mean to avoid the resolution issue, -> sum proportional to batch)

            elif 'L2' == self.type_L:
                # loss_sum_L = (self.fake_B.sum() - self.real_B.sum())**2 # sum loss follow the L type
                # loss_sum_li.append(((fake_B.mean([2,3]) - self.real_B.mean([2,3]))**2).mean() * self.opt.lambda_sum)    # average to image, pixel, then diff, sqrt, then mean of such sqrt loss,
                if self.opt.if_phyMean == 'y':
                    loss_sum_li.append(((fake_B.mean([2,3]) - self.real_B.mean([2,3]))**2).sum() * self.opt.lambda_sum)    # average to image, pixel, then diff, sqrt, then mean of such sqrt loss,
                else:
                    loss_sum_li.append(((fake_B.sum([2,3]) - self.real_B.sum([2,3]))**2).sum() * self.opt.lambda_sum)    # average to image, pixel, then diff, sqrt, then mean of such sqrt loss,
            # self.loss_sum_L =loss_sum_L * self.opt.lambda_sum   # direct sum loss calculation
            # ssim the higher the better, so use neg
            loss_ssim_li.append((1-self.crit_ssim(fake_B, self.real_B)) * self.opt.lambda_ssim)
            # self.loss_ssim = - crit_ssim(self.fake_B, self.real_B) * self.opt.lambda_ssim
        self.loss_G_L = sum(loss_L_li)      # mandatory  regression loss
        # loss_G_li.append(self.loss_G_L)
        self.loss_G = self.loss_G_L     # initialize hold G_L value
        if self.if_sum:     # get all G loss accordingly
            # self.loss_G += self.loss_sum_L
            self.loss_sum_L = sum(loss_sum_li)
            self.loss_G = self.loss_G + self.loss_sum_L      # modify the original tensor  not good
        if self.opt.lambda_ssim:    # default 0 , if there is  add it.
            self.loss_ssim = sum(loss_ssim_li)
            self.loss_G = self.loss_G + self.loss_ssim
        if self.if_D == 'D':
            self.loss_G_GAN = sum(loss_GAN_li)
            self.loss_G = self.loss_G + self.loss_G_GAN
        self.loss_G.backward()       # all pushing back

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update D
        if self.if_D =='D':   # only when D is needed
            self.set_requires_grad(self.netD, True)  # enable backprop for D
            self.optimizer_D.zero_grad()     # set D's gradients to zero
            self.backward_D()                # calculate gradients for D
            self.optimizer_D.step()          # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights
