import torch
from .base_model import BaseModel
from . import networks
from collections import OrderedDict

import torch
import torch.nn as nn

def make_layers(block, no_relu_layers):
    layers = []
    for layer_name, v in block.items():     # layer names in here
        if 'pool' in layer_name:
            layer = nn.MaxPool2d(kernel_size=v[0], stride=v[1],
                                    padding=v[2])
            layers.append((layer_name, layer))
        else:
            conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1],
                               kernel_size=v[2], stride=v[3],
                               padding=v[4])
            layers.append((layer_name, conv2d))
            if layer_name not in no_relu_layers:
                layers.append(('relu_'+layer_name, nn.ReLU(inplace=True)))

    return nn.Sequential(OrderedDict(layers))

class bodypose_model(nn.Module):
    '''
    This is a modified version of open pose for pm estimation.
    PAF branches are pruned which is not available in pm task. So only branch 1 left.
    Confidence map is reduced to ch_out instead of 19 joint heatmaps.
    '''
    def __init__(self, ch_in=3, ch_out=1):
        super(bodypose_model, self).__init__()

        # these layers have no relu layer
        no_relu_layers = ['conv5_5_CPM_L1', 'conv5_5_CPM_L2', 'Mconv7_stage2_L1',\
                          'Mconv7_stage2_L2', 'Mconv7_stage3_L1', 'Mconv7_stage3_L2',\
                          'Mconv7_stage4_L1', 'Mconv7_stage4_L2', 'Mconv7_stage5_L1',\
                          'Mconv7_stage5_L2', 'Mconv7_stage6_L1', 'Mconv7_stage6_L1']
        blocks = {}
        block0 = OrderedDict({'conv1_1': [ch_in, 64, 3, 1, 1],  # in_c, out_c, kernel , stride,  padding
                  'conv1_2': [64, 64, 3, 1, 1],
                  'pool1_stage1': [2, 2, 0],    # kenrel stride, padd
                  'conv2_1': [64, 128, 3, 1, 1],
                  'conv2_2': [128, 128, 3, 1, 1],
                  'pool2_stage1': [2, 2, 0],        # kernel stride padding
                  'conv3_1': [128, 256, 3, 1, 1],
                  'conv3_2': [256, 256, 3, 1, 1],
                  'conv3_3': [256, 256, 3, 1, 1],
                  'conv3_4': [256, 256, 3, 1, 1],
                  'pool3_stage1': [2, 2, 0],
                  'conv4_1': [256, 512, 3, 1, 1],
                  'conv4_2': [512, 512, 3, 1, 1],
                  'conv4_3_CPM': [512, 256, 3, 1, 1],
                  'conv4_4_CPM': [256, 128, 3, 1, 1]})      # 128 output suppose to be VGG structure


        # Stage 1
        block1_1 = OrderedDict({'conv5_1_CPM_L1': [128, 128, 3, 1, 1],
                    'conv5_2_CPM_L1': [128, 128, 3, 1, 1],
                    'conv5_3_CPM_L1': [128, 128, 3, 1, 1],
                    'conv5_4_CPM_L1': [128, 512, 1, 1, 0],
                    'conv5_5_CPM_L1': [512, ch_out, 1, 1, 0]})      # ch_out is 38 in original work

        # block1_2 = OrderedDict({'conv5_1_CPM_L2': [128, 128, 3, 1, 1],
        #             'conv5_2_CPM_L2': [128, 128, 3, 1, 1],
        #             'conv5_3_CPM_L2': [128, 128, 3, 1, 1],
        #             'conv5_4_CPM_L2': [128, 512, 1, 1, 0],
        #             'conv5_5_CPM_L2': [512, 19, 1, 1, 0]})
        blocks['block1_1'] = block1_1
        # blocks['block1_2'] = block1_2

        self.model0 = make_layers(block0, no_relu_layers)

        # Stages 2 - 6
        for i in range(2, 7):
            blocks['block%d_1' % i] = OrderedDict({
                'Mconv1_stage%d_L1' % i: [128+ch_out, 128, 7, 1, 3],   # 128 + 19 + 38     so need to cut off
                'Mconv2_stage%d_L1' % i: [128, 128, 7, 1, 3],
                'Mconv3_stage%d_L1' % i: [128, 128, 7, 1, 3],
                'Mconv4_stage%d_L1' % i: [128, 128, 7, 1, 3],
                'Mconv5_stage%d_L1' % i: [128, 128, 7, 1, 3],
                'Mconv6_stage%d_L1' % i: [128, 128, 1, 1, 0],
                'Mconv7_stage%d_L1' % i: [128, ch_out, 1, 1, 0]})

            # blocks['block%d_2' % i] = OrderedDict({
            #     'Mconv1_stage%d_L2' % i: [185, 128, 7, 1, 3],
            #     'Mconv2_stage%d_L2' % i: [128, 128, 7, 1, 3],
            #     'Mconv3_stage%d_L2' % i: [128, 128, 7, 1, 3],
            #     'Mconv4_stage%d_L2' % i: [128, 128, 7, 1, 3],
            #     'Mconv5_stage%d_L2' % i: [128, 128, 7, 1, 3],
            #     'Mconv6_stage%d_L2' % i: [128, 128, 1, 1, 0],
            #     'Mconv7_stage%d_L2' % i: [128, 19, 1, 1, 0]})       # confidence map only

        for k in blocks.keys():
            blocks[k] = make_layers(blocks[k], no_relu_layers)

        self.model1_1 = blocks['block1_1']
        self.model2_1 = blocks['block2_1']
        self.model3_1 = blocks['block3_1']
        self.model4_1 = blocks['block4_1']
        self.model5_1 = blocks['block5_1']
        self.model6_1 = blocks['block6_1']
        self.ups = nn.Upsample(scale_factor=8, mode='bilinear')
        # branch 2 deprecated. we change the PAF branch for PM
        # self.model1_2 = blocks['block1_2']
        # self.model2_2 = blocks['block2_2']
        # self.model3_2 = blocks['block3_2']
        # self.model4_2 = blocks['block4_2']
        # self.model5_2 = blocks['block5_2']
        # self.model6_2 = blocks['block6_2']


    def forward(self, x):
        # ups = self.ups
        # outs = []
        out1 = self.model0(x)
        # outs.append(self.upSample(out1))

        out1_1 = self.model1_1(out1)
        # out1_2 = self.model1_2(out1)
        # out2 = torch.cat([out1_1, out1_2, out1], 1)
        out2 = torch.cat([out1_1, out1],1)

        out2_1 = self.model2_1(out2)
        # out2_2 = self.model2_2(out2)
        # out3 = torch.cat([out2_1, out2_2, out1], 1)
        out3 = torch.cat([out2_1, out1], 1)

        out3_1 = self.model3_1(out3)
        # out3_2 = self.model3_2(out3)
        # out4 = torch.cat([out3_1, out3_2, out1], 1)
        out4 = torch.cat([out3_1, out1], 1)

        out4_1 = self.model4_1(out4)
        # out4_2 = self.model4_2(out4)
        # out5 = torch.cat([out4_1, out4_2, out1], 1)
        out5 = torch.cat([out4_1, out1],1)

        out5_1 = self.model5_1(out5)
        # out5_2 = self.model5_2(out5)
        # out6 = torch.cat([out5_1, out5_2, out1], 1)
        out6 = torch.cat([out5_1, out1], 1)

        out6_1 = self.model6_1(out6)
        # out6_2 = self.model6_2(out6)
        outs = [out1_1, out2_1, out3_1, out4_1, out5_1, out6_1]      # return a list s
        outs = list(map(self.ups, outs))
        return outs

class OpenPoseModel(BaseModel):
    """ This class is a wrapper of memNet from original work of pytorch implementation.
    network updaing is wrapped as a function, intermediate result is saved in class to interface with
    our shared training code .
    The model training requires '--dataset_mode PM' dataset.
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.
        nothingi happen for memNet
        """

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        self.loss_names = ['G_L']

        self.visual_names = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        self.opt = opt      # keep all opt
        self.model_names = ['G']
        self.criterionL = torch.nn.MSELoss(size_average=False)
        # self.netG = MemNet(opt.output_nc, 64, 6, 6)  # add residue, so only same channel is possible.
        self.netG = bodypose_model(opt.input_nc, opt.output_nc)
        self.netG = networks.init_net(self.netG, gpu_ids=opt.gpu_ids)    # turn to gpu
        # the attribute name corresponds to 'net_' + models_name[i]
        if self.isTrain:
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            # follow original work's hyperparameters,  hardwired here not in proj opt.
            # lr 0.1  mmt 0.9 and wdc 1e-4
            # self.optimizer_G = torch.optim.SGD(self.netG.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)  #
            self.optimizers.append(self.optimizer_G)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        # make it to single channel,
        self.real_A = input['A' if AtoB else 'B'].to(self.device)       # make color to gray for PM
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.phyVec = input['phyVec'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']  # path also included

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        # self.fake_B = self.netG(self.real_A, self.phyVec)  # G(A) model(x) similar,
        self.outs = self.netG(self.real_A)
        self.fake_B = self.outs[-1]


    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        self.loss_G_L = torch.tensor(0.0).cuda()     # empty
        for out in self.outs:
            self.loss_G_L += self.criterionL(out, self.real_B)
        # self.loss_G_L = self.criterionL(self.fake_B, self.real_B)
        self.loss_G_L.backward()       # all pushing back
    def optimize_parameters(self):
        self.forward()
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        # there is clipping oeration in original work   default to 0.4
        # nn.utils.clip_grad_norm_(self.netG.parameters(), 0.4)
        self.optimizer_G.step()             # udpate G's weights
