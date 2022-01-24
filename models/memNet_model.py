import torch
from .base_model import BaseModel
from . import networks
import os
import pytorch_ssim
import torch.nn as nn

## memnet implementation
class MemNet(nn.Module):
    def __init__(self, in_channels, channels, num_memblock, num_resblock):
        super(MemNet, self).__init__()
        self.feature_extractor = BNReLUConv(in_channels, channels, True)  # FENet: staic(bn)+relu+conv1
        self.reconstructor = BNReLUConv(channels, in_channels, True)  # ReconNet: static(bn)+relu+conv
        self.dense_memory = nn.ModuleList(
            [MemoryBlock(channels, num_resblock, i + 1) for i in range(num_memblock)]
        )
        # ModuleList can be indexed like a regular Python list, but modules it contains are
        # properly registered, and will be visible by all Module methods.

        self.weights = nn.Parameter((torch.ones(1, num_memblock) / num_memblock), requires_grad=True)  # same weights
        # output1,...,outputn corresponding w1,...,w2

    # Multi-supervised MemNet architecture
    def forward(self, x):
        residual = x
        out = self.feature_extractor(x)
        w_sum = self.weights.sum(1)
        mid_feat = []  # A lsit contains the output of each memblock
        ys = [out]  # A list contains previous memblock output(long-term memory)  and the output of FENet
        for memory_block in self.dense_memory:
            out = memory_block(out, ys)  # out is the output of GateUnit  channels=64
            mid_feat.append(out);
        # pred = Variable(torch.zeros(x.shape).type(dtype),requires_grad=False)
        pred = (self.reconstructor(mid_feat[0]) + residual) * self.weights.data[0][0] / w_sum
        for i in range(1, len(mid_feat)):
            pred = pred + (self.reconstructor(mid_feat[i]) + residual) * self.weights.data[0][i] / w_sum
        return pred
    # Base MemNet architecture
    '''
    def forward(self, x):
        residual = x   #input data 1 channel
        out = self.feature_extractor(x)
        ys = [out]  #A list contains previous memblock output and the output of FENet
        for memory_block in self.dense_memory:
            out = memory_block(out, ys)
        out = self.reconstructor(out)
        out = out + residual

        return out
    '''


class MemoryBlock(nn.Module):
    """Note: num_memblock denotes the number of MemoryBlock currently"""

    def __init__(self, channels, num_resblock, num_memblock):
        super(MemoryBlock, self).__init__()
        self.recursive_unit = nn.ModuleList(
            [ResidualBlock(channels) for i in range(num_resblock)]
        )
        # self.gate_unit = BNReLUConv((num_resblock+num_memblock) * channels, channels, True)  #kernel 3x3
        self.gate_unit = GateUnit((num_resblock + num_memblock) * channels, channels, True)  # kernel 1x1

    def forward(self, x, ys):
        """ys is a list which contains long-term memory coming from previous memory block
        xs denotes the short-term memory coming from recursive unit
        """
        xs = []
        residual = x
        for layer in self.recursive_unit:
            x = layer(x)
            xs.append(x)

        # gate_out = self.gate_unit(torch.cat([xs,ys], dim=1))
        gate_out = self.gate_unit(torch.cat(xs + ys, 1))  # where xs and ys are list, so concat operation is xs+ys
        ys.append(gate_out)
        return gate_out


class ResidualBlock(torch.nn.Module):
    """ResidualBlock
    introduced in: https://arxiv.org/abs/1512.03385
    x - Relu - Conv - Relu - Conv - x
    """

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.relu_conv1 = BNReLUConv(channels, channels, True)
        self.relu_conv2 = BNReLUConv(channels, channels, True)

    def forward(self, x):
        residual = x
        out = self.relu_conv1(x)
        out = self.relu_conv2(out)
        out = out + residual
        return out


class BNReLUConv(nn.Sequential):
    def __init__(self, in_channels, channels, inplace=True):
        super(BNReLUConv, self).__init__()
        self.add_module('bn', nn.BatchNorm2d(in_channels))
        self.add_module('relu',
                        nn.ReLU(inplace=inplace))  # tureL: direct modified x, false: new object and the modified
        self.add_module('conv',
                        nn.Conv2d(in_channels, channels, 3, 1, 1))  # bias: defautl: ture on pytorch, learnable bias


class GateUnit(nn.Sequential):
    def __init__(self, in_channels, channels, inplace=True):
        super(GateUnit, self).__init__()
        self.add_module('bn', nn.BatchNorm2d(in_channels))
        self.add_module('relu', nn.ReLU(inplace=inplace))
        self.add_module('conv', nn.Conv2d(in_channels, channels, 1, 1, 0))

# memnet wrapper
# get from modellib.__dict__.items(): , case not that matter
class MemNetModel(BaseModel):
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
        self.netG = MemNet(opt.output_nc, 64, 6, 6)  # add residue, so only same channel is possible.
        self.netG = networks.init_net(self.netG, gpu_ids=opt.gpu_ids)    # turn to gpu
        # self.netG = networks.init_net(self.netG, gpu_ids=opt.gpu_ids)
        # the attribute name corresponds to 'net_' + models_name[i]
        if self.isTrain:

            # self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            # follow original work's hyperparameters,  hardwired here not in proj opt.
            # lr 0.1  mmt 0.9 and wdc 1e-4
            self.optimizer_G = torch.optim.SGD(self.netG.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)  #
            self.optimizers.append(self.optimizer_G)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        # make it to single channel,
        self.real_A = input['A' if AtoB else 'B'].mean(1, keepdim=True).to(self.device)       # make color to gray for PM
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.phyVec = input['phyVec'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']  # path also included

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        # self.fake_B = self.netG(self.real_A, self.phyVec)  # G(A) model(x) similar,
        self.fake_B = self.netG(self.real_A)


    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator

        self.loss_G_L = self.criterionL(self.fake_B, self.real_B)
        self.loss_G_L.backward()       # all pushing back
    def optimize_parameters(self):
        self.forward()
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        # there is clipping oeration in original work   default to 0.4
        nn.utils.clip_grad_norm_(self.netG.parameters(), 0.4)
        self.optimizer_G.step()             # udpate G's weights
