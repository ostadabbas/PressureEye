# for project test purpose
import numpy as np
import os
from options.train_options import TrainOptions
from options.test_options import TestOptions
import util.utils_PM as utils_PM
# project modules
import time
from options.train_options import TrainOptions
from data import create_dataset # load the corresponding class according to opts
from models import create_model
from util.visualizer import Visualizer
import torch
from skimage import io
import matplotlib.pyplot as plt
import cv2
from util import utils_PM
from scipy.ndimage import gaussian_filter
import skimage.io as io
import os.path as osp
import json


# utils_PM test
# imgPth = utils_PM.getPth()
# print(imgPth)

# physique parameters and sub matrix
# dsFd = r'G:\My Drive\ACLab Shared GDrive\datasetPM\danaLab'
# physiques = np.load(os.path.join(dsFd, 'physiqueData.npy'))
# # print(physiques[0])     # all numbers 1 for male npy format
# phy_sub = physiques[[0,2,4],:][:,[0,2,4]]   # index by two times
# print(physiques[:5,:5])
# print(phy_sub)

# reparsing test
# opt_train = TrainOptions().parse()      # options can be parsed multiple times
# opt_test = TestOptions().parse()


if __name__ == '__main__':
    opt = TrainOptions().parse()
    opt_test = TestOptions().parse()
    out_fd = 'output_tmp'

    opt.phase = 'test'
    # opt = TestOptions().parse()
    if 'pm' == opt.dataset_mode: # if pm, update the ch_in/out
        opt.input_nc = len(opt.mod_src)
        opt.output_nc = len(opt.mod_tar)
    n_bch = 3

    ## # presetting
    # dataset = create_dataset(opt)  # test phase
    # # # model = create_model(opt)
    # # # model.setup(opt)  # regular setup: load and print networks; create schedulers
    # #
    # # ## test dataset
    # idx = 3
    # idx_subj = idx//135
    # sample = dataset.dataset.__getitem__(idx) # idx from 0  'A','B','A_paths', 'B_paths'
    # print('haha')
    # A = sample['A'] # A is only 2D not 3D
    # B = sample['B']#  PM
    # wt = sample['wt_pwrs']
    # nm_cm = plt.get_cmap('jet')

    ## check the hist
    with open('output_tmp/hist_pwrs100.json','r') as f:
        rst = json.load(f)
    hist_ave = np.array(rst['hist_ave'])
    print(hist_ave)
    hist_ave_gau = gaussian_filter(hist_ave,sigma=1)
    bins = np.arange(len(hist_ave))
    st_hist = 0
    print(hist_ave_gau)

    hist_show = hist_ave
    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)
    # ax1.bar(bins[st_hist:] + 0.5, hist_ave[st_hist:], width=0.7)
    # ax2.bar(bins[st_hist:] + 0.5, hist_ave_gau[st_hist:], width=0.7)
    ax1.plot(1/(hist_show[st_hist:]+1))
    ax2.plot(1/(hist_show[st_hist:]+5))      # tune the curve

    plt.show()
    ## check the weight
    # print('A shape', A.shape)
    # imgA = utils_PM.ts2Img(A, if_bch=False)    #3 256 256
    # imgB = utils_PM.ts2Img(B, nm_cm=nm_cm, if_bch=False)
    # wt = wt.numpy().squeeze()
    # wt_gau = gaussian_filter(wt, sigma=3)   # sig 3 pix
    # cm = plt.get_cmap('jet')
    # wt_clr = cm(wt)
    # wt_gau_clr = cm(wt_gau)
    # io.imsave(osp.join(out_fd, 'wt.png'), wt_clr)
    # io.imsave(osp.join(out_fd, 'wt_gau.png'), wt_gau_clr)
    # print('image saved in ', out_fd)

    # colored_image = cm(RGB)     # 255 x 255 x3 x4  ( seems each channel to a 4 channel with alpha)

    # cv2.imshow('test', colored_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    #
    # phyVec = sample['phyVec']
    # print('sum is', B.sum())  # 96 .452
    # print('wt 0 is', phyVec[0])
    # print('rt is ', phyVec[0]/B.sum())

    # print('check physiques')
    # print(phyVec)
    # print(physiques[idx_subj])
    # print('ds length', dataset.dataset.__len__())
    # plt.figure()
    # plt.imshow(IR)
    # plt.figure()
    # plt.imshow(depthRaw)
    # plt.figure()
    # plt.imshow(PM)
    # plt.show()

    # iteration test
    # src = torch.randn(n_bch,1,256,256)
    # tar = torch.ones(n_bch, 1,256,256)
    # phyVec = torch.randn(n_bch, 10)
    # inp = {'A':src, 'B':tar, 'A_paths':['Apath'] * n_bch, 'B_paths':['B_paths']*n_bch, 'phyVec':phyVec}
    # for i in range(10):
    #     model.set_input(inp)
    #     model.optimize_parameters()
    #     losses = model.get_current_losses()
    #     print('at iteration {}, loss is'.format(i), losses.items())


    ## test result data # good
    # expNm = 'vis2PM_ts1_uc_RGB-2-PMarray_clip01_w-align_phy1_3stg_woactiFn_100.0L2_auto-whtL100_1e-07sum_10.0ssim_D3_epoch1_0_bch20'
    # rstFd= os.path.join('results', expNm, 'test_latest')
    # rst = np.load(os.path.join(rstFd, 'rst0.05.npz'))
    # test_diff = np.load(os.path.join(rstFd, 'test_diffV2.npz'))
    # print(test_diff.keys())