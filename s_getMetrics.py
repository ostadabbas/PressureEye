# get all metrics (MSE, MSE_efs, PCS0.05 and PCS0.01 of test_diff result.
# old version obsolete.XXXXXXXXXXXXXXXXX
import os
import numpy as np

import util.utils_PM as utils_PM

# pre setting
rstFd = 'results'
senseRg = 2
clipMod = 1     # 1 for clip11 0 for clip 01
PMthreshRt = 0.05
PCSrt1 = 0
PCSrt2 = 0.1
rgPCSrt = 0.1
if clipMod:
    print('use clip11 mode')
    bs_sensing = -1
    rg_sensing = 2
else:
    bs_sensing = 0
    rg_sensing = 1

PM_thresh = bs_sensing + rg_sensing * PMthreshRt
# PCS = opt.PCS * rg_sensing  # miss base here? only absolute thresh here
# rg_PCS = opt.rg_PCS * rg_sensing
############# AR for cycleGAN  ************
# rst_li = [
#     'cycle_gan_uc_unm_IR-2-PMarray_clip11_D3_epoch25_5'
# ]
# legend_li = ['cycleGAN']


################ RGB test
# rst_li = [
#     'vis2PM_exp_RGB-2-PMarray_phy-concat-1_nGated1_whtMode-n-100_wo-sumLoss100_clip11_w-actiFn_nD3_epoch25_5_bch70',
#     'vis2PM_exp_RGB-2-PMarray_phy-concat-1_nGated1_whtMode-autoWht-100_wo-sumLoss100_clip11_w-actiFn_nD3_epoch25_5_bch70',
#     'vis2PM_exp_RGB-2-PMarray_phy-concat-1_nGated1_whtMode-autoWht-100_w-sumLoss100_clip11_w-actiFn_nD3_epoch25_5_bch70',
#     'vis2PM_exp_RGB-2-PMarray_phy-concat-1_nGated1_whtMode-autoWht-100_w-sumLoss100_clip11_w-actiFn_D3_epoch25_5_bch70',
#     'vis2PM_exp_RGB-2-PMarray_phy-concat-0_nGated1_whtMode-n-100_wo-sumLoss100_clip11_w-actiFn_nD3_epoch25_5_bch70',
#     'vis2PM_exp_RGB-2-PMarray_phy-concat-1_nGated1_whtMode-n-100_w-sumLoss100_clip11_w-actiFn_nD3_epoch25_5_bch70',
#     'vis2PM_exp_RGB-2-PMarray_phy-concat-1_nGated1_whtMode-n-100_wo-sumLoss100_clip11_w-actiFn_D3_epoch25_5_bch70',
#     'pix2pix_exp_RGB-2-PMarray_clip11_D3_epoch25_5_bch70',
#     'cycle_gan_exp_uc_RGB-2-PMarray_clip11_D3_epoch25_5'
# ]

#-- for IR
rst_li = [
    'vis2PM_uc_unm_IR-2-PMarray_phy-concat-1_nGated1_whtMode-n-100_wo-sumLoss100_clip11_w-actiFn_nD3_epoch25_5_bch70',
    'vis2PM_uc_unm_IR-2-PMarray_phy-concat-1_nGated1_whtMode-autoWht-100_wo-sumLoss100_clip11_w-actiFn_nD3_epoch25_5_bch70',
    'vis2PM_uc_unm_IR-2-PMarray_phy-concat-1_nGated1_whtMode-autoWht-100_w-sumLoss100_clip11_w-actiFn_nD3_epoch25_5_bch70',
    'vis2PM_uc_unm_IR-2-PMarray_phy-concat-1_nGated1_whtMode-autoWht-100_w-sumLoss100_clip11_w-actiFn_D3_epoch25_5_bch70',
'vis2PM_uc_unm_IR-2-PMarray_phy-concat-0_nGated1_whtMode-n-100_wo-sumLoss100_clip11_w-actiFn_nD3_epoch25_5_bch70',
    'vis2PM_uc_unm_IR-2-PMarray_phy-concat-1_nGated1_whtMode-n-100_w-sumLoss100_clip11_w-actiFn_nD3_epoch25_5_bch70',
    'vis2PM_uc_unm_IR-2-PMarray_phy-concat-1_nGated1_whtMode-n-100_wo-sumLoss100_clip11_w-actiFn_D3_epoch25_5_bch70',
    'pix2pix_uc_unm_IR-2-PMarray_clip11_D3_epoch25_5_bch70',
    'cycle_gan_uc_unm_IR-2-PMarray_clip11_D3_epoch25_5'
]

legend_li = [
    'base',
    'autoW',
    'autoW-sum',
    'autoW-sum-D',
    'noBeta',
    'sum',
    'D',
    'pix2pix',
    'cycleGAN'
]
print('{:>12} {:>12} {:>12} {:>12} {:>12}'.format('model','PCS{}'.format(PCSrt1), 'PCS{}'.format(PCSrt2), 'MSE', 'MSE_efs'))

for i, name in enumerate(rst_li):
    pth = os.path.join(rstFd, name, 'test_latest', 'test_diff.npz')
    data = np.load(pth)
    diff_dStk = data['diff_dStk']
    real_dStk = data['real_dStk']
    PCStest = 0.1
    acc1 = (diff_dStk[real_dStk > PM_thresh] < PCSrt1*rg_sensing).sum() / diff_dStk[real_dStk > PM_thresh].size
    acc2 = (diff_dStk[real_dStk > PM_thresh] < PCSrt2*rg_sensing).sum() / diff_dStk[real_dStk > PM_thresh].size
    mse = (diff_dStk ** 2).mean()
    mse_efs = (diff_dStk[real_dStk > PM_thresh] ** 2).mean()
    # print('number {} name is {}'.format(i, name))
    # print('{}: final test PCS {} test accuracy is {:4f}, PCS{} is {:4f}, MSE is {:4f}, MSE_efs is {:4f}'.format(legend_li[i], PCSrt1, acc1, PCSrt2, acc2, mse,mse_efs))
    print('{:>12} {:>12.4f} {:>12.4f} {:>12.8f} {:>12.8f}'.format(legend_li[i], acc1, acc2, mse, mse_efs))