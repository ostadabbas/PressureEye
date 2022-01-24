'''
Order the peak values of the dataset, put in order so we can select with better effect

'''
import os
import argparse
import numpy as np
import math
import pytorch_ssim
import torch

parser = argparse.ArgumentParser()
# parser.add_argument('--outNm', default='RGB', help='name for this results, I think you can set it here anyway as you have to set all the rest list down below')   # direct wire is better
parser.add_argument('--efs_rt', type=float, default=0.1,
                    help='the thresh for pressure needed to be taken into consideration')
parser.add_argument('--R', type=float, default=1,
                    help='dynamic range of the data, cycleGan and pix2pix employs central normalization')
parser.add_argument('--pcs_test', type=float, default=0.05, help='PCS ratio for print out')

### users setting
rstFd = 'results'
step = 256  #  or the resolution
# L2 RGB version
outNm = 'idxRGB'
# for current list
# rst_li = [
#     'vis2PM_ts1_uc_RGB-2-PMarray_clip01_w-align_phy1_3stg_woactiFn_100.0L2_auto-whtL100_1e-07sum_10.0ssim_D3_epoch1_0_bch20'
# ]

rst_li = [
'vis2PM_exp_uc_RGB-2-PMarray_clip01_w-align_n_phy1_3stg_woactiFn_100.0L2_n-whtL100_0.0sum_0.0ssim_nD3_epoch25_5_bch30',          # clip01
# 'vis2PM_exp_uc_RGB-2-PMarray_clip01_w-align_n_phy1_3stg_woactiFn_100.0L2_auto-whtL100_0.0sum_0.0ssim_nD3_epoch25_5_bch30',       # sa
# 'vis2PM_exp_uc_RGB-2-PMarray_clip01_w-align_n_phy1_3stg_woactiFn_100.0L2_auto-whtL100_10000.0sum_0.0ssim_nD3_epoch25_5_bch30',    # sa_sum
# 'vis2PM_exp_uc_RGB-2-PMarray_clip01_w-align_n_phy1_3stg_woactiFn_100.0L2_auto-whtL100_10000.0sum_10.0ssim_nD3_epoch25_5_bch30',   # sa_sum_ssim
# 'vis2PM_exp_uc_RGB-2-PMarray_clip01_w-align_n_phy1_3stg_woactiFn_100.0L2_auto-whtL100_10000.0sum_10.0ssim_D3_epoch25_5_bch20',    # sa_sum_ssim_D
# 'vis2PM_exp_uc_RGB-2-PMarray_clip01_w-align_n_phy0_3stg_woactiFn_100.0L2_n-whtL100_0.0sum_0.0ssim_nD3_epoch25_5_bch30',          # nPhy0
# 'vis2PM_exp_uc_RGB-2-PMarray_clip01_w-align_n_phy1_3stg_woactiFn_100.0L2_n-whtL100_10000.0sum_0.0ssim_nD3_epoch25_5_bch30',       # sum
# 'vis2PM_exp_uc_RGB-2-PMarray_clip01_w-align_n_phy1_3stg_woactiFn_100.0L2_n-whtL100_0.0sum_10.0ssim_nD3_epoch25_5_bch30',         # ssim
# 'vis2PM_exp_uc_RGB-2-PMarray_clip01_w-align_n_phy1_3stg_woactiFn_100.0L2_n-whtL100_0.0sum_0.0ssim_D3_epoch25_5_bch20',           # D
# 'openPose_exp_uc_RGB-2-PMarray_epoch25_5_bch30',
# 'memNet_exp_uc_RGB-2-PMarray_epoch25_5_bch2',
#     'pix2pix_exp_uc_RGB-2-PMarray_epoch25_5_bch70',
#     'cycle_gan_exp_uc_RGB-2-PMarray_epoch25_5_bch3',
]

### params
metricNms = [
    'mse',
    'mse_efs',
    'psnr',
    'psnr_efs',
    'pcs_efs',
    'pcs_efs01',
    'ssim',
    'fp'
]
R2_li = ['pix2pix', 'cycle_gan']
outFd = os.path.join(rstFd, 'metricRsts')
if not os.path.exists(outFd):
    os.makedirs(outFd)

### scripts
args = parser.parse_args()
efs_rt = args.efs_rt
pcs_test = args.pcs_test

outNm = outNm + '_efs' + str(efs_rt) + '.txt'   # outNm_efs[val].txt eg RGB_efs0.1.txt
f = open(os.path.join(outFd, outNm), "w+")

for rstNm in rst_li:
    rstPth = os.path.join(rstFd, rstNm, 'test_latest', 'test_diffV2.npz')
    expFd = os.path.join(rstFd, rstNm, 'test_latest')  # the experiment folder
    dataL = np.load(rstPth)
    fake_vStk = dataL['fake_vStk']
    real_vStk = dataL['real_vStk']
    # diff_vStk = np.abs(fake_vStk - real_vStk)
    max_li = real_vStk.max(axis=(1, 2, 3)).tolist()    # keep only batch dim
    order_li = sorted(range(len(max_li)), key = lambda k: max_li[k], reverse=True)

    sup_li = [e for e in order_li if math.floor(e)%3 ==0]
    L_li = [e for e in order_li if math.floor(e)%3 ==1]
    R_li = [e for e in order_li if math.floor(e)%3 ==2]
    # write into files.
    for li in [sup_li, L_li, R_li]:
        for e in li:
            f.write('{}\t'.format(e))
        f.write('\n')
print('write idx order in supine left right order with decreasing peak value in ', os.path.join(outFd, outNm))
f.close()


