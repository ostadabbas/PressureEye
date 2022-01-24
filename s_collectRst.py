'''
This script collects all the metric results save in rst.npz files and print it with tab into a txt file for import convenience
'''
import os
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--outNm', default='L1_epoch25_5', help='name for this results, I think you can set it here anyway as you have to set all the rest list down below')
parser.add_argument('--modSrc', default='RGB', help='the source mode')

opt, _ = parser.parse_known_args()
rstFd = 'results'

# L1 version
# rst_li = [
# 'vis2PM_exp_uc_RGB-2-PM_clip01_w-align_n_phy1_3stg_woactiFn_100.0L1_n-whtL100_0.0sum_0.0ssim_nD3_epoch25_5_bch30',          # clip01
# 'vis2PM_exp_uc_RGB-2-PM_clip01_w-align_n_phy1_3stg_woactiFn_100.0L1_auto-whtL100_0.0sum_0.0ssim_nD3_epoch25_5_bch30',       # sa
# 'vis2PM_exp_uc_RGB-2-PM_clip01_w-align_n_phy1_3stg_woactiFn_100.0L1_auto-whtL100_0.0001sum_0.0ssim_nD3_epoch25_5_bch30',    # sa_sum
# 'vis2PM_exp_uc_RGB-2-PM_clip01_w-align_n_phy1_3stg_woactiFn_100.0L1_auto-whtL100_0.0001sum_10.0ssim_nD3_epoch25_5_bch30',   # sa_sum_ssim
# 'vis2PM_exp_uc_RGB-2-PM_clip01_w-align_n_phy1_3stg_woactiFn_100.0L1_auto-whtL100_0.0001sum_10.0ssim_D3_epoch25_5_bch20',    # sa_sum_ssim_D
# 'vis2PM_exp_uc_RGB-2-PM_clip01_w-align_n_phy0_3stg_woactiFn_100.0L1_n-whtL100_0.0sum_0.0ssim_nD3_epoch25_5_bch30',          # nPhy0
# 'vis2PM_exp_uc_RGB-2-PM_clip01_w-align_n_phy1_3stg_woactiFn_100.0L1_n-whtL100_0.0001sum_0.0ssim_nD3_epoch25_5_bch30',       # sum
# 'vis2PM_exp_uc_RGB-2-PM_clip01_w-align_n_phy1_3stg_woactiFn_100.0L1_n-whtL100_0.0sum_10.0ssim_nD3_epoch25_5_bch30',         # ssim
# 'vis2PM_exp_uc_RGB-2-PM_clip01_w-align_n_phy1_3stg_woactiFn_100.0L1_n-whtL100_0.0sum_0.0ssim_D3_epoch25_5_bch20',           # D
# ]

# L2 RGB
# outNm = 'RGBrst'
# rst_li = [
# 'vis2PM_exp_uc_RGB-2-PMarray_clip01_w-align_n_phy1_3stg_woactiFn_100.0L2_n-whtL100_0.0sum_0.0ssim_nD3_epoch25_5_bch30',          # clip01
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
# ]

## beta study
# outNm = 'beta_RGB'
# rst_li = [
# 'vis2PM_exp_uc_RGB-2-PMarray_clip01_w-align_n_phy1_3stg_woactiFn_100.0L2_auto-whtL100_10000.0sum_0.0ssim_nD3_epoch25_5_bch30',    # sa_sum
# 'vis2PM_exp_uc_RGB-2-PMarray_clip01_w-align_n_phy2_3stg_woactiFn_100.0L2_auto-whtL100_10000.0sum_0.0ssim_nD3_epoch25_5_bch30',    # sa_sum
# 'vis2PM_exp_uc_RGB-2-PMarray_clip01_w-align_n_phy3_3stg_woactiFn_100.0L2_auto-whtL100_10000.0sum_0.0ssim_nD3_epoch25_5_bch30',    # sa_sum
# 'vis2PM_exp_uc_RGB-2-PMarray_clip01_w-align_n_phy10_3stg_woactiFn_100.0L2_auto-whtL100_10000.0sum_0.0ssim_nD3_epoch25_5_bch30',    # sa_sum
# ]

# IR version
# outNm = 'IRrst'
# rst_li = [
# 'vis2PM_exp_uc_IR-2-PMarray_clip01_w-align_n_phy1_3stg_woactiFn_100.0L2_n-whtL100_0.0sum_0.0ssim_nD3_epoch25_5_bch30',          # clip01
# 'vis2PM_exp_uc_IR-2-PMarray_clip01_w-align_n_phy1_3stg_woactiFn_100.0L2_auto-whtL100_0.0sum_0.0ssim_nD3_epoch25_5_bch30',       # sa
# 'vis2PM_exp_uc_IR-2-PMarray_clip01_w-align_n_phy1_3stg_woactiFn_100.0L2_auto-whtL100_10000.0sum_0.0ssim_nD3_epoch25_5_bch30',    # sa_sum
# 'vis2PM_exp_uc_IR-2-PMarray_clip01_w-align_n_phy1_3stg_woactiFn_100.0L2_auto-whtL100_10000.0sum_10.0ssim_nD3_epoch25_5_bch30',   # sa_sum_ssim
# 'vis2PM_exp_uc_IR-2-PMarray_clip01_w-align_n_phy1_3stg_woactiFn_100.0L2_auto-whtL100_10000.0sum_10.0ssim_D3_epoch25_5_bch20',    # sa_sum_ssim_D
# 'vis2PM_exp_uc_IR-2-PMarray_clip01_w-align_n_phy0_3stg_woactiFn_100.0L2_n-whtL100_0.0sum_0.0ssim_nD3_epoch25_5_bch30',          # nPhy0
# 'vis2PM_exp_uc_IR-2-PMarray_clip01_w-align_n_phy1_3stg_woactiFn_100.0L2_n-whtL100_10000.0sum_0.0ssim_nD3_epoch25_5_bch30',       # sum
# 'vis2PM_exp_uc_IR-2-PMarray_clip01_w-align_n_phy1_3stg_woactiFn_100.0L2_n-whtL100_0.0sum_10.0ssim_nD3_epoch25_5_bch30',         # ssim
# 'vis2PM_exp_uc_IR-2-PMarray_clip01_w-align_n_phy1_3stg_woactiFn_100.0L2_n-whtL100_0.0sum_0.0ssim_D3_epoch25_5_bch20',           # D
# 'openPose_exp_uc_IR-2-PMarray_epoch25_5_bch30',
#     'memNet_exp_uc_IR-2-PMarray_epoch25_5_bch2',
#     'pix2pix_exp_uc_IR-2-PMarray_epoch25_5_bch70',
#     'cycle_gan_exp_uc_IR-2-PMarray_epoch25_5_bch3',
# ]

## IR beta study
# outNm = 'beta_IR'
# rst_li = [
# 'vis2PM_exp_uc_IR-2-PMarray_clip01_w-align_n_phy1_3stg_woactiFn_100.0L2_auto-whtL100_10000.0sum_0.0ssim_nD3_epoch25_5_bch30',    # sa_sum
# 'vis2PM_exp_uc_IR-2-PMarray_clip01_w-align_n_phy2_3stg_woactiFn_100.0L2_auto-whtL100_10000.0sum_0.0ssim_nD3_epoch25_5_bch30',    # sa_sum
# 'vis2PM_exp_uc_IR-2-PMarray_clip01_w-align_n_phy3_3stg_woactiFn_100.0L2_auto-whtL100_10000.0sum_0.0ssim_nD3_epoch25_5_bch30',    # sa_sum
# 'vis2PM_exp_uc_IR-2-PMarray_clip01_w-align_n_phy10_3stg_woactiFn_100.0L2_auto-whtL100_10000.0sum_0.0ssim_nD3_epoch25_5_bch30',    # sa_sum
# ]


# RGB2IR version
# IR version
# outNm = 'RGB2IRrst'
# rst_li = [
# 'vis2PM_exp_uc_RGB-2-IR_clip01_w-align_n_phy1_3stg_woactiFn_100.0L2_n-whtL100_0.0sum_0.0ssim_nD3_epoch25_5_bch30',          # clip01
# 'vis2PM_exp_uc_RGB-2-IR_clip01_w-align_n_phy1_3stg_woactiFn_100.0L2_auto-whtL100_0.0sum_0.0ssim_nD3_epoch25_5_bch30',       # sa
# 'vis2PM_exp_uc_RGB-2-IR_clip01_w-align_n_phy1_3stg_woactiFn_100.0L2_auto-whtL100_10000.0sum_0.0ssim_nD3_epoch25_5_bch30',    # sa_sum
# 'vis2PM_exp_uc_RGB-2-IR_clip01_w-align_n_phy1_3stg_woactiFn_100.0L2_auto-whtL100_10000.0sum_10.0ssim_nD3_epoch25_5_bch30',   # sa_sum_ssim
# 'vis2PM_exp_uc_RGB-2-IR_clip01_w-align_n_phy1_3stg_woactiFn_100.0L2_auto-whtL100_10000.0sum_10.0ssim_D3_epoch25_5_bch20',    # sa_sum_ssim_D
# 'vis2PM_exp_uc_RGB-2-IR_clip01_w-align_n_phy0_3stg_woactiFn_100.0L2_n-whtL100_0.0sum_0.0ssim_nD3_epoch25_5_bch30',          # nPhy0
# 'vis2PM_exp_uc_RGB-2-IR_clip01_w-align_n_phy1_3stg_woactiFn_100.0L2_n-whtL100_10000.0sum_0.0ssim_nD3_epoch25_5_bch30',       # sum
# 'vis2PM_exp_uc_RGB-2-IR_clip01_w-align_n_phy1_3stg_woactiFn_100.0L2_n-whtL100_0.0sum_10.0ssim_nD3_epoch25_5_bch30',         # ssim
# 'vis2PM_exp_uc_RGB-2-IR_clip01_w-align_n_phy1_3stg_woactiFn_100.0L2_n-whtL100_0.0sum_0.0ssim_D3_epoch25_5_bch20',           # D
# 'openPose_exp_uc_RGB-2-IR_epoch25_5_bch30',
#     'memNet_exp_uc_RGB-2-IR_epoch25_5_bch2',
#     'pix2pix_exp_uc_RGB-2-IR_epoch25_5_bch70',
#     'cycle_gan_exp_uc_RGB-2-IR_epoch25_5_bch3',
# ]

## pwrs RGB
# outNm = 'pwrs_RGB'
modSrc = opt.modSrc
lambda_sum = 1e-6
## pwrs {}
outNm = 'pwrs_{}'.format(modSrc)
rst_li = [
    'vis2PM_exp_uc_{}-2-PMarray_n_phy1_stg3_whtL-n-100.0L2_lap0.001_sum0.0_ssim0.0_D0.0L3'.format(modSrc), # mse
    'vis2PM_exp_uc_{}-2-PMarray_n_phy1_stg3_whtL-pwrs-100.0L2_lap0.0_sum0.0_ssim0.0_D0.0L3'.format(modSrc), # pwrs0
    'vis2PM_exp_uc_{}-2-PMarray_n_phy1_stg3_whtL-pwrs-100.0L2_lap0.001_sum0.0_ssim0.0_D0.0L3'.format(modSrc), # pwrs
    'vis2PM_exp_uc_{}-2-PMarray_n_phy1_stg3_whtL-pwrs-100.0L2_lap0.001_sum1e-06_ssim0.0_D0.0L3'.format(modSrc), #pwrs sum
    'vis2PM_exp_uc_{}-2-PMarray_n_phy1_stg3_whtL-pwrs-100.0L2_lap0.001_sum1e-06_ssim10.0_D0.0L3'.format(modSrc), # pwrs+sum+sim
    'vis2PM_exp_uc_{}-2-PMarray_n_phy1_stg3_whtL-pwrs-100.0L2_lap0.001_sum1e-06_ssim10.0_D1.0L3'.format(modSrc), # pwrs+sum+sim+D
    # 'vis2PM_exp_uc_{}-2-PMarray_n_phy0_stg3_whtL-n-100.0L2_lap0.001_sum0.0_ssim0.0_D0.0L3'.format(modSrc), # phy0+n
    'vis2PM_exp_uc_{}-2-PMarray_n_phy1_stg3_whtL-n-100.0L2_lap0.001_sum1e-06_ssim0.0_D0.0L3'.format(modSrc), # sum
    'vis2PM_exp_uc_{}-2-PMarray_n_phy1_stg3_whtL-n-100.0L2_lap0.001_sum0.0_ssim10.0_D0.0L3'.format(modSrc), # sim
    'vis2PM_exp_uc_{}-2-PMarray_n_phy1_stg3_whtL-n-100.0L2_lap0.001_sum0.0_ssim0.0_D1.0L3'.format(modSrc), # D 8th
    # 'vis2PM_exp_uc_{}-2-PMarray_n_phy0_stg3_whtL-pwrs-100.0L2_lap0.001_sum1e-06_ssim0.0_D0.0L3'.format(modSrc),   # pwrs sum_nPhy0
    'vis2PM_exp_uc_{}-2-PMarray_n_phy2_stg3_whtL-pwrs-100.0L2_lap0.001_sum1e-06_ssim0.0_D0.0L3'.format(modSrc),  # pwrs sum_nPhy2
    'vis2PM_exp_uc_{}-2-PMarray_n_phy3_stg3_whtL-pwrs-100.0L2_lap0.001_sum1e-06_ssim0.0_D0.0L3'.format(modSrc),  # pwrs sum3
    'vis2PM_exp_uc_{}-2-PMarray_n_phy10_stg3_whtL-pwrs-100.0L2_lap0.001_sum1e-06_ssim0.0_D0.0L3'.format(modSrc),  # pwrs sum_nPhy10
    # 'openPose_exp_uc_{}-2-PMarray_epoch25_5_bch30'.format(modSrc),  # 13
    # 'memNet_exp_uc_{}-2-PMarray_epoch25_5_bch2'.format(modSrc),
    # 'pix2pix_exp_uc_{}-2-PMarray_epoch25_5_bch70'.format(modSrc),
    # 'cycle_gan_exp_uc_{}-2-PMarray_epoch25_5_bch3'.format(modSrc),
    'vis2PM_exp_uc_{}-2-PMarray_n_phy1_stg1_whtL-pwrs-100.0L2_lap0.001_sum1e-06_ssim0.0_D0.0L3'.format(modSrc),     # pwrs sum
    'vis2PM_exp_uc_{}-2-PMarray_n_phy1_stg2_whtL-pwrs-100.0L2_lap0.001_sum1e-06_ssim0.0_D0.0L3'.format(modSrc),     # pwrs sum
]



# whole settings
predNm = 'rst0.05.npz'
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

args = parser.parse_args()
# outNm = args.outNm + '.txt'
outNm = outNm + '.txt'   # given in file directly


outFd = os.path.join(rstFd, 'metricRsts')
if not os.path.exists(outFd):
    os.makedirs(outFd)

f = open(os.path.join(outFd, outNm), "w+")      # create new

for rstNm in rst_li:
    rstPth = os.path.join(rstFd, rstNm, 'test_latest', predNm)
    if rstNm:
        rst = np.load(rstPth)
        for metric in metricNms:
            f.write('{}\t'.format(rst[metric]))
    f.write('\n')
f.close()


