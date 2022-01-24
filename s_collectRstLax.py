'''
This script collects all the metric results save in rst.npz files and print it with tab into a txt file for import convenience. This version is also formed with & separation for lax format, it is also scaled and formated with need.
'''
import os
import argparse
import numpy as np

parser = argparse.ArgumentParser()
# parser.add_argument('--outNm', default='pwrs_RGB_lax', help='name for this results, I think you can set it here anyway as you have to set all the rest list down below')
parser.add_argument('--modSrc', default='RGB', help='the source mode')

rstFd = 'results'
opt, _ = parser.parse_known_args()
modSrc = opt.modSrc
# L1 version

## pwrs RGB
# outNm = 'pwrs_RGB'
# lambda_sum = 1e-6
# rst_li = [
#     'vis2PM_exp_uc_RGB-2-PMarray_n_phy1_stg3_whtL-n-100.0L2_lap0.001_sum0.0_ssim0.0_D0.0L3', # mse
#     'vis2PM_exp_uc_RGB-2-PMarray_n_phy1_stg3_whtL-pwrs-100.0L2_lap0.0_sum0.0_ssim0.0_D0.0L3', # pwrs0
#     'vis2PM_exp_uc_RGB-2-PMarray_n_phy1_stg3_whtL-pwrs-100.0L2_lap0.001_sum0.0_ssim0.0_D0.0L3', # pwrs
#     'vis2PM_exp_uc_RGB-2-PMarray_n_phy1_stg3_whtL-pwrs-100.0L2_lap0.001_sum1e-06_ssim0.0_D0.0L3', #pwrs sum
#     'vis2PM_exp_uc_RGB-2-PMarray_n_phy1_stg3_whtL-pwrs-100.0L2_lap0.001_sum1e-06_ssim10.0_D0.0L3', # pwrs+sum+sim
#     'vis2PM_exp_uc_RGB-2-PMarray_n_phy1_stg3_whtL-pwrs-100.0L2_lap0.001_sum1e-06_ssim10.0_D1.0L3', # pwrs+sum+sim+D
#     'vis2PM_exp_uc_RGB-2-PMarray_n_phy0_stg3_whtL-n-100.0L2_lap0.001_sum0.0_ssim0.0_D0.0L3', # phy0+n
#     'vis2PM_exp_uc_RGB-2-PMarray_n_phy1_stg3_whtL-n-100.0L2_lap0.001_sum1e-06_ssim0.0_D0.0L3', # sum
#     'vis2PM_exp_uc_RGB-2-PMarray_n_phy1_stg3_whtL-n-100.0L2_lap0.001_sum0.0_ssim10.0_D0.0L3', # sim
#     'vis2PM_exp_uc_RGB-2-PMarray_n_phy1_stg3_whtL-n-100.0L2_lap0.001_sum0.0_ssim0.0_D1.0L3', # D
#     'vis2PM_exp_uc_RGB-2-PMarray_n_phy2_stg3_whtL-pwrs-100.0L2_lap0.001_sum1e-06_ssim0.0_D0.0L3',  # pwrs sum2
#     'vis2PM_exp_uc_RGB-2-PMarray_n_phy3_stg3_whtL-pwrs-100.0L2_lap0.001_sum1e-06_ssim0.0_D0.0L3',  # pwrs sum3
#     'vis2PM_exp_uc_RGB-2-PMarray_n_phy10_stg3_whtL-pwrs-100.0L2_lap0.001_sum1e-06_ssim0.0_D0.0L3',  # pwrs sum10
#     'openPose_exp_uc_RGB-2-PMarray_epoch25_5_bch30',
#     'memNet_exp_uc_RGB-2-PMarray_epoch25_5_bch2',
#     'pix2pix_exp_uc_RGB-2-PMarray_epoch25_5_bch70',
#     'cycle_gan_exp_uc_RGB-2-PMarray_epoch25_5_bch3',
#     'vis2PM_exp_uc_RGB-2-PMarray_n_phy1_stg3_whtL-pwrs-100.0L2_lap0.001_sum100.0_ssim0.0_D0.0L3',
#     'vis2PM_exp_uc_RGB-2-PMarray_n_phy1_stg3_whtL-pwrs-100.0L2_lap0.001_sum10.0_ssim0.0_D0.0L3',
#     'vis2PM_exp_uc_RGB-2-PMarray_n_phy1_stg3_whtL-pwrs-100.0L2_lap0.001_sum1.0_ssim0.0_D0.0L3',
# ]
## pwrs IR
# outNm = 'pwrs_IR'
# outNm ='pwrs_sota_{}_lax'.format(modSrc) #
outNm ='stg_{}_lax'.format(modSrc) #
lambda_sum = 1e-6
rst_li = [
    # 'vis2PM_exp_uc_{}-2-PMarray_n_phy1_stg3_whtL-n-100.0L2_lap0.001_sum0.0_ssim0.0_D0.0L3'.format(modSrc), # mse
    ## 'vis2PM_exp_uc_{}-2-PMarray_n_phy1_stg3_whtL-pwrs-100.0L2_lap0.0_sum0.0_ssim0.0_D0.0L3'.format(modSrc), # pwrs0
    # 'vis2PM_exp_uc_{}-2-PMarray_n_phy1_stg3_whtL-pwrs-100.0L2_lap0.001_sum0.0_ssim0.0_D0.0L3'.format(modSrc), # pwrs
    'vis2PM_exp_uc_{}-2-PMarray_n_phy1_stg1_whtL-pwrs-100.0L2_lap0.001_sum1e-06_ssim0.0_D0.0L3'.format(modSrc), #pwrs sum
    'vis2PM_exp_uc_{}-2-PMarray_n_phy1_stg2_whtL-pwrs-100.0L2_lap0.001_sum1e-06_ssim0.0_D0.0L3'.format(modSrc), #pwrs sum
    'vis2PM_exp_uc_{}-2-PMarray_n_phy1_stg3_whtL-pwrs-100.0L2_lap0.001_sum1e-06_ssim0.0_D0.0L3'.format(modSrc), #pwrs sum
    # 'vis2PM_exp_uc_{}-2-PMarray_n_phy1_stg3_whtL-pwrs-100.0L2_lap0.001_sum1e-06_ssim10.0_D0.0L3'.format(modSrc), # pwrs+sum+sim
    # 'vis2PM_exp_uc_{}-2-PMarray_n_phy1_stg3_whtL-pwrs-100.0L2_lap0.001_sum1e-06_ssim10.0_D1.0L3'.format(modSrc), # pwrs+sum+sim+D
    ## 'vis2PM_exp_uc_{}-2-PMarray_n_phy0_stg3_whtL-n-100.0L2_lap0.001_sum0.0_ssim0.0_D0.0L3'.format(modSrc), # phy0+n
    # 'vis2PM_exp_uc_{}-2-PMarray_n_phy1_stg3_whtL-n-100.0L2_lap0.001_sum1e-06_ssim0.0_D0.0L3'.format(modSrc), # sum
    # 'vis2PM_exp_uc_{}-2-PMarray_n_phy1_stg3_whtL-n-100.0L2_lap0.001_sum0.0_ssim10.0_D0.0L3'.format(modSrc), # sim
    # 'vis2PM_exp_uc_{}-2-PMarray_n_phy1_stg3_whtL-n-100.0L2_lap0.001_sum0.0_ssim0.0_D1.0L3'.format(modSrc), # D
    # 'vis2PM_exp_uc_{}-2-PMarray_n_phy2_stg3_whtL-pwrs-100.0L2_lap0.001_sum1e-06_ssim0.0_D0.0L3'.format(modSrc),  # pwrs sum2
    # 'vis2PM_exp_uc_{}-2-PMarray_n_phy3_stg3_whtL-pwrs-100.0L2_lap0.001_sum1e-06_ssim0.0_D0.0L3'.format(modSrc),  # pwrs sum3
    # 'vis2PM_exp_uc_{}-2-PMarray_n_phy10_stg3_whtL-pwrs-100.0L2_lap0.001_sum1e-06_ssim0.0_D0.0L3'.format(modSrc),  # pwrs sum10
    # 'openPose_exp_uc_{}-2-PMarray_epoch25_5_bch30'.format(modSrc),
    # 'memNet_exp_uc_{}-2-PMarray_epoch25_5_bch2'.format(modSrc),
    # 'pix2pix_exp_uc_{}-2-PMarray_epoch25_5_bch70'.format(modSrc),
    # 'cycle_gan_exp_uc_{}-2-PMarray_epoch25_5_bch3'.format(modSrc),
]

# for the test li
# rst_li = ['vis2PM_exp_uc_{}-2-PMarray_n_phy0_stg3_whtL-pwrs-100.0L2_lap0.001_sum1e-06_ssim0.0_D0.0L3'.format(modSrc)]

# give the names for row header
# nms = [
#     'base',
#     'pwrs',
#     'pwrs-phy',
#     'pwrs-phy-ssim',
#     'pwrs-phy-ssim-D',
#     'phy',
#     'ssim',
#     'D'
# ]

# sota
# nms = [
#     'pwrs-phy',
#     'openPose',
#     'memNet',
#     'pix2pix',
#     'cycleGan',
# ]

# stgs
nms = [
    'stage1',
    'stage2',
    'stage3'
]


# whole settings
predNm = 'rst0.05.npz'
# name only for value retrieval not important for illustration
metricNms_scalForms = [     # metric nam, scale and format
    # ['mse', 1000, '{:.2f} \t'],
    ['mse_efs', 1000, '{:.2f} \t'],
    ['psnr', 1, '{:.2f} \t'],
    # ['psnr_efs',1 , '{:.3f} \t'],
    ['pcs_efs', 1, '{:.3f} \t'],
    ['pcs_efs01', 1, '{:.3f} \t'],
    ['ssim', 1, '{:.3f} \t'],
    # ['fp', 1, '{:.3f} \t'],
]

outNm = outNm + '.txt'   # given in file directly
outFd = os.path.join(rstFd, 'metricRsts')
if not os.path.exists(outFd):
    os.makedirs(outFd)

f = open(os.path.join(outFd, outNm), "w+")      # create new

for i, rstNm in enumerate(rst_li):
    nm_row = nms[i]
    rstPth = os.path.join(rstFd, rstNm, 'test_latest', predNm)
    rst = np.load(rstPth)
    f.write('{}\t'.format(nm_row + ' &'))  # head
    for j ,metric_scal_fmt in enumerate(metricNms_scalForms):
        metric = metric_scal_fmt[0]
        scale = metric_scal_fmt[1]
        fmt = metric_scal_fmt[2]
        f.write(fmt.format(rst[metric]*scale))
        if j<len(metricNms_scalForms)-1:
            f.write('&')

    f.write('\\\\ \n')
f.close()


