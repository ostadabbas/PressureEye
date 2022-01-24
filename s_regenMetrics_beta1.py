'''
re generate metric result based on different threshold, with generated testing results.  Especially kesi, for the PCS and fp calculation.
regen rst.npz, you can rename it for backcup purpose
rename to outNm_efs[val].npz

'''
import os
import argparse
import numpy as np
import math
import pytorch_ssim
import torch

parser = argparse.ArgumentParser()
# parser.add_argument('--outNm', default='RGB', help='name for this results, I think you can set it here anyway as you have to set all the rest list down below')   # direct wire is better
parser.add_argument('--efs_rt', type=float, default=0.05,
                    help='the thresh for pressure needed to be taken into consideration')
parser.add_argument('--R', type=float, default=1,
                    help='dynamic range of the data, cycleGan and pix2pix employs central normalization')
parser.add_argument('--pcs_test', type=float, default=0.05, help='PCS ratio for print out')

### users setting
rstFd = 'results'

# L2 RGB version
# outNm = 'RGB'
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
# 'pix2pix_exp_uc_RGB-2-PMarray_epoch25_5_bch70',
# 'cycle_gan_exp_uc_RGB-2-PMarray_epoch25_5_bch3',
# #     'cycle_gan_exp_uc_RGB-2-PMarray_clip11_D3_epoch25_5',     # old cycle version
# ]

# IR version
outNm = 'IR'
rst_li = [
'vis2PM_exp_uc_IR-2-PMarray_clip01_w-align_n_phy1_3stg_woactiFn_100.0L2_n-whtL100_0.0sum_0.0ssim_nD3_epoch25_5_bch30',          # clip01
'vis2PM_exp_uc_IR-2-PMarray_clip01_w-align_n_phy1_3stg_woactiFn_100.0L2_auto-whtL100_0.0sum_0.0ssim_nD3_epoch25_5_bch30',       # sa
'vis2PM_exp_uc_IR-2-PMarray_clip01_w-align_n_phy1_3stg_woactiFn_100.0L2_auto-whtL100_10000.0sum_0.0ssim_nD3_epoch25_5_bch30',    # sa_sum
'vis2PM_exp_uc_IR-2-PMarray_clip01_w-align_n_phy1_3stg_woactiFn_100.0L2_auto-whtL100_10000.0sum_10.0ssim_nD3_epoch25_5_bch30',   # sa_sum_ssim
'vis2PM_exp_uc_IR-2-PMarray_clip01_w-align_n_phy1_3stg_woactiFn_100.0L2_auto-whtL100_10000.0sum_10.0ssim_D3_epoch25_5_bch20',    # sa_sum_ssim_D
'vis2PM_exp_uc_IR-2-PMarray_clip01_w-align_n_phy0_3stg_woactiFn_100.0L2_n-whtL100_0.0sum_0.0ssim_nD3_epoch25_5_bch30',          # nPhy0
'vis2PM_exp_uc_IR-2-PMarray_clip01_w-align_n_phy1_3stg_woactiFn_100.0L2_n-whtL100_10000.0sum_0.0ssim_nD3_epoch25_5_bch30',       # sum
'vis2PM_exp_uc_IR-2-PMarray_clip01_w-align_n_phy1_3stg_woactiFn_100.0L2_n-whtL100_0.0sum_10.0ssim_nD3_epoch25_5_bch30',         # ssim
'vis2PM_exp_uc_IR-2-PMarray_clip01_w-align_n_phy1_3stg_woactiFn_100.0L2_n-whtL100_0.0sum_0.0ssim_D3_epoch25_5_bch20',           # D
'openPose_exp_uc_IR-2-PMarray_epoch25_5_bch30',
'memNet_exp_uc_IR-2-PMarray_epoch25_5_bch2',
'pix2pix_exp_uc_IR-2-PMarray_epoch25_5_bch70',
'cycle_gan_exp_uc_IR-2-PMarray_epoch25_5_bch3',
#     'cycle_gan_exp_uc_IR-2-PMarray_clip11_D3_epoch25_5',     # old cycle version
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
    cnt = fake_vStk.shape[0] # how many images

    # suppose to normalize all data to 0 1 for a common comparison
    # v1 for R based comparison
    # if any(subStr in rstNm for subStr in R2_li):
    #     R = 2       # cycle and pix fixed preprocessing
    #     # print('range 2')
    # else:
    #     R = args.R
    #     # print('range 1')
    # if R == 2:
    #     thr = -1 + R*efs_rt  # the threshold
    # else:
    #     thr = R * efs_rt

    #
    # v1 for R based comparison
    if any(subStr in rstNm for subStr in R2_li):    # all to range 1
        fake_vStk = (fake_vStk+1)/2
        real_vStk = (real_vStk+1)/2
    # all to std R
    R = 1
    thr = R * efs_rt
    # accumulator
    mse_sum = 0
    mse_efs_sum = 0
    psnr_sum = 0
    psnr_efs_sum = 0
    pcs_efs_sum = 0
    pcs_efs01_sum = 0   # a default pcs0.1
    ssim_sum = 0
    fp_sum = 0

    for i in range(cnt):
        real_B = real_vStk[i]
        fake_B = fake_vStk[i]
        diff_abs = np.abs(real_B - fake_B)
        # get the tensor form
        # if len(real_B.shape) < 3:   # gray
        #     real_B_3c = np.expand_dims(real_B, axis=0)
        #     fake_B_3c = np.expand_dims(fake_B, axis=0)
        # else:
        #     real_B_3c = real_B  # direct reference, tensor will copy
        #     fake_B_3c = fake_B
        ts_real_B = torch.tensor(real_B).unsqueeze(0)
        # print('realB shape', real_B.shape)
        ts_fake_B = torch.tensor(fake_B).unsqueeze(0)
        # print('fakeB shape', fake_B.shape)
        # get current metric, add to list
        # metrics
        mseT = (diff_abs ** 2).mean()
        # print('MSE passed')
        psnrT = 20 * math.log10(R / mseT)  # R depends on clip
        # print('PSNR passed')
        mse_efsT = (diff_abs[real_B > thr] ** 2).mean()  #
        psnr_efsT = 20 * math.log10(R / mse_efsT)  # R depends on clip
        n_efs = diff_abs[real_B > thr].size
        pcs_efsT = (diff_abs[real_B > thr] < R * pcs_test).sum() / n_efs
        pcs_efs01T = (diff_abs[real_B > thr] < R * 0.1).sum() / n_efs
        # print('pcs efs01 passed')
        ssimT = pytorch_ssim.ssim(ts_real_B, ts_fake_B).item()  # get value, no fake image !! , need to rewrite the evaluation, to redo all evaluations. save real fake vstk will be good now.
        # print('SSIM passed')
        fpT = (fake_B[real_B < thr] > thr).sum() / real_B.size

        # accumulation
        mse_sum += mseT
        mse_efs_sum += mse_efsT
        psnr_sum += psnrT
        psnr_efs_sum += psnr_efsT
        pcs_efs_sum += pcs_efsT
        pcs_efs01_sum += pcs_efs01T
        ssim_sum += ssimT
        fp_sum += fpT
        # print('round {} operation finished'.format(i))
    # reduce to final metric
    mse = mse_sum / cnt
    mse_efs = mse_efs_sum / cnt
    psnr = psnr_sum / cnt
    psnr_efs = psnr_efs_sum /cnt
    pcs_efs = pcs_efs_sum / cnt
    pcs_efs01 = pcs_efs01_sum /cnt
    ssim = ssim_sum / cnt
    fp = fp_sum / cnt

    # save to target folder
    np.savez(os.path.join(expFd, 'rst{}.npz'.format(efs_rt)), mse=mse, mse_efs=mse_efs, psnr=psnr,
             psnr_efs=psnr_efs, pcs_efs=pcs_efs, pcs_efs01=pcs_efs01, ssim=ssim, fp=fp) # rst[efs_rt].npz'  # save to result to save calculation for result next time.
    f.write('{}\t'.format(mse))
    f.write('{}\t'.format(mse_efs))
    f.write('{}\t'.format(psnr))
    f.write('{}\t'.format(psnr_efs))
    f.write('{}\t'.format(pcs_efs))
    f.write('{}\t'.format(pcs_efs01))
    f.write('{}\t'.format(ssim))
    f.write('{}\t'.format(fp))
    f.write('\n')
    # write to file

print('write metrics with given order', os.path.join(outFd, outNm))
f.close()


