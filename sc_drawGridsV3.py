import util.utils_PM as utils_PM
import os
import argparse
'''
This version is for the multi stage version where naming rule changes with stage number in. So call the drawGrids v2
This is for the ECCV20 version 
'''

parser = argparse.ArgumentParser()
parser.add_argument('--modSrc', default='RGB', help='the source mode')

opt, _ = parser.parse_known_args()
modSrc = opt.modSrc

if_local = False
n_stg = 3
if_comb = True      # if draw combined version of the output
dm = opt.modSrc
scale = 2       # scaling of pressure for contrast
clrBg = 'black'
thr_bg = 0.05

if dm =='RGB':
    if_gray=False
else:
    if_gray=True

rstFd = 'results'
rst_li = [
    'vis2PM_exp_uc_{}-2-PMarray_n_phy1_stg3_whtL-n-100.0L2_lap0.001_sum0.0_ssim0.0_D0.0L3'.format(modSrc), # mse
    ## 'vis2PM_exp_uc_{}-2-PMarray_n_phy1_stg3_whtL-pwrs-100.0L2_lap0.0_sum0.0_ssim0.0_D0.0L3'.format(modSrc), # pwrs0
    # 'vis2PM_exp_uc_{}-2-PMarray_n_phy1_stg3_whtL-pwrs-100.0L2_lap0.001_sum0.0_ssim0.0_D0.0L3'.format(modSrc), # pwrs
    'vis2PM_exp_uc_{}-2-PMarray_n_phy1_stg3_whtL-pwrs-100.0L2_lap0.001_sum1e-06_ssim0.0_D0.0L3'.format(modSrc),
    # pwrs sum
    # 'vis2PM_exp_uc_{}-2-PMarray_n_phy1_stg3_whtL-pwrs-100.0L2_lap0.001_sum1e-06_ssim10.0_D0.0L3'.format(modSrc), # pwrs+sum+sim
    'vis2PM_exp_uc_{}-2-PMarray_n_phy1_stg3_whtL-pwrs-100.0L2_lap0.001_sum1e-06_ssim10.0_D1.0L3'.format(modSrc), # pwrs+sum+sim+D
    ## 'vis2PM_exp_uc_{}-2-PMarray_n_phy0_stg3_whtL-n-100.0L2_lap0.001_sum0.0_ssim0.0_D0.0L3'.format(modSrc), # phy0+n
    # 'vis2PM_exp_uc_{}-2-PMarray_n_phy1_stg3_whtL-n-100.0L2_lap0.001_sum1e-06_ssim0.0_D0.0L3'.format(modSrc), # sum
    # 'vis2PM_exp_uc_{}-2-PMarray_n_phy1_stg3_whtL-n-100.0L2_lap0.001_sum0.0_ssim10.0_D0.0L3'.format(modSrc), # sim
    # 'vis2PM_exp_uc_{}-2-PMarray_n_phy1_stg3_whtL-n-100.0L2_lap0.001_sum0.0_ssim0.0_D1.0L3'.format(modSrc), # D
    # 'vis2PM_exp_uc_{}-2-PMarray_n_phy2_stg3_whtL-pwrs-100.0L2_lap0.001_sum1e-06_ssim0.0_D0.0L3'.format(modSrc),  # pwrs sum2
    # 'vis2PM_exp_uc_{}-2-PMarray_n_phy3_stg3_whtL-pwrs-100.0L2_lap0.001_sum1e-06_ssim0.0_D0.0L3'.format(modSrc),  # pwrs sum3
    # 'vis2PM_exp_uc_{}-2-PMarray_n_phy10_stg3_whtL-pwrs-100.0L2_lap0.001_sum1e-06_ssim0.0_D0.0L3'.format(modSrc),  # pwrs sum10
    'openPose_exp_uc_{}-2-PMarray_epoch25_5_bch30'.format(modSrc),
    'memNet_exp_uc_{}-2-PMarray_epoch25_5_bch2'.format(modSrc),
    'pix2pix_exp_uc_{}-2-PMarray_epoch25_5_bch70'.format(modSrc),
    'cycle_gan_exp_uc_{}-2-PMarray_epoch25_5_bch3'.format(modSrc),
]

figNm_li = [
    'base',
    # 'sa',
    'pwrs_phy',
    # 'sa_phy_ssim',
    'pwrs_phy_ssim_D',
    # # 'nPhy0',
    # 'phy',
    # 'ssim',
    # 'D',
    'openPose',
    'memNet',
    'pix2pix',
    'cycleGAN',
]

# idx_li = [45,60,75,90,105,120]        # EVENLY SAMPLED
# idx_li = [0, 6, 15, 30, 1]      # specific sampled to have failure case
idx_li = [0, 6, 30]      # for short version of RGB and IR
# subNm = 'RGB_234-283-284'
suffix = '{}'.format(idx_li[0])
for idx in idx_li[1:]:
    suffix += '-' + str(idx)
# subNm = '{}_14-32'.format(dm)
subNm = '{}_'.format(dm) + suffix
# idx_li = [234, 283, 284]
subNm = subNm + '_scale{}'.format(scale)
if clrBg:
    subNm += '_' + clrBg
if not os.path.exists(rstFd):
    os.makedirs(rstFd)
# idxLs = list(range(6))

# idxLs = [90,105,120,135,150,165]
# idxLs = [6,18]
dpi = len(idx_li) * 50
if if_comb:
    utils_PM.drawGridV3(nmLs=rst_li, idxLs=idx_li, nmSht_li=figNm_li, rstFd=rstFd, name='pred', subNm=subNm, dpi=dpi, scale=scale, clrBg=clrBg, thr_bg=thr_bg)
else:
    for i, nm in enumerate(rst_li): # for each line version
        utils_PM.drawGridV3(nmLs=[nm], idxLs=idx_li, nmSht_li=[figNm_li[i]], rstFd=rstFd, name=figNm_li[i], subNm=subNm, dpi=dpi, scale=scale,
                            clrBg=clrBg, thr_bg=thr_bg)

utils_PM.drawGridV3(nmLs= [rst_li[0]], idxLs=idx_li, nmSht_li=['INPUT'], rstFd=rstFd, name='input', mod=1, subNm=subNm, if_gray=if_gray, dpi=dpi, scale=1)    # input not enhancing, mod control output, input or gt
utils_PM.drawGridV3(nmLs= [rst_li[0]], idxLs=idx_li, nmSht_li=['GT'], rstFd=rstFd, name='label', mod=2, subNm=subNm, dpi=dpi, scale=scale, clrBg=clrBg,
                    thr_bg=thr_bg)

