# for version two post NIPS19 study
import util.utils_PM as utils_PM
import argparse


### for RGB and IR test
## RGB version
parser = argparse.ArgumentParser()
parser.add_argument('--modSrc', default='RGB', help='The input modality')
parser.add_argument('--cmp_mode', default='abla', help='the comparison mode of PCS lots [abla|sota|beta]')

opt, _ = parser.parse_known_args()
modSrc = opt.modSrc
cmp_mode = opt.cmp_mode

if 'abla' == cmp_mode:
    idxs = list(range(8))   # 0 to 7 8 plots
elif 'sota' == cmp_mode:
    # idxs = [2,8,9,10,11]
    idxs = [0,11,12,13,14]      # give leading one the candidates
elif 'beta' == cmp_mode:
    idxs = [0,8,9,10]
    rgSt = 0.048
    rgTo = 0.052

pltNm = 'pcs_{}_{}'.format(modSrc, cmp_mode)       # file name

# titleNm = 'ablation study via RGB'
rst_li = [
    'vis2PM_exp_uc_{}-2-PMarray_n_phy1_stg3_whtL-pwrs-100.0L2_lap0.001_sum1e-06_ssim0.0_D0.0L3'.format(modSrc),
    # pwrs sum
    'vis2PM_exp_uc_{}-2-PMarray_n_phy1_stg3_whtL-n-100.0L2_lap0.001_sum0.0_ssim0.0_D0.0L3'.format(modSrc),  # mse
    # 'vis2PM_exp_uc_{}-2-PMarray_n_phy1_stg3_whtL-pwrs-100.0L2_lap0.0_sum0.0_ssim0.0_D0.0L3'.format(modSrc),  # pwrs0
    'vis2PM_exp_uc_{}-2-PMarray_n_phy1_stg3_whtL-pwrs-100.0L2_lap0.001_sum0.0_ssim0.0_D0.0L3'.format(modSrc),  # pwrs
    'vis2PM_exp_uc_{}-2-PMarray_n_phy1_stg3_whtL-pwrs-100.0L2_lap0.001_sum1e-06_ssim10.0_D0.0L3'.format(modSrc),
    # pwrs+sum+sim
    'vis2PM_exp_uc_{}-2-PMarray_n_phy1_stg3_whtL-pwrs-100.0L2_lap0.001_sum1e-06_ssim10.0_D1.0L3'.format(modSrc),
    # pwrs+sum+sim+D
    # 'vis2PM_exp_uc_{}-2-PMarray_n_phy0_stg3_whtL-n-100.0L2_lap0.001_sum0.0_ssim0.0_D0.0L3'.format(modSrc),  # phy0+n
    'vis2PM_exp_uc_{}-2-PMarray_n_phy1_stg3_whtL-n-100.0L2_lap0.001_sum1e-06_ssim0.0_D0.0L3'.format(modSrc),  # sum
    'vis2PM_exp_uc_{}-2-PMarray_n_phy1_stg3_whtL-n-100.0L2_lap0.001_sum0.0_ssim10.0_D0.0L3'.format(modSrc),  # sim
    'vis2PM_exp_uc_{}-2-PMarray_n_phy1_stg3_whtL-n-100.0L2_lap0.001_sum0.0_ssim0.0_D1.0L3'.format(modSrc),  # D 8th
    # 'vis2PM_exp_uc_{}-2-PMarray_n_phy0_stg3_whtL-pwrs-100.0L2_lap0.001_sum1e-06_ssim0.0_D0.0L3'.format(modSrc),    # pwrs sum_nPhy0
    'vis2PM_exp_uc_{}-2-PMarray_n_phy2_stg3_whtL-pwrs-100.0L2_lap0.001_sum1e-06_ssim0.0_D0.0L3'.format(modSrc),     # pwrs sum2
    'vis2PM_exp_uc_{}-2-PMarray_n_phy3_stg3_whtL-pwrs-100.0L2_lap0.001_sum1e-06_ssim0.0_D0.0L3'.format(modSrc),     # pwrs sum3
    'vis2PM_exp_uc_{}-2-PMarray_n_phy10_stg3_whtL-pwrs-100.0L2_lap0.001_sum1e-06_ssim0.0_D0.0L3'.format(modSrc),    # pwrs sum10 11th
    'openPose_exp_uc_{}-2-PMarray_epoch25_5_bch30'.format(modSrc),
    'memNet_exp_uc_{}-2-PMarray_epoch25_5_bch2'.format(modSrc),
    'pix2pix_exp_uc_{}-2-PMarray_epoch25_5_bch70'.format(modSrc),
    'cycle_gan_exp_uc_{}-2-PMarray_epoch25_5_bch3'.format(modSrc),
    'vis2PM_exp_uc_{}-2-PMarray_n_phy1_stg1_whtL-pwrs-100.0L2_lap0.001_sum1e-06_ssim0.0_D0.0L3'.format(modSrc),
    # pwrs sum stg1
    'vis2PM_exp_uc_{}-2-PMarray_n_phy1_stg2_whtL-pwrs-100.0L2_lap0.001_sum1e-06_ssim0.0_D0.0L3'.format(modSrc),
    # pwrs sum stg2
]
legend_li = [
    'pwrs_phy',
    'base',
    'pwrs',
    'pwrs_phy_ssim',
    'pwrs_phy_ssim_D',
    'phy',
    'ssim',
    'D',
    'beta-2',
    'beta-3',
    'beta-10',
    'openPose',
    'memNet',
    'pix2pix',
    'cycleGAN',
]
rst_li = [rst_li[i] for i in idxs]
legend_li = [legend_li[i] for i in idxs]
if 'beta' == cmp_mode:      # rename it to show beta difference
    legend_li[0] = 'beta-1'

print('under compare mode', cmp_mode)
for nm in rst_li:
    print(nm)


## phy number test########################
# RGB---------------
# rst_li = [
# 'vis2PM_exp_RGB-2-PMarray_phy-concat-1_nGated1_whtMode-autoWht-100_w-sumLoss100_clip11_w-actiFn_nD3_epoch25_5_bch70',
# 'vis2PM_exp_RGB-2-PMarray_phy-concat-2_nGated1_whtMode-autoWht-100_w-sumLoss100_clip11_w-actiFn_nD3_epoch25_5_bch70',
# 'vis2PM_exp_RGB-2-PMarray_phy-concat-3_nGated1_whtMode-autoWht-100_w-sumLoss100_clip11_w-actiFn_nD3_epoch25_5_bch70',
# 'vis2PM_exp_RGB-2-PMarray_phy-concat-10_nGated1_whtMode-autoWht-100_w-sumLoss100_clip11_w-actiFn_nD3_epoch25_5_bch70',
# ]
#
# legend_li = [
#     'phy-1',
#     'phy-2',
#     'phy-3',
#     'phy-10'
# ]
# pltNm = 'phyNum_RGB'
# titleNm =r'PADS-PM RGB with varying $\beta$'
# # draw func
# utils_PM.drawPCS(rst_li, legend_li, pltNm, titleNm=titleNm)
# utils_PM.drawPCS(rst_li, legend_li, pltNm + '_zoomIn', rgSt=0.048, rgTo=0.052, titleNm=titleNm)

# IR ---------------------
# rst_li = [
# 'vis2PM_uc_unm_IR-2-PMarray_phy-concat-1_nGated1_whtMode-autoWht-100_w-sumLoss100_clip11_w-actiFn_nD3_epoch25_5_bch70',
# 'vis2PM_uc_unm_IR-2-PMarray_phy-concat-2_nGated1_whtMode-autoWht-100_w-sumLoss100_clip11_w-actiFn_nD3_epoch25_5_bch70',
# 'vis2PM_uc_unm_IR-2-PMarray_phy-concat-3_nGated1_whtMode-autoWht-100_w-sumLoss100_clip11_w-actiFn_nD3_epoch25_5_bch70',
# 'vis2PM_uc_unm_IR-2-PMarray_phy-concat-10_nGated1_whtMode-autoWht-100_w-sumLoss100_clip11_w-actiFn_nD3_epoch25_5_bch70',
# ]
#
# legend_li = [
#     'phy-1',
#     'phy-2',
#     'phy-3',
#     'phy-10'
# ]
# pltNm = 'phyNum_IR'
# titleNm =r'PADS-PM IR with varying $\beta$'
# # draw func
# utils_PM.drawPCS(rst_li, legend_li, pltNm, titleNm=titleNm)
# utils_PM.drawPCS(rst_li, legend_li, pltNm + '_zoomIn', rgSt=0.048, rgTo=0.052, titleNm=titleNm)


utils_PM.drawPCSv2(rst_li, legend_li, pltNm, sz_lgd=15)          # add an emphasizing number
if 'beta' == cmp_mode:      # add additional zoom in plot
    utils_PM.drawPCSv2(rst_li, legend_li, pltNm + '_zoomIn', rgSt=0.048, rgTo=0.052)



