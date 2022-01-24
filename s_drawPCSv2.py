'''
ECCV20 version
'''
# for version two post NIPS19 study
import util.utils_PM as utils_PM


### for RGB and IR test
## RGB version
cmp_mode = 'beta'       # choose abla or IR
dm = 'RGB'
if 'abla' == cmp_mode:
    idxs = list(range(8))   # 0 to 7 8 plots
elif 'sota' == cmp_mode:
    idxs = [2,8,9,10,11]
elif 'beta' == cmp_mode:
    idxs = [2,12,13,14]
    rgSt = 0.048
    rgTo = 0.052

pltNm = 'pcs_{}_{}'.format(dm, cmp_mode)       # file name
# titleNm = 'ablation study via RGB'
if 'RGB' == dm:
    rst_li = [
    'vis2PM_exp_uc_RGB-2-PMarray_clip01_w-align_n_phy1_3stg_woactiFn_100.0L2_auto-whtL100_10000.0sum_0.0ssim_nD3_epoch25_5_bch30',
        # sa_sum
    'vis2PM_exp_uc_RGB-2-PMarray_clip01_w-align_n_phy1_3stg_woactiFn_100.0L2_n-whtL100_0.0sum_0.0ssim_nD3_epoch25_5_bch30',          # clip01
    'vis2PM_exp_uc_RGB-2-PMarray_clip01_w-align_n_phy1_3stg_woactiFn_100.0L2_auto-whtL100_0.0sum_0.0ssim_nD3_epoch25_5_bch30',       # sa

    'vis2PM_exp_uc_RGB-2-PMarray_clip01_w-align_n_phy1_3stg_woactiFn_100.0L2_auto-whtL100_10000.0sum_10.0ssim_nD3_epoch25_5_bch30',   # sa_sum_ssim
    'vis2PM_exp_uc_RGB-2-PMarray_clip01_w-align_n_phy1_3stg_woactiFn_100.0L2_auto-whtL100_10000.0sum_10.0ssim_D3_epoch25_5_bch20',    # sa_sum_ssim_D
    # 'vis2PM_exp_uc_RGB-2-PMarray_clip01_w-align_n_phy0_3stg_woactiFn_100.0L2_n-whtL100_0.0sum_0.0ssim_nD3_epoch25_5_bch30',   # D
    'vis2PM_exp_uc_RGB-2-PMarray_clip01_w-align_n_phy1_3stg_woactiFn_100.0L2_n-whtL100_10000.0sum_0.0ssim_nD3_epoch25_5_bch30',       # sum
    'vis2PM_exp_uc_RGB-2-PMarray_clip01_w-align_n_phy1_3stg_woactiFn_100.0L2_n-whtL100_0.0sum_10.0ssim_nD3_epoch25_5_bch30',         # ssim
    'vis2PM_exp_uc_RGB-2-PMarray_clip01_w-align_n_phy1_3stg_woactiFn_100.0L2_n-whtL100_0.0sum_0.0ssim_D3_epoch25_5_bch20',           # D
    'openPose_exp_uc_RGB-2-PMarray_epoch25_5_bch30',
    'memNet_exp_uc_RGB-2-PMarray_epoch25_5_bch2',
    'pix2pix_exp_uc_RGB-2-PMarray_epoch25_5_bch70',
    'cycle_gan_exp_uc_RGB-2-PMarray_epoch25_5_bch3',
    'vis2PM_exp_uc_RGB-2-PMarray_clip01_w-align_n_phy2_3stg_woactiFn_100.0L2_auto-whtL100_10000.0sum_0.0ssim_nD3_epoch25_5_bch30',
    'vis2PM_exp_uc_RGB-2-PMarray_clip01_w-align_n_phy3_3stg_woactiFn_100.0L2_auto-whtL100_10000.0sum_0.0ssim_nD3_epoch25_5_bch30',
    'vis2PM_exp_uc_RGB-2-PMarray_clip01_w-align_n_phy10_3stg_woactiFn_100.0L2_auto-whtL100_10000.0sum_0.0ssim_nD3_epoch25_5_bch30',
    ]
elif 'IR' == dm:
    rst_li = [
        'vis2PM_exp_uc_IR-2-PMarray_clip01_w-align_n_phy1_3stg_woactiFn_100.0L2_auto-whtL100_10000.0sum_0.0ssim_nD3_epoch25_5_bch30',
        # sa_sum
        'vis2PM_exp_uc_IR-2-PMarray_clip01_w-align_n_phy1_3stg_woactiFn_100.0L2_n-whtL100_0.0sum_0.0ssim_nD3_epoch25_5_bch30',
        # clip01
        'vis2PM_exp_uc_IR-2-PMarray_clip01_w-align_n_phy1_3stg_woactiFn_100.0L2_auto-whtL100_0.0sum_0.0ssim_nD3_epoch25_5_bch30',
        # sa
        'vis2PM_exp_uc_IR-2-PMarray_clip01_w-align_n_phy1_3stg_woactiFn_100.0L2_auto-whtL100_10000.0sum_10.0ssim_nD3_epoch25_5_bch30',
        # sa_sum_ssim
        'vis2PM_exp_uc_IR-2-PMarray_clip01_w-align_n_phy1_3stg_woactiFn_100.0L2_auto-whtL100_10000.0sum_10.0ssim_D3_epoch25_5_bch20',
        # sa_sum_ssim_D
        # 'vis2PM_exp_uc_IR-2-PMarray_clip01_w-align_n_phy0_3stg_woactiFn_100.0L2_n-whtL100_0.0sum_0.0ssim_nD3_epoch25_5_bch30',          # nPhy0
        'vis2PM_exp_uc_IR-2-PMarray_clip01_w-align_n_phy1_3stg_woactiFn_100.0L2_n-whtL100_10000.0sum_0.0ssim_nD3_epoch25_5_bch30',
        # sum
        'vis2PM_exp_uc_IR-2-PMarray_clip01_w-align_n_phy1_3stg_woactiFn_100.0L2_n-whtL100_0.0sum_10.0ssim_nD3_epoch25_5_bch30',
        # ssim
        'vis2PM_exp_uc_IR-2-PMarray_clip01_w-align_n_phy1_3stg_woactiFn_100.0L2_n-whtL100_0.0sum_0.0ssim_D3_epoch25_5_bch20',
        # D
        'openPose_exp_uc_IR-2-PMarray_epoch25_5_bch30',
        'memNet_exp_uc_IR-2-PMarray_epoch25_5_bch2',
        'pix2pix_exp_uc_IR-2-PMarray_epoch25_5_bch70',
        'cycle_gan_exp_uc_IR-2-PMarray_epoch25_5_bch3',
        'vis2PM_exp_uc_IR-2-PMarray_clip01_w-align_n_phy2_3stg_woactiFn_100.0L2_auto-whtL100_10000.0sum_0.0ssim_nD3_epoch25_5_bch30',
        'vis2PM_exp_uc_IR-2-PMarray_clip01_w-align_n_phy3_3stg_woactiFn_100.0L2_auto-whtL100_10000.0sum_0.0ssim_nD3_epoch25_5_bch30',
        'vis2PM_exp_uc_IR-2-PMarray_clip01_w-align_n_phy10_3stg_woactiFn_100.0L2_auto-whtL100_10000.0sum_0.0ssim_nD3_epoch25_5_bch30',
    ]
legend_li = [
    'sa_phy',
    'base',
    'sa',
    'sa_phy_ssim',
    'sa_phy_ssim_D',
    'phy',
    'ssim',
    'D',
    'openPose',
    'memNet',
    'pix2pix',
    'cycleGAN',
    'beta-2',
    'beta-3',
    'beta-10',
]
rst_li = [rst_li[i] for i in idxs]
legend_li = [legend_li[i] for i in idxs]
if 'beta' == cmp_mode:      # rename it to show beta difference
    legend_li[0] = 'beta-1'



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
if 'beta' == cmp_mode:
    utils_PM.drawPCSv2(rst_li, legend_li, pltNm + '_zoomIn', rgSt=0.048, rgTo=0.052)



