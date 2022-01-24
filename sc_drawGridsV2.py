import util.utils_PM as utils_PM
import os
'''
This version is for the multi stage version where naming rule changes with stage number in. So call the drawGrids v2
This is for the ECCV20 version 
'''
if_local = False
n_stg = 3
if_comb = True      # if draw combined version of the output
dm = 'IR'
scale = 2
if if_local:
    # local test
    rst_li = [
        # 'vis2PM_exp_RGB-2-PMarray_phy-concat-1_nGated1_whtMode-autoWht-100_w-sumLoss100_clip11_w-actiFn_nD3_epoch25_5_bch70',
        # 'pix2pix_exp_IR-2-PMarray_clip11_D3_epoch25_5_bch70',
        # 'vis2PM_exp_IR-2-PMarray_phy-concat-1_nGated1_whtMode-autoWht-100_w-sumLoss100_clip01_w-actiFn_nD3_epoch25_5_bch70'
        # 'cycle_gan_exp_uc_RGB-2-PMarray_clip11_D3_epoch25_5',
        # 'cycle_gan_rst_test',

    ]
    figNm_li = ['cycle_test']
    # rstFd = r'S:\ACLab\rst_model\vis2PM\results'
    rstFd = 'results'
    subNm = 'cycleTest'
    if_gray = False     # for input saving method
    idx_li = [0, 1]

else:   # remote    only use 4 to give same size comparison
    rstFd = 'results'
    # RGB2PM
    ## L2 version
    if 'RGB' == dm:
        rst_li = [
            'vis2PM_exp_uc_RGB-2-PMarray_clip01_w-align_n_phy1_3stg_woactiFn_100.0L2_n-whtL100_0.0sum_0.0ssim_nD3_epoch25_5_bch30',
            # 'vis2PM_exp_uc_RGB-2-PMarray_clip01_w-align_n_phy1_3stg_woactiFn_100.0L2_auto-whtL100_0.0sum_0.0ssim_nD3_epoch25_5_bch30',
            'vis2PM_exp_uc_RGB-2-PMarray_clip01_w-align_n_phy1_3stg_woactiFn_100.0L2_auto-whtL100_10000.0sum_0.0ssim_nD3_epoch25_5_bch30',
            # 'vis2PM_exp_uc_RGB-2-PMarray_clip01_w-align_n_phy1_3stg_woactiFn_100.0L2_auto-whtL100_10000.0sum_10.0ssim_nD3_epoch25_5_bch30',
            'vis2PM_exp_uc_RGB-2-PMarray_clip01_w-align_n_phy1_3stg_woactiFn_100.0L2_auto-whtL100_10000.0sum_10.0ssim_D3_epoch25_5_bch20',
            # # 'vis2PM_exp_uc_RGB-2-PMarray_clip01_w-align_n_phy0_3stg_woactiFn_100.0L2_n-whtL100_0.0sum_0.0ssim_nD3_epoch25_5_bch30', # n_phy0
            # 'vis2PM_exp_uc_RGB-2-PMarray_clip01_w-align_n_phy1_3stg_woactiFn_100.0L2_n-whtL100_10000.0sum_0.0ssim_nD3_epoch25_5_bch30',
            # 'vis2PM_exp_uc_RGB-2-PMarray_clip01_w-align_n_phy1_3stg_woactiFn_100.0L2_n-whtL100_0.0sum_10.0ssim_nD3_epoch25_5_bch30',
            # 'vis2PM_exp_uc_RGB-2-PMarray_clip01_w-align_n_phy1_3stg_woactiFn_100.0L2_n-whtL100_0.0sum_0.0ssim_D3_epoch25_5_bch20',
            'openPose_exp_uc_RGB-2-PMarray_epoch25_5_bch30',
            'memNet_exp_uc_RGB-2-PMarray_epoch25_5_bch2',
            'pix2pix_exp_uc_RGB-2-PMarray_epoch25_5_bch70',
            'cycle_gan_exp_uc_RGB-2-PMarray_epoch25_5_bch3'
        ]
        if_gray = False  # for input saving method
    elif 'IR' == dm:
    # outNm = 'IRrst'
        rst_li = [
        'vis2PM_exp_uc_IR-2-PMarray_clip01_w-align_n_phy1_3stg_woactiFn_100.0L2_n-whtL100_0.0sum_0.0ssim_nD3_epoch25_5_bch30',          # clip01
        # 'vis2PM_exp_uc_IR-2-PMarray_clip01_w-align_n_phy1_3stg_woactiFn_100.0L2_auto-whtL100_0.0sum_0.0ssim_nD3_epoch25_5_bch30',       # sa
        'vis2PM_exp_uc_IR-2-PMarray_clip01_w-align_n_phy1_3stg_woactiFn_100.0L2_auto-whtL100_10000.0sum_0.0ssim_nD3_epoch25_5_bch30',    # sa_sum
        # 'vis2PM_exp_uc_IR-2-PMarray_clip01_w-align_n_phy1_3stg_woactiFn_100.0L2_auto-whtL100_10000.0sum_10.0ssim_nD3_epoch25_5_bch30',   # sa_sum_ssim
        'vis2PM_exp_uc_IR-2-PMarray_clip01_w-align_n_phy1_3stg_woactiFn_100.0L2_auto-whtL100_10000.0sum_10.0ssim_D3_epoch25_5_bch20',    # sa_sum_ssim_D
        # 'vis2PM_exp_uc_IR-2-PMarray_clip01_w-align_n_phy0_3stg_woactiFn_100.0L2_n-whtL100_0.0sum_0.0ssim_nD3_epoch25_5_bch30',          # nPhy0
        # 'vis2PM_exp_uc_IR-2-PMarray_clip01_w-align_n_phy1_3stg_woactiFn_100.0L2_n-whtL100_10000.0sum_0.0ssim_nD3_epoch25_5_bch30',       # sum
        # 'vis2PM_exp_uc_IR-2-PMarray_clip01_w-align_n_phy1_3stg_woactiFn_100.0L2_n-whtL100_0.0sum_10.0ssim_nD3_epoch25_5_bch30',         # ssim
        # 'vis2PM_exp_uc_IR-2-PMarray_clip01_w-align_n_phy1_3stg_woactiFn_100.0L2_n-whtL100_0.0sum_0.0ssim_D3_epoch25_5_bch20',           # D
        'openPose_exp_uc_IR-2-PMarray_epoch25_5_bch30',
            'memNet_exp_uc_IR-2-PMarray_epoch25_5_bch2',
            'pix2pix_exp_uc_IR-2-PMarray_epoch25_5_bch70',
            'cycle_gan_exp_uc_IR-2-PMarray_epoch25_5_bch3',
        ]
        if_gray = True  # for input saving method
    elif 'RGB2IR' == dm:
        rst_li = [
            'vis2PM_exp_uc_RGB-2-IR_clip01_w-align_n_phy1_3stg_woactiFn_100.0L2_n-whtL100_0.0sum_0.0ssim_nD3_epoch25_5_bch30',
            # clip01
            'vis2PM_exp_uc_RGB-2-IR_clip01_w-align_n_phy1_3stg_woactiFn_100.0L2_auto-whtL100_0.0sum_0.0ssim_nD3_epoch25_5_bch30',
            # sa
            'vis2PM_exp_uc_RGB-2-IR_clip01_w-align_n_phy1_3stg_woactiFn_100.0L2_auto-whtL100_10000.0sum_0.0ssim_nD3_epoch25_5_bch30',
            # # sa_sum
            # 'vis2PM_exp_uc_RGB-2-IR_clip01_w-align_n_phy1_3stg_woactiFn_100.0L2_auto-whtL100_10000.0sum_10.0ssim_nD3_epoch25_5_bch30',
            # # sa_sum_ssim
            'vis2PM_exp_uc_RGB-2-IR_clip01_w-align_n_phy1_3stg_woactiFn_100.0L2_auto-whtL100_10000.0sum_10.0ssim_D3_epoch25_5_bch20',
            # # sa_sum_ssim_D
            # # 'vis2PM_exp_uc_RGB-2-IR_clip01_w-align_n_phy0_3stg_woactiFn_100.0L2_n-whtL100_0.0sum_0.0ssim_nD3_epoch25_5_bch30',
            # # nPhy0
            # 'vis2PM_exp_uc_RGB-2-IR_clip01_w-align_n_phy1_3stg_woactiFn_100.0L2_n-whtL100_10000.0sum_0.0ssim_nD3_epoch25_5_bch30',
            # # sum
            # 'vis2PM_exp_uc_RGB-2-IR_clip01_w-align_n_phy1_3stg_woactiFn_100.0L2_n-whtL100_0.0sum_10.0ssim_nD3_epoch25_5_bch30',
            # # ssim
            # 'vis2PM_exp_uc_RGB-2-IR_clip01_w-align_n_phy1_3stg_woactiFn_100.0L2_n-whtL100_0.0sum_0.0ssim_D3_epoch25_5_bch20',
            # # D
            'openPose_exp_uc_RGB-2-IR_epoch25_5_bch30',
            # 'memNet_exp_uc_RGB-2-IR_epoch25_5_bch2',
            'pix2pix_exp_uc_RGB-2-IR_epoch25_5_bch70',
            'cycle_gan_exp_uc_RGB-2-IR_epoch25_5_bch3',
        ]
        if_gray = False
    if 'RGB2IR' == dm:
        figNm_li = [
            'base',
            'sa',
            'sa_phy',
            # 'sa_phy_ssim',
            'sa_phy_ssim_D',
            # # 'nPhy0',
            # 'phy',
            # 'ssim',
            # 'D',
            'openPose',
            # 'memNet',
            'pix2pix',
            'cycleGAN',
        ]
        # idx_li = [14, 32, 15]
        idx_li = [450, 465, 480]
    else:
        figNm_li = [
            'base',
            # 'sa',
            'sa_phy',
            # 'sa_phy_ssim',
            'sa_phy_ssim_D',
            # # 'nPhy0',
            # 'phy',
            # 'ssim',
            # 'D',
            'openPose',
            'memNet',
            'pix2pix',
            'cycleGAN',
        ]
        # idx_li = [14, 32]
        idx_li = [45,60,75,90,105,120]
    # subNm = 'RGB_234-283-284'
    suffix = '{}'.format(idx_li[0])
    for idx in idx_li[1:]:
        suffix += '-' + str(idx)
    # subNm = '{}_14-32'.format(dm)
    subNm = '{}_'.format(dm) + suffix
    # idx_li = [234, 283, 284]
    subNm = subNm + '_scale{}'.format(scale)
if not os.path.exists(rstFd):
    os.makedirs(rstFd)
# idxLs = list(range(6))

# idxLs = [90,105,120,135,150,165]
# idxLs = [6,18]
dpi = len(idx_li) * 50
if if_comb:
    utils_PM.drawGridV3(nmLs=rst_li, idxLs=idx_li, nmSht_li=figNm_li, rstFd=rstFd, name='pred', subNm=subNm, dpi=dpi, scale=scale)
else:
    for i, nm in enumerate(rst_li): # for each line version
        utils_PM.drawGridV3(nmLs=[nm], idxLs=idx_li, nmSht_li=[figNm_li[i]], rstFd=rstFd, name=figNm_li[i], subNm=subNm, dpi=dpi, scale=scale)

utils_PM.drawGridV3(nmLs= [rst_li[0]], idxLs=idx_li, nmSht_li=['INPUT'], rstFd=rstFd, name='input', mod=1, subNm=subNm, if_gray=if_gray, dpi=dpi, scale=1)  # input not enhancing
utils_PM.drawGridV3(nmLs= [rst_li[0]], idxLs=idx_li, nmSht_li=['GT'], rstFd=rstFd, name='label', mod=2, subNm=subNm, dpi=dpi, scale=scale)

