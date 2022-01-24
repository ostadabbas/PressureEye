import util.utils_PM as utils_PM

if_local = False
if if_local:
    # local test
    rst_li = [
        # 'vis2PM_exp_RGB-2-PMarray_phy-concat-1_nGated1_whtMode-autoWht-100_w-sumLoss100_clip11_w-actiFn_nD3_epoch25_5_bch70',
        # 'pix2pix_exp_IR-2-PMarray_clip11_D3_epoch25_5_bch70',
        # 'vis2PM_exp_IR-2-PMarray_phy-concat-1_nGated1_whtMode-autoWht-100_w-sumLoss100_clip01_w-actiFn_nD3_epoch25_5_bch70'
        'cycle_gan_exp_uc_RGB-2-PMarray_clip11_D3_epoch25_5'
    ]
    figNmLs = ['RGB','IR1','IR2', 'clip01']
    rstFd = r'S:\ACLab\rst_model\vis2PM\results'
else:   # remote    only use 4 to give same size comparison
    rstFd = 'results'
    # RGB2PM
    # rst_li = [
    #     'vis2PM_exp_RGB-2-PMarray_phy-concat-1_nGated1_whtMode-n-100_wo-sumLoss100_clip11_w-actiFn_nD3_epoch25_5_bch70',
    #     'vis2PM_exp_RGB-2-PMarray_phy-concat-1_nGated1_whtMode-autoWht-100_w-sumLoss100_clip11_w-actiFn_nD3_epoch25_5_bch70',
    #     # 'vis2PM_exp_RGB-2-PMarray_phy-sfg-1_nGated1_whtMode-autoWht-100_w-sumLoss100_clip01_w-actiFn_nD3_epoch25_5_bch70',
    #     'pix2pix_exp_RGB-2-PMarray_clip11_D3_epoch25_5_bch70',
    #     'cycle_gan_exp_uc_RGB-2-PMarray_clip11_D3_epoch25_5',
    # ]
    # figNmLs = ['base', 'autoW_sum',
    #            # 'autoW_sum_sfg1',
    #            'pix2pix', 'cycleGAN']
    # subNm = 'RGB2PM'
    # if_gray = False     # for input saving method

    # for RGB2IR study  *********************
    # rst_li = [
    #     'vis2PM_exp_RGB-2-IR_phy-concat-1_nGated1_whtMode-n-100_wo-sumLoss100_clip11_w-actiFn_nD3_epoch25_5_bch70',
    #     'vis2PM_exp_RGB-2-IR_phy-concat-1_nGated1_whtMode-autoWht-100_wo-sumLoss100_clip11_w-actiFn_nD3_epoch25_5_bch70',
    #     'pix2pix_exp_RGB-2-IR_clip11_D3_epoch25_5_bch70',
    #     'cycle_gan_exp_RGB-2-IR_clip11_D3_epoch25_5',
    # ]
    # figNmLs = ['base', 'autoW', 'pix2pix', 'cycleGAN']
    # rstFd = 'results'
    # subNm = 'RGB2IR'
    # if_gray = False

    # for IR2PM study
    # v1
    # rst_li = [
    #     'vis2PM_exp_IR-2-PMarray_phy-concat-1_nGated1_whtMode-n-100_wo-sumLoss100_clip11_w-actiFn_nD3_epoch25_5_bch70',
    #     'vis2PM_exp_IR-2-PMarray_phy-concat-1_nGated1_whtMode-autoWht-100_w-sumLoss100_clip11_w-actiFn_nD3_epoch25_5_bch70',
    #     # 'vis2PM_exp_IR-2-PMarray_phy-sfg-1_nGated1_whtMode-autoWht-100_w-sumLoss100_clip01_w-actiFn_nD3_epoch25_5_bch70',
    #     'pix2pix_exp_IR-2-PMarray_clip11_D3_epoch25_5_bch70',
    #     'cycle_gan_exp_IR-2-PMarray_clip11_D3_epoch25_5',
    # ]

    # v2 for uc_unm version
    rst_li = [
        'vis2PM_uc_unm_IR-2-PMarray_phy-concat-1_nGated1_whtMode-n-100_wo-sumLoss100_clip11_w-actiFn_nD3_epoch25_5_bch70',
        'vis2PM_uc_unm_IR-2-PMarray_phy-concat-1_nGated1_whtMode-autoWht-100_w-sumLoss100_clip11_w-actiFn_nD3_epoch25_5_bch70',
        # 'vis2PM_uc_unm_IR-2-PMarray_phy-concat-1_nGated1_whtMode-autoWht-100_w-sumLoss100_clip11_w-actiFn_D3_epoch25_5_bch70',
        'pix2pix_uc_unm_IR-2-PMarray_clip11_D3_epoch25_5_bch70',
        'cycle_gan_uc_unm_IR-2-PMarray_clip11_D3_epoch25_5'
    ]
    figNmLs = ['base', 'autoW_sum',
               # 'autoW_sum_sfg1',
               'pix2pix', 'cycleGAN']
    subNm = 'IR2PM'     # which folder to put int   # sub folder in result session
    if_gray = True

    # for multi-stage

# idxLs = list(range(6))
# idxLs = [0, 15, 30, 45, 60, 75] # only 3 images for supine, left and right
# NIPS v1
idxLs = [6,18]
dpi = len(idxLs)*50
for i, nm in enumerate(rst_li):
    utils_PM.drawGrid(nmLs=[nm], idxLs=idxLs, rstFd=rstFd, name=figNmLs[i], subNm=subNm, dpi=dpi)
utils_PM.drawGrid(nmLs= [rst_li[0]], idxLs=idxLs, rstFd=rstFd, name='input', mod=1, subNm=subNm, if_gray=if_gray, dpi=dpi)
utils_PM.drawGrid(nmLs= [rst_li[0]], idxLs=idxLs, rstFd=rstFd, name='label', mod=2, subNm=subNm, dpi=dpi)

