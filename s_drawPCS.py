# draw PCS results for selected curves
# this version keeps most candidates of the result for NIPS19 version.
# We open a new version 2 for new study to avoid confusing.
import util.utils_PM as utils_PM

# compare setting version 1
# rst_li=[
#     'pix2pix_expriment_test_IR-2-PMarray_clip11_D3_epoch25_5',  # use a new trained one to see if batch
#     'cycle_gan_experiment_name_IR-2-PMarray_clip11_D3_epoch25_5',
#     'vis2PM_experiment_name_IR-2-PMarray_phy-concat-0_nGated1_whtMode-n-100_clip11_w-actiFn_D3_epoch25_5_bch70',        # non phy model
#     'vis2PM_experiment_name_IR-2-PMarray_phy-concat-1_nGated1_whtMode-n-100_clip11_w-actiFn_nD3_epoch25_5',
#     'vis2PM_experiment_name_IR-2-PMarray_phy-concat-1_nGated1_whtMode-n-100_clip11_w-actiFn_D3_epoch25_5',
#     'vis2PM_experiment_name_IR-2-PMarray_phy-sim_gated-1_nGated1_whtMode-n-100_clip11_w-actiFn_nD3_epoch25_5_bch70',
#     'vis2PM_experiment_name_IR-2-PMarray_phy-sim_gated-1_nGated1_whtMode-n-100_clip11_w-actiFn_D3_epoch25_5',
#     'vis2PM_experiment_name_IR-2-PMarray_phy-fc_gated-1_nGated1_whtMode-n-100_clip11_w-actiFn_nD3_epoch25_5',
#     'vis2PM_experiment_name_IR-2-PMarray_phy-fc_gated-1_nGated2_whtMode-n-100_clip11_w-actiFn_nD3_epoch25_5',
#     'vis2PM_experiment_name_IR-2-PMarray_phy-concat-1_nGated1_whtMode-autoWht-100_clip11_w-actiFnnD3_epoch25_5_bch70'
#     ]
#
# legend_li = [
#     'pix2pix',
#     'cycleGAN',
#     'vis2PMas_pix2pix',
#     'vis2PM-concat1-nD',
#     'vis2Pm-concat1-D',
#     'simGated-nD',
#     'simGated-D',
#     'fc_gated-l1',
#     'fc_gated-l2',
#     'autoWht'
# ]
# pltNm = 'vis2PM_con_gate_pix2pix'

# IR test 2
rst_li=[
    'pix2pix_expriment_test_IR-2-PMarray_clip11_D3_epoch25_5',  # use a new trained one to see if batch
    'cycle_gan_experiment_name_IR-2-PMarray_clip11_D3_epoch25_5',
    'vis2PM_experiment_name_IR-2-PMarray_phy-concat-0_nGated1_whtMode-n-100_clip11_w-actiFn_D3_epoch25_5_bch70',        # non phy model
    'vis2PM_experiment_name_IR-2-PMarray_phy-concat-1_nGated1_whtMode-n-100_clip11_w-actiFn_nD3_epoch25_5',
    'vis2PM_experiment_name_IR-2-PMarray_phy-concat-1_nGated1_whtMode-n-100_clip11_w-actiFn_D3_epoch25_5',
    'vis2PM_experiment_name_IR-2-PMarray_phy-sim_gated-1_nGated1_whtMode-n-100_clip11_w-actiFn_nD3_epoch25_5_bch70',
    'vis2PM_experiment_name_IR-2-PMarray_phy-sim_gated-1_nGated1_whtMode-n-100_clip11_w-actiFn_D3_epoch25_5',
    'vis2PM_experiment_name_IR-2-PMarray_phy-fc_gated-1_nGated1_whtMode-n-100_clip11_w-actiFn_nD3_epoch25_5',
    'vis2PM_experiment_name_IR-2-PMarray_phy-fc_gated-1_nGated2_whtMode-n-100_clip11_w-actiFn_nD3_epoch25_5',
    'vis2PM_experiment_name_IR-2-PMarray_phy-concat-1_nGated1_whtMode-autoWht-100_clip11_w-actiFnnD3_epoch25_5_bch70'
    ]

legend_li = [
    'pix2pix',
    'cycleGAN',
    'vis2PMas_pix2pix',
    'vis2PM-concat1-nD',
    'vis2Pm-concat1-D',
    'simGated-nD',
    'simGated-D',
    'fc_gated-l1',
    'fc_gated-l2',
    'autoWht'
]
pltNm = 'vis2PM_con_gate_pix2pix'

######   CONFIG test ############
#----RGB
# rst_li = [
#     'vis2PM_exp_RGB-2-PMarray_phy-concat-1_nGated1_whtMode-n-100_wo-sumLoss100_clip11_w-actiFn_nD3_epoch25_5_bch70',
#     'vis2PM_exp_RGB-2-PMarray_phy-concat-0_nGated1_whtMode-n-100_wo-sumLoss100_clip11_w-actiFn_nD3_epoch25_5_bch70',
#     'vis2PM_exp_RGB-2-PMarray_phy-concat-1_nGated1_whtMode-autoWht-100_w-sumLoss100_clip11_w-actiFn_D3_epoch25_5_bch70',
#     'vis2PM_exp_RGB-2-PMarray_phy-concat-1_nGated1_whtMode-autoWht-100_w-sumLoss100_clip11_w-actiFn_nD3_epoch25_5_bch70',
#     'vis2PM_exp_RGB-2-PMarray_phy-concat-1_nGated1_whtMode-autoWht-100_wo-sumLoss100_clip11_w-actiFn_nD3_epoch25_5_bch70',
#     'vis2PM_exp_RGB-2-PMarray_phy-concat-1_nGated1_whtMode-n-100_w-sumLoss100_clip11_w-actiFn_nD3_epoch25_5_bch70',
#     'vis2PM_exp_RGB-2-PMarray_phy-concat-1_nGated1_whtMode-n-100_wo-sumLoss100_clip11_w-actiFn_D3_epoch25_5_bch70'
#
# ]
# legend_li = [
#     'base',
#     'noBeta',
#     'autoW-sum-D',
#     'autoW-sum',
#     'autoW',
#     'sum',
#     'D'
# ]
# pltNm = 'vis2PM_configs'
# titleNm = 'vis2PM with different configurations'
#----RGB V2
# rst_li = [
#     'vis2PM_exp_RGB-2-PMarray_phy-concat-1_nGated1_whtMode-n-100_wo-sumLoss100_clip11_w-actiFn_nD3_epoch25_5_bch70',
#     'vis2PM_exp_RGB-2-PMarray_phy-concat-1_nGated1_whtMode-autoWht-100_w-sumLoss100_clip11_w-actiFn_nD3_epoch25_5_bch70',
#     'vis2PM_exp_RGB-2-PMarray_phy-concat-1_nGated1_whtMode-autoWht-100_w-sumLoss100_clip11_w-actiFn_D3_epoch25_5_bch70',
#     # 'pix2pix_exp_RGB-2-PMarray_clip11_D3_epoch25_5_bch70',
#     # 'cycle_gan_exp_uc_RGB-2-PMarray_clip11_D3_epoch25_5'
# ]
# legend_li = [
#     'base',
#     # 'autoW',
#     'autoW-sum',
#     'autoW-sum-D',
#     # 'pix2pix',
#     # 'cycleGAN'
# ]
# pltNm = 'PADS-PM_RGB'
# titleNm = 'PADS-PM RGB configurations'
# utils_PM.drawPCS(rst_li, legend_li, pltNm, titleNm=titleNm)
#----------IR--
# rst_li = [
#     'vis2PM_uc_unm_IR-2-PMarray_phy-concat-1_nGated1_whtMode-n-100_wo-sumLoss100_clip11_w-actiFn_nD3_epoch25_5_bch70',
#     'vis2PM_uc_unm_IR-2-PMarray_phy-concat-1_nGated1_whtMode-autoWht-100_w-sumLoss100_clip11_w-actiFn_nD3_epoch25_5_bch70',
#     'vis2PM_uc_unm_IR-2-PMarray_phy-concat-1_nGated1_whtMode-autoWht-100_w-sumLoss100_clip11_w-actiFn_D3_epoch25_5_bch70',
#     # 'pix2pix_uc_unm_IR-2-PMarray_clip11_D3_epoch25_5_bch70',
#     # 'cycle_gan_uc_unm_IR-2-PMarray_clip11_D3_epoch25_5'
# ]
#
# legend_li = [
#     'base',
#     'autoW-sum',
#     'autoW-sum-D',
#     # 'pix2pix',
#     # 'cycleGAN'
# ]
# pltNm = 'PADS-PM_IR'
# titleNm = 'PADS-PM IR configurations'
# utils_PM.drawPCS(rst_li, legend_li, pltNm, titleNm=titleNm)
# ####  gate layer test ##############
# rst_li = [
#     'vis2PM_exp_RGB-2-PMarray_phy-concat-1_nGated1_whtMode-autoWht-100_w-sumLoss100_clip11_w-actiFn_nD3_epoch25_5_bch70',
#     'vis2PM_exp_RGB-2-PMarray_phy-dcfg-1_nGated1_whtMode-autoWht-100_w-sumLoss100_clip01_w-actiFn_nD3_epoch25_5_bch70',
#     'vis2PM_exp_RGB-2-PMarray_phy-sfg-1_nGated1_whtMode-autoWht-100_w-sumLoss100_clip01_w-actiFn_nD3_epoch25_5_bch70',
#     # 'vis2PM_exp_RGB-2-PMarray_phy-sfg-1_nGated1_w-posInit_whtMode-autoWht-100_w-sumLoss100_clip01_w-actiFn_nD3_epoch25_5_bch70'
# ]
# legend_li = [
#     'autoW-sum',
#     'autoW-sum-dcfg1',
#     'autoW-sum-sfg1',
#     # 'autoW-sum-sfg1-posInit'
# ]
# pltNm = 'vis2PM_gate'
# titleNm = 'vis2PM with gating structure'
# utils_PM.drawPCS(rst_li, legend_li, pltNm, titleNm=titleNm)
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
rst_li = [
'vis2PM_uc_unm_IR-2-PMarray_phy-concat-1_nGated1_whtMode-autoWht-100_w-sumLoss100_clip11_w-actiFn_nD3_epoch25_5_bch70',
'vis2PM_uc_unm_IR-2-PMarray_phy-concat-2_nGated1_whtMode-autoWht-100_w-sumLoss100_clip11_w-actiFn_nD3_epoch25_5_bch70',
'vis2PM_uc_unm_IR-2-PMarray_phy-concat-3_nGated1_whtMode-autoWht-100_w-sumLoss100_clip11_w-actiFn_nD3_epoch25_5_bch70',
'vis2PM_uc_unm_IR-2-PMarray_phy-concat-10_nGated1_whtMode-autoWht-100_w-sumLoss100_clip11_w-actiFn_nD3_epoch25_5_bch70',
]

legend_li = [
    'phy-1',
    'phy-2',
    'phy-3',
    'phy-10'
]
pltNm = 'phyNum_IR'
titleNm =r'PADS-PM IR with varying $\beta$'
# draw func
utils_PM.drawPCS(rst_li, legend_li, pltNm, titleNm=titleNm)
utils_PM.drawPCS(rst_li, legend_li, pltNm + '_zoomIn', rgSt=0.048, rgTo=0.052, titleNm=titleNm)




##### state of the art ##########
#--------RGB
# rst_li = [
#     'vis2PM_exp_RGB-2-PMarray_phy-concat-1_nGated1_whtMode-autoWht-100_w-sumLoss100_clip11_w-actiFn_nD3_epoch25_5_bch70',
#     'pix2pix_exp_RGB-2-PMarray_clip11_D3_epoch25_5_bch70',
#     'cycle_gan_exp_uc_RGB-2-PMarray_clip11_D3_epoch25_5',
# ]
# legend_li = [
#     'autoW-sum',
#     'pix2pix',
#     'cycleGAN'
# ]
# pltNm = 'PADS-PM_RGB_soa'
# titleNm = 'RGB state-of-the-art'
# utils_PM.drawPCS(rst_li, legend_li, pltNm, idx_bold=0, titleNm=titleNm)          # add an emphasizing number

# -----------IR
# rst_li = [
#     # 'vis2PM_uc_unm_IR-2-PMarray_phy-concat-1_nGated1_whtMode-n-100_wo-sumLoss100_clip11_w-actiFn_nD3_epoch25_5_bch70',
#     'vis2PM_uc_unm_IR-2-PMarray_phy-concat-1_nGated1_whtMode-autoWht-100_w-sumLoss100_clip11_w-actiFn_nD3_epoch25_5_bch70',
#     # 'vis2PM_uc_unm_IR-2-PMarray_phy-concat-1_nGated1_whtMode-autoWht-100_w-sumLoss100_clip11_w-actiFn_D3_epoch25_5_bch70',
#     'pix2pix_uc_unm_IR-2-PMarray_clip11_D3_epoch25_5_bch70',
#     'cycle_gan_uc_unm_IR-2-PMarray_clip11_D3_epoch25_5'
# ]
# legend_li = [
#     # 'base',
#     'autoW-sum',
#     # 'autoW-sum-D',
#     'pix2pix',
#     'cycleGAN'
# ]
# pltNm = 'PADS-PM_IR_soa'
# titleNm = 'IR state-of-the-art'
# utils_PM.drawPCS(rst_li, legend_li, pltNm, idx_bold=0, titleNm=titleNm)          # add an emphasizing number


