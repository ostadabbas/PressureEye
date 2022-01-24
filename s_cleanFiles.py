'''
clean up the folder with certain patterns.
'''
import os
import shutil
import os.path as osp

fds_kept = [
	# --------------RGB
	'vis2PM_exp_uc_RGB-2-PMarray_clip01_w-align_n_phy1_3stg_woactiFn_100.0L2_n-whtL100_0.0sum_0.0ssim_nD3_epoch25_5_bch30', # base
	'vis2PM_exp_uc_RGB-2-PMarray_clip01_w-align_n_phy1_3stg_woactiFn_100.0L2_auto-whtL100_0.0sum_0.0ssim_nD3_epoch25_5_bch30', # sa-100sum
	'vis2PM_exp_uc_RGB-2-PMarray_clip01_w-align_n_phy1_3stg_woactiFn_100.0L2_auto-whtL100_10000.0sum_0.0ssim_nD3_epoch25_5_bch30', # sa-sum
	'vis2PM_exp_uc_RGB-2-PMarray_clip01_w-align_n_phy1_3stg_woactiFn_100.0L2_auto-whtL100_10000.0sum_10.0ssim_nD3_epoch25_5_bch30', # sa sum sim
	'vis2PM_exp_uc_RGB-2-PMarray_clip01_w-align_n_phy1_3stg_woactiFn_100.0L2_auto-whtL100_10000.0sum_10.0ssim_D3_epoch25_5_bch20',  # sa sum sim  D
	'vis2PM_exp_uc_RGB-2-PMarray_clip01_w-align_n_phy0_3stg_woactiFn_100.0L2_n-whtL100_0.0sum_0.0ssim_nD3_epoch25_5_bch30', # n_phy0
	'vis2PM_exp_uc_RGB-2-PMarray_clip01_w-align_n_phy1_3stg_woactiFn_100.0L2_n-whtL100_10000.0sum_0.0ssim_nD3_epoch25_5_bch30',  # sum
	'vis2PM_exp_uc_RGB-2-PMarray_clip01_w-align_n_phy1_3stg_woactiFn_100.0L2_n-whtL100_0.0sum_10.0ssim_nD3_epoch25_5_bch30',  # sim
	'vis2PM_exp_uc_RGB-2-PMarray_clip01_w-align_n_phy1_3stg_woactiFn_100.0L2_n-whtL100_0.0sum_0.0ssim_D3_epoch25_5_bch20',    # D
	'openPose_exp_uc_RGB-2-PMarray_epoch25_5_bch30',
	'memNet_exp_uc_RGB-2-PMarray_epoch25_5_bch2',
	'pix2pix_exp_uc_RGB-2-PMarray_epoch25_5_bch70',
	'cycle_gan_exp_uc_RGB-2-PMarray_epoch25_5_bch3',
	# -------------IR
	'vis2PM_exp_uc_IR-2-PMarray_clip01_w-align_n_phy1_3stg_woactiFn_100.0L2_n-whtL100_0.0sum_0.0ssim_nD3_epoch25_5_bch30',
	# clip01
	'vis2PM_exp_uc_IR-2-PMarray_clip01_w-align_n_phy1_3stg_woactiFn_100.0L2_auto-whtL100_0.0sum_0.0ssim_nD3_epoch25_5_bch30',       # sa
	'vis2PM_exp_uc_IR-2-PMarray_clip01_w-align_n_phy1_3stg_woactiFn_100.0L2_auto-whtL100_10000.0sum_0.0ssim_nD3_epoch25_5_bch30',
	# sa_sum
	'vis2PM_exp_uc_IR-2-PMarray_clip01_w-align_n_phy1_3stg_woactiFn_100.0L2_auto-whtL100_10000.0sum_10.0ssim_nD3_epoch25_5_bch30',   # sa_sum_ssim
	'vis2PM_exp_uc_IR-2-PMarray_clip01_w-align_n_phy1_3stg_woactiFn_100.0L2_auto-whtL100_10000.0sum_10.0ssim_D3_epoch25_5_bch20',
	# sa_sum_ssim_D
	'vis2PM_exp_uc_IR-2-PMarray_clip01_w-align_n_phy0_3stg_woactiFn_100.0L2_n-whtL100_0.0sum_0.0ssim_nD3_epoch25_5_bch30',          # nPhy0
	'vis2PM_exp_uc_IR-2-PMarray_clip01_w-align_n_phy1_3stg_woactiFn_100.0L2_n-whtL100_10000.0sum_0.0ssim_nD3_epoch25_5_bch30',       # sum
	'vis2PM_exp_uc_IR-2-PMarray_clip01_w-align_n_phy1_3stg_woactiFn_100.0L2_n-whtL100_0.0sum_10.0ssim_nD3_epoch25_5_bch30',         # ssim
	'vis2PM_exp_uc_IR-2-PMarray_clip01_w-align_n_phy1_3stg_woactiFn_100.0L2_n-whtL100_0.0sum_0.0ssim_D3_epoch25_5_bch20',           # D
	'openPose_exp_uc_IR-2-PMarray_epoch25_5_bch30',
	'memNet_exp_uc_IR-2-PMarray_epoch25_5_bch2',
	'pix2pix_exp_uc_IR-2-PMarray_epoch25_5_bch70',
	'cycle_gan_exp_uc_IR-2-PMarray_epoch25_5_bch3',
	# --------------RGB2IR
	'vis2PM_exp_uc_RGB-2-IR_clip01_w-align_n_phy1_3stg_woactiFn_100.0L2_n-whtL100_0.0sum_0.0ssim_nD3_epoch25_5_bch30', 	# clip01
	'vis2PM_exp_uc_RGB-2-IR_clip01_w-align_n_phy1_3stg_woactiFn_100.0L2_auto-whtL100_0.0sum_0.0ssim_nD3_epoch25_5_bch30', 	# sa
	'vis2PM_exp_uc_RGB-2-IR_clip01_w-align_n_phy1_3stg_woactiFn_100.0L2_auto-whtL100_10000.0sum_0.0ssim_nD3_epoch25_5_bch30', 	# # sa_sum
	'vis2PM_exp_uc_RGB-2-IR_clip01_w-align_n_phy1_3stg_woactiFn_100.0L2_auto-whtL100_10000.0sum_10.0ssim_nD3_epoch25_5_bch30', 	# # sa_sum_ssim
	'vis2PM_exp_uc_RGB-2-IR_clip01_w-align_n_phy1_3stg_woactiFn_100.0L2_auto-whtL100_10000.0sum_10.0ssim_D3_epoch25_5_bch20', 	# # sa_sum_ssim_D
	'vis2PM_exp_uc_RGB-2-IR_clip01_w-align_n_phy0_3stg_woactiFn_100.0L2_n-whtL100_0.0sum_0.0ssim_nD3_epoch25_5_bch30', 	# # nPhy0
	'vis2PM_exp_uc_RGB-2-IR_clip01_w-align_n_phy1_3stg_woactiFn_100.0L2_n-whtL100_10000.0sum_0.0ssim_nD3_epoch25_5_bch30', 	# # sum
	'vis2PM_exp_uc_RGB-2-IR_clip01_w-align_n_phy1_3stg_woactiFn_100.0L2_n-whtL100_0.0sum_10.0ssim_nD3_epoch25_5_bch30', 	# ssim
	'vis2PM_exp_uc_RGB-2-IR_clip01_w-align_n_phy1_3stg_woactiFn_100.0L2_n-whtL100_0.0sum_0.0ssim_D3_epoch25_5_bch20', # D
	'openPose_exp_uc_RGB-2-IR_epoch25_5_bch30',
	'memNet_exp_uc_RGB-2-IR_epoch25_5_bch2',
	'pix2pix_exp_uc_RGB-2-IR_epoch25_5_bch70',
	'cycle_gan_exp_uc_RGB-2-IR_epoch25_5_bch3',
	# ------------ pwrs  exp
	'vis2PM_exp_uc_RGB-2-PMarray_clip01_n_phy1_stg3_whtL-pwrs-100.0L2_lap0.0_sum0.0_ssim0.0_D0.0L3', # pwrs0
	'vis2PM_exp_uc_RGB-2-PMarray_clip01_n_phy1_stg3_whtL-pwrs-100.0L2_lap0.001_sum0.0_ssim0.0_D0.0L3', # pwrs1e-3
]

# src_fd = 'checkpoints'
src_fd = 'results'
tar_fd = 'ckpt_kept'
rst_lst = os.listdir(src_fd)

# check if in there
# for file in fds_kept:       # the pwrs nor found, regen OK. reanme the clip01 thing
# 	if file not in rst_lst:
# 		print('not found', file)

# check not found
for file in os.listdir(src_fd):
	if file in fds_kept:
		print('found', file)
	else:
		print('not found, try remove', file)
		try:
			shutil.rmtree(osp.join(src_fd, file))
		except OSError as e:
			print("Error: %s - %s." % (e.filename, e.strerror))