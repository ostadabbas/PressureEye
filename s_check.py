'''
mainly used to check results with comparison.
'''
import os.path as osp
import cv2
import numpy as np
import argparse
import os
import os.path as osp

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--lambda_sum', type=float, default=0.0, help='sum loss coeff')
parser.add_argument('--lambda_ssim', type=float, default=0.0, help='sum loss coeff')
parser.add_argument('--lambda_lap', type=float, default=0.001, help='sum loss coeff')
parser.add_argument('--idx', type=int, default=0, help='result sample index 0 ~ 499')
parser.add_argument('--kdeMode', type=int, default=0, help='result sample index 0 ~ 499')
parser.add_argument('--h_mode', type=int, default=0, help='result sample index 0 ~ 499')
parser.add_argument('--h_base', type=float, default=1.0, help='result sample index 0 ~ 499')
parser.add_argument('--scale_img', type=float, default=2.0, help='scale the image range for visualization')
parser.add_argument('--name',default='exp', help='exp name')
parser.add_argument('--modSrc',default='RGB', help='input modalities')
parser.add_argument('--if_svImg',default='n', help='if save the image [y|n]')
parser.add_argument('--thr_pseudo', type=float, default=0.05, help='the threshold for the pseudo color map')



# get the basic options
opt, _ = parser.parse_known_args()
print(opt)
lambda_sum = opt.lambda_sum
kdeMode = opt.kdeMode
h_mode = opt.h_mode
h_base = opt.h_base
lambda_lap = opt.lambda_lap
modSrc = opt.modSrc
## pwrs no kkde
# expNm = 'vis2PM_exp_uc_{modSrc}-2-PMarray_n_phy1_stg3_whtL-pwrs-100.0L2_lap{lambda_lap}_sum{lambda_sum}_ssim0.0_D0.0L3'.format(**vars(opt))        # ver 1 name
## base typeWht n
# expNm = 'vis2PM_exp_uc_{modSrc}-2-PMarray_n_phy1_stg3_whtL-n-100.0L2_lap{lambda_lap}_sum{lambda_sum}_ssim0.0_D0.0L3'.format(**vars(opt))        # ver 1 name
## kde version
# expNm = 'vis2PM_exp_uc_RGB-2-PMarray_n_phy1_stg3_whtL-pwrs-100.0L2_kde{}_lap{}_sum{}_ssim0.0_D0.0L3'.format(kdeMode,opt.lambda_lap, lambda_sum)        # ver 1 name
## h mode
# expNm='vis2PM_{name}_uc_RGB-2-PMarray_n_phy1_stg3_whtL-pwrs-100.0L2_kde{kdeMode}_hMode{h_mode}-{h_base}_lap{lambda_lap}_sum{lambda_sum}_ssim0.0_D0.0L3'.format(**vars(opt))
## eccv version
# expNm = 'ECCV20/vis2PM_exp_uc_RGB-2-PMarray_clip01_w-align_n_phy1_3stg_woactiFn_100.0L2_auto-whtL100_0.0sum_0.0ssim_nD3_epoch25_5_bch30'
## openPose
# expNm = 'openPose_exp_uc_RGB-2-PMarray_epoch25_5_bch30'

## for the stage comparison
expNm = 'vis2PM_exp_uc_{modSrc}-2-PMarray_n_phy1_stg{n_stg}_whtL-pwrs-100.0L2_lap0.001_sum1e-06_ssim0.0_D0.0L3'
# expNm = 'vis2PM_exp_uc_RGB-2-PMarray_n_phy1_stg1_whtL-pwrs-100.0L2_lap0.001_sum1e-06_ssim0.0_D0.0L3'      # stg 1
# 'vis2PM_exp_uc_RGB-2-PMarray_n_phy1_stg3_whtL-pwrs-100.0L2_kde-1_lap0.001_sum1.0_ssim0.0_D0.0L3'
# print('kde mode is {}'.format(kdeMode))
# print('read from expNm {}'.format(expNm))

# print('read from file')
# print(expNm)
sv_fd = 'results/demoImgs'
if not osp.exists(sv_fd):
	os.makedirs(sv_fd)

expFd = osp.join('results', expNm, 'test_latest', 'demoImgs')
tar_idx = opt.idx
scale_img = opt.scale_img
if_svImg = opt.if_svImg
thr_pseudo = opt.thr_pseudo

# check diffs
li_diff = []
for idx in range(tar_idx, tar_idx+1): # single setting
# for idx in range(500):        # loop setting
	# idx = 0
	## real fake compare
	# fakeNm = 'demo{}_fake_B2.png'.format(idx)
	# realNm = 'demo{}_real_B.png'.format(idx)
	# fake = cv2.imread(osp.join(expFd, fakeNm), flags=cv2.IMREAD_GRAYSCALE)
	# real = cv2.imread(osp.join(expFd, realNm), flags=cv2.IMREAD_GRAYSCALE)
	# fake = (fake*scale_img).clip(0, 255).astype(np.uint8)
	# real = (real*scale_img).clip(0, 255).astype(np.uint8)
	# fake = cv2.applyColorMap(fake, cv2.COLORMAP_JET)
	# real = cv2.applyColorMap(real, cv2.COLORMAP_JET)
	# img_cmb = np.concatenate([real, fake], axis=1)

	## read in directly
	# cmbNm = 'demo{}_wht_cmb.png'.format(idx)
	# img_cmb = cv2.imread(osp.join(expFd, cmbNm), flags=cv2.IMREAD_GRAYSCALE)
	# img_cmb = cv2.applyColorMap(img_cmb, cv2.COLORMAP_JET)


	## stg comp
	li_img = []
	for i in range(1,4):
		nm = expNm.format(n_stg=i, **vars(opt)) # src + stg
		expFd = osp.join('results', nm, 'test_latest', 'demoImgs')
		print("read from", nm)
		fakeNm = 'demo{}_fake_B{}.png'.format(idx, i-1)
		realNm = 'demo{}_real_B.png'.format(idx)
		fake = cv2.imread(osp.join(expFd, fakeNm), flags=cv2.IMREAD_GRAYSCALE)
		fake = (fake * scale_img).clip(0, 255).astype(np.uint8)
		imgM = fake.copy()
		fake = cv2.applyColorMap(fake, cv2.COLORMAP_JET)
		if thr_pseudo>0:
			fake[imgM<thr_pseudo*255]=0
		if i == 1:
			real = cv2.imread(osp.join(expFd, realNm), flags=cv2.IMREAD_GRAYSCALE)
			real = (real * scale_img).clip(0, 255).astype(np.uint8)
			imgM = real.copy()
			real = cv2.applyColorMap(real, cv2.COLORMAP_JET)
			if thr_pseudo >0 :
				real[imgM<thr_pseudo*255] = 0
			li_img.append(real)
		li_img.append(fake)
		img_cmb = np.concatenate(li_img, axis=1)

	if 'y' == if_svImg:
		cv2.imwrite(osp.join(sv_fd, 'stgs_{}.png'.format(modSrc)), img_cmb)



	cv2.imshow('compare', img_cmb)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	# print('real sum is', real.sum())
	# print('fake sum is', fake.sum())
	# print('real min {} max {}'.format(real.min(), real.max()))
	# print('fake min {} max {}'.format(fake.min(), fake.max()))
	# diff = fake.sum() - real.sum()
	# print('sum difference is', diff)
	# li_diff.append(diff)

# arr_diff = np.array(li_diff)
# print('mean error is', np.abs(arr_diff).mean())