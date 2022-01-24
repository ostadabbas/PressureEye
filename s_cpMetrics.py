'''
mainly used to check results with comparison of certain metric. For hyperparameter search.
'''
import os.path as osp
import cv2
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--lambda_sum', type=float, default=0.0, help='sum loss coeff')
parser.add_argument('--lambda_lap', type=float, default=0.001, help='sum loss coeff')
parser.add_argument('--idx', type=int, default=0, help='result sample index 0 ~ 499')
parser.add_argument('--kdeMode', type=int, default=0, help='result sample index 0 ~ 499')
parser.add_argument('--h_mode', type=int, default=0, help='result sample index 0 ~ 499')
parser.add_argument('--h_base', type=float, default=1.0, help='result sample index 0 ~ 499')
parser.add_argument('--scale_img', type=float, default=2.0, help='scale the image range for visualization')
parser.add_argument('--name',default='exp', help='exp name')

# only compare the sum0 and sum0.00001 to find pcs candidate

# get the basic options
opt, _ = parser.parse_known_args()
print(opt)
lambda_sum = opt.lambda_sum
kdeMode = opt.kdeMode
h_mode = opt.h_mode
h_base = opt.h_base
lambda_lap = opt.lambda_lap

## pwrs no kkde
# expNm = 'vis2PM_exp_uc_RGB-2-PMarray_n_phy1_stg3_whtL-pwrs-100.0L2_lap{lambda_lap}_sum{lambda_sum}_ssim0.0_D0.0L3'.format(**vars(opt))        # ver 1 name
## kde version
# expNm = 'vis2PM_exp_uc_RGB-2-PMarray_n_phy1_stg3_whtL-pwrs-100.0L2_kde{}_lap{}_sum{}_ssim0.0_D0.0L3'.format(kdeMode,opt.lambda_lap, lambda_sum)        # ver 1 name
## h mode
# expNm='vis2PM_{name}_uc_RGB-2-PMarray_n_phy1_stg3_whtL-pwrs-100.0L2_kde{kdeMode}_hMode{h_mode}-{h_base}_lap{lambda_lap}_sum{lambda_sum}_ssim0.0_D0.0L3'.format(**vars(opt))
## eccv version
# expNm = 'ECCV20/vis2PM_exp_uc_RGB-2-PMarray_clip01_w-align_n_phy1_3stg_woactiFn_100.0L2_auto-whtL100_0.0sum_0.0ssim_nD3_epoch25_5_bch30'


# expNmT = 'vis2PM_exp_uc_IR-2-PMarray_n_phy1_stg3_whtL-pwrs-100.0L2_lap{lambda_lap}_sum{lambda_sum}_ssim0.0_D0.0L3'\
expNmT = 'vis2PM_exp_uc_RGB-2-PMarray_n_phy1_stg3_whtL-pwrs-100.0L2_lap{lambda_lap}_sum{lambda_sum}_ssim0.0_D0.0L3'\
	# .format(	**vars(opt))  # ver 1 name
# print('read from file')
# print(expNm)

rstFd = 'results'
predNm = 'rst0.05.npz'      # for pcs0.05
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

args = parser.parse_args()
# outNm = args.outNm + '.txt'
li_lSum = [0.,0.1,0.01, 0.001, 0.0001, 0.00001, 1e-6, 1e-7, 1e-8, 1e-9]
# li_lSum = [0., 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]
# li_lLap = [0., 0.00001]
li_pcs01 = []
for lapSum in li_lSum:
	rstNm = expNmT.format(lambda_sum=lapSum,lambda_lap=lambda_lap)
	rstPth = os.path.join(rstFd, rstNm, 'test_latest', predNm)
	rst = np.load(rstPth)
	pcs01 = rst['pcs_efs01'].flatten()[0]
	li_pcs01.append(pcs01)

print('lap {} sum 0 ~ 0.00001'.format(lambda_lap))
print(li_pcs01)