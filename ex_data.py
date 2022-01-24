'''
check the data
'''

from util import utils_PM
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os.path as path


idx_subj = 1
idx_frm = 4
A = 0.0102 **2
g = 9.8

dsFd = r'S:\ACLab\datasets\SLP\danaLab'
PM = utils_PM.getImg_dsPM(dsFd, idx_subj, 'PMarray', 'uncover', idx_frm)  # interface literally from idx 1

phys_arr = np.load(path.join(dsFd, 'physiqueData.npy'))
pth_cali = path.join(dsFd, '{:05d}'.format(idx_subj), 'PMcali.npy')
caliPM = np.load(pth_cali)
caliPM_t = caliPM[0][3] # the 4 th

wt = phys_arr[idx_subj-1][2]    # wt
sm = PM.sum()
print('cali rt is', caliPM_t)
 # rt = wt * g / A / PMarr.sum() / 1000  # kPa/unit  transformer
print('array  sum is', PM.sum())
print('wt is', wt)
print('sum is',  PM.sum())
print("recalculate rt",  wt*g/A/sm/1000)

