'''
generate the histogram comparison with RGB and PM data
'''

from util import utils_PM
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os.path as path

font = {'family': 'Times New Roman',
        'weight': 'bold',
        'size': 22,
        }
matplotlib.rc('font', **font)

# dsFd =r'/home/liu.shu/datasets/datasetPM/danaLab'   # for  discovery
dsFd = r'S:\ACLab\datasets\SLP\danaLab'
outFd = 'output_tmp'

# imgRGB = utils_PM.getImg_dsPM(dsFd, 0, mod, cov, idx_frm)  # interface literally from idx 1
RGB = utils_PM.getImg_dsPM(dsFd, 1, 'RGB', 'uncover', 1)  # interface literally from idx 1
PM = utils_PM.getImg_dsPM(dsFd, 1, 'PMarray', 'uncover', 1)  # interface literally from idx 1
PM = PM.astype(np.uint8)
# PM is float64
bins = np.arange(256) - 0.5      # 1 more than the window value
colors = ['red', 'green', 'blue']
RGB_flt = RGB[-1, :]        # 576 x 3 why not correct hist

# fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2)
fig, ax0 = plt.subplots()
ax0.hist(RGB.reshape([-1, 3]), bins=50, density=True, color=colors, label=colors)    # show density, no read doesn't make sense???
# ax0.set_title('RGB histogram')
ax0.legend(prop={'size': 30})
fig.tight_layout()
fig.savefig(path.join(outFd, 'histRGB.pdf'))

fig, ax1 = plt.subplots()
ax1.hist(PM.flatten(), bins=50, density=True)  # show density, no read doesn't make sense???
# ax1.set_title('PM histogram')
# ax.legend(prop={'size': 10})
fig.tight_layout()
# plt.show()
fig.savefig(path.join(outFd, 'histPM.pdf'))

# plt.show()
# vis checking
# cv2.imshow('test', PM)      # black and white ?
# cv2.waitKey(0)
# cv2.destroyAllWindows()