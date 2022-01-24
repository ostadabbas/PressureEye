'''
image processing related.
'''
import os
import skimage
from skimage import io
from skimage import transform
from skimage import color
import math
import cv2


## test how to write  text on a image
rstFd = 'results'
outFd = 'out_test'
if not os.path.exists(outFd):
    os.makedirs(outFd)

# mdlNm = 'cycle_gan_rst_test'
# mdlNm = 'vis2PM_ts1_uc_IR-2-PM_clip01_w-align_phy1_3stg_woactiFn_100.0L2_auto-whtL100_1e-07sum_10.0ssim_D3_epoch1_0_bch20'
mdlNm = 'vis2PM_ts1_uc_RGB-2-PMarray_clip01_w-align_phy1_3stg_woactiFn_100.0L2_auto-whtL100_1e-07sum_10.0ssim_D3_epoch1_0_bch20'
imgNm = 'demo0_real_B.png'
text = 'test'
if_gray = True
pth_img = os.path.join(rstFd, mdlNm, 'test_latest', 'demoImgs', imgNm)
img = io.imread(pth_img)
imgCv2 = skimage.img_as_ubyte(img[:, :, ::-1])     # make it to uint8   gray works
imgCv2_cmap = cv2.applyColorMap(imgCv2, cv2.COLORMAP_JET)

## different behavior when gray_img transferred to different range 0 to 1 and -1 to 1
print('min max before trans', img.min(), img.max())
img01 = skimage.img_as_float(img)
img11 = img01*2-1
io.imsave(os.path.join(outFd, 'img01.png'), img01)
io.imsave(os.path.join(outFd, 'img11.png'), img11)


# if if_gray:
#     if len(img.shape) > 2:
#         img = color.rgb2gray(img)
# elif len(img.shape) < 3:

## for putting text on test
# img = color.gray2rgb(img)
# if if_gray:
#     img = color.rgb2gray(img)
#     img = skimage.img_as_ubyte(img)
#     img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
# cv2.putText(img, text, (5,20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

# cv2.imshow('test text', imgCv2_cmap)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
