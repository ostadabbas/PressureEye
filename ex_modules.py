# ex for modules
from PIL import Image
import visdom
import numpy as np
import util.utils_PM as utils_PM
import argparse
import torchvision.transforms as transforms
from util import utils_PM
import cv2
import torch.nn as nn
import os
import torch
from util import utils_PM
import skimage
import skimage.io as io
import math
import matplotlib.pyplot as plt
import matplotlib
import math
# from skimage.metrics import structural_similarity as ssim # only works for 0.16dev!!
import pytorch_ssim


# visdom
# vis = visdom.Visdom()
# vis.text('Hello, world!')
# vis.image(np.ones((3, 10, 10)))

# imshow
# imgPth = utils_PM.getPth()
# img = Image.open(imgPth)
if not os.path.exists('rst'):
    os.mkdir('rst')
dsFd = r'G:\My Drive\ACLab Shared GDrive\datasetPM\danaLab'
# modality = 'depthRaw'
modality = 'RGB'
depth_min_li = []
depth_max_li = []
# if modality in {'RGB', 'IR'}:
#     readFunc = io.imread
# else:
#     readFunc = np.load

# image operations
for i in range(1):
    pthPTr = os.path.join(dsFd, '{:05d}'.format(i+1), 'align_PTr_{}.npy'.format('depth'))
    PTr_depth = np.load(pthPTr)
    for j in range(1):
        # IR = utils_PM.getImg_dsPM(modality='IR')
        # depthRaw = utils_PM.getImg_dsPM(modality='depthRaw')
        # print(IR.shape)
        # # IR_rot = skimage.transform.rotate(IR, 30)
        # # IR_test = skimage.util.crop(IR, ((-10,-10)))    # can't negative values
        # simTsfm = skimage.transform.SimilarityTransform(scale=1, rotation=math.radians(45), translation=(20, 20))
        # IR_test = skimage.transform.warp(IR, simTsfm)
        # io.imshow(IR_test)
        # io.show()
        # print(depthRaw.shape)
        # pth = utils_PM.getPth(idx_subj=i+1, modality=modality, idx_frm=j+1)
        # depthRaw = np.load(pth)
        # img = utils_PM.getImg_dsPM(idx_subj=i+1, modality=modality, idx_frm=j+1)
        # print(img.dtype)
        # print(img.min())
        # print(img.max())
        # if np.uint8 == img.dtype:
        #     img = skimage.img_as_float(img)
        # print('after change')
        # print(img.dtype)
        # print(img.min())
        # print(img.max())
        # h,w  = depthRaw.shape
        # c_x = (w+1)/2
        # c_y = (h+1)/2
        # # plt.figure()
        # # plt.imshow(depthRaw)
        # sc = 1.5
        # # A to C_s, center scale then move center back
        # M = np.array([
        #     [sc, 0 , (1-sc)*c_x],
        #                   [0, sc,  (1-sc)*c_y],
        #                   [0, 0 , 1]])
        # tsfm = skimage.transform.AffineTransform(M)
        # tsfm2 = skimage.transform.AffineTransform(scale =(sc,sc))
        # # depth_warp = skimage.transform.warp(depthRaw, tsfm2)
        # # depth_warp = skimage.transform.rescale(depthRaw, sc)
        # depth_warp = utils_PM.affineImg(depthRaw, 1.1, 20, (50,50))
        # plt.figure()
        # plt.imshow(depth_warp)
        # plt.show()


        # depthWarp = cv2.warpPerspective(depth, PTr_depth, (84, 192))
        # cv2.imshow('warpedDepth', depthWarp)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
#         # test the Image read in
#         # Image, channel first
#         print('depthRaw, max', depthWarp.max())
#         img_depth = Image.fromarray(depthWarp)
#         print('img_depth dimension', img_depth.size)    # recovered correctly
#         depth_rec = np.array(img_depth)
#         print('max recover depth is', depth_rec.max())
#         # img_depth = Image.fromarray(depthWarp.transpose((2,0,1)))
#         # img_depth.show()        # all white color,  perhaps store original high value
#         tsfm = transforms.Grayscale(1)  # result still 18 to 182
#         # tsfm = transforms.Normalize((0.5,), (0.5,)) # normalize to 0.5 ? only for tensor image
#         tsfm = transforms.ToTensor()
#         img_depth = tsfm(depthWarp.astype('int')) # to tensor if image more than 255, return the
#         arr_im = np.array(img_depth)  # default is 0 to 182 uint8
#         print('arr_im max', arr_im.max(), 'min', arr_im.min())  # check the IR image,
        # 0 to
        # skimage.io.imshow(depthWarp)        # can't show image object
        # skimage.io.show()
        # viewer = ImageViewer(depthWarp)
        # viewer.show()
        pass

# plot function test
# test against plot
# plt.plot([1, 2, 3, 4])
# plt.ylabel('some numbers')
# plt.savefig('haha.pdf')
# matplotlib.rc('xtick', labelsize=12)
# matplotlib.rc('ytick', labelsize=12)
# matplotlib.rc('axes',labelsize=15)
# matplotlib.rc('legend', fontsize=15)
# t = np.linspace(0, 2*math.pi, 10)
# # for i in range(len(t)):
# #     print('t value in order', t[i])
# a = np.sin(t)
# b = np.cos(t)
# c = a + b
# plt.plot(t, a, label='plot r') # plotting t, a separately
# plt.plot(t, b, label='plot b') # plotting t, b separately
# plt.plot(t, c, label='plot g') # plotting t, c separately
# plt.legend(loc='upper left')
# plt.xlabel('PCS')
# plt.ylabel('Accuracy(%)')
# plt.show()

# print(im.size)
# im.crop((0,0,150,100)).show()       # larger x?
# im.rotate(45).show()        # default BMP app to open

# test parser
# parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# parser.add_argument('--test_li', nargs='+', default=1, help='test list argument')
# opt = parser.parse_args()
# print('test_li types', type(opt.test_li))
# print(opt.test_li)
# print(len(opt.test_li))

# test torch.nn
# nn_li = nn.ModuleList()
# for i in range(3):
#     nn_li.append(nn.Linear(3,3))
# net = nn.Sequential(*nn_li) # open list   can be expanded
# print(net)
# def changeInput(ts):
#     x = ts
#     x[0,0] = 100
# arr1 = np.array([1,2,3])
# ts1 = torch.Tensor(arr1[:0])        # empty tensor
# ts2 = torch.Tensor([[1,2,3], [4,5,6]])
# ts3 = torch.cat((ts1, ts2)) # can be concatenated, higher dim also work
# x = ts3
# x = ts1     # assign will pointer to not affect original
# ts4 = ts3   # reference changes
# ts4[0,0] = 100  # will affect ts3
# changeInput(ts3)    # will affect ts3
# print(ts1)
# print(ts2)
# print(ts3)
# print(ts3[0,2])
# ts1 = torch.Tensor([-11,2,3])
# ts2 = torch.Tensor([4,5])
# print(ts1 * ts2)    # same length works
# print(ts1.abs()) # worked
# print(torch.abs(ts1))

# test plot ###########
# import numpy as np
# import matplotlib.pyplot as plt
# plt.rc('font', family='Times New Roman')
# N = 50
# area = np.pi * (15 * np.random.rand(N)) ** 2
# plt.scatter(np.random.rand(N),
#             np.random.rand(N),
#             s=area,
#             c=np.random.rand(N),
#             alpha=0.5)
# plt.title("Sample title")
# plt.ylabel("Random Value")
# plt.xlabel("Random Variable")
# plt.show()

# pytorch ssim
# img1 = torch.Variable(torch.rand(1, 1, 256, 256))
# img2 = torch.Variable(torch.rand(1, 1, 256, 256))
# img1 = torch.rand(1,1,256,256)
# img2 = torch.rand(1,1,256,256)
# ones1 = torch.ones(1, 1,256,256)
# ones2 = torch.ones(1, 1,256,256)
# if torch.cuda.is_available():
#     img1 = img1.cuda()
#     img2 = img2.cuda()
# print(pytorch_ssim.ssim(img1, img2))
# ssim_loss = pytorch_ssim.SSIM(window_size = 11)
# print(ssim_loss(img1, img2))
# print('ones ssim', pytorch_ssim.ssim(ones1, ones2).item()) # must dim 4

## scalar can be saved viw nzp
# x=2
# y=10
# np.savez('haha.npz',x=x, y=y)
# file_loaded = np.load('haha.npz')
# print(file_loaded['x'])

# path join test.
print(os.path.join('haha','', 'mama'))

