'''
for key points visualization. Also visualizer for visdom class.
'''
import os
import os.path as osp
import cv2
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
import ntpath
import time
# from . import utils_tool, html
from subprocess import Popen, PIPE
# from scipy.misc import imresize
from skimage.transform import resize  # misc deprecated e


def vis_keypoints(img, kps, kps_lines, kp_thresh=0.4, alpha=1):
	'''

	:param img:
	:param kps: 3 * n_jts
	:param kps_lines:
	:param kp_thresh:
	:param alpha:
	:return:
	'''
	# Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
	cmap = plt.get_cmap('rainbow')
	colors = [cmap(i) for i in np.linspace(0, 1, len(kps_lines) + 2)]
	colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

	# Perform the drawing on a copy of the image, to allow for blending.
	kp_mask = np.copy(img)

	# Draw the keypoints.
	for l in range(len(kps_lines)):
		i1 = kps_lines[l][0]
		i2 = kps_lines[l][1]
		p1 = kps[0, i1].astype(np.int32), kps[1, i1].astype(np.int32)
		p2 = kps[0, i2].astype(np.int32), kps[1, i2].astype(np.int32)
		if kps[2, i1] > kp_thresh and kps[2, i2] > kp_thresh:
			cv2.line(
				kp_mask, p1, p2,
				color=colors[l], thickness=2, lineType=cv2.LINE_AA)
		if kps[2, i1] > kp_thresh:
			cv2.circle(
				kp_mask, p1,
				radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)
		if kps[2, i2] > kp_thresh:
			cv2.circle(
				kp_mask, p2,
				radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)

	# Blend the keypoints.
	return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)


def vis_3d_skeleton(kpt_3d, kpt_3d_vis, kps_lines, filename=None, input_shape=(256, 256), if_dsFmt=True):
	# worked mainly for ds format with range set properly
	# vis with x, z , -y
	# plt.clf()
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')

	# Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
	cmap = plt.get_cmap('rainbow')
	colors = [cmap(i) for i in np.linspace(0, 1, len(kps_lines) + 2)]
	colors = [np.array((c[0], c[1], c[2])) for c in colors]  # array list

	for l in range(len(kps_lines)):
		i1 = kps_lines[l][0]
		i2 = kps_lines[l][1]
		x = np.array([kpt_3d[i1, 0], kpt_3d[i2, 0]])
		y = np.array([kpt_3d[i1, 1], kpt_3d[i2, 1]])
		z = np.array([kpt_3d[i1, 2], kpt_3d[i2, 2]])

		if kpt_3d_vis[i1, 0] > 0 and kpt_3d_vis[i2, 0] > 0:
			ax.plot(x, z, -y, c=colors[l], linewidth=2)
		if kpt_3d_vis[i1, 0] > 0:
			ax.scatter(kpt_3d[i1, 0], kpt_3d[i1, 2], -kpt_3d[i1, 1], c=[colors[l]], marker='o')
		if kpt_3d_vis[i2, 0] > 0:
			ax.scatter(kpt_3d[i2, 0], kpt_3d[i2, 2], -kpt_3d[i2, 1], c=[colors[l]], marker='o')

	x_r = np.array([0, input_shape[1]], dtype=np.float32)
	y_r = np.array([0, input_shape[0]], dtype=np.float32)
	z_r = np.array([0, 1], dtype=np.float32)

	if filename is None:
		ax.set_title('3D vis')
	else:
		ax.set_title(filename)

	ax.set_xlabel('X Label')
	ax.set_ylabel('Z Label')
	ax.set_zlabel('Y Label')
	if if_dsFmt:  # if ds format , then form it this way
		ax.set_xlim([0, input_shape[1]])
		ax.set_ylim([0, 1])
		ax.set_zlim([-input_shape[0], 0])
	# ax.legend()

	plt.show()
	cv2.waitKey(0)


def vis_entry(entry_dict):
	'''
	from the entry dict plot the images
	:param entry_dict:
	:return:
	'''


if sys.version_info[0] == 2:
	VisdomExceptionBase = Exception
else:
	VisdomExceptionBase = ConnectionError


def save_images(webpage, visuals, image_path, aspect_ratio=1.0, width=256):
	"""Save images to the disk. Also to webpage

	Parameters:
		webpage (the HTML class) -- the HTML webpage class that stores these imaegs (see html.py for more details)
		visuals (OrderedDict)    -- an ordered dictionary that stores (name, images (either tensor or numpy) ) pairs
		image_path (str)         -- the string is used to create image paths
		aspect_ratio (float)     -- the aspect ratio of saved images
		width (int)              -- the images will be resized to width x width

	This function will save images stored in 'visuals' to the HTML file specified by 'webpage'.
	"""
	image_dir = webpage.get_image_dir()
	short_path = ntpath.basename(image_path[0])
	name = os.path.splitext(short_path)[0]

	webpage.add_header(name)
	ims, txts, links = [], [], []

	for label, im_data in visuals.items():
		im = utils_tool.tensor2im(im_data)
		image_name = '%s_%s.png' % (name, label)
		save_path = os.path.join(image_dir, image_name)
		h, w, _ = im.shape
		if aspect_ratio > 1.0:
			im = resize(im, (h, int(w * aspect_ratio)))
		if aspect_ratio < 1.0:
			im = resize(im, (int(h / aspect_ratio), w))
		utils_tool.save_image(im, save_path)

		ims.append(image_name)
		txts.append(label)
		links.append(image_name)
	webpage.add_images(ims, txts, links, width=width)


def ipyth_imshow(img):
	# use ipython to show an image
	import cv2
	import IPython
	_, ret = cv2.imencode('.jpg', img)
	i = IPython.display.Image(data=ret)
	IPython.display.display(i)
