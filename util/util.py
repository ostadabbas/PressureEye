"""This module contains simple helper functions """
from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os


def tensor2im(input_image, imtype=np.uint8, clipMod='clip11'):
	""""Converts a Tensor array into a numpy image array.

	Parameters:
		input_image (tensor) --  the input image tensor array
		imtype (type)        --  the desired type of the converted numpy array
	"""
	if not isinstance(input_image, np.ndarray):
		if isinstance(input_image, torch.Tensor):  # get the data from a variable
			image_tensor = input_image.data
		else:
			return input_image
		image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
		if image_numpy.shape[0] == 1:  # grayscale to RGB
			image_numpy = np.tile(image_numpy, (3, 1, 1))
		if 'clip11' == clipMod:
			image_numpy = (np.transpose(image_numpy,
			                            (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling, 01 mod
		else:  # for clip 11 operation
			image_numpy = (np.transpose(image_numpy, (1, 2, 0)) * 255.0)  # 01 scale directly   11 mod, save clip 01 so possibly large value in training
	else:  # if it is a numpy array, do nothing
		image_numpy = input_image
	return image_numpy.astype(imtype)


def diagnose_network(net, name='network'):
	"""Calculate and print the mean of average absolute(gradients)

	Parameters:
		net (torch network) -- Torch network
		name (str) -- the name of the network
	"""
	mean = 0.0
	count = 0
	for param in net.parameters():
		if param.grad is not None:
			mean += torch.mean(torch.abs(param.grad.data))
			count += 1
	if count > 0:
		mean = mean / count
	print(name)
	print(mean)


def save_image(image_numpy, image_path):
	"""Save a numpy image to the disk

	Parameters:
		image_numpy (numpy array) -- input numpy array
		image_path (str)          -- the path of the image
	"""
	image_pil = Image.fromarray(image_numpy)
	image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
	"""Print the mean, min, max, median, std, and size of a numpy array

	Parameters:
		val (bool) -- if print the values of the numpy array
		shp (bool) -- if print the shape of the numpy array
	"""
	x = x.astype(np.float64)
	if shp:
		print('shape,', x.shape)
	if val:
		x = x.flatten()
		print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
			np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
	"""create empty directories if they don't exist

	Parameters:
		paths (str list) -- a list of directory paths
	"""
	if isinstance(paths, list) and not isinstance(paths, str):
		for path in paths:
			mkdir(path)
	else:
		mkdir(paths)


def mkdir(path):
	"""create a single empty directory if it didn't exist

	Parameters:
		path (str) -- a single directory path
	"""
	if not os.path.exists(path):
		os.makedirs(path)


def cam2pixel(cam_coord, f, c):
	x = cam_coord[..., 0] / cam_coord[..., 2] * f[0] + c[0]
	y = cam_coord[..., 1] / cam_coord[..., 2] * f[1] + c[1]
	z = cam_coord[..., 2]

	return x, y, z


def pixel2cam(pixel_coord, f, c):
	x = (pixel_coord[..., 0] - c[0]) / f[0] * pixel_coord[..., 2]
	y = (pixel_coord[..., 1] - c[1]) / f[1] * pixel_coord[..., 2]
	z = pixel_coord[..., 2]

	return x, y, z


def get_bbox(joint_img):
	# bbox extract from keypoint coordinates
	bbox = np.zeros((4))
	xmin = np.min(joint_img[:, 0])
	ymin = np.min(joint_img[:, 1])
	xmax = np.max(joint_img[:, 0])
	ymax = np.max(joint_img[:, 1])
	width = xmax - xmin - 1
	height = ymax - ymin - 1

	bbox[0] = (xmin + xmax) / 2. - width / 2 * 1.2
	bbox[1] = (ymin + ymax) / 2. - height / 2 * 1.2
	bbox[2] = width * 1.2
	bbox[3] = height * 1.2

	return bbox


def nameToIdx(name_tuple, joints_name):  # test, tp,
	'''
	from reference joints_name, change current name list into index form
	:param name_tuple:  The query tuple like tuple(int,) or tuple(tuple(int, ...), )
	:param joints_name:
	:return:
	'''
	jtNm = joints_name
	if type(name_tuple[0]) == tuple:
		# Transer name_tuple to idx
		return tuple(tuple([jtNm.index(tpl[0]), jtNm.index(tpl[1])]) for tpl in name_tuple)
	else:
		# direct transfer
		return tuple(jtNm.index(tpl) for tpl in name_tuple)
