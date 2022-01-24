'''
loop through the datasets to for point estimation.
1. for pixel value prob density over the dataset.  loop all, get the  hist(density), average them , save
'''
from options.test_options import TestOptions
from options.train_options import TrainOptions
from data import create_dataset
import matplotlib.pyplot as plt
import numpy as np
from data.pm_dataset import PmDataset
from tqdm import tqdm
import argparse
import json

if __name__ == '__main__':
	opt = TrainOptions().parse()
	out_fd = 'output_tmp'

	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--st_hist', type=int, default=0, help='the starting point of hist')
	parser.add_argument('--i_end', type=int, default=5, help='the starting point of hist')
	opt_ct, _ = parser.parse_known_args()


	# opt.phase = 'test'
	# opt = TestOptions().parse()

	# ds = create_dataset(opt)
	ds = PmDataset(opt)
	nbin = 100
	st_hist = opt_ct.st_hist        # try higher end
	i_end = opt_ct.i_end

	if i_end<0:     #
		i_end = len(ds)

	li_hist = []
	print('loop through {} samples'.format(i_end))
	for i in tqdm(range(i_end)): # iterater too slow?
		input = ds.__getitem__(i)
		B = input['B'].numpy().squeeze()
		B_np = B * nbin  # 0 ~ 1 so map to 0 ~ nbins
		bins = np.arange(nbin+2)-0.5        # -0.5 ~  nbins+0.5
		# print("bin gotten")
		# print('B_np min {} max {}'.format(B_np.min(), B_np.max()))
		hist, bins_o = np.histogram(B_np.flatten(), bins)
		# print("hist {} gotten".format(i))
		li_hist.append(hist)

	# print(li_hist)
	hist = li_hist[0]
	arr_hist = np.array(li_hist)
	hist_ave = arr_hist.mean(axis=0)
	print(len(hist_ave))
	print(hist_ave[st_hist:])

	# save to json
	rst = {'hists':arr_hist.tolist(), 'hist_ave': hist_ave.tolist()}
	with open('output_tmp/hist_pwrs.json', 'w') as f:      # raw hist
		json.dump(rst, f, allow_nan=True)

	# show
	# plt.bar(bins[:-1], hist, width=0.7)
	ax1 = plt.subplot(1, 2, 1)
	ax2 = plt.subplot(1, 2, 2)
	ax1.bar(bins[st_hist:-1]+0.5, hist[st_hist:], width=0.7)
	ax2.bar(bins[st_hist:-1]+0.5, hist_ave[st_hist:], width=0.7)


	plt.show()