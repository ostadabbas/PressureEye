"""
generate and save the metrics  mse, pcs ..
"""
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import util.utils_PM as utils_PM
matplotlib.rc('xtick', labelsize=15)
matplotlib.rc('ytick', labelsize=15)
matplotlib.rc('axes',labelsize=18)

if __name__ == '__main__':
    opt_test = TestOptions().parse()  # get test options
    # nm = opt_test.name

    # hard-code some parameters for test
    opt_test.num_threads = 0   # test code only supports num_threads = 1
    opt_test.batch_size = 1    # test code only supports batch_size = 1
    opt_test.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt_test.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt_test.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset_test = create_dataset(opt_test)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt_test)      # create a model given opt.model and other options
    model.setup(opt_test)               # regular setup: load and print networks; create schedulers
    # create a website
    web_dir = os.path.join(opt_test.results_dir, opt_test.name, '%s_%s' % (opt_test.phase, opt_test.epoch))  # define the website directory
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt_test.name, opt_test.phase, opt_test.epoch))
    if not os.path.exists(web_dir):  # check and make dir
        os.makedirs(web_dir)

    if opt_test.eval:
        model.eval()
    if 'clip11' == opt_test.pmDsProc:
        bs_sensing = -1
        rg_sensing = 2
    elif 'clip01' == opt_test.pmDsProc:
        bs_sensing = 0
        rg_sensing = 1
    else:
        print('no such pmDsProc, exit1')
        os.exit(-1)
    print('Initiate final test')
    fake_vStk, real_vStk, mse, mse_efs, psnr, psnr_efs, pcs_efs, pcs_efs01, ssim, fp = utils_PM.test(model, dataset_test, opt_test.num_test, web_dir, opt_test.if_saveImg, R=rg_sensing, efs_rt=opt_test.efs_rt, pcs_test=opt_test.pcs_test)
    # ver2 save and print
    print('{:6} {:10} {:10} {:10} {:10} {:10} {:10} {:10} {:10}'.format('epoch','mse','mse_efs', 'psnr', 'psnr_efs', 'pcs_efs{}'.format(opt_test.pcs_test),'pcs_efs01','ssim', 'fp'))
    print('{:6} {:10.5f} {:10.5f} {:10.5f} {:10.5f} {:10.5f} {:10.5f} {:10.5f} {:10.5f}'.format('final', mse, mse_efs, psnr, psnr_efs, pcs_efs, pcs_efs01, ssim, fp))
    # save all the result
    np.savez(os.path.join(web_dir, 'rst{}.npz'.format(opt_test.efs_rt)), mse=mse, mse_efs=mse_efs, psnr=psnr, psnr_efs=psnr_efs, pcs_efs=pcs_efs, pcs_efs01=pcs_efs01, ssim=ssim, fp=fp)
    if opt_test.if_saveDiff == 'y':
        np.savez(os.path.join(web_dir, opt_test.predNm), fake_vStk=fake_vStk, real_vStk=real_vStk)

