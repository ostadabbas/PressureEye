"""
ds_loader, model, init/load in.
test included. So testPM not necessary at the moment.

"""
import time
from options.train_options import TrainOptions
from options.test_options import TestOptions
from data import create_dataset # load the corresponding class according to opts
from models import create_model
from util.visualizer import Visualizer
import torch
import numpy as np
from util.visualizer import save_images
import os
import tqdm
import util.utils_PM as utils_PM
import datetime

if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    opt_test = TestOptions().parse()
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options, only return Image obj no normal
    dataset_test = create_dataset(opt_test) # change bch to 1 according to phase.
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers, only continue train will load, otherwise scratch
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations

    if 'clip11' == opt.pmDsProc:
        bs_sensing = -1
        rg_sensing = 2
    elif 'clip01' == opt.pmDsProc:
        bs_sensing = 0
        rg_sensing = 1
    else:
        print('no such pmDsProc, exit1')
        os.exit(-1)
    thr = bs_sensing + rg_sensing * opt.efs_rt
    web_dir = os.path.join(opt_test.results_dir, opt_test.name,
                           '%s_%s' % (opt_test.phase, opt_test.epoch))  # define the website directory rst folder is from the test opt, so according to that bch setting
    print('beginning training at ' + str(datetime.datetime.now()))      # make a time record in  output log
    epoch = opt.epoch_count # keep oinitial could be latest.
    if 'n' == opt.if_test:  # if not only for test purpose
        for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
            epoch_start_time = time.time()  # timer for entire epoch
            iter_data_time = time.time()    # timer for data loading per iteration
            epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
            tm_st = time.time()
            for i, data in enumerate(dataset):  # inner loop within one epoch
                if i == opt.n_train:        # break out if reached max training samples.
                    break
                iter_start_time = time.time()  # timer for computation per iteration
                if total_iters % opt.print_freq == 0:
                    t_data = iter_start_time - iter_data_time
                visualizer.reset()
                # total_iters += opt.batch_size     # iter as sample number
                total_iters += 1    # total use the iter without considering the batch size
                # epoch_iter += opt.batch_size
                epoch_iter += 1
                model.set_input(data)         # unpack data from dataset and apply preprocessing
                model.optimize_parameters()   # calculate loss functions, get gradients, update network weights
                if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                    save_result = total_iters % opt.update_html_freq == 0   # 20 update
                    model.compute_visuals()
                    visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)
                if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                    losses = model.get_current_losses()
                    t_comp = (time.time() - iter_start_time) / opt.batch_size
                    visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                    if opt.display_id > 0:
                        visualizer.plot_current_losses(epoch, float(epoch_iter) * opt.batch_size / dataset_size,
                                                       losses)  # ratio of epoch

                if total_iters % opt.save_latest_freq == 0 and opt.save_latest_freq>0:   # cache our latest model every <save_latest_freq> iterations
                    print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                    save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                    model.save_networks(save_suffix)
                iter_data_time = time.time()
            if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
                print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
                model.save_networks('latest')       # always keep a copy of the last epoch model, only save at certain iter instead of last.
                model.save_networks(epoch)
            # -- check the sfg layer ---
            if opt.phyMode == 'sfg' and False:  # check weight only, inner wired here for debug purpose
                if len(opt.gpu_ids) > 0:
                    fc_gate = model.netG.module.fc_gate
                else:
                    fc_gate = model.netG.fc_gate
                print('the fc_gate linear weight is')
                for i, module in enumerate(fc_gate.children()):  # single layer not as list?
                    if type(module) == torch.nn.Linear:
                        print(module.weight)
            #----- test in traing -----
            model.eval()
            diff_vStk, real_vStk, mse, mse_efs, psnr, psnr_efs, pcs_efs, pcs_efs01, ssim, fp = utils_PM.test(model, dataset_test, opt.n_train, web_dir, R=rg_sensing, efs_rt=opt.efs_rt, pcs_test=opt.pcs_test)  # pcs_test the ratio to print out      # simply follow n_train image save with loading order index
            # write a formatted result two lines
            print('{:6}, {:10}, {:10}, {:10}, {:10}, {:12}, {:12}, {:10}, {:10}'.format('epoch','mse','mse_efs', 'psnr', 'psnr_efs', 'pcs_efs{}'.format(opt.pcs_test),'pcs_efs01','ssim', 'fp'))
            print('{:6}, {:10.5f}, {:10.5f}, {:10.5f}, {:10.5f}, {:12.5f}, {:12.5f}, {:10.5f}, {:10.5f}'.format(epoch, mse, mse_efs, psnr, psnr_efs, pcs_efs, pcs_efs01, ssim, fp))

            model.train()   # back to train, small test over
            print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
            model.update_learning_rate()                     # update learning rates at the end of every epoch.

    # ran a test session to the diff result--------
    if not os.path.exists(web_dir):  # check and make dir
        os.makedirs(web_dir)
    model.eval()

    if opt.if_saveWhtCmb == 'y':
        if_svWhtCmb = True
    else:
        if_svWhtCmb = False
    fake_vStk, real_vStk, mse, mse_efs, psnr, psnr_efs, pcs_efs, pcs_efs01, ssim, fp = utils_PM.test(model, dataset_test, opt_test.n_train, web_dir, opt_test.if_saveImg, R=rg_sensing, efs_rt=opt.efs_rt, pcs_test=opt.pcs_test, if_svWhtCmb=if_svWhtCmb)
    # for fast check , num_test is moore accurate
    # result
    print('{:6} {:10} {:10} {:10} {:10} {:10} {:10} {:10} {:10}'.format('epoch','mse','mse_efs', 'psnr', 'psnr_efs', 'pcs_efs{}'.format(opt.pcs_test),'pcs_efs01','ssim', 'fp'))
    print('{:6} {:10.5f} {:10.5f} {:10.5f} {:10.5f} {:10.5f} {:10.5f} {:10.5f} {:10.5f}'.format(epoch, mse, mse_efs, psnr, psnr_efs, pcs_efs, pcs_efs01, ssim, fp))
    # save all the result
    np.savez(os.path.join(web_dir, 'rst{}.npz'.format(opt_test.efs_rt)), mse=mse, mse_efs=mse_efs, psnr=psnr, psnr_efs=psnr_efs, pcs_efs=pcs_efs, pcs_efs01=pcs_efs01, ssim=ssim, fp=fp)
    if opt_test.if_saveDiff == 'y':    # save the numpy format
        np.savez(os.path.join(web_dir, opt.predNm), fake_vStk=fake_vStk, real_vStk=real_vStk)

