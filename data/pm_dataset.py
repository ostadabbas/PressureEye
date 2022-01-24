from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import cv2
from util import utils_PM
import numpy as np
import os
import torch
import skimage
from skimage import transform
import random
import math
import skimage.io as io
import matplotlib.pyplot as plt
import skimage
from scipy.ndimage import gaussian_filter
import json

def genPTr_dict(subj_li, mod_li, dsFd=r'G:\My Drive\ACLab Shared GDrive\datasetPM\danaLab'):
    '''
    loop idx_li, loop mod_li then generate dictionary {mod[0]:PTr_li[...], mod[1]:PTr_li[...]}
    :param subj_li:
    :param mod_li:
    :return:
    '''
    PTr_dct_li_src = {}  # a dict
    for modNm in mod_li:        # initialize the dict_li
        if not 'PM' in modNm:   # all PM immue
            PTr_dct_li_src[modNm] = []  # make empty list  {md:[], md2:[]...}
    for i in subj_li:
        for mod in mod_li:  # add mod PTr
            if not 'PM' in mod:  # not PTr for 'PM'
                if 'IR' in mod:
                    nm_mod_PTr = 'IR'
                elif 'depth' in mod:
                    nm_mod_PTr = 'depth'
                elif 'RGB' in mod:
                    nm_mod_PTr = 'RGB'
                else:
                    print('no such modality', mod)
                    exit(-1)
                pth_PTr = os.path.join(dsFd, '{:05d}'.format(i + 1), 'align_PTr_{}.npy'.format(nm_mod_PTr))
                PTr = np.load(pth_PTr)
                PTr_dct_li_src[mod].append(PTr)
    return PTr_dct_li_src

class PmDataset(BaseDataset):
    """This dataset class can load a set of images specified by the path --dataroot /path/to/data.

    It can be used for generating CycleGAN results only for one side with the model option '-model test'.
    """
    def __init__(self, opt):
        """Initialize this dataset class.
        only pth_desc_li relative length, all inside is abs index from 1
        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        # hardwired separations
        n_subj = 102    # total subjects
        n_frm = 45      # each cover 45 images
        self.h = 192    # std PM size, everything to PM, keep as it is
        self.w = 84
        self.rg_shf =(-25,25)   # the range of allowed shift
        self.rg_rot = (-45,45)    # allowed rotation range
        self.rg_scale =(1, opt.crop_size/self.h)  # random scale but largest almost full fit
        self.roomH = 2600    # in mm, about 2.6m, this number can be achieved from the dataset statistically such histogram, should be peak.
        dsFd  = opt.dataroot
        self.dsFd = dsFd
        self.stdSz = opt.crop_size  # 256
        self.mod_src = opt.mod_src
        self.mod_tar = opt.mod_tar
        self.kdeMode = opt.kdeMode
        self.sig_kde = opt.sig_kde      #kde sigma
        self.n_subj = n_subj    # total number of subject in the set
        self.depth_max = 2127   # stands for the bed height 2101 in SLP
        self.span_depth = 2127 -1891  # depth max - min
        self.PMmax = 94     # 94 kpa to normalize PM
        self.nbin_pwrs = 100     # to 100 be better?
        if opt.if_align == 'w':
            self.if_align = True        # use bool but not charater 'w'
        else:
            self.if_align = False
        if 'w' == opt.if_normPhy:
            self.if_normPhy = True
        else:
            self.if_normPhy = False
        self.normVec = np.ones([10])*100        # norm vec all 100 except gender 1
        self.normVec[2]=1       # all normalized to around 1 but only the gender as 1 directly

        if 'train' == opt.phase:
            idx_subj = range(90)   # to load index should add one as folder start from 1
        else:
            idx_subj = range(90, min(90+opt.n_testPM, n_subj))    # max 12 test, choose 5 for quick results
        idx_subj_all = range(n_subj)        # all index
        # all set PTr
        self.PTr_dct_li_src = genPTr_dict(idx_subj_all, opt.mod_src, opt.dataroot)  # could be empty dict for tar {mod:PTr_li, ...}
        self.PTr_dct_li_tar = genPTr_dict(idx_subj_all, opt.mod_tar, opt.dataroot)
        self.PTr_dct_li = genPTr_dict(idx_subj_all, opt.mod_src + opt.mod_tar, opt.dataroot)    # keep all PTr dict, can be used accordingly
        phys_arr = np.load(os.path.join(opt.dataroot, 'physiqueData.npy'))  # physic list of the references
        # phys_arr[:,2], phys_arr[:,0] = phys_arr[:,0], phys_arr[:, 2] # swap weight to first -->  w, h, gender, ...
        phys_arr[:, [2, 0]] = phys_arr[:, [0, 2]]       # swap  weight first
        self.phys_arr = phys_arr.astype(np.float)    # all set ori gender height weight
        # for caliPM_li
        # get the h_hist
        with open('misc/hist_pwrs100.json', 'r') as f:
            rst = json.load(f)
        self.hist_ave = np.array(rst['hist_ave'])

        caliPM_li = []
        for i in idx_subj_all:
            pth_cali = os.path.join(dsFd,'{:05d}'.format(i+1), 'PMcali.npy')
            caliPM = np.load(pth_cali)
            caliPM_li.append(caliPM)
        self.caliPM_li = caliPM_li  # all cali files in order
        # gen the descriptor list   [[ i_subj,   cov,  i_frm  ]]
        pthDesc_li = []     # pth descriptor, make abs from 1
        for i in idx_subj:
            for cov in opt.cov_li:  # add pth Descriptor
                for j in range(n_frm):
                    pthDesc_li.append([i+1,cov,j+1])
        self.pthDesc_li = pthDesc_li

    def makeDummyPth(self, index, mod_li):
        '''
        makea dummy path. Path is only employed for saving purpose
        <idx_subj>_<cov>_<idx_frm>_<mod_li>
        :param index:
        :param mod_li:
        :return:
        '''
        pth_desc = self.pthDesc_li[index]  # all index from 0
        return '_'.join(str(e) for e in pth_desc + mod_li)

    def getImg(self, index, mod_li, augParam={'scale':1, 'deg':0, 'shf':(0,0)}):
        '''
        according to index, get in image, according to mod, do basic process, if there is transform apply the transform to the image. Return stacked tensor. shift and rotate according to parameters. Preprocess, depth depends on mode, PM preserve original.
        :param img:
        :param mod:
        :return: concated tensor of all modalities
        '''
        # for A gen
        cov_dict={'uncover': 0, 'cover1': 1, 'cover2': 2}

        idx_subj, cov, idx_frm = self.pthDesc_li[index]  # all index from 1, abs coord
        stdSz = self.stdSz
        tsCat = torch.Tensor([])
        scal_caliPM = self.caliPM_li[idx_subj-1][cov_dict[cov], idx_frm-1]
        for mod in mod_li:
            img = utils_PM.getImg_dsPM(self.dsFd, idx_subj, mod, cov, idx_frm)  # interface literally from idx 1
            if np.uint8 == img.dtype: # im image type
                img = skimage.img_as_float(img) # for RGB version, should be 0 ~ 1
            if not 'PM' in mod:  # homography align not PM
                # PTr = self.PTr_dct_li_src[mod][idx_subj-1]  # from 0
                PTr = self.PTr_dct_li[mod][idx_subj - 1]  # use combined version
                persTran = transform.ProjectiveTransform(np.linalg.inv(PTr))   # ski transform
                if self.if_align:   # align, other wise simply rescale to std_z, as vision has larger margin
                    img = transform.warp(img, persTran, output_shape=(self.h,self.w), preserve_range=True ) #   value nearly 0 , perhaps position issue?   align operation 192 x 84 standard
                else:   # no warpping will be larger to stdZ here
                    h, w = img.shape[:2]    # all tall image, use h as std
                    w_new = math.floor(stdSz/h *w)
                    img = transform.resize(img, [stdSz, w_new])
            # normalize image around 0 and 1 -- original read in
            if 'depthRaw' == mod:   # discard later
                if self.opt.depthPreproc == 'normalize':
                    img = (-img+self.depth_max)/self.span_depth     # how much above bed
                    img[img > 3] = 0
                    img[img < -3] = 0     # outsider 3 span away  assign to base
            # print('img sum', img.sum())
            # print("cali scalar is", scal_caliPM)
            if 'PMarray' == mod:
                img = img * scal_caliPM # scale to size
                img = img/self.PMmax # roughly to -1 to 1 first
                # print('after scale sum',img.sum())
            if 'clip11' == self.opt.pmDsProc:
                img = img * 2 - 1  # roughly to -1 and 1
                img = img.clip(-1,1)
            elif 'clip01' == self.opt.pmDsProc:
                img = img.clip(0,1)  # clip to 0 and 1

            # mapping operation
            n_dim = len(img.shape)
            h,w = img.shape[:2]
            r_st = (stdSz-h)//2
            c_st = (stdSz-w)//2
            # pad to std
            if n_dim == 3:    # for RGB
                c = img.shape[2]  # last dimension
                img_new = np.zeros((self.stdSz, self.stdSz, c))     # just 0
            else:
                # img_new = np.zeros((self.stdSz, self.stdSz)) + np.quantile(img, 0.1)
                img_new = np.zeros((self.stdSz, self.stdSz))   # add min as bg
            img_new[r_st:r_st+h,c_st:c_st+w] = img # copy image in, should be float already
            # print('after image fill', img_new.sum())
            img_new = utils_PM.affineImg(img_new, augParam['scale'], augParam['deg'], augParam['shf'])      # change the total value
            # print('after scaling', img_new.sum())
            # to tensor

            ts_tmp = torch.Tensor(img_new)      # torch tensor will change to float32 from float64
            if len(ts_tmp.size())<3:    # no 3rd dimension
                ts_tmp.unsqueeze_(0) # put ch at 0   unsqueeze no working??
            else:   # if the image has channel dim,  swap
                ts_tmp = ts_tmp.permute((2, 0, 1)) # channel first first RGB, no inplace !!

            tsCat = torch.cat((tsCat, ts_tmp)) # keep cat along channel

        # print('readin image is', img.dtype)
        # print('new image is', img_new.dtype)
        # print('the tensor image is', tsCat.dtype)
        return tsCat

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A and A_paths
            A(tensor) - - an image in one domain
            A_paths(str) - - the path of the image
        """
        idx_subj, cov, idx_frm = self.pthDesc_li[index]  # all index from 0
        proc_str = self.opt.preprocess
        if_train = 'train'==self.opt.phase
        # idx_subj, cov, idx_frm = self.pthDesc_li[index]  # all index from 0
        if 'scale' in proc_str and if_train:    # not for test
            scale = random.uniform(*self.rg_scale)
        else:
            scale = self.rg_scale[1]    # the fix scale to the fullfill
        if 'rotate' in proc_str and if_train:
            deg = random.uniform(*self.rg_rot)
        else:
            deg = 0
        if 'shift' in proc_str and if_train:
            x_shf = random.uniform(*self.rg_shf)
            y_shf = random.uniform(*self.rg_shf)
            shf = (x_shf, y_shf)
        else:
            shf = (0, 0)        # don't move
        # augParam={'scale':1.2, 'deg':30, 'shf':(50,50)}
        augParam = {'scale': scale, 'deg': deg, 'shf': shf}
        A = self.getImg(index, self.opt.mod_src, augParam)
        B = self.getImg(index, self.opt.mod_tar, augParam) # B will be 1 x Mx N tensor for pm
        # get weight    for B only
        nbin = self.nbin_pwrs
        B_np = B.numpy()* nbin       # 0 ~ 1 so map to 50
        B_np = B_np.astype(int)         # is float to int good?
        wt = np.ones_like(B_np, dtype=np.float) # keep rare case (0 case)  1 , otherwise just 1/hist    float 32
        bins = np.arange(nbin+2)-0.5        # put in center, fix bin mth, ( easy to map to value), arange integer?
        h_mode = self.opt.h_mode
        h_base = self.opt.h_base
        if 0 == h_mode:
            hist, bins_o = np.histogram(B_np.flatten(), bins)   # each bin a
        else:
            hist = self.hist_ave + h_base
        if 2 == h_mode:
            hist = gaussian_filter(hist, sigma=1)

        # plt.bar(bins_o[:-1]+0.5, hist, width=0.7)
        # plt.show()    # hist correct
        # for i in range(nbin):
        #     if hist[i] >0 :
        #         wt[B_np==i] = 1./hist[i]
        shp = wt.shape
        for i in range(shp[1]): # loop to update        # 0 ~ 50  so 51 entries, 52
            for j in range(shp[2]):
                v_p = B_np[0][i][j]
                if hist[v_p]>0:
                    wt[0][i][j] = 1./hist[v_p] + self.opt.lambda_lap        # either 1  or add lambda_lap to it.  test auto W first.

        # skimage.io.imshow(wt.squeeze())
        # plt.show()

        # check kde mode, 0 nothing < 0 a
        wt_rst = wt.copy()      # raw pwrs with lambda
        if self.kdeMode>0:      # stacked
            sig = self.sig_kde
            for i in range(self.kdeMode):
                wt_t = gaussian_filter(wt, sig)
                wt_rst += wt_t
                sig = sig * 2   # scaling up the rf
        elif self.kdeMode<0:
            sig = self.sig_kde*(-self.kdeMode)
            wt_rst = gaussian_filter(wt, sig)


        # image path only for name saving purpose
        A_paths = self.makeDummyPth(index, self.opt.mod_src)
        B_paths = self.makeDummyPth(index, self.opt.mod_tar)
        phyVec_np = self.phys_arr[idx_subj-1,:]

        # print('phy vec raw 0 is', phyVec_np[0])
        if self.if_normPhy:
            phyVec_np = phyVec_np/self.normVec
        # phyVec = torch.Tensor(self.phys_arr[idx_subj-1,:])   # one vec
        # print('phyVec is', phyVec_np)
        phyVec = torch.Tensor(phyVec_np)
        # wt = torch.Tensor(wt)
        wt_rst = torch.Tensor(wt_rst)

        return {'A':A, 'B':B, 'A_paths':A_paths, 'B_paths':B_paths, 'phyVec': phyVec,'wt_pwrs': wt_rst}

    def __len__(self):
        """Return the total number of images in the dataset."""
        # return len(self.A_paths)
        return len(self.pthDesc_li)