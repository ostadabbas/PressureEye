'''
rename demo images of V1 with new naming rule of multi stage version with demo<idx>_fake<nStg>.png
'''
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--rstFd', default='results/cycle_gan_rst_test', help='input result folder to be rename with new naming rule.Default one is empty')
parser.add_argument('--nStg', default='2', help='the default stage in string format, default 2 stage')
args = parser.parse_args()
imgFd = os.path.join(args.rstFd, 'test_latest', 'demoImgs')
str_nStg = args.nStg
# csvFd = r'G:\My Drive\ACLab Shared GDrive\datasetPM\danaLab\00001\PMcsv'

ls_fNm = os.listdir(imgFd)
for fNm in ls_fNm:
    # if fNm.endswith('.csv') and '_F' not in fNm:
    #     os.rename(os.path.join(csvFd,fNm), os.path.join(csvFd,
#  fNm[:-4]+'_F'+'.csv'))
    if fNm.endswith('.png'):
        # eg:  demo_real_A_0.png
        # to:  demo0_real_A.png
        #      demo0_fake_B2.png
        # print(fNm)       #  only base name with ext here
        demo, strFR, strDm, strIdx = fNm.split('_')
        # print(demo,strFR, strDm, strIdx)
        idx = int(strIdx.split('.')[0])
        # print(demo, strFR, strDm, idx)
        if 'fake' in fNm:
            str_stgT = str_nStg
        else:
            str_stgT = ''

        nm_tar = '{}{}_{}_{}{}.png'.format(demo, idx, strFR, strDm, str_stgT)
        os.rename(os.path.join(imgFd, fNm), os.path.join(imgFd, nm_tar))

