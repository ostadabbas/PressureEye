# rename files in a folder according to certain pattern
import os
tarFd = 'pmScripts_depthRaw'

ls_fNm = os.listdir(tarFd)
for i, fNm in enumerate(ls_fNm):
    # print(fNm)
    # print('after rename')
    fNm_new = fNm[4:]
    # print(i, fNm_new)
    # if fNm.endswith('.csv') and '_F' not in fNm:
    os.rename(os.path.join(tarFd, fNm), os.path.join(tarFd,fNm_new))

