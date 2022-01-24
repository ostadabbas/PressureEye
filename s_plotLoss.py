'''
plot the loss function of log file
'''
import matplotlib.pyplot as plt
import re
import os

## find a specific string test
# strT = '(epoch: 3, iters: 80, time: 0.191, data: 0.022) G_L: 2.334'
# m = re.search('G_L: (\d+\.\d+)', strT)
# print(m.group(1))       # is a string

## plot from one local file
# pth = 'loss_log.txt'
# loss= []
# with open(pth) as f:
#     _ =f.readline()     # get rid of first line
#     for line in f:
#         m = re.search('G_L: (\d+\.\d+)', line)
#         loss.append(float(m.group(1)))
# print('loss len is', len(loss))
# plt.plot(loss)
# plt.show()

## func to deploy


rstFd = 'checkpoints'
rst_li = [
'vis2PM_exp_uc_RGB-2-PM_clip01_w-align_n_phy1_3stg_woactiFn_100.0L2_n-whtL100_100.0sum_0.0ssim_nD3_epoch25_5_bch30',
'vis2PM_exp_uc_RGB-2-PM_clip01_w-align_n_phy1_3stg_woactiFn_100.0L2_auto-whtL100_100.0sum_0.0ssim_nD3_epoch25_5_bch30'       # sa
#     'test'
]
labels = ['base',
          'sa',
          ]
loss_nms = ['G_L',
            'sum_L']
len_loss = len(loss_nms)
outFd = 'results/lossPlots'
outNm = 'baseAuto.png'
if not os.path.exists(outFd):
    os.makedirs(outFd)

loss_total = []   # for total result list
if_xed = False
for i, rstNm in enumerate(rst_li):
    pth = os.path.join(rstFd, rstNm, 'loss_log.txt')
    # lossT = []        # temp
    with open(pth) as f:
        _ = f.readline()
        lossT = []  # temp
        for j in range(len_loss):
            lossT.append([])  # give empty ones
        for line in f:
            for j, loss_nm in enumerate(loss_nms):
                m = re.search(loss_nm+': (\d+\.\d+)', line)
                lossT[j].append(float(m.group(1)))
        if not if_xed:
            x = list(range(1*10, (len(lossT[0])+1)*10, 10))
            if_xed = True
        for j in range(len_loss):
            plt.plot(x, lossT[j], label=labels[i]+'_'+loss_nm)     # plot on

    loss_total.append(lossT)

plt.legend()
plt.xlabel('iteration')
plt.ylabel('loss')
plt.savefig(os.path.join(outFd,outNm))
# plt.plot(loss)
# plt.show()